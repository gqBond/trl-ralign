"""
Evaluate a trained generative judge on standard preference benchmarks.

Supports:
    - reward_bench    : allenai/reward-bench
    - reward_bench_2  : allenai/reward-bench-2
    - judge_bench     : ScalerLab/JudgeBench
    - ralign_bmk      : lyn22333/R-Align-BMK (rationale-aware; F-Score/S-Corr
                        requires METAJUDGE_* env vars, reuses ralign_reward)

For each example: format as (prompt, response_A, response_B), generate with
vLLM, parse \\boxed{A|B}, compare to ground truth.

Output:
    <output_dir>/<bench>/predictions.jsonl   one row per example
    <output_dir>/summary.json                aggregated per-benchmark accuracy

Usage:
    python scripts/ralign/eval_benchmarks.py \
        --model ./outputs/ralign-final \
        --benchmarks reward_bench judge_bench \
        --output_dir ./eval_out \
        --tensor_parallel_size 4

Rationale-aware mode (F-Score on R-Align-BMK):
    METAJUDGE_BASE_URL=... METAJUDGE_API_KEY=... METAJUDGE_MODEL=... \
    python scripts/ralign/eval_benchmarks.py --model ... \
        --benchmarks ralign_bmk --rationale_aware
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "You are an impartial judge. You will be given a user prompt and two "
    "candidate responses (A and B). Carefully analyse each response, then "
    "explain your reasoning step by step and finish with your final verdict "
    "on a new line in the exact format \\boxed{A} or \\boxed{B}."
)

LABEL_RE = re.compile(r"\\?boxed\{\s*([AB])\s*\}", re.IGNORECASE)


def extract_label(text: str) -> str | None:
    m = LABEL_RE.search(text or "")
    return m.group(1).upper() if m else None


def build_user_prompt(prompt_text: str, a: str, b: str) -> str:
    return (
        f"<|User Prompt|>\n{prompt_text}\n\n"
        f"<|The Start of Assistant A's Answer|>\n{a}\n<|The End of Assistant A's Answer|>\n\n"
        f"<|The Start of Assistant B's Answer|>\n{b}\n<|The End of Assistant B's Answer|>\n"
    )


# ----------------------------------------------------------------------------
# Benchmark loaders — normalize each row to:
#   {prompt, response_A, response_B, gt_label, category, golden_rationale?}
# ----------------------------------------------------------------------------


def _load_reward_bench(split: str = "filtered"):
    ds = load_dataset("allenai/reward-bench", split=split)
    rng = random.Random(0)
    for row in ds:
        swap = rng.random() < 0.5
        a, b = (row["rejected"], row["chosen"]) if swap else (row["chosen"], row["rejected"])
        gt = "B" if swap else "A"
        yield {
            "prompt": row["prompt"],
            "response_A": a, "response_B": b,
            "gt_label": gt,
            "category": row.get("subset", "unknown"),
        }


def _load_reward_bench_2(split: str = "train"):
    ds = load_dataset("allenai/reward-bench-2", split=split)
    rng = random.Random(0)
    for row in ds:
        # schema: chosen (str), rejected (list[str]) — evaluate against first rejected
        chosen = row["chosen"]
        rejected = row["rejected"][0] if isinstance(row["rejected"], list) else row["rejected"]
        swap = rng.random() < 0.5
        a, b = (rejected, chosen) if swap else (chosen, rejected)
        gt = "B" if swap else "A"
        yield {
            "prompt": row["prompt"],
            "response_A": a, "response_B": b,
            "gt_label": gt,
            "category": row.get("subset", "unknown"),
        }


def _normalize_pairwise_label(raw: str) -> str | None:
    """Map labels like 'A>B', 'B>A', 'A', 'B' to the winning letter."""
    s = (raw or "").strip().upper()
    if ">" in s:
        return s.split(">", 1)[0].strip() or None
    if "<" in s:
        return s.split("<", 1)[1].strip() or None
    if s in ("A", "B"):
        return s
    return None


def _load_judge_bench(split: str = "test"):
    ds = load_dataset("ScalerLab/JudgeBench", split=split)
    for row in ds:
        # schema: question, response_A, response_B, label ("A>B"/"B>A"), source/category
        label = _normalize_pairwise_label(row.get("label") or row.get("winner"))
        if label is None:
            continue  # skip ties or unparseable rows
        yield {
            "prompt": row.get("question") or row.get("prompt"),
            "response_A": row.get("response_A") or row.get("answer_A"),
            "response_B": row.get("response_B") or row.get("answer_B"),
            "gt_label": label,
            "category": row.get("category") or row.get("source", "unknown"),
        }


def _load_ralign_bmk():
    ds = load_dataset("lyn22333/R-Align-BMK", split="train")
    for row in ds:
        last_user = ""
        convs = row.get("conversations") or []
        if convs:
            last_user = convs[-1].get("content", "")
        yield {
            "prompt": last_user,
            "response_A": row["response_A"],
            "response_B": row["response_B"],
            "gt_label": str(row["gt_label"]).strip().upper(),
            "category": row.get("source", "ralign"),
            "golden_rationale": row.get("gt_judgment", ""),
        }


LOADERS = {
    "reward_bench": _load_reward_bench,
    "reward_bench_2": _load_reward_bench_2,
    "judge_bench": _load_judge_bench,
    "ralign_bmk": _load_ralign_bmk,
}


# ----------------------------------------------------------------------------
# Meta-Judge (for F-Score on ralign_bmk)
# ----------------------------------------------------------------------------


async def _rationale_verdicts(examples, generations):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ralign_reward import META_JUDGE_PROMPT, _call_meta_judge, _parse_verdict  # noqa: E402
    import os, httpx  # noqa: E402

    base_url = os.environ["METAJUDGE_BASE_URL"]
    api_key = os.environ["METAJUDGE_API_KEY"]
    model = os.environ.get("METAJUDGE_MODEL", "gemini-3-pro-thinking")
    max_tokens = int(os.environ.get("METAJUDGE_MAX_TOKENS", "2048"))
    timeout = float(os.environ.get("METAJUDGE_TIMEOUT", "120"))
    retries = int(os.environ.get("METAJUDGE_RETRIES", "3"))

    sem = asyncio.Semaphore(int(os.environ.get("METAJUDGE_CONCURRENCY", "32")))

    async def one(ex, gen):
        if not ex.get("golden_rationale"):
            return None
        meta_prompt = META_JUDGE_PROMPT.format(
            context_and_responses=build_user_prompt(ex["prompt"], ex["response_A"], ex["response_B"]),
            golden_explanation=ex["golden_rationale"],
            genrm_explanation=gen,
        )
        async with sem:
            try:
                out = await _call_meta_judge(
                    client, base_url=base_url, api_key=api_key, model=model,
                    max_tokens=max_tokens, prompt=meta_prompt, retries=retries,
                )
                return _parse_verdict(out)
            except Exception:
                return False

    async with httpx.AsyncClient(timeout=timeout) as client:
        return await asyncio.gather(*[one(ex, g) for ex, g in zip(examples, generations)])


# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to merged model (or HF id).")
    parser.add_argument("--benchmarks", nargs="+", default=["reward_bench"],
                        choices=sorted(LOADERS.keys()))
    parser.add_argument("--output_dir", default="./eval_out")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Cap per-benchmark size (debug).")
    parser.add_argument("--rationale_aware", action="store_true",
                        help="Compute F-Score / S-Corr via Meta-Judge API (ralign_bmk only).")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    summary = {}
    for bench in args.benchmarks:
        print(f"\n=== {bench} ===")
        examples = list(LOADERS[bench]())
        if args.limit:
            examples = examples[: args.limit]

        prompts = []
        for ex in examples:
            user = build_user_prompt(ex["prompt"], ex["response_A"], ex["response_B"])
            prompts.append(tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            ))

        outputs = llm.generate(prompts, sampling)
        generations = [o.outputs[0].text for o in outputs]

        rationale_ok = None
        if args.rationale_aware and bench == "ralign_bmk":
            print("Running Meta-Judge for rationale fidelity...")
            rationale_ok = asyncio.run(_rationale_verdicts(examples, generations))

        rows = []
        n_label_ok = 0
        n_fidelity_ok = 0
        n_spurious = 0
        by_cat: dict[str, dict[str, int]] = {}
        for i, (ex, gen) in enumerate(zip(examples, generations)):
            pred = extract_label(gen)
            label_ok = pred is not None and pred == ex["gt_label"]
            fid = rationale_ok[i] if rationale_ok else None
            row = {
                "idx": i, "category": ex["category"], "gt_label": ex["gt_label"],
                "predicted_label": pred, "label_correct": label_ok,
                "generation": gen,
            }
            if fid is not None:
                row["rationale_correct"] = bool(fid)
            rows.append(row)
            n_label_ok += int(label_ok)
            if fid is not None:
                n_fidelity_ok += int(label_ok and fid)
                n_spurious += int(label_ok and not fid)
            cat = ex["category"]
            c = by_cat.setdefault(cat, {"n": 0, "label_ok": 0})
            c["n"] += 1
            c["label_ok"] += int(label_ok)

        bench_dir = out_root / bench
        bench_dir.mkdir(parents=True, exist_ok=True)
        with (bench_dir / "predictions.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        total = len(rows)
        bench_summary = {
            "n": total,
            "label_accuracy": n_label_ok / total if total else 0.0,
            "by_category": {
                cat: {"n": v["n"], "label_accuracy": v["label_ok"] / v["n"]}
                for cat, v in by_cat.items()
            },
        }
        if rationale_ok is not None:
            bench_summary["fidelity_score"] = n_fidelity_ok / total
            bench_summary["spurious_correctness"] = n_spurious / total
        summary[bench] = bench_summary

        print(f"  n={total}  L-Acc={bench_summary['label_accuracy']:.4f}", end="")
        if rationale_ok is not None:
            print(f"  F-Score={bench_summary['fidelity_score']:.4f}  S-Corr={bench_summary['spurious_correctness']:.4f}")
        else:
            print()

    with (out_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary -> {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
