"""
Download lyn22333/R-Align-RL-Data and convert it into two datasets:

1. SFT warmup dataset (conversational `messages` format):
       system + user(prompt+A+B) -> assistant(rationale + \boxed{label})
2. GRPO dataset (conversational `prompt` format + extra columns):
       `prompt`: system + user(prompt+A+B)
       `gt_label`, `gt_judgment`: passed through to the reward function.

Both datasets are saved with `save_to_disk` so they can be loaded via
`datasets.load_from_disk(path)` and passed to TRL scripts.

Usage:
    python scripts/ralign/prepare_data.py --output_dir ./data/ralign
"""

import argparse
import random
from pathlib import Path

from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are an impartial judge. You will be given a user prompt and two "
    "candidate responses (A and B). Carefully analyse each response, then "
    "explain your reasoning step by step and finish with your final verdict "
    "on a new line in the exact format \\boxed{A} or \\boxed{B}."
)


def format_conversation(conversation):
    """Render a list of {role, content} turns into a readable string."""
    lines = []
    for turn in conversation:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


def build_user_prompt(conversation, response_a, response_b):
    if len(conversation) > 1:
        context = (
            "<|Dialogue Context|>\n"
            f"{format_conversation(conversation[:-1])}\n"
            "<|End of Dialogue Context|>\n\n"
        )
    else:
        context = ""
    last_user = conversation[-1].get("content", "") if conversation else ""
    return (
        f"{context}"
        f"<|User Prompt|>\n{last_user}\n\n"
        f"<|The Start of Assistant A's Answer|>\n{response_a}\n<|The End of Assistant A's Answer|>\n\n"
        f"<|The Start of Assistant B's Answer|>\n{response_b}\n<|The End of Assistant B's Answer|>\n"
    )


def make_example(row, swap_prob=0.5, seed=0):
    """Apply random A/B swap to reduce position bias, then build messages."""
    rng = random.Random(seed + hash(row.get("response_A", "")) % (2**31))
    swap = rng.random() < swap_prob

    a, b = row["response_A"], row["response_B"]
    label = row["gt_label"].strip().upper()
    if swap:
        a, b = b, a
        label = "B" if label == "A" else "A"

    user_prompt = build_user_prompt(row["conversations"], a, b)
    assistant_text = f"{row['gt_judgment'].strip()}\n\n\\boxed{{{label}}}"

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ],
        "gt_label": label,
        "gt_judgment": row["gt_judgment"].strip(),
        "response_A": a,
        "response_B": b,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lyn22333/R-Align-RL-Data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="./data/ralign")
    parser.add_argument("--eval_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    ds = ds.shuffle(seed=args.seed)

    ds = ds.map(
        make_example,
        fn_kwargs={"seed": args.seed},
        remove_columns=[c for c in ds.column_names if c not in ()],
        desc="Formatting R-Align data",
    )

    splits = ds.train_test_split(test_size=args.eval_size, seed=args.seed)
    out = Path(args.output_dir)

    sft_cols = ["messages"]
    grpo_cols = ["prompt", "gt_label", "gt_judgment", "response_A", "response_B"]

    splits_sft = splits.remove_columns([c for c in splits["train"].column_names if c not in sft_cols])
    splits_grpo = splits.remove_columns([c for c in splits["train"].column_names if c not in grpo_cols])

    (out / "sft").mkdir(parents=True, exist_ok=True)
    (out / "grpo").mkdir(parents=True, exist_ok=True)
    splits_sft.save_to_disk(out / "sft")
    splits_grpo.save_to_disk(out / "grpo")

    print(f"Saved SFT dataset   -> {out / 'sft'}  (train={len(splits_sft['train'])}, test={len(splits_sft['test'])})")
    print(f"Saved GRPO dataset  -> {out / 'grpo'} (train={len(splits_grpo['train'])}, test={len(splits_grpo['test'])})")


if __name__ == "__main__":
    main()
