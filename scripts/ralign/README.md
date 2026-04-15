# R-Align reproduction (TRL + API Meta-Judge)

Minimal TRL implementation of R-Align (arXiv:2602.06763) sized for 4x RTX 3090
(24GB each). Upstream repo (`lyn22333/R-Align`) only ships evaluation code;
the training pipeline below is reconstructed from the paper and uses an
OpenAI-compatible API (Gemini-3-Pro recommended) as the Meta-Judge.

## Files

- `prepare_data.py`       download `lyn22333/R-Align-RL-Data`, build SFT + GRPO splits
- `ralign_reward.py`      async reward: label check + Meta-Judge API call
- `merge_adapter.py`      merge a QLoRA adapter back into the base (bf16)
- `eval_benchmarks.py`    eval on RewardBench / JudgeBench / R-Align-BMK via vLLM
- `accelerate_4x3090.yaml` DeepSpeed ZeRO-2, 4 processes, bf16
- `run_sft.sh`            Stage 1: SFT warmup on Golden Rationales (QLoRA)
- `run_grpo.sh`           Stage 2: GRPO with R-Align reward (QLoRA + vLLM colocate)
- `run_eval.sh`           wrapper for eval_benchmarks.py

## Quickstart

```bash
# 1. Install extras (if not already)
pip install peft bitsandbytes httpx vllm

# 2. Build datasets (saves to ./data/ralign/{sft,grpo})
python scripts/ralign/prepare_data.py --output_dir ./data/ralign

# 3. Stage 1 — SFT warmup
bash scripts/ralign/run_sft.sh

# 4. Merge SFT adapter into base so GRPO starts from a clean checkpoint
python scripts/ralign/merge_adapter.py \
    --adapter_dir ./outputs/ralign-sft \
    --output_dir  ./outputs/ralign-sft-merged

# 5. Stage 2 — R-Align GRPO (set API credentials first)
export METAJUDGE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
export METAJUDGE_API_KEY=...
export METAJUDGE_MODEL=gemini-3-pro-thinking
MODEL=./outputs/ralign-sft-merged bash scripts/ralign/run_grpo.sh

# 6. After GRPO, merge the GRPO adapter for vLLM serving / evaluation
python scripts/ralign/merge_adapter.py \
    --adapter_dir ./outputs/ralign-grpo \
    --output_dir  ./outputs/ralign-final
```

## Notes for 4x3090

- Default base model is `Qwen3-4B-Instruct`. Qwen3-8B is feasible with QLoRA
  but GRPO rollout is slow; 14B will not fit. Override with `MODEL=...`.
- `vllm_mode colocate` shares GPUs between training and rollout — required
  here because 3090 clusters rarely have spare cards for a dedicated server.
- Meta-Judge cost: for each rollout, only samples whose predicted label
  matches `gt_label` are sent to the API (label-gated reward). Expect ~50%
  of rollouts to hit the API after warmup. Monitor spend.
- If you hit OOM, lower `--max_prompt_length` to 2048 or drop `num_generations`
  to 2.

## Evaluation

Standalone evaluation (no API required for label accuracy):

```bash
MODEL=./outputs/ralign-final TP=4 bash scripts/ralign/run_eval.sh
```

Benchmarks default to `reward_bench reward_bench_2 judge_bench`. Per-example
predictions go to `eval_out/<bench>/predictions.jsonl`; aggregated metrics
(label accuracy overall + per category) go to `eval_out/summary.json`.

For rationale-aware metrics (F-Score / S-Corr on `lyn22333/R-Align-BMK`),
enable the Meta-Judge API:

```bash
export METAJUDGE_BASE_URL=... METAJUDGE_API_KEY=... METAJUDGE_MODEL=gemini-3-pro-thinking
RATIONALE_AWARE=1 MODEL=./outputs/ralign-final bash scripts/ralign/run_eval.sh
```

Expected after R-Align: **F-Score up**, **S-Corr down** vs. the SFT-only
baseline, while Label Accuracy stays roughly flat — the central claim of
the paper. You can also point the original `lyn22333/R-Align` eval repo at
your vLLM-served model for a second opinion.
