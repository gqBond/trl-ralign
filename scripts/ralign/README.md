# R-Align reproduction (TRL + API Meta-Judge)

Minimal TRL implementation of R-Align (arXiv:2602.06763) sized for 4x RTX 3090
(24GB each). Upstream repo (`lyn22333/R-Align`) only ships evaluation code;
the training pipeline below is reconstructed from the paper and uses an
OpenAI-compatible API (Gemini-3-Pro recommended) as the Meta-Judge.

## Files

- `prepare_data.py`       download `lyn22333/R-Align-RL-Data`, build SFT + GRPO splits
- `ralign_reward.py`      async reward: label check + Meta-Judge API call
- `accelerate_4x3090.yaml` DeepSpeed ZeRO-2, 4 processes, bf16
- `run_sft.sh`            Stage 1: SFT warmup on Golden Rationales (QLoRA)
- `run_grpo.sh`           Stage 2: GRPO with R-Align reward (QLoRA + vLLM colocate)

## Quickstart

```bash
# 1. Install extras (if not already)
pip install peft bitsandbytes httpx vllm

# 2. Build datasets (saves to ./data/ralign/{sft,grpo})
python scripts/ralign/prepare_data.py --output_dir ./data/ralign

# 3. Stage 1 — SFT warmup
bash scripts/ralign/run_sft.sh

# 4. Stage 2 — R-Align GRPO (set API credentials first)
export METAJUDGE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
export METAJUDGE_API_KEY=...
export METAJUDGE_MODEL=gemini-3-pro-thinking
bash scripts/ralign/run_grpo.sh
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

After training, score the model on the Rationale-Aware Benchmark using the
original repo:

```bash
git clone https://github.com/lyn22333/R-Align ./third_party/R-Align
# serve the trained model via vLLM (OpenAI-compatible), then point
# gen_base_url / gen_api_key / gen_model in third_party/R-Align/run.sh
# at your endpoint and run:
bash third_party/R-Align/run.sh
python third_party/R-Align/show_result.py
```

Expected: **F-Score up**, **S-Corr down** vs. the SFT-only baseline, while
Label Accuracy stays roughly flat — the central claim of the paper.
