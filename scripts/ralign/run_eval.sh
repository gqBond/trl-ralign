#!/usr/bin/env bash
# Evaluate a merged R-Align checkpoint on RewardBench, JudgeBench, and
# (optionally) the R-Align Rationale-Aware Benchmark.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."

MODEL=${MODEL:-./outputs/ralign-final}
OUTPUT=${OUTPUT:-./eval_out}
TP=${TP:-4}
BENCHMARKS=${BENCHMARKS:-"reward_bench reward_bench_2 judge_bench"}

extra=()
if [[ "${RATIONALE_AWARE:-0}" == "1" ]]; then
  : "${METAJUDGE_BASE_URL:?set METAJUDGE_BASE_URL for rationale-aware eval}"
  : "${METAJUDGE_API_KEY:?set METAJUDGE_API_KEY for rationale-aware eval}"
  extra+=(--rationale_aware)
  BENCHMARKS="$BENCHMARKS ralign_bmk"
fi

python scripts/ralign/eval_benchmarks.py \
  --model "$MODEL" \
  --benchmarks $BENCHMARKS \
  --output_dir "$OUTPUT" \
  --tensor_parallel_size "$TP" \
  --max_model_len 8192 \
  --max_tokens 1024 \
  "${extra[@]}"
