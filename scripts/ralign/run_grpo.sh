#!/usr/bin/env bash
# Stage 2: GRPO with Meta-Judge API reward.
# 4x3090 + vLLM colocate + QLoRA + DeepSpeed ZeRO-2.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."

MODEL=${MODEL:-./outputs/ralign-sft}
DATA=${DATA:-./data/ralign/grpo}
OUTPUT=${OUTPUT:-./outputs/ralign-grpo}

: "${METAJUDGE_BASE_URL:?set METAJUDGE_BASE_URL, e.g. https://generativelanguage.googleapis.com/v1beta/openai}"
: "${METAJUDGE_API_KEY:?set METAJUDGE_API_KEY}"
export METAJUDGE_MODEL=${METAJUDGE_MODEL:-gemini-3-pro-thinking}
export METAJUDGE_MAX_TOKENS=${METAJUDGE_MAX_TOKENS:-2048}

# Expose the custom reward module to dotted-path import.
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

accelerate launch \
  --config_file "$HERE/accelerate_4x3090.yaml" \
  trl/scripts/grpo.py \
    --model_name_or_path "$MODEL" \
    --dataset_name "$DATA" \
    --reward_funcs ralign_reward.r_align_reward_async \
    --learning_rate 5e-6 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_prompt_length 3072 \
    --max_completion_length 1024 \
    --use_vllm \
    --vllm_mode colocate \
    --bf16 \
    --gradient_checkpointing \
    --use_peft \
    --load_in_4bit \
    --lora_r 32 --lora_alpha 64 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --logging_steps 5 \
    --save_steps 200 \
    --output_dir "$OUTPUT"
