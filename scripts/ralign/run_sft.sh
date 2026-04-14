#!/usr/bin/env bash
# Stage 1: SFT warmup on Golden Rationales.
# 4x3090 (24GB each) + QLoRA + DeepSpeed ZeRO-2.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."

MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct}
DATA=${DATA:-./data/ralign/sft}
OUTPUT=${OUTPUT:-./outputs/ralign-sft}

accelerate launch \
  --config_file "$HERE/accelerate_4x3090.yaml" \
  trl/scripts/sft.py \
    --model_name_or_path "$MODEL" \
    --dataset_name "$DATA" \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --max_length 4096 \
    --bf16 \
    --use_peft \
    --load_in_4bit \
    --lora_r 32 --lora_alpha 64 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --logging_steps 10 \
    --save_steps 500 \
    --output_dir "$OUTPUT"
