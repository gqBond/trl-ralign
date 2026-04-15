"""
Merge a QLoRA adapter back into its base model to produce a standalone
bf16 checkpoint suitable as the base for the next training stage or for
vLLM serving.

Notes for QLoRA:
    Merging a bf16 LoRA adapter into a 4-bit quantized base would be
    lossy. We dequantize by loading the base in bf16 (not 4-bit) first,
    then apply + merge the adapter on top.

Usage:
    python scripts/ralign/merge_adapter.py \
        --adapter_dir ./outputs/ralign-sft \
        --output_dir  ./outputs/ralign-sft-merged

    # base model is read from the adapter's adapter_config.json; override
    # with --base_model if needed.
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default=None,
                        help="Override base model; defaults to adapter_config.json's base_model_name_or_path.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    base_id = args.base_model
    if base_id is None:
        from peft import PeftConfig
        base_id = PeftConfig.from_pretrained(args.adapter_dir).base_model_name_or_path
    print(f"Loading base model: {base_id} (dtype={args.dtype})")

    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype, device_map=args.device_map)

    print(f"Applying adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, args.adapter_dir)

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model -> {out}")
    model.save_pretrained(out, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    tokenizer.save_pretrained(out)
    print("Done.")


if __name__ == "__main__":
    main()
