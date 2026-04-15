"""Merge base Llama model + LoRA adapter into a separate standalone model folder.

This script keeps original adapter files untouched and writes merged weights to
an output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--adapter",
        default="model/simplifier-4090",
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output",
        default="model/simplifier-merged",
        help="Directory to write merged model",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Optional base model override (otherwise read from adapter_config.json)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Dtype used while loading base model before merge",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Device placement for merge load. Use cpu to avoid offload-related merge issues.",
    )
    parser.add_argument(
        "--offload-dir",
        default="model/simplifier-merge-offload",
        help="Directory for Accelerate/PEFT disk offload files if needed.",
    )
    return parser.parse_args()


def dtype_from_name(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def read_base_model_name(adapter_path: Path) -> str:
    cfg = adapter_path / "adapter_config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    data = json.loads(cfg.read_text(encoding="utf-8"))
    base = data.get("base_model_name_or_path")
    if not base:
        raise ValueError("base_model_name_or_path missing in adapter_config.json")
    return str(base)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    adapter_path = (root / args.adapter).resolve()
    output_path = (root / args.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    offload_path = (root / args.offload_dir).resolve()
    offload_path.mkdir(parents=True, exist_ok=True)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    base_model_name = args.base or read_base_model_name(adapter_path)
    dtype = dtype_from_name(args.dtype)

    print(f"Adapter path : {adapter_path}")
    print(f"Output path  : {output_path}")
    print(f"Offload path : {offload_path}")
    print(f"Base model   : {base_model_name}")
    print(f"Load dtype   : {dtype}")
    print(f"Load device  : {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.device == "cpu":
        # Use a non-offloaded path to avoid meta/offload key mismatches during PEFT load.
        device_map = None
        low_cpu_mem_usage = False
    elif args.device == "cuda":
        device_map = {"": 0}
        low_cpu_mem_usage = True
    else:
        device_map = "auto"
        low_cpu_mem_usage = True

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        low_cpu_mem_usage=False,
        device_map=device_map,
        offload_dir=str(offload_path),
    )
    merged_model = model.merge_and_unload()

    merged_model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(str(output_path))

    print("Merge complete.")
    print(f"Merged model saved to: {output_path}")


if __name__ == "__main__":
    main()
