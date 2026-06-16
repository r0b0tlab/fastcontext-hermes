#!/usr/bin/env python3
"""
NVFP4 Quantization Script for FastContext-1.0-4B-RL

Model: microsoft/FastContext-1.0-4B-RL (Qwen3-4B-Instruct base, dense, BF16)
Tool:  NVIDIA Model Optimizer (nvidia-modelopt >= 0.44.0)
Config: NVFP4_DEFAULT_CFG (W4A4, group_size=16)

This is a standard dense Qwen3 transformer — no MoE, no multimodal.
Simplest NVFP4 quantization case in our pipeline.

Usage:
    # In a ModelOpt venv (nvidia-modelopt[hf]>=0.44.0, torch+cu130)
    python quantize_fastcontext_nvfp4.py \
        --model-path microsoft/FastContext-1.0-4B-RL \
        --output-dir /home/r0b0tdgx/models/llm/nvfp4/r0b0tlab/FastContext-1.0-4B-RL-NVFP4 \
        --calib-samples 512 \
        --batch-size 16 \
        --seq-len 1024

Expected output: ~2.4 GB NVFP4 checkpoint (3.3× compression from ~8 GB BF16)
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="NVFP4 quantize FastContext-1.0-4B-RL for SM121"
    )
    parser.add_argument(
        "--model-path",
        default="microsoft/FastContext-1.0-4B-RL",
        help="HF model ID or local path to BF16 source",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for NVFP4 checkpoint",
    )
    parser.add_argument("--calib-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    args = parser.parse_args()

    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  FastContext-1.0-4B-RL → NVFP4 Quantization             ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print(f"\nSource:  {args.model_path}")
    print(f"Output:  {args.output_dir}")
    print(f"Config:  NVFP4_DEFAULT_CFG (W4A4, group_size=16)")
    print(f"Calib:   {args.calib_samples} samples × {args.seq_len} tokens × batch {args.batch_size}")
    print()

    # ─── Load model on CPU first (GB10 unified memory safe) ───────────
    print("[1/5] Loading BF16 model (CPU-first for unified memory)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    # Fix architectures if missing (ModelOpt export requirement)
    if not getattr(model.config, "architectures", None):
        model.config.architectures = ["Qwen3ForCausalLM"]
        print(f"  → Set architectures to {model.config.architectures}")

    # Move to CUDA (on GB10 unified memory this is near-instant)
    print("[2/5] Moving model to CUDA...")
    for name, param in model.named_parameters():
        param.data = param.data.to("cuda")
    for name, buf in model.named_buffers():
        buf.data = buf.data.to("cuda")
    model.eval()

    # ─── Configure NVFP4 quantization ────────────────────────────────
    print("[3/5] Configuring NVFP4 quantization...")
    import modelopt.torch.quantization as mtq

    quant_cfg = mtq.NVFP4_DEFAULT_CFG

    # Qwen3-4B is dense, text-only, standard transformer.
    # NVFP4_DEFAULT_CFG quantizes all linear layers (attention QKV/O + MLP).
    # ModelOpt defaults already exclude: norms, biases, lm_head.
    #
    # embed_tokens: 152064 × 2560 = divisible by 16, CAN be quantized.
    # Keep it quantized for max compression. If quality regresses,
    # re-run with --exclude-embed-tokens.
    if args.exclude_embed_tokens:
        print("  → Excluding *embed_tokens* from quantization")
        quant_cfg["quant_cfg"].append(
            {"quantizer_name": "*embed_tokens*", "enable": False}
        )

    print(f"  → Quantization config: NVFP4_DEFAULT_CFG (W4A4, group16)")

    # ─── Calibration ─────────────────────────────────────────────────
    print(f"[4/5] Calibrating with cnn_dailymail ({args.calib_samples} samples)...")
    calib_data = load_dataset(
        "cnn_dailymail", "3.0.0", split=f"train[:{args.calib_samples}]"
    )

    def forward_loop(model):
        for i in range(0, len(calib_data), args.batch_size):
            batch = calib_data[i : i + args.batch_size]["article"]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.seq_len,
            ).to("cuda")
            with torch.no_grad():
                model(**inputs)
            if (i // args.batch_size) % 5 == 0:
                print(
                    f"  → Calibrated {(i + args.batch_size) // args.batch_size}/{len(calib_data) // args.batch_size} batches"
                )

    mtq.quantize(model, quant_cfg, forward_loop)
    print("  → Quantization complete")

    # ─── Export ──────────────────────────────────────────────────────
    print(f"[5/5] Exporting NVFP4 checkpoint to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    from modelopt.torch.export import export_hf_checkpoint

    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=args.output_dir)
    print("  → Export complete")

    # ─── Post-export fixes (known ModelOpt pitfalls) ─────────────────

    # Fix 1: Ensure quant_method is in hf_quant_config.json
    quant_config_path = Path(args.output_dir) / "hf_quant_config.json"
    if quant_config_path.exists():
        with open(quant_config_path) as f:
            qc = json.load(f)
        if "quant_method" not in qc:
            qc["quant_method"] = "modelopt_fp4"
            with open(quant_config_path, "w") as f:
                json.dump(qc, f, indent=2)
            print("  → Added quant_method: modelopt_fp4 to hf_quant_config.json")

    # Fix 2: Verify lm_head.weight is present
    index_path = Path(args.output_dir) / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            idx = json.load(f)
        if "lm_head.weight" not in idx.get("weight_map", {}):
            print("  ⚠️  lm_head.weight MISSING from export — copying from source...")
            _copy_lm_head(args.model_path, args.output_dir)
        else:
            print("  ✓ lm_head.weight present in checkpoint")

    # Fix 3: Copy essential tokenizer/chat files from source
    essential_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ]
    from huggingface_hub import snapshot_download

    source_snapshot = snapshot_download(
        repo_id=args.model_path,
        allow_patterns=essential_files,
    )
    for fname in essential_files:
        src = Path(source_snapshot) / fname
        dst = Path(args.output_dir) / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  → Copied {fname}")

    # ─── Summary ────────────────────────────────────────────────────
    total_size = sum(
        f.stat().st_size
        for f in Path(args.output_dir).rglob("*")
        if f.is_file()
    )
    print(f"\n{'═' * 60}")
    print(f"✅ NVFP4 checkpoint ready: {args.output_dir}")
    print(f"   Total size: {total_size / 1e9:.2f} GB")
    print(f"   Compression: ~{8.0 / (total_size / 1e9):.1f}× from BF16")
    print(f"\n   Next: serve with vLLM 0.23.0 on SM121")
    print(f"   vllm serve {args.output_dir} \\")
    print(f"       --quantization modelopt --tensor-parallel-size 1 \\")
    print(f"       --kv-cache-dtype fp8 --attention-backend flashinfer")
    print(f"{'═' * 60}")


def _copy_lm_head(source_path: str, output_dir: str):
    """Copy lm_head.weight from BF16 source if ModelOpt dropped it."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    # Find the shard containing lm_head.weight in source
    from huggingface_hub import snapshot_download

    source = snapshot_download(
        repo_id=source_path,
        allow_patterns=["model-*.safetensors", "model.safetensors"],
    )
    for sf in sorted(Path(source).glob("*.safetensors")):
        with safe_open(sf, framework="pt") as f:
            if "lm_head.weight" in f.keys():
                lm_head = f.get_tensor("lm_head.weight")
                # Append to last shard or create new shard
                save_file(
                    {"lm_head.weight": lm_head},
                    str(Path(output_dir) / "lm_head.safetensors"),
                )
                # Update index
                idx_path = Path(output_dir) / "model.safetensors.index.json"
                if idx_path.exists():
                    with open(idx_path) as f:
                        idx = json.load(f)
                    idx["weight_map"]["lm_head.weight"] = "lm_head.safetensors"
                    idx["metadata"]["total_size"] += (
                        lm_head.numel() * lm_head.element_size()
                    )
                    with open(idx_path, "w") as f:
                        json.dump(idx, f, indent=2)
                print("  → Copied lm_head.weight to lm_head.safetensors")
                return
    print("  ⚠️  Could not find lm_head.weight in source either!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--exclude-embed-tokens", action="store_true")
    args_extra, _ = parser.parse_known_args()
    main.__wrapped__ = None  # type: ignore
    sys.exit()
