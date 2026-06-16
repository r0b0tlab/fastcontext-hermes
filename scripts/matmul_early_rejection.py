#!/usr/bin/env python3
"""
Matmul-Level Early Rejection Test for NVFP4 on SM121

CRITICAL: For small-dimension dense models (hidden_size=2560, intermediate=9728),
BF16 tensor cores may outperform NVFP4 FP4 tensor cores on GB10. This was observed
with HiDream-O1/Qwen3VL (0.87× slower with NVFP4).

This script benchmarks the largest representative matmuls BEFORE committing to
a full multi-hour quantization. If NVFP4 is slower at the matmul level, stop.

Run in a venv with torch+cu130 and comfy_kitchen (or use the vLLM container):

    python matmul_early_rejection.py

Decision rule: If ratio < 1.0×, NVFP4 will NOT be faster for the full model.
→ Re-evaluate: the value-add is then memory savings (3× smaller) + power savings,
  not throughput speedup. Document this honestly.
"""

import time
import sys

try:
    import torch
    import comfy_kitchen as ck
    from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX
except ImportError:
    print("ERROR: This script requires torch + comfy_kitchen.")
    print("Run inside the ComfyUI Docker container:")
    print("  docker exec -w /opt/ComfyUI comfyui python3 matmul_early_rejection.py")
    print("Or install: pip install torch comfy_kitchen")
    sys.exit(1)


def benchmark_matmul(name, w_shape, x_shape, n_iter=100):
    """Benchmark BF16 vs NVFP4 for a single matmul layer."""
    print(f"\n{'─' * 50}")
    print(f"Layer: {name}")
    print(f"Weight: {w_shape}, Input: {x_shape}")

    # Create weights and inputs
    w_bf16 = torch.randn(*w_shape, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(*x_shape, device="cuda", dtype=torch.bfloat16)
    x_flat = x.reshape(-1, x_shape[-1])

    # ─── Quantize weight to NVFP4 ────────────────────────────────────
    pts_w = (w_bf16.abs().amax() / (F8_E4M3_MAX * F4_E2M1_MAX)).unsqueeze(0).float()
    w_nvfp4, w_block_scale = ck.quantize_nvfp4(w_bf16, pts_w, pad_16x=True)

    # ─── Quantize activations to NVFP4 (required for scaled_mm_nvfp4) ──
    pts_a = (x_flat.abs().amax() / (F8_E4M3_MAX * F4_E2M1_MAX)).unsqueeze(0).float()
    x_nvfp4, x_block_scale = ck.quantize_nvfp4(x_flat, pts_a, pad_16x=True)

    # ─── Warmup ──────────────────────────────────────────────────────
    for _ in range(5):
        _ = torch.nn.functional.linear(x, w_bf16)
    for _ in range(5):
        _ = ck.scaled_mm_nvfp4(x_nvfp4, w_nvfp4, pts_a, pts_w,
                                x_block_scale, w_block_scale,
                                out_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    # ─── Benchmark BF16 ──────────────────────────────────────────────
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        _ = torch.nn.functional.linear(x, w_bf16)
    torch.cuda.synchronize()
    t_bf16 = (time.time() - t0) / n_iter

    # ─── Benchmark NVFP4 ─────────────────────────────────────────────
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        _ = ck.scaled_mm_nvfp4(x_nvfp4, w_nvfp4, pts_a, pts_w,
                                x_block_scale, w_block_scale,
                                out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t_nvfp4 = (time.time() - t0) / n_iter

    ratio = t_bf16 / t_nvfp4
    verdict = "✅ NVFP4 FASTER" if ratio >= 1.0 else "❌ NVFP4 SLOWER"

    print(f"BF16:  {t_bf16 * 1000:.3f} ms")
    print(f"NVFP4: {t_nvfp4 * 1000:.3f} ms")
    print(f"Ratio: {ratio:.2f}×  {verdict}")

    return name, ratio


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  NVFP4 Matmul Early Rejection Test — FastContext 4B    ║")
    print("║  Architecture: Qwen3, hidden=2560, intermediate=9728   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    device = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"\nGPU: {device} (SM {cap[0]}.{cap[1]})")

    # Representative batch: 1 sequence × 1024 tokens
    # This matches typical FastContext exploration: single query, moderate context
    batch_shape = (1, 1024)

    results = []

    # Largest layers by parameter count (these dominate inference time)
    # nn.functional.linear(x, w): x=[B,T,in], w=[out,in]
    # Qwen3-4B: hidden=2560, intermediate=9728, n_heads=32, n_kv=8, head_dim=128

    # 1. MLP down_proj: maps intermediate→hidden, weight [2560, 9728]
    results.append(benchmark_matmul(
        "mlp_down_proj [2560, 9728]",
        w_shape=(2560, 9728),
        x_shape=(*batch_shape, 9728),
    ))

    # 2. MLP gate_proj: maps hidden→intermediate, weight [9728, 2560]
    results.append(benchmark_matmul(
        "mlp_gate_proj [9728, 2560]",
        w_shape=(9728, 2560),
        x_shape=(*batch_shape, 2560),
    ))

    # 3. Attention Q proj: maps hidden→n_heads*head_dim, weight [4096, 2560]
    results.append(benchmark_matmul(
        "attn_q_proj [4096, 2560]",
        w_shape=(4096, 2560),
        x_shape=(*batch_shape, 2560),
    ))

    # 4. Attention O proj: maps n_heads*head_dim→hidden, weight [2560, 4096]
    results.append(benchmark_matmul(
        "attn_o_proj [2560, 4096]",
        w_shape=(2560, 4096),
        x_shape=(*batch_shape, 4096),
    ))

    # ─── Summary ─────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("SUMMARY")
    print(f"{'═' * 60}")
    print(f"{'Layer':<35} {'Ratio':>8}  {'Verdict'}")
    print(f"{'─' * 60}")
    all_pass = True
    for name, ratio in results:
        verdict = "✅" if ratio >= 1.0 else "❌"
        print(f"{name:<35} {ratio:>7.2f}×  {verdict}")
        if ratio < 1.0:
            all_pass = False

    print(f"\n{'─' * 60}")
    if all_pass:
        print("✅ ALL LAYERS FASTER WITH NVFP4 — proceed with full quantization")
    else:
        print("⚠️  SOME LAYERS SLOWER WITH NVFP4")
        print("   NVFP4 throughput benefit is NOT guaranteed for this model size.")
        print("   Proceed ONLY if memory savings (3× smaller) justify the trade-off.")
        print("   Document honestly: 'NVFP4 provides 3× memory compression but")
        print("   may not improve throughput on SM121 for 4B dense dimensions.'")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
