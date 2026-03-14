"""Benchmark FP8 overhead: weight quantization, activation quantization, and W1 matmul."""
import torch
import torch.utils.benchmark as benchmark
import time

device = "cuda"
torch.set_default_device(device)

# Model dimensions
model_dim = 768
mlp_hdim = 3072
E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# Test across all 3 stage sizes
stages = {
    "stage1": 16384,   # 131K tokens / 8 GPUs
    "stage2": 32768,   # 262K / 8
    "stage3": 49152,   # 393K / 8
}

for stage_name, M in stages.items():
    print(f"\n{'='*60}")
    print(f"{stage_name}: M={M}, K={model_dim}, N={mlp_hdim}")
    print(f"{'='*60}")

    # Create test tensors
    x_bf16 = torch.randn(M, model_dim, dtype=torch.bfloat16)
    W1_bf16 = torch.randn(mlp_hdim, model_dim, dtype=torch.bfloat16)
    mlp_bank = torch.randn(12, 2, mlp_hdim, model_dim, dtype=torch.bfloat16)

    # Pre-quantized
    W1_f8 = W1_bf16.to(torch.float8_e4m3fn)
    x_f8 = x_bf16.to(torch.float8_e4m3fn)

    # ---- 1. Weight quantization (called every step) ----
    def weight_quant():
        flat = mlp_bank.view(24, -1)
        scales = flat.abs().amax(dim=1).clamp(min=1e-12) / E4M3_MAX
        _ = (mlp_bank / scales.view(12, 2, 1, 1)).to(torch.float8_e4m3fn)

    # ---- 2. Activation quantization (called per layer) ----
    def act_quant_dynamic():
        amax = x_bf16.abs().max().clamp(min=1e-12)
        _ = (x_bf16 * (448.0 / amax)).to(torch.float8_e4m3fn)

    # ---- 3. BF16 W1 matmul ----
    def matmul_bf16():
        return x_bf16 @ W1_bf16.T

    # ---- 4. FP8 W1 matmul via _scaled_mm ----
    # _scaled_mm: (M,K) @ (K,N) where second is col-major
    # W1 is (N=3072, K=768). We want (K=768, N=3072) col-major.
    # Col-major (768,3072) = row-major (3072,768) = W1 itself, viewed as transpose
    W1_f8_for_smm = W1_f8.contiguous().t()  # (768, 3072) col-major
    scale_a = torch.ones(1, dtype=torch.float32, device=device)
    scale_b = torch.ones(1, dtype=torch.float32, device=device)

    def matmul_fp8():
        return torch._scaled_mm(x_f8, W1_f8_for_smm, scale_a=scale_a, scale_b=scale_b,
                                out_dtype=torch.bfloat16, use_fast_accum=True)

    # ---- 5. Full FP8 path: act quant + fp8 matmul (per layer) ----
    def full_fp8_path():
        amax = x_bf16.abs().max().clamp(min=1e-12)
        x_q = (x_bf16 * (448.0 / amax)).to(torch.float8_e4m3fn)
        s = (amax / 448.0).float()
        return torch._scaled_mm(x_q, W1_f8_for_smm, scale_a=s.unsqueeze(0), scale_b=scale_b,
                                out_dtype=torch.bfloat16, use_fast_accum=True)

    # Warmup
    for fn in [weight_quant, act_quant_dynamic, matmul_bf16, matmul_fp8, full_fp8_path]:
        for _ in range(5):
            fn()
    torch.cuda.synchronize()

    # Benchmark each
    results = {}
    for name, fn in [
        ("weight_quant (1x/step)", weight_quant),
        ("act_quant (12x/step)", act_quant_dynamic),
        ("matmul_bf16 W1 (12x/step)", matmul_bf16),
        ("matmul_fp8 W1 (12x/step)", matmul_fp8),
        ("full_fp8_path (12x/step)", full_fp8_path),
    ]:
        # Use CUDA events for accurate GPU timing
        n_iters = 20
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(5):
            fn()
        torch.cuda.synchronize()

        start.record()
        for _ in range(n_iters):
            fn()
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / n_iters
        results[name] = elapsed_ms
        print(f"  {name:35s}: {elapsed_ms*1000:8.1f} μs")

    # Summary
    print(f"\n  --- Per-step overhead analysis ---")
    wq = results["weight_quant (1x/step)"]
    aq = results["act_quant (12x/step)"]
    mbf16 = results["matmul_bf16 W1 (12x/step)"]
    mfp8 = results["matmul_fp8 W1 (12x/step)"]
    full = results["full_fp8_path (12x/step)"]

    bf16_total = 12 * mbf16
    fp8_total = wq + 12 * full
    print(f"  BF16 path (12 × matmul):           {bf16_total*1000:8.1f} μs")
    print(f"  FP8 path (wq + 12 × (aq+matmul)):  {fp8_total*1000:8.1f} μs")
    print(f"  Difference:                         {(fp8_total - bf16_total)*1000:+8.1f} μs")
    print(f"  FP8 matmul speedup:                 {mbf16/mfp8:.2f}x")
    print(f"  FP8 full path speedup:              {mbf16/full:.2f}x")
