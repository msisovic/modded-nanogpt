"""Benchmark FP8 overhead: compiled vs uncompiled, and tile-wise vs tensor-wise scaling."""
import torch
import time

device = "cuda"
torch.set_default_device(device)

model_dim = 768
mlp_hdim = 3072
E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# --- Compiled versions ---

@torch.compile(dynamic=False)
def act_quant_compiled(x):
    amax = x.detach().abs().max().clamp(min=1e-12)
    x_f8 = (x.detach() * (448.0 / amax)).to(torch.float8_e4m3fn)
    scale = (amax / 448.0).float()
    return x_f8, scale

@torch.compile(dynamic=False)
def weight_quant_compiled(mlp_bank):
    flat = mlp_bank.view(24, -1)
    scales = flat.abs().amax(dim=1).clamp(min=1e-12) / E4M3_MAX
    w_f8 = (mlp_bank / scales.view(12, 2, 1, 1)).to(torch.float8_e4m3fn)
    return w_f8, scales

# Non-compiled versions for comparison
def act_quant_eager(x):
    amax = x.abs().max().clamp(min=1e-12)
    x_f8 = (x * (448.0 / amax)).to(torch.float8_e4m3fn)
    scale = (amax / 448.0).float()
    return x_f8, scale

def weight_quant_eager(mlp_bank):
    flat = mlp_bank.view(24, -1)
    scales = flat.abs().amax(dim=1).clamp(min=1e-12) / E4M3_MAX
    w_f8 = (mlp_bank / scales.view(12, 2, 1, 1)).to(torch.float8_e4m3fn)
    return w_f8, scales

# Fixed scale (no amax needed)
@torch.compile(dynamic=False)
def act_quant_fixed_compiled(x):
    x_f8 = (x * 50.0).to(torch.float8_e4m3fn)  # fixed scale
    return x_f8

def act_quant_fixed_eager(x):
    x_f8 = (x * 50.0).to(torch.float8_e4m3fn)
    return x_f8

def bench(fn, args, n_warmup=10, n_iters=50, label=""):
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    elapsed_us = start.elapsed_time(end) / n_iters * 1000
    print(f"  {label:45s}: {elapsed_us:8.1f} μs")
    return elapsed_us

stages = {
    "stage1": 16384,
    "stage2": 32768,
    "stage3": 49152,
}

for stage_name, M in stages.items():
    print(f"\n{'='*60}")
    print(f"{stage_name}: M={M}")
    print(f"{'='*60}")

    x = torch.randn(M, model_dim, dtype=torch.bfloat16)
    mlp_bank = torch.randn(12, 2, mlp_hdim, model_dim, dtype=torch.bfloat16)
    W1 = torch.randn(mlp_hdim, model_dim, dtype=torch.bfloat16)
    W1_f8 = W1.to(torch.float8_e4m3fn)
    W1_f8_smm = W1_f8.contiguous().t()

    scale_one = torch.ones(1, dtype=torch.float32, device=device)

    print("\n  -- Activation quantization --")
    aq_eager = bench(act_quant_eager, (x,), label="eager (dynamic scale)")
    aq_compiled = bench(act_quant_compiled, (x,), label="compiled (dynamic scale)")
    aq_fixed_eager = bench(act_quant_fixed_eager, (x,), label="eager (fixed scale)")
    aq_fixed_compiled = bench(act_quant_fixed_compiled, (x,), label="compiled (fixed scale)")

    print("\n  -- Weight quantization (all 24 matrices) --")
    wq_eager = bench(weight_quant_eager, (mlp_bank,), label="eager")
    wq_compiled = bench(weight_quant_compiled, (mlp_bank,), label="compiled")

    print("\n  -- Matmul --")

    def mm_bf16():
        return x @ W1.T

    def mm_fp8():
        return torch._scaled_mm(x.to(torch.float8_e4m3fn), W1_f8_smm,
                                scale_a=scale_one, scale_b=scale_one,
                                out_dtype=torch.bfloat16, use_fast_accum=True)

    m_bf16 = bench(mm_bf16, (), label="BF16 matmul")
    m_fp8 = bench(mm_fp8, (), label="FP8 matmul (pre-quantized)")

    print(f"\n  -- Per-step total (W1 forward only, 12 layers) --")
    bf16_total = 12 * m_bf16
    fp8_eager_total = wq_eager + 12 * (aq_eager + m_fp8)
    fp8_compiled_total = wq_compiled + 12 * (aq_compiled + m_fp8)
    fp8_fixed_total = wq_compiled + 12 * (aq_fixed_compiled + m_fp8)

    print(f"  BF16 baseline:                       {bf16_total:8.1f} μs")
    print(f"  FP8 eager quant:                     {fp8_eager_total:8.1f} μs  ({fp8_eager_total - bf16_total:+.1f})")
    print(f"  FP8 compiled quant:                  {fp8_compiled_total:8.1f} μs  ({fp8_compiled_total - bf16_total:+.1f})")
    print(f"  FP8 fixed scale + compiled:          {fp8_fixed_total:8.1f} μs  ({fp8_fixed_total - bf16_total:+.1f})")
