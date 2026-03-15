# FP8 W2 (c_proj) Matmul Optimization — Notes

## Goal
Use FP8 for the second MLP matmul (`post @ W2` where post = relu^2 output, shape [M, 3072] @ [3072, 768]) to speed up training.

## Branch
`fp8-w2-only` — changes in `triton_kernels.py` and `train_gpt.py`.

## Result
**~0.4% faster overall** (468.71ms vs 470.67ms master, 1 GPU). Val loss identical (3.2806 vs 3.2794). Not worth the complexity in current form.

### Per-stage breakdown

| Segment | Batch M | FP8 vs Master |
|---------|---------|---------------|
| 0-250 (stage 1) | 16384 | +1.8% slower |
| 250-500 (stage 1) | 16384 | +0.5% slower |
| 500-750 (stage 2) | 32768 | -0.6% faster |
| 750-1000 (stage 2) | 32768 | -0.7% faster |
| 1000-1250 (stage 3) | 49152 | -1.2% faster |
| 1250-1490 (stage 3) | 49152 | -1.1% faster |

Stage 1 losses eat into stage 2-3 gains.

## What works
- Raw FP8 W2 matmul is **1.88x faster** than BF16 (bench_w2_matmul.py, bench_scaling.py)
- Fusing FP8 cast into the W1 Triton kernel eliminates the separate Pointwise kernel (~556us for [49152, 3072])
- Pre-cached FP8 weights (updated after optimizer step) avoid per-forward quantization
- 1-dim `[1]` scale tensors avoid inductor 0-dim unsqueeze bug
- Pre-allocated scale_a buffer avoids 85us/call `new_tensor` allocation overhead
- Under `torch.compile(fullgraph=True)`: no graph breaks, compiles cleanly

## What costs
1. **W1 kernel regression**: Fusing FP8 emit requires `num_stages=3` (down from 4) due to shared memory pressure (245KB needed vs 232KB limit). Costs 3-7% on W1 kernel across 12 layers.
   - Alternative: `BLOCK_SIZE_K=32` with 4 stages fits in shared memory but performs similarly.
2. **`update_mlp_w2_cache()` per step**: ~400us (amax over 12×[3072,768], division, fp8 cast, permute, copy).
3. At small M (stage 1), overhead > savings. At large M (stage 3), savings win by ~1.2%.

## Key bugs found along the way
- **Inductor 0-dim scale bug**: `torch._scaled_mm` with 0-dim scale tensors crashes in inductor template heuristics (`triton.py:1878-1882`). The `unsqueeze` on 0-dim triggers `resolve_negative_size` assertion. Fix: use 1-dim `[1]` tensors.
- **NaN from zero-init W2**: `amax=0 → scale=0 → NaN`. Fix: `.clamp(min=1e-12)` on scales.
- **`new_tensor` allocation overhead**: `post.new_tensor([...])` inside forward = CUDA alloc every call = 85us. Fix: pre-allocate as buffer.
- **Shared memory overflow**: `BLOCK_SIZE_M=128, N=256, K=64` with 4 stages + FP8 aux tensor = 245KB > 232KB H100 limit. Fix: reduce `num_stages` to 3 or `BLOCK_SIZE_K` to 32.

## Implementation summary
- `triton_kernels.py`: Added `EMIT_F8` param to `linear_relu_square_kernel`, fuses `tl.minimum(post * (448/10000), 448).to(tl.float8e4nv)` into the kernel. `FusedLinearReLUSquareFunction.forward` accepts `W2_f8`, `W2_scale`, `scale_a` and calls `torch._scaled_mm`.
- `train_gpt.py`: Registers `mlp_w2_f8_T`, `mlp_w2_scales`, `mlp_w2_scale_a` buffers. `update_mlp_w2_cache()` quantizes `mlp_bank[:, 1]` after each optimizer step. Forward unbinds cache and passes to `ReLUSqrdMLP`.

## Possible improvements to explore
- Only enable FP8 for stages 2-3 (M >= 32768), use BF16 for stage 1 — would eliminate the stage 1 regression
- Find a way to keep `num_stages=4` with EMIT_F8 (smaller block sizes, different tiling)
- Skip `update_mlp_w2_cache` when weights haven't changed much (stale cache with periodic refresh)
- On 8 GPUs, M per GPU is 1/8th — FP8 gains would be even smaller

## Benchmark scripts (in repo root)
- `bench_w2_matmul.py` — raw BF16 vs FP8 W2 matmul + chrome traces
- `bench_scaling.py` — how M affects FP8 vs BF16 speedup
- `bench_emit_f8.py` — W1 kernel overhead from EMIT_F8
- `bench_full_mlp.py` — full MLP through autograd Function
- `bench_compiled.py` — MLP under torch.compile
- `bench_breakdown.py` — piece-by-piece breakdown
- `bench_isolate_overhead.py` — isolates new_tensor allocation cost
