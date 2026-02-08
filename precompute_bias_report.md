# Precompute Residual Bias Optimization — Master Branch Report

## Change

Moved `x0_lambdas[i] * x0 + bigram_lambdas[i] * x0_bigram` outside the main sequential
loop into a precomputed tuple. Inside the loop, the residual update simplifies from:

```python
# Before (inside loop)
x = resid_lambdas[i] * x + x0_lambdas[i] * x0 + bigram_lambdas[i] * x0_bigram

# After (precomputed + inside loop)
residual_bias = tuple(x0_lambdas[i] * x0 + bigram_lambdas[i] * x0_bigram
                      for i in range(1, num_layers))
...
x = resid_lambdas[i] * x + residual_bias[i - 1]
```

Layer 0 is unchanged (special case where `x == x0`).

## Hypothesis

Moving x0/bigram bias computation outside the sequential loop reduces the backward pass
critical path, since gradient computation for x0_lambdas and bigram_lambdas no longer
blocks the sequential chain through the loop iterations. This optimization yielded
~1.2ms/step savings in the Hyper-Connections (HC) 2-lane variant.

## Results (4xH200, 2 runs each)

| Config | Run | step_avg | val_loss | train_time |
|--------|-----|----------|----------|------------|
| Master (baseline) | 1 | 112.19ms | 3.2763 | 174,452ms |
| Master (baseline) | 2 | 112.24ms | 3.2787 | 174,538ms |
| **Baseline avg** | | **112.22ms** | **3.2775** | **174,495ms** |
| Precompute bias | 1 | 113.12ms | 3.2808 | 175,896ms |
| Precompute bias | 2 | 113.27ms | 3.2788 | 176,139ms |
| **Precompute avg** | | **113.20ms** | **3.2798** | **176,018ms** |

**Delta: +0.98ms/step (+0.87%), +1,523ms total training time (+0.87%)**

## Why It Didn't Help

The optimization **helped HC** (~1.2ms/step faster) but **hurt master** (~1ms/step slower)
because of fundamental differences in workload:

### HC (2-lane variant) — optimization helps
- 2 residual lanes × 4 bias terms per sublayer = heavy per-iteration work
- 22 sublayers (2 per layer × 11 layers) = long critical path
- Removing bias computation from the loop provides meaningful critical path reduction
- Benefit > materialization cost

### Master (single stream) — optimization hurts
- 1 residual stream, 2 scalar-tensor multiplies per layer
- 11 layers = shorter critical path
- `torch.compile` fuses `resid * x + x0_lambda * x0 + bigram * x0_bigram` into a **single
  element-wise kernel** — the three terms are computed in one pass
- Precomputing **breaks this fusion**: the bias tensor must be materialized to memory,
  then loaded back in a separate kernel for the addition
- Materialization cost > critical path reduction

### Key Insight

The precompute optimization only helps when:
1. There is enough per-iteration work that the backward critical path is the bottleneck
2. The materialization cost (writing + reading the precomputed tensors) is smaller than
   the critical path reduction

For master's simple single-stream architecture, `torch.compile` already generates
near-optimal fused kernels, and the precompute introduces unnecessary memory traffic.

## Conclusion

**Not recommended for master.** The optimization is specific to the HC 2-lane architecture
where the per-sublayer work is heavy enough to benefit from critical path reduction.
