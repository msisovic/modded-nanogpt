# Hyper-Connections Experiment Log

**NEW Baseline (master, post-merge):** val_loss=3.2769 @ 1555 steps (1515 scheduled + 40 extension)
**Old Baseline (pre-merge):** val_loss=3.2769 @ 1600 steps
**Best HC (Exp 17, old baseline):** val_loss=3.2720 @ 1600 steps

---

> **NOTE (Baseline Change):** Experiments 1-19 below were run against the OLD baseline
> (3.2769 @ 1600 steps). The master branch was updated with new improvements (single value
> embedding weight, mimetic init, x0_lambdas separation, Yarn refactor, 1555 steps).
> These old results are **outdated** — the step count was higher (1600 vs 1555) and the
> code base has changed. Experiments 20+ use the NEW merged baseline.

---

## Exp 1: n_lanes=2 + vectorized + adam_betas fix
- KILLED at step 250 (val_loss:4.5722). Marginal difference.

## Exp 2: Pre-Apply Permanent (Option B) + 4 lanes
- val_loss: **3.2883** @ 1600. WORSE than prev best (+0.007). Pre-apply hurts.

## Exp 3: Post-Apply + Depth-Ramped Post Init (1.0→2.2)
- val_loss: **3.2831** @ 1600. Better than exp 2 but still worse than prev best.
- Late post converged to 3.5 (from init 2.14). Model aggressively increases post.
- Beat baseline at step 750 (3.8928 vs 3.8985) but fell behind after.

## Exp 4: Previous Best Config + adam_betas Fix Only
**Config:** Exact previous best (post-apply, 4 lanes, identity res, one-hot pre, flat post=1.0)
**Only change:** adam_betas for hyper_x0_bias: [0.9, 0.99] -> [0.65, 0.95] (match baseline x0_lambdas)
**Also:** Vectorized ops (einsum, should be equivalent)

**Hypothesis:** Isolate adam_betas effect. Previous best never had this fix.
Baseline x0_lambdas use beta1=0.65 for faster adaptation; HC biases used 0.9 (slower).

**Result:** val_loss=3.2867 @ 1600. WORSE (+0.006). adam_betas change hurts. Reverted.

## Exp 5: Separate Residual Lambda (sqrt(1.1) init)
**Config:** Previous best + new `hyper_resid_lambda` scalar per sublayer
**Changes from previous best:**
1. NEW: `hyper_resid_lambda` per sublayer, init sqrt(1.1)≈1.0488 (matches baseline's 1.1/block)
2. REVERTED: x0_bias adam_betas back to [0.9, 0.99] (exp 4 showed [0.65, 0.95] hurts)
3. Optimizer: adam_betas=[0.9, 0.99], lr_mul=5.0 (same as baseline scalars)

**Hypothesis:** HC's w_res naturally decays to ~0.95/sublayer, losing the gain that baseline's
resid_lambda=1.1 provides. Adding a separate scalar lambda at the mixing point provides this
gain independently, while w_res handles lane mixing. This directly addresses the gradient
attenuation problem (0.95^22 ≈ 0.32 vs baseline's 1.1^11 ≈ 2.85).

**Result:** val_loss=**3.2813** @ 1600. Best of new exps, nearly tied with prev best (3.2809).
**Learned resid_lambda:** Early [3.52, 3.66], Middle ~0.5-0.85, Late ~1.0-1.2.
Cumulative gain product=0.28 (net decay). Model amplifies early layers massively but kills gain later.
Off-diagonal drift still small (~0.03-0.05). Optimizer fights gain in later sublayers.

## Exp 6: Learnable Residual Lambda (init 1.1/sublayer, higher than Exp 5)
**Config:** Same as Exp 5 but with HIGHER init: 1.1 per sublayer (= 1.21/block, vs baseline's 1.1/block)
**Note:** Baseline's resid_lambda IS learnable (part of scalars group), confirmed from master branch.
**Changes from Exp 5:**
1. resid_lambda init: 1.1 per sublayer (was sqrt(1.1)≈1.0488)
2. Cumulative init gain: 1.1^22 ≈ 7.4 (vs baseline's 1.1^11 ≈ 2.85)

**Hypothesis:** Exp 5 early sublayers learned 3.5+ from init 1.0488, suggesting the model wants
higher initial gain. With 1.1 init, model starts with stronger amplification and can learn
to reduce specific layers if needed.

**Result:** val_loss=**3.2846** @ 1600. Worse than Exp 5. Higher init didn't help.
Learned: Early [1.84, 2.15], cumulative=0.29. Similar pattern to Exp 5 regardless of init.

## Exp 7: Higher LR for HC Matrices (lr_mul 1.0 → 5.0)
**Config:** Exp 5 base + increased lr_mul for w_res/w_pre/w_post from 1.0 to 5.0
**Changes from Exp 5:**
1. hyper_res lr_mul: 1.0 → 5.0
2. hyper_pre lr_mul: 1.0 → 5.0
3. hyper_post lr_mul: 1.0 → 5.0
4. resid_lambda: back to sqrt(1.1) init (best from Exp 5)

**Hypothesis:** HC matrix params at lr_mul=1.0 learn too slowly to break lane symmetry.
Off-diagonal drift was only 0.03-0.05 across all experiments. Baseline scalars (including
resid_lambda) use lr_mul=5.0. With faster learning, w_res might develop meaningful
off-diagonal mixing and w_pre/w_post might specialize per lane.

**Result:** KILLED at step 750 (val_loss=3.9392). Much worse. High lr_mul for matrices confirmed bad.

## Exp 7b: n_lanes=2 + resid_lambda (natural lane specialization)
**Config:** Exp 5 base but n_lanes=2 instead of 4
**Key insight:** With 2 lanes + round-robin one-hot pre:
- ALL attn sublayers (even idx) read from lane 0
- ALL MLP sublayers (odd idx) read from lane 1
- This naturally creates "attn-view" and "MLP-view" lanes
- No symmetry breaking needed - the architecture enforces specialization
**Changes from Exp 5:**
1. n_lanes: 4 → 2
2. HC matrix lr_mul: 1.0 (reverted from 7a)
3. resid_lambda: sqrt(1.1) per sublayer (same as Exp 5)

**Hypothesis:** 4 lanes with symmetric init = 4 copies of same thing = wasted params.
2 lanes naturally specialize (attn vs MLP views), requiring less from the optimizer.
Less memory overhead may also help training.

**Result:** val_loss=**3.2797** @ 1600. NEW BEST! Gap to baseline narrowed to +0.0028.
**Key findings:**
- Cumulative resid_lambda gain = 0.9484 (nearly 1.0!) vs 0.28 with 4 lanes
- Matrix drift 0.08-0.13 (2-3x more than 4 lanes) - lanes actually mixing
- Lanes genuinely specialized: different x0_bias, bigram_bias, pre weights per lane
- Early layer: lane 0 x0_bias=0.98 vs lane 1=0.66 (different roles)
- Middle layer: lane 0 x0_bias=-1.77 (!) vs lane 1=0.29
- Pre learned to mix both lanes (no longer pure one-hot)
- Step time: ~816ms early → 1572ms final (slightly faster than 4 lanes)

## Exp 8: 2 Lanes WITHOUT resid_lambda (ablation)
**Config:** Same as Exp 7b but resid_lambda disabled (fixed buffer = 1.0)
**Changes from Exp 7b:**
1. hyper_resid_lambda: learnable → fixed buffer of 1.0 (no gain)
2. Removed from optimizer

**Hypothesis:** Exp 7b showed 2-lane cumulative gain ≈ 0.95 (near neutral).
With only 2 lanes, the model may not need explicit gain since w_res stays closer to identity.
Fewer learnable params = cleaner optimization. If this matches 7b, resid_lambda isn't needed.

## Exp 9: Reproduce Exp 7b on single H200 (nproc=1)
**Config:** Faithful Exp 7b reproduction (2 lanes + resid_lambda sqrt(1.1))
- n_lanes=2, w_res=I_2, round-robin pre, flat post=1.0
- resid_lambda: sqrt(1.1) per sublayer, lr_mul=5.0, adam_betas=[0.9, 0.99]
- x0_bias: lr_mul=1.0, bigram_bias: lr_mul=1.0
- HC matrices (res/pre/post): lr_mul=1.0
- Single GPU (nproc_per_node=1), grad_accum_steps=8

**Result:** val_loss=**3.2734** @ 1600. **BEATS BASELINE** (3.2769) by 0.0035!
```
step:250/1600  val_loss:4.5792  step_avg:268ms
step:500/1600  val_loss:4.2458  step_avg:267ms
step:750/1600  val_loss:3.8984  step_avg:336ms
step:1000/1600 val_loss:3.5918  step_avg:375ms
step:1250/1600 val_loss:3.4131  step_avg:503ms
step:1500/1600 val_loss:3.3033  step_avg:585ms
step:1600/1600 val_loss:3.2734  step_avg:609ms
```
**HC Analysis:**
- Late layer: w_post=[1.80, 1.66] (high write strength), matrix drift=0.107
- Middle: resid_lambda=0.87, pre learned to mix both lanes [0.29, 0.92]
- Late: resid_lambda=0.78, pre flipped signs [-0.40, 0.39] (creative routing!)
- Strong lane specialization throughout

## Exp 10: x0_bias and bigram_bias lr_mul 1.0→5.0
**Config:** Exp 9 + faster learning for x0_bias and bigram_bias
**Result:** val_loss=**3.2810** @ 1600. Worse. Overshoot—x0_bias grew to 1.15/-2.23. Reverted.

## Exp 11: Lane average for skip_in and backout
**Config:** Exp 9 + skip_in/backout use lane average instead of lane 0
**Result:** val_loss=**3.2838** @ 1600. Worse. Reverted.

## Exp 12: Lower resid_lambda init (1.0 vs sqrt(1.1))
**Config:** Exp 9 but resid_lambda init 1.0 (neutral) instead of sqrt(1.1)
**Result:** val_loss=**3.2828** @ 1600. Worse. sqrt(1.1) init is better. Reverted.

## Exp 13: Frozen w_res (identity buffer, not learned)
**Config:** Exp 9 but w_res is a fixed buffer (identity), removed from optimizer
**Hypothesis:** w_res only achieves 0.03-0.11 drift in 1600 steps. Learning it wastes
optimizer capacity. Freezing it lets pre/post/resid_lambda learn more effectively.
**Result:** val_loss=**3.2727** @ 1600. **NEW BEST!** Beats baseline by 0.0042.
```
step:250/1600  val_loss:4.5867
step:500/1600  val_loss:4.2304
step:750/1600  val_loss:3.8829  ← beats baseline 3.8985 by 0.016!
step:1000/1600 val_loss:3.5884
step:1250/1600 val_loss:3.4105
step:1500/1600 val_loss:3.3019
step:1600/1600 val_loss:3.2727
```
Cumulative resid_lambda gain: 0.1236 (heavy decay). Fewer params = better convergence.

## Exp 14: Depth-ramped w_post init (0.8→1.8)
**Result:** val_loss=**3.2788** @ 1600. Worse. Flat 1.0 init is better. Reverted.

## Exp 15: Per-lane resid_lambda
**Result:** val_loss=**3.2748** @ 1600. Slightly worse than Exp 13. Extra params hurt. Reverted.

## Exp 16: Frozen w_res + frozen w_post
**Result:** val_loss=**3.2769** @ 1600. Exactly baseline. w_post IS important — freezing it
removes our advantage completely. Learned w_post is critical for depth-dependent weighting.

## Exp 17: Frozen w_res + frozen w_pre (keep w_post learned)
**Config:** Exp 13 base + also freeze w_pre at round-robin one-hot
**Learned params:** w_post, resid_lambda, x0_bias, bigram_bias only
**Result:** val_loss=**3.2720** @ 1600. **NEW BEST!** Beats baseline by 0.0049.
```
step:250/1600  val_loss:4.5879
step:500/1600  val_loss:4.2436
step:750/1600  val_loss:3.8823
step:1000/1600 val_loss:3.5880
step:1250/1600 val_loss:3.4097
step:1500/1600 val_loss:3.3013
step:1600/1600 val_loss:3.2720
```
Pattern: frozen w_res + frozen w_pre + learned w_post = sweet spot.
Fewer HC params = more optimizer capacity for the params that matter.

## Exp 18: w_post lr_mul 1.0→5.0
**Config:** Exp 17 + w_post lr_mul increased from 1.0 to 5.0
**Hypothesis:** w_post is the most important learned HC param. Faster learning might help.
**Result:** val_loss=**3.2723** @ 1600. Nearly identical to Exp 17 (3.2720). No benefit. Reverted to lr_mul=1.0.

## Exp 19: Learned output weights (lane combination)
**Config:** Exp 17 + learned `hyper_output_w` per-lane scalar for final output (init 0.5 each)
**Hypothesis:** Equal lane averaging may not be optimal. Let model learn output weighting.
**Result:** val_loss=**3.2750** @ 1600. Worse than Exp 17. Equal averaging is fine. Reverted.

---

## === NEW BASELINE (post-merge) ===
## Baseline: val_loss=3.2769 @ 1555 steps (1515 scheduled + 40 extension)
## Master changes: single value_embed, mimetic init, x0_lambdas separation, Yarn refactor

---

## Exp 20: Verify Exp 17 config on new baseline
**Config:** Exact Exp 17 config (frozen w_res, frozen w_pre, learned w_post, 2 lanes, resid_lambda sqrt(1.1))
merged onto new master (single value_embed, mimetic init, 1555 steps).
**Result:** val_loss=**3.2745** @ 1555. **BEATS NEW BASELINE** by 0.0024.
```
step:0/1555    val_loss:10.8294
step:250/1555  val_loss:4.5614  step_avg:308ms
step:500/1555  val_loss:4.2414  step_avg:308ms
step:750/1555  val_loss:3.8653  step_avg:365ms
step:1000/1555 val_loss:3.5690  step_avg:394ms
step:1250/1555 val_loss:3.3934  step_avg:458ms
step:1500/1555 val_loss:3.2908  step_avg:501ms
step:1555/1555 val_loss:3.2745  step_avg:509ms
```
HC still works on new baseline. Margin reduced from -0.0049 (old) to -0.0024 (new),
partly because we have 45 fewer steps (1555 vs 1600).

## Exp 21: Match x0_bias/bigram_bias optimizer to baseline (adam_betas=[0.65,0.95], lr_mul=5.0)
**Config:** Exp 20 + hyper_x0_bias adam_betas [0.9,0.99]→[0.65,0.95], lr_mul 1.0→5.0;
hyper_bigram_bias lr_mul 1.0→5.0
**Result:** val_loss=**3.2785** @ 1555. Worse than Exp 20 (3.2745). Faster optimizer settings hurt.
Consistent with old findings (Exp 4, 10). Reverted.

## Exp 22: Remove hyper_bigram_bias, restore baseline bigram_lambdas in scalars
**Config:** Exp 20 but bigram contribution from per-layer scalars instead of per-sublayer HC param
**Result:** KILLED at step 1000 (val_loss=3.7700). Catastrophically worse. Per-sublayer bigram
granularity is critical. Reverted.

## Exp 22b: Replace hyper_x0_bias with baseline x0_lambdas (per-layer)
**Config:** Exp 20 but x0 contribution from per-layer x0_lambdas (baseline adam_betas/lr_mul)
instead of per-sublayer per-lane hyper_x0_bias.
**Result:** KILLED at step 250 (val_loss=4.7388 vs Exp 20's 4.5614). Much worse. Reverted.

## Exp 23: Lower resid_lambda init (1.02 vs sqrt(1.1))
**Config:** Exp 20 but resid_lambda init 1.02 per sublayer (cumulative ~1.55) instead of sqrt(1.1) (~2.78)
**Result:** val_loss=**3.2794** @ 1555. Worse than Exp 20. sqrt(1.1) init remains optimal. Reverted.

## Exp 24: Scalar x0_bias and bigram_bias (shared across lanes)
**Config:** Exp 20 but x0_bias and bigram_bias are per-sublayer scalars (not per-lane).
Reduces HC params from 44+44 to 22+22. Same optimizer settings.
**Result:** val_loss=**3.2811** @ 1555. Worse than Exp 20 (3.2745). Per-lane specialization
of x0_bias/bigram_bias is valuable. Reverted.

## Exp 25b: Combine HC + baseline x0_lambdas
**Config:** Exp 20 + baseline x0_lambdas (per-layer, adam_betas=[0.65,0.95], lr_mul=5.0).
x0 contribution = (x0_lambdas[i] + x0_bias[o]) * x0. Two signals: per-layer coarse + per-sublayer per-lane fine.
**Result:** val_loss=**3.2796** @ 1555. Worse than Exp 20. Redundant x0 signals interfere. Reverted.

## Exp 26: Vectorized 4D broadcast (optimize step time, attempt 1)
**Config:** Exp 20 but forward pass rewritten: eliminated Python loops, used 4D tensor broadcasting
with `h.unsqueeze(2) * w_post.view(1,1,2,1)` pattern. Removed w_pre/w_res unbinding.
**Result:** val_loss=**3.2731** @ 1555, step_avg=**614ms** (WORSE than 509ms!).
Broadcasting created expensive 4D intermediates that hurt torch.compile fusion. Reverted approach.

## Exp 27: Explicit scalar mults with separate lane tensors (optimize step time, attempt 2)
**Config:** Exp 20 forward pass rewritten: two separate (B,S,D) tensors (`lane0`, `lane1`)
instead of one (B,S,2,D) `x_lanes`. All operations are scalar-tensor multiplies on 3D tensors.
No 4D broadcasting, no unsqueeze, no torch.stack. Pre-split per-lane scalars via unbind.
**Result:** val_loss=**3.2746** @ 1555, step_avg=**468ms** (down from 509ms, **8% faster**).
```
step:0/1555    val_loss:10.8291
step:250/1555  val_loss:4.5551  step_avg:279ms
step:500/1555  val_loss:4.2529  step_avg:274ms
step:750/1555  val_loss:3.8619  step_avg:332ms
step:1000/1555 val_loss:3.5651  step_avg:361ms
step:1250/1555 val_loss:3.3933  step_avg:419ms
step:1500/1555 val_loss:3.2908  step_avg:460ms
step:1555/1555 val_loss:3.2746  step_avg:468ms
```
HC crosses baseline loss (3.2769) around step ~1548. ~6% overhead vs baseline (see AB test below).

## Exp 28: Remove frozen buffers + fold bigram into main expression
**Config:** Exp 27 + removed hyper_w_res/hyper_w_pre registered buffers (hardcoded identity/one-hot),
folded bigram addition into main residual update (no separate `if idx > 0:` branch).
**Result:** val_loss=**3.2741** @ 1555, step_avg=**470ms**. No speed improvement from removing frozen buffers.

## Exp 29: Block-based forward_hc method
**Config:** Exp 28 + moved HC forward logic into Block.forward_hc() method. Hypothesis: kernel fusion
would improve if attention+residual are in the same function scope.
**Result:** val_loss=**3.2766** @ 1540 steps (num_scheduled_iterations=1500), step_avg=**487ms**.
SLOWER than inline approach (470ms). Function boundaries don't help torch.compile. Reverted to inline.

## Exp 30: Clean refactored code, 1540 steps
**Config:** Cleaned up code: removed dead forward_hc method, removed frozen buffers, short var names,
inline HC forward. Fixed double-bigram bug at sublayer 0 (refactor dropped `if idx > 0` guard).
num_scheduled_iterations=1500 (1540 total).
**Result:** val_loss=**3.2776** @ 1540, step_avg=**465ms**. Does NOT beat baseline (3.2769).
Pre-refactor code got 3.2765 at same step count — 0.0011 gap is likely run-to-run variance.
```
step:0/1540    val_loss:10.8303
step:250/1540  val_loss:4.5489  step_avg:259ms
step:500/1540  val_loss:4.2274  step_avg:256ms
step:750/1540  val_loss:3.8582  step_avg:321ms
step:1000/1540 val_loss:3.5618  step_avg:354ms
step:1250/1540 val_loss:3.3922  step_avg:417ms
step:1500/1540 val_loss:3.2890  step_avg:459ms
step:1540/1540 val_loss:3.2776  step_avg:465ms
```
**Verification run (pre-refactor code at 72178d5, same 1540 steps):** val_loss=**3.2795**.
Pre-refactor is WORSE — confirms refactor is clean; 1540 steps is borderline (run-to-run variance ~0.003).

---

## Performance Analysis: AB Test (HC Overhead Isolation)

Ran 4 variants for 100 steps each (num_scheduled_iterations=60, 3 stages of 20 steps + 40 extension)
to isolate HC overhead. Per-stage incremental step times (excluding outliers >800ms):

| Variant | Stage 0 (batch=8) | Stage 1 (batch=16) | Stage 2 (batch=24) |
|---------|------------------|-------------------|-------------------|
| **Master baseline** | 232ms | 432ms | 652ms |
| **A** (1 lane, no HC) | 234ms | 416ms | 663ms |
| **B** (2 lanes, no scalars) | 259ms | 428ms | 631ms |
| **C** (2 lanes, full HC) | 279ms | 457ms | 665ms |

**Key Finding: HC overhead is ~6%, not ~2x as previously estimated.**

The "2x overhead" claim was based on comparing step_avg across different run lengths with different
stage distributions. The actual master baseline step_avg at 1555 steps is ~444ms (not ~220ms).
HC full configuration predicts ~472ms at 1555 steps, matching the measured 470ms.

Breakdown of HC overhead at Stage 2:
- 2-lane tensors: +0ms (within noise of master baseline)
- HC scalar multiplications: +13ms (~2%)
- Total HC overhead: +13ms per step at stage 2

**Corrected summary:**
- Previous "step_avg=220ms baseline" was incorrect (may have been from an earlier codebase/hardware)
- Master actual: ~444ms step_avg @ 1555 steps
- HC actual: ~470ms step_avg @ 1555 steps
- Real overhead: ~26ms or ~6%

## Exp 31: HC at 1545 steps + full master baseline wall time comparison
**HC Config:** Clean refactored code, num_scheduled_iterations=1505 (1545 total).
**Master:** Standard master branch, num_scheduled_iterations=1515 (1555 total).
Both runs on same hardware (single H200), back-to-back.

| | HC | Master | Delta |
|---|---|---|---|
| **val_loss** | **3.2759** | 3.2820 | **-0.0061** |
| **steps** | 1545 | 1555 | -10 |
| **train_time** | 713.9s | 678.4s | +35.5s (+5.2%) |
| **step_avg** | 462ms | 436ms | +26ms (+6.0%) |

HC trajectories:
```
step:0/1545    val_loss:10.8311
step:250/1545  val_loss:4.5661  step_avg:254ms
step:500/1545  val_loss:4.2236  step_avg:253ms
step:750/1545  val_loss:3.8570  step_avg:319ms
step:1000/1545 val_loss:3.5622  step_avg:352ms
step:1250/1545 val_loss:3.3908  step_avg:414ms
step:1500/1545 val_loss:3.2896  step_avg:456ms
step:1545/1545 val_loss:3.2759  step_avg:462ms
```
Master trajectories:
```
step:0/1555    val_loss:10.8305
step:250/1555  val_loss:4.5364  step_avg:230ms
step:500/1555  val_loss:4.2263  step_avg:229ms
step:750/1555  val_loss:3.8806  step_avg:294ms
step:1000/1555 val_loss:3.5755  step_avg:327ms
step:1250/1555 val_loss:3.4016  step_avg:388ms
step:1500/1555 val_loss:3.2982  step_avg:429ms
step:1555/1555 val_loss:3.2820  step_avg:436ms
```
**Note:** Both runs show higher val_loss than historical best (HC 3.2741, master 3.2769).
Run-to-run variance is ~0.003-0.005. HC beats master in this head-to-head by 0.0061.
Wall time cost: +35.5s (+5.2%) for 10 fewer steps and better loss.

---
### Summary of Post-Merge Experiments
**Baseline (historical):** 3.2769 @ 1555 steps, step_avg ~436ms
**Best HC (val_loss):** Exp 28 = 3.2741 @ 1555 steps (**-0.0028**, step_avg=470ms, ~6% overhead)
**HC at 1545 steps:** Exp 31 = 3.2759 @ 1545 (**-0.0010**, +35.5s wall time vs master)
**HC at 1540 steps:** Exp 30 = 3.2776 @ 1540 (borderline, run-to-run variance ~0.003)

Wall time: HC costs ~5% more wall time than master for the same val_loss or better.
On 8xH100 cluster, overhead should be smaller (2-3%) due to compute-bound per-GPU workload.

All modifications to Exp 20 val_loss config have been worse. The config is at a local optimum:
- Frozen w_res (identity), frozen w_pre (round-robin one-hot)
- Learned w_post, x0_bias, bigram_bias (per-sublayer per-lane)
- Learned resid_lambda (per-sublayer, init sqrt(1.1))
- All HC params: adam_betas=[0.9, 0.99], lr_mul=1.0 (except resid_lambda lr_mul=5.0)
- Forward pass: separate lane0/lane1 tensors with explicit scalar mults (Exp 27/28)

---

## === 4xH200 OPTIMIZATION PHASE ===
## Goal: Beat master wall time (175,060ms @ 1555 steps, step_avg=112.58ms)
## HC baseline on 4xH200: 183,519ms @ 1540 steps, step_avg=119.17ms, val_loss=3.2748
## Master on 4xH200: 175,060ms @ 1555 steps, step_avg=112.58ms, val_loss=3.2774

---

### Profiling: Forward/Backward/Optimizer Split (1 GPU, Stage 2, batch=24)

Instrumented forward, backward, and optimizer with `torch.cuda.synchronize()` + `time.perf_counter()`.
5-step warmup, 20-step average. Single GPU, grad_accum_steps=1.

| Variant | Forward | Backward | Optimizer | Total |
|---------|---------|----------|-----------|-------|
| **Master** | 54.3ms | 102.2ms | 4.9ms | 161.4ms |
| **HC original** | 55.5ms | 109.7ms | 4.9ms | 170.1ms |
| **HC precompute** | 55.3ms | 107.7ms | 4.9ms | 167.9ms |
| **HC detach x0+bg** | 54.9ms | 106.5ms | 4.9ms | 166.3ms |
| **HC precompute+bg_detach** | 55.0ms | 106.0ms | 4.9ms | 165.9ms |

**Key finding: Backward is 86% of HC overhead** (+7.5ms bwd vs +1.2ms fwd).
HC adds 154 scalar gradient reductions (22 sublayers × 7 scalars) in backward.

---

## Exp 32: Detach x0 and x0_bigram (cut gradient flow through embeddings)
**Config:** HC original + `x0.detach()` and `x0_bigram.detach()` at HC bias terms.
**Hypothesis:** Embedding gradients through HC bias terms may be redundant with direct embedding gradients.
**Result (4xH200):** step_avg=**117.31ms**, val_loss=**3.2911**. Loss degradation of +0.016.
**Conclusion:** Embedding gradients through HC terms provide critical deep supervision signal. **Reverted.**

## Exp 33: Precompute bias terms (mathematically identical)
**Config:** HC original but precompute `hc_bias0[i] = x0 * xb0[2*i] + x0_bigram * bb0[2*i]` before
the main loop. Separates bias gradient computation from sequential loop backward.
**Result (4xH200):** step_avg=**117.96ms**, val_loss=**3.2734**. Clean win — identical loss, ~1.2ms faster.
train_time=181,663ms.

## Exp 34: Precompute + bigram-only detach
**Config:** Exp 33 + `x0_bigram.detach()` in bias terms (keep x0 gradients, detach bigram only).
**Hypothesis:** Bigram embedding gradients through HC may be less important than x0 gradients.
**Result (4xH200):** step_avg=**117.59ms**, val_loss=**3.2886**. Loss degradation of +0.014.
Bigram gradient flow is also critical. **Reverted.**

## Exp 35: Stacked lanes tensor (2,T,D) for batched backward
**Config:** Merge lane0/lane1 into a single (2,T,D) tensor for batched operations.
**Result:** Backward time DOUBLED (~200ms vs ~110ms). torch.compile generates terrible
kernels for the (2,T,D) broadcast pattern. **Completely abandoned.**

## Exp 36: Attn-only bias (precompute + skip bias for MLP sublayers)
**Config:** Exp 33 + remove x0_bias and bigram_bias from MLP sublayer updates.
MLP becomes `lane = rl * lane + h * wp` (no bias injection).
Attn keeps full `lane = rl * lane + h * wp + precomputed_bias`.
**Hypothesis:** MLP sublayers may not need bias terms; removing them eliminates
22 tensor additions + their backward gradients from the sequential loop.
**Result (4xH200):** step_avg=**115.84ms**, val_loss=**3.2755**, train_time=178,395ms.
**Best config: saved 3.3ms/step** vs original HC. Loss still beats master (3.2774).
Gap to master: 178,395 - 175,060 = 3,335ms (+1.9%).

## Exp 37: Attn-only bias + no MLP w_post
**Config:** Exp 36 + remove w_post from MLP (MLP becomes `lane = rl * lane + h`).
**Hypothesis:** Simplifying MLP update further might save more time.
**Result (4xH200):** step_avg=**115.42ms**, val_loss=**3.2785**. 0.4ms faster but loss
regressed above master's 3.2774. MLP w_post matters for lane differentiation. **Reverted.**

---
### 4xH200 Optimization Summary
| Config | step_avg | val_loss | train_time | vs master |
|--------|----------|----------|------------|-----------|
| Master (1555 steps) | 112.58ms | 3.2774 | 175,060ms | baseline |
| HC original (1540) | 119.17ms | 3.2748 | 183,519ms | +4.8% |
| HC precompute (Exp 33) | 117.96ms | 3.2734 | 181,663ms | +3.8% |
| **HC attn-only bias (Exp 36)** | **115.84ms** | **3.2755** | **178,395ms** | **+1.9%** |
| HC attn-only+no mlp wp (Exp 37) | 115.42ms | 3.2785 | 177,742ms | +1.5% |
| HC full detach (Exp 32) | 117.31ms | 3.2911 | 180,661ms | BROKEN |
| HC precompute+bg detach (Exp 34) | 117.59ms | 3.2886 | 181,091ms | BROKEN |

## Exp 38: Remove MLP resid_lambda + trim HC params
**Config:** Exp 36 + resid_lambda only for attn sublayers (11→11, MLP fixed at 1.0).
x0_bias/bigram_bias trimmed from (22,2,1) to (11,2,1) since only attn sublayer values used.
MLP update: `lane = lane + h * wp` (no rl scaling).
**Result (4xH200):** step_avg=**116.36ms**, val_loss=**3.2762**, train_time=179,189ms.
Within noise of Exp 36. torch.compile already fuses the rl multiply with surrounding ops,
so removing it doesn't save time. **No benefit. Reverted.**

## Exp 39: Reduced step count (num_scheduled=1470, total=1510)
**Config:** Exp 38 config + fewer training steps to close wall time gap.
**Hypothesis:** HC converges faster so fewer steps might still beat master loss.
**Result (4xH200):** step_avg=**116.34ms**, val_loss=**3.2803**, train_time=175,673ms.
val_loss above master's 3.2774. Wall time still over by 613ms.
**HC needs ~1534 steps to reach master's loss, too many for wall time parity.** Reverted.

## Exp 40: Partial HC (layers 0-6 only, standard residual for 7-10)
**Config:** HC active for first 7 layers only. Layers 7-10 use plain `lane += h`.
Hypothesis: HC is most active in early layers (rl=3.5 at layer 0); late layers might not need it.
**Result (4xH200):** step_avg=**114.64ms**, val_loss=**3.2793**, train_time=176,539ms.
Saved ~1.2ms/step but val_loss regressed above master (3.2793 > 3.2774).
Late layer HC contributes meaningful loss quality. **Reverted.**

## Exp 41: Custom torch.autograd.Function for HC update
**Config:** Exp 36 + fused backward: `HCUpdate.apply(lane0, lane1, h, rl, wp0, wp1)`.
Custom backward batches scalar gradient reductions (1 reduction for rl instead of 2).
**Hypothesis:** Batched reductions + fused grad_h saves kernel launches.
**Result (4xH200):** step_avg=**116.06ms**, val_loss=**3.2801**, train_time=178,732ms.
No improvement — torch.compile already generates near-optimal fused kernels.
Custom Function may hinder cross-sublayer fusion. **Reverted.**

---
### Final 4xH200 Optimization Summary
| Config | step_avg | val_loss | train_time | vs master |
|--------|----------|----------|------------|-----------|
| Master (1555 steps) | 112.58ms | 3.2774 | 175,060ms | baseline |
| HC original (1540) | 119.17ms | 3.2748 | 183,519ms | +4.8% |
| HC precompute (Exp 33) | 117.96ms | 3.2734 | 181,663ms | +3.8% |
| **HC attn-only bias (Exp 36)** | **115.84ms** | **3.2755** | **178,395ms** | **+1.9%** |
| HC attn-only+no mlp wp (Exp 37) | 115.42ms | 3.2785 | 177,742ms | +1.5% |
| HC no MLP rl (Exp 38) | 116.36ms | 3.2762 | 179,189ms | +2.4% |
| HC 1510 steps (Exp 39) | 116.34ms | 3.2803 | 175,673ms | +0.4% (loss worse) |
| HC partial layers 0-6 (Exp 40) | 114.64ms | 3.2793 | 176,539ms | +0.8% (loss worse) |
| HC custom autograd (Exp 41) | 116.06ms | 3.2801 | 178,732ms | +2.1% |

**Current best: Exp 36 (attn-only bias).** 1.9% slower wall time but 0.002 better val_loss.

### Analysis: Why the ~3ms/step gap is fundamental on 4xH200

The 2-lane HC architecture requires:
- 2× residual stream tensors (lane0, lane1) through all forward+backward ops
- Per-sublayer scalar-tensor multiplies (rl, wp) for both lanes
- Bias precompute and injection at attention sublayers

Optimizations achieved: **3.33ms/step saved** (119.17→115.84ms, 50% of original gap):
1. Precompute bias outside sequential loop: -1.2ms
2. Skip bias for MLP sublayers: -2.1ms

What doesn't work (remaining 3.26ms gap):
- Removing scalar multiplies: torch.compile fuses them; overhead is memory bandwidth, not compute
- Custom autograd Function: compiler already generates near-optimal kernels
- Partial HC: removing HC from any layers degrades loss below master
- Fewer steps: HC can't reach master's loss fast enough at reduced step count

The residual 2.9% overhead maps to the irreducible memory bandwidth cost of maintaining
2 lane tensors. On a larger cluster (8x GPUs), communication becomes a larger fraction
of step time, reducing HC's relative overhead to ~1-2% where it would likely break even.

## Exp 42: Reduce extension from 40→20 (1520 total steps)
**Config:** Exp 36 + num_extension_iterations=20 (was 40). Total steps: 1520 (was 1540).
**Hypothesis:** HC converges during the scheduled 1500 steps; the extension phase can be shorter.
**Result (4xH200):** step_avg=**115.39ms**, val_loss=**3.2771**, train_time=175,394ms.
val_loss BEATS master (3.2771 < 3.2774)! Train time only 334ms over master (+0.19%).
**Nearly wall-time parity.** The last 20 extension steps contributed negligible improvement.

## Exp 43: HC lr_mul=2.0 + extension=20 (1520 steps)
**Config:** Exp 42 + lr_mul=2.0 for hyper_post, hyper_x0_bias, hyper_bigram_bias (was 1.0).
**Hypothesis:** Higher LR for HC params → faster convergence → room to cut more steps.
**Result (4xH200):** step_avg=**115.46ms**, val_loss=**3.2824**, train_time=175,506ms.
val_loss above master (3.2824 > 3.2774). lr_mul=2.0 overshoots, consistent with Exp 10. **Reverted.**

## Exp 44: HC beta2=0.95 + extension=20 (1520 steps)
**Config:** Exp 42 + adam_betas=[0.9, 0.95] for ALL HC params (was [0.9, 0.99]).
**Hypothesis:** Lower beta2 → faster second-moment adaptation → faster convergence.
**Result (4xH200):** step_avg=**115.33ms**, val_loss=**3.2750**, train_time=175,295ms.
val_loss beats master by 0.0024! Train time only 235ms over master (+0.13%).
**Best yet!** beta2=0.95 helps HC convergence.

## Exp 45: Master-style x0_bias optimizer (beta=[0.65,0.95], lr_mul=5.0)
**Config:** Exp 44 + x0_bias matching master's x0_lambdas optimizer: beta=[0.65, 0.95], lr_mul=5.0.
**Hypothesis:** Master's x0_lambdas use very aggressive settings; matching them might help.
**Result (4xH200):** step_avg=**115.11ms**, val_loss=**3.2831**, train_time=174,961ms.
Aggressive x0_bias settings hurt loss. **Reverted.**

## Exp 46: beta2=0.95 + reduced scheduled=1470 (1510 total)
**Config:** Exp 44 beta2=0.95 + num_scheduled=1470, extension=40. Total 1510 steps.
**Hypothesis:** beta2=0.95 convergence boost lets us cut 30 scheduled steps.
**Result (4xH200):** step_avg=**115.96ms**, val_loss=**3.2788**, train_time=175,106ms.
Train time only 46ms over master! val_loss 0.0014 above master (within noise).

## Exp 47: beta2=0.95 + reduced scheduled=1480 (1520 total)
**Config:** Same but num_scheduled=1480, extension=40. Total 1520 steps.
**Hypothesis:** 10 more steps than Exp 46 should improve loss.
**Result (4xH200):** step_avg=**116.05ms**, val_loss=**3.2807**, train_time=176,393ms.
Worse than both Exp 46 and Exp 44. Cutting scheduled steps (compressing LR schedule)
hurts more than cutting extension steps. Exp 44 (sched=1500, ext=20) is better than
Exp 47 (sched=1480, ext=40) at the same 1520 total steps.

## Exp 48: beta2=0.95 at full 1540 steps (baseline)
**Config:** Exp 36 + beta2=0.95 for ALL HC params. Full 1540 steps (sched=1500, ext=40).
**Hypothesis:** Establish beta2=0.95 baseline at original step count to measure headroom.
**Result (4xH200):** step_avg=**116.03ms**, val_loss=**3.2748**, train_time=178,684ms.
Tiny improvement over Exp 36 (3.2755). beta2=0.95 benefit is small at full step count.

## Exp 49: beta2=0.95 + extension=15 (1515 total steps)
**Config:** Exp 48 + num_extension_iterations=15. Total 1515 steps.
**Hypothesis:** Extension steps contribute diminishing returns; 15 should be sufficient.
**Result (4xH200):** step_avg=**115.13ms**, val_loss=**3.2780**, train_time=174,427ms.
**BEATS MASTER WALL TIME!** 174,427ms < 175,060ms (-633ms, -0.36%).
val_loss only 0.0006 above master (3.2780 vs 3.2774), within +-0.003 noise.

## Exp 50: beta2=0.95 + extension=10 (1510 total)
**Config:** Same as Exp 49 but ext=10. Total 1510 steps.
**Result (4xH200):** step_avg=**114.97ms**, val_loss=**3.2821**, train_time=173,607ms.
Too aggressive — val_loss 0.005 above master. ext=10 not viable.

## Exp 51: bigram_bias init=0.1 + ext=15 (1515 total)
**Config:** Exp 49 + bigram_bias init 0.1 (from 0.05).
**Hypothesis:** Higher bigram injection at init helps early convergence.
**Result (4xH200):** step_avg=**115.10ms**, val_loss=**3.2777**, train_time=174,380ms.
Tiny improvement over Exp 49 (3.2780→3.2777). Within noise. Beats master wall time.

## Exp 52: bigram_bias init=0.15 + ext=15
**Config:** Same as Exp 51 but bigram=0.15.
**Result (4xH200):** step_avg=**115.17ms**, val_loss=**3.2810**, train_time=174,483ms. Overshoots.

## Exp 53: bigram_bias init=0.1 + ext=20 (1520 total)
**Config:** Exp 51 config at ext=20.
**Result (4xH200):** step_avg=**115.55ms**, val_loss=**3.2778**, train_time=175,640ms.
No improvement over Exp 44 (3.2750). bigram=0.1 doesn't clearly help.

## Exp 54: resid_lambda lr_mul=10.0
**Config:** Exp 44 + resid_lambda lr_mul doubled (5.0→10.0).
**Result (4xH200):** step_avg=**115.47ms**, val_loss=**3.2886**, train_time=175,520ms.
Way too high. Overshoots badly. **Reverted.**

## Exp 55: cooldown_frac=0.50
**Config:** Exp 44 + cooldown_frac reduced (0.55→0.50). More peak LR time.
**Result (4xH200):** step_avg=**115.41ms**, val_loss=**3.2781**, train_time=175,422ms.
Worse than 0.55 (3.2781 vs 3.2750).

## Exp 56: cooldown_frac=0.60
**Config:** Exp 44 + cooldown_frac increased (0.55→0.60). More cooldown time.
**Result (4xH200):** step_avg=**115.43ms**, val_loss=**3.2781**, train_time=175,460ms.
Same as 0.50. cooldown_frac=0.55 is already optimal.

## Exp 57: x0_bias init=0.05 (nonzero init)
**Config:** Exp 44 + x0_bias init 0.05 (from 0.0).
**Result (4xH200):** step_avg=**115.39ms**, val_loss=**3.2809**, train_time=175,397ms.
Hurts convergence. x0_bias init=0 is correct.

---
### Convergence Optimization Summary (Exps 42-57)
| Config | Steps | val_loss | train_time | vs master |
|--------|-------|----------|------------|-----------|
| Master | 1555 | 3.2774 | 175,060ms | baseline |
| **Exp 44 (beta2=0.95, ext=20)** | 1520 | **3.2750** | 175,295ms | loss -0.0024, time +0.13% |
| **Exp 49 (beta2=0.95, ext=15)** | 1515 | 3.2780 | **174,427ms** | loss +0.0006, time **-0.36%** |
| Exp 46 (beta2=0.95, sched=1470) | 1510 | 3.2788 | 175,106ms | loss +0.0014, time +0.03% |
| Exp 48 (beta2=0.95, 1540) | 1540 | 3.2748 | 178,684ms | loss -0.0026, time +2.1% |

**Best config: Exp 44** (beta2=0.95, sched=1500, ext=20, 1520 total steps).
Beats master on val_loss (3.2750 vs 3.2774) with only 0.13% wall time overhead.
All other tuning attempts (init, cooldown_frac, LR) were within noise or worse.

## Exp 58: sched=1470, ext=40, cooldown=0.60
**Config:** Exp 46 + cooldown_frac=0.60 (from 0.55). Keeps ext=40 per user preference.
**Result (4xH200):** step_avg=**115.85ms**, val_loss=**3.2788**, train_time=174,934ms.
**Beats master wall time!** 174,934 < 175,060 (-126ms). val_loss 0.0014 above master (within noise).

## Exp 59: sched=1470, ext=40, cooldown=0.65
**Result (4xH200):** step_avg=**116.00ms**, val_loss=**3.2801**, train_time=175,164ms. Worse. cd=0.65 too much.

## Exp 60: Shorter stage 1 (1/4→1/3→5/12 stage split)
**Config:** More time in stage 3 (batch=24) at expense of stage 1 (batch=8).
**Result (4xH200):** step_avg=**124.31ms**, val_loss=**3.2643** (!), train_time=187,706ms.
Amazing loss but step_avg explodes (larger batch = slower steps). Not viable for wall time.

## Exp 61: sched=1460, ext=40 (1500 total), cooldown=0.60
**Result (4xH200):** step_avg=**115.97ms**, val_loss=**3.2797**, train_time=173,961ms.
Great time but loss deteriorating with fewer scheduled steps.

## Exp 62: resid_lambda lr_mul=7.0
**Result (4xH200):** step_avg=**116.09ms**, val_loss=**3.2825**, train_time=175,296ms. Too high, overshoots.

## Exp 63: HC beta1=0.85 (from 0.9)
**Result (4xH200):** step_avg=**115.95ms**, val_loss=**3.2789**, train_time=175,090ms. No difference.

## Exp 64: Base LR=0.0085 (from 0.008)
**Result (4xH200):** step_avg=**115.99ms**, val_loss=**3.2805**, train_time=175,142ms. Slight hurt.

## Exp 65: sched=1450, ext=60 (1510 total), cooldown=0.60
**Result (4xH200):** step_avg=**116.62ms**, val_loss=**3.2801**, train_time=176,099ms.
More extension, fewer scheduled = worse. Compressed schedule hurts more than extra flat-LR helps.

## Exp 66: sched=1465, ext=40, cooldown=0.60 (1505 total)
**Result (4xH200):** step_avg=**116.21ms**, val_loss=**3.2792**, train_time=174,900ms.
Beats master by 160ms. Same loss as 1470-sched configs.

## Exp 67: sched=1465, ext=40, cooldown=0.55 (1505 total)
**Result (4xH200):** step_avg=**116.17ms**, val_loss=**3.2792**, train_time=174,830ms.
**Best ext=40 config.** Beats master by 230ms. val_loss 0.0018 above master (within noise).

---
### Best ext=40 Summary (Exps 58-67)
| Config | Steps | val_loss | train_time | vs master |
|--------|-------|----------|------------|-----------|
| Master | 1555 | 3.2774 | 175,060ms | baseline |
| Exp 58 (sched=1470, cd=0.60) | 1510 | 3.2788 | 174,934ms | -126ms |
| **Exp 67 (sched=1465, cd=0.55)** | **1505** | **3.2792** | **174,830ms** | **-230ms** |
| Exp 46 (sched=1470, cd=0.55) | 1510 | 3.2788 | 175,106ms | +46ms |

Verification runs of sched=1470 config: 3.2788, 3.2788, 3.2792 → avg 3.2789 (robust).
val_loss gap (0.0015-0.0018) is within single-run noise (+-0.003).

---
## Overhead Analysis: Master Precompute Result Challenges "Memory Bandwidth" Hypothesis

### Key Finding
The same precompute-outside-loop optimization was applied to master's residual update
(`x = resid_lambdas[i] * x + x0_lambdas[i] * x0 + bigram_lambdas[i] * x0_bigram`).
On master, it **added** +0.98ms/step (regression). See `msisovic/precompute-bias` branch.

| Version | step_avg (2-run avg) | Delta |
|---------|---------------------|-------|
| Master baseline | 112.22ms | — |
| Master + precompute | 113.20ms | **+0.98ms (slower!)** |
| HC baseline (Exp 36) | 115.84ms | +3.62ms vs master |
| HC + precompute (Exp 33) | 117.96ms → 115.84ms | -1.2ms (helped) |

### Why Precompute Helps HC but Hurts Master
**Master:** torch.compile fuses `resid * x + x0_lambda * x0 + bigram * x0_bigram` into a
single kernel. Precomputing BREAKS this fusion (bias materialized to memory, then loaded back).
**HC:** The 4 lane updates per layer (2 lanes × 2 sublayers) create a longer backward
critical path. Precomputing moves x0/bigram gradient computation OFF this path.

### What This Tells Us About the HC Overhead
The ~3.6ms/step gap is NOT simply "2 lanes = 2× memory bandwidth." If it were, precomputing
would also help master (since it reduces memory ops). The real overhead comes from:

**1. Longer backward critical path (primary)**
HC has 4 sequential lane updates per layer (attn-lane0, attn-lane1, mlp-lane0, mlp-lane1).
Each update creates a new dependency node in the backward graph. Master has just 2 simple
additions per layer (x += attn_out, x += mlp_out) inside a single compiled Block.forward().

**2. More scalar gradient reductions (secondary)**
HC: 6 scalar reductions per layer × 11 layers = 66 reductions (rl, wp0, wp1 for attn+MLP)
Master: 3 scalar reductions per layer × 11 layers = 33 reductions
Each reduction sums a (1,2048,1024) tensor. Extra 33 reductions ≈ 0.5-1.0ms.

**3. Breaking Block compilation unit (structural)**
Master calls `self.blocks[i](x, ...)` — a single function where attn+residual+MLP+residual
are compiled together. HC calls `block.attn(...)` and `block.mlp(...)` separately with
explicit lane updates between them. While fullgraph=True traces everything, the lane
updates between attn and MLP may prevent certain kernel fusions.

### Optimization Implications
The overhead is NOT from simple memory bandwidth but from backward pass complexity.
Potential approaches:
- Reduce the number of sequential lane updates (e.g., attn writes only to lane0, MLP only to lane1)
- Merge the 2 lane updates per sublayer into fewer operations
- Further reduce scalar parameters (fewer gradient reductions)

---

## Exp 68: No cross-lane writes (attn→lane0 only, MLP→lane1 only)
**Config:** Remove all cross-lane writes entirely. Attn only updates lane0, MLP only updates lane1.
No rl, no wp, no bias on the "other" lane. Isolates cross-lane overhead.
**Result (4xH200):** step_avg=**113.65ms**, val_loss=**3.5142**, train_time=175,099ms.
Loss completely broken — confirms cross-lane writes are essential for HC quality.
**Key finding:** Cross-lane writes account for ~2.19ms of ~3.6ms total HC overhead (113.65 vs 115.84).

## Exp 69: Simplified cross-lane (no rl, no bias on cross-lane)
**Config:** Keep cross-lane writes but simplify: attn cross-lane = `lane1 += h * wp1[si]` (no rl, no bias).
MLP cross-lane = `lane0 += h * wp0[si+1]` (no rl, no bias). Home lane keeps full rl + wp + bias.
**Result (4xH200):** step_avg=**114.57ms**, val_loss=**3.2777**, train_time=176,563ms.
Saves 1.27ms/step vs full HC. Loss borderline (0.0003 above master). The rl scalar on cross-lane
is the expensive part, not the bias addition.

## Exp 70: Cross-lane with rl but no bias
**Config:** Cross-lane writes include rl but no bias: `lane1 = rl[si] * lane1 + h * wp1[si]`.
Tests whether bias or rl is responsible for cross-lane overhead.
**Result (4xH200):** step_avg=**115.86ms**, val_loss=similar to full HC.
Same speed as full HC — confirms **rl is the expensive part** (scalar gradient reductions), not bias.

## Exp 71: Merge-before-MLP (merge lanes before MLP, split after)
**Config:** Before MLP sublayer, merge lanes: `merged = (lane0 + lane1) * 0.5`, run MLP on merged,
then split back to lanes. Reduces sequential lane updates.
**Result (4xH200):** step_avg=**115.15ms**, val_loss=**3.3659**, train_time=~177k ms.
Loss broken — merging destroys lane-specific information.

## Exp 72: No-rl cross-lane + 1555 steps (compensate with more training)
**Config:** Exp 69 config (simplified cross-lane, no rl) + sched=1500, ext=55 (1555 total steps).
Adds 15 extra steps to compensate for loss penalty.
**Result (4xH200):** step_avg=**114.57ms**, val_loss=**3.2767**, train_time=178,908ms.
Loss recovered but extra steps negate the speed benefit (178,908 > 175,060ms master).

---
### Cross-Lane Write Analysis Summary (Exps 68-72)
| Config | step_avg | val_loss | vs full HC |
|--------|----------|----------|------------|
| Full HC (Exp 36) | 115.84ms | 3.2755 | baseline |
| No cross-lane (Exp 68) | 113.65ms | 3.5142 | -2.19ms, loss broken |
| **No-rl cross-lane (Exp 69)** | **114.57ms** | **3.2777** | **-1.27ms, borderline loss** |
| rl + no bias cross-lane (Exp 70) | 115.86ms | ~same | +0.02ms, same speed |
| Merge-before-MLP (Exp 71) | 115.15ms | 3.3659 | -0.69ms, loss broken |
| No-rl + extra steps (Exp 72) | 114.57ms | 3.2767 | -1.27ms, but +3.9s wall |

**Conclusion:** The rl scalar multiplication on cross-lane updates accounts for ~1.3ms of overhead
(via scalar gradient reductions in backward). Removing it saves per-step time but costs ~0.002 loss,
and adding steps to compensate negates the wall-time gain. The full cross-lane config remains optimal.

---

## 5-Run Reliability Test: beta2=0.95, sched=1500, ext=40 (1540 total)
Full HC config with adam_betas=[0.9, 0.95] for all HC params.
| Run | val_loss |
|-----|----------|
| 1 | 3.2772 |
| 2 | 3.2776 |
| 3 | 3.2760 |
| 4 | 3.2760 |
| 5 | 3.2760 |
| **Avg** | **3.2766** |

4/5 runs under 3.277. Run 2 barely missed at 3.2776. Need a small margin improvement.

---

## Post 5-Run Testing: Convergence Tuning (Exp 73+)
## Goal: Beat master wall time (175,060ms @ 1555 steps) with val_loss ≤ 3.278 (p < 0.01)

## Exp 73: sched=1475, ext=40 (1515 total steps)
**Config:** Full HC config with beta2=0.95, attn-only bias precompute (Exp 67 base).
num_scheduled_iterations=1475, num_extension_iterations=40.
**Result (4xH200):** step_avg=**116.60ms**, val_loss=**3.2799**, train_time=176,649ms.
**Analysis:** Too slow (+1,589ms vs master, +0.91%). Loss also above target (3.2799 > 3.278).
Need fewer steps to save time.

## Exp 74: sched=1480, ext=40 (1520 total steps)
**Config:** Same as Exp 73, num_scheduled_iterations=1480.
**Result (4xH200):** step_avg=**116.27ms**, val_loss=**3.2788**, train_time=176,735ms.
**Analysis:** Even slower (+1,675ms vs master, +0.96%). Loss improved slightly (3.2788 vs 3.2799)
but still above target. Both Exp 73 and 74 are slower than Exp 67 (sched=1465, 174,830ms).

**Trend Analysis (sched, total steps, train_time, val_loss):**
- Exp 67: 1465, 1505, 174,830ms, 3.2792 ← BEST (beats master by 230ms)
- Exp 73: 1475, 1515, 176,649ms, 3.2799
- Exp 74: 1480, 1520, 176,735ms, 3.2788

**Conclusion:** Increasing scheduled steps beyond 1465 hurts wall time more than it helps convergence.
The step_avg is roughly constant (~116ms), so more steps = worse wall time.

## Exp 75: sched=1460, ext=40, cooldown=0.55 (1500 total steps)
**Config:** Testing if cd=0.55 improves upon Exp 61's cd=0.60 at sched=1460.
**Result (4xH200):** step_avg=**116.47ms**, val_loss=**3.2802**, train_time=174,712ms.
**Analysis:** WORSE than Exp 61 (cd=0.60) on both metrics:
- Exp 61 (cd=0.60): 173,961ms, val_loss=3.2797
- Exp 75 (cd=0.55): 174,712ms, val_loss=3.2802 (+751ms, +0.0005 loss)

**Finding:** At very low step counts (1460), higher cooldown_frac (0.60) helps more than 0.55.
But still worse than Exp 67 (sched=1465, cd=0.55): 174,830ms, val_loss=3.2792.

## Exp 76: sched=1465, ext=40, cooldown=0.50 (1505 total steps)
**Config:** Testing if cd=0.50 improves upon Exp 67's cd=0.55 at sched=1465.
**Result (4xH200):** step_avg=**116.50ms**, val_loss=**3.2798**, train_time=175,338ms.
**Analysis:** WORSE than both cd=0.55 and cd=0.60:
- Exp 67 (cd=0.55): 174,830ms, val_loss=3.2792 ✓ BEST
- Exp 66 (cd=0.60): 174,900ms, val_loss=3.2792
- Exp 76 (cd=0.50): 175,338ms, val_loss=3.2798 (+508ms, +0.0006 loss)

**Finding:** At sched=1465, cd=0.55 is optimal. Lower cooldown (0.50) hurts both time and loss.

**Conclusion:** Exp 67 (sched=1465, cd=0.55) is the optimal configuration after exhaustive search.

---

## Exp 67 Multi-Run Validation (sched=1465, cd=0.55, 1505 total steps)
Goal: Verify mean val_loss ≤ 3.278 with p < 0.01

| Run | val_loss | train_time | step_avg |
|-----|----------|------------|----------|
| 1 | 3.2798 | 175,096ms | 116.34ms |
| 2 | 3.2815 | 175,245ms | 116.44ms |
| 3 | 3.2800 | 175,105ms | 116.35ms |
| 4 | 3.2811 | 175,232ms | 116.43ms |
| 5 | 3.2801 | 175,212ms | 116.42ms |
| **Mean (5)** | **3.2805** | **175,178ms** | **116.39ms** |
| **Std Dev** | **±0.0008** | **±67ms** | - |

**Statistics:**
- Mean: 3.2805, Target: 3.278, **Gap: +0.0025** (0.076% above target)
- Std Dev: 0.0008 (very consistent - systematic, not random)
- Train time: 175,178ms vs master 175,060ms (+118ms, +0.07%)

**Conclusion:**
✅ Wall time parity achieved (+0.07% overhead)
❌ Loss **consistently 0.0025 above target** - need HC initialization tuning
- Original Exp 67: val_loss=3.2792 (single run outlier)
- Mean of 5 runs: 3.2805 (true performance)

---

## HC Initialization Tuning (Exps 77+)
Goal: Shift loss down by ~0.0025 to reach target 3.278

### Exp 77: resid_lambda init = 1.02 (vs sqrt(1.1)≈1.0488)
**Config:** Exp 67 base (sched=1465, cd=0.55) + resid_lambda init lowered to 1.02
**Result:** val_loss=**3.2840**, train_time=175,152ms
**Analysis:** WORSE (+0.0035 vs baseline mean 3.2805). Lower init hurts. Try higher.

### Exp 78: resid_lambda init = 1.06 (vs sqrt(1.1)≈1.0488)
**Config:** Exp 67 base + resid_lambda init increased to 1.06
**Result:** val_loss=**3.2795**, train_time=175,203ms
**Analysis:** ✓ BETTER! (-0.001 vs baseline 3.2805). Higher init helps. Try even higher.

### Exp 79: resid_lambda init = 1.08
**Config:** Exp 67 base + resid_lambda init=1.08
**Result:** val_loss=**3.2821**, train_time=175,173ms
**Analysis:** WORSE (+0.0026 vs 1.06). Too high. **1.06 is optimal for resid_lambda.**

**resid_lambda init search:**
- 1.02: 3.2840 ❌
- sqrt(1.1)≈1.0488: 3.2805 (baseline)
- **1.06: 3.2795** ✓ **BEST**
- 1.08: 3.2821 ❌

### Exp 80: resid_lambda=1.06 + w_post init = 0.98 (vs 1.0)
**Config:** Exp 67 base + resid_lambda init=1.06 + w_post init=0.98
**Result:** val_loss=**3.2784**, train_time=175,079ms
**Analysis:** ✓✓ EVEN BETTER! (-0.0021 vs baseline 3.2805, -0.0011 vs rl=1.06 alone)
**Gap to target:** 3.2784 vs 3.278 = +0.0004 (very close!)

### Exp 81: resid_lambda=1.06 + w_post init = 0.96
**Config:** Exp 67 base + resid_lambda=1.06 + w_post=0.96
**Result:** val_loss=**3.2810**, train_time=175,546ms
**Analysis:** WORSE (+0.0026 vs wp=0.98). Too low. **w_post=0.98 is optimal.**

---

## Summary of Initialization Tuning:
**Best Configuration (Exp 80):**
- resid_lambda init: 1.06 (vs sqrt(1.1)≈1.0488)
- w_post init: 0.98 (vs 1.0)
- **Result: val_loss=3.2784** (single run)
- **Gap to target: +0.0004** (essentially at target!)
- Improvement from baseline: -0.0021 (3.2805 → 3.2784)

**Next:** Run multiple iterations of Exp 80 config for statistical validation.

---

## === 2xH200 EXPERIMENTS (Exps 82+) ===
## Baseline config: rl=1.06, wp=0.98, sched=1465, ext=40, cd=0.55 (1505 total steps)
## Master on origin/master: sched=1515, ext=40 (1555 total steps)

---

### Exp 82: x0_bias lr_mul=5.0 + bigram_bias lr_mul=5.0
**Config:** Exp 80 base + x0_bias lr_mul 1.0→5.0, bigram_bias lr_mul 1.0→5.0 (match master's lambda LR)
**Hypothesis:** Master uses lr_mul=5.0 for x0_lambdas/bigram_lambdas. HC uses 1.0. Maybe higher LR helps.
**Result (2xH200):** val_loss=**3.2801** @ 1505 steps. WORSE. Consistent with Exps 10, 21, 45. **Reverted.**

### Exp 83: rl=1.05, wp=0.95 (round numbers)
**Config:** resid_lambda init=1.05, w_post init=0.95
**Result (2xH200):** val_loss=**3.2783** @ 1505 steps. Essentially tied with Exp 80 (3.2784). Not clearly better.

### Exp 84: rl=1.10, wp=0.95
**Config:** resid_lambda init=1.10, w_post init=0.95
**Result (2xH200):** val_loss=**3.2807** @ 1505 steps. WORSE. rl=1.10 too high, consistent with Exp 79.

### Exp 85: rl=1.05, wp=0.98
**Config:** resid_lambda init=1.05, w_post init=0.98
**Result (2xH200):** val_loss=**3.2800** @ 1505 steps. WORSE than Exp 80. rl=1.06 is better than 1.05.

### Exp 86: rl=1.10, wp=0.98
**Config:** resid_lambda init=1.10, w_post init=0.98
**Result (2xH200):** val_loss=**3.2806** @ 1505 steps. WORSE. Confirms rl=1.10 is too high.

### Exp 87: rl=1.07, wp=0.97
**Config:** resid_lambda init=1.07, w_post init=0.97
**Result (2xH200):** val_loss=**3.2807** @ 1505 steps. WORSE. rl=1.06 + wp=0.98 is the sweet spot.

### Exp 88: rl=1.06, wp=0.95
**Config:** resid_lambda init=1.06, w_post init=0.95
**Result (2xH200):** val_loss=**3.2802** @ 1505 steps. WORSE. wp=0.98 is optimal.

---

### Init Sweep Summary (2xH200, 1505 steps)
| Exp | rl | wp | Other | val_loss |
|-----|------|------|-------|----------|
| **80** | **1.06** | **0.98** | **baseline** | **3.2784** |
| 82 | 1.06 | 0.98 | lr_mul=5.0 biases | 3.2801 |
| 83 | 1.05 | 0.95 | | 3.2783 |
| 84 | 1.10 | 0.95 | | 3.2807 |
| 85 | 1.05 | 0.98 | | 3.2800 |
| 86 | 1.10 | 0.98 | | 3.2806 |
| 87 | 1.07 | 0.97 | | 3.2807 |
| 88 | 1.06 | 0.95 | | 3.2802 |

**Conclusions:**
- rl=1.10 consistently worse (~3.2807). rl≥1.08 overshoots.
- wp=0.95 not clearly better than wp=0.98 (3.2783 vs 3.2784 within noise).
- lr_mul=5.0 for x0_bias/bigram_bias still hurts (consistent across all experiments).
- **rl=1.06, wp=0.98 remains optimal.**

---

### 7-Run Validation: rl=1.06, wp=0.98 @ 1505 steps (2xH200)
| Run | val_loss |
|-----|----------|
| 1 | 3.2800 |
| 2 | 3.2809 |
| 3 | 3.2824 |
| 4 | 3.2810 |
| 5 | 3.2812 |
| 6 | 3.2796 |
| 7 | 3.2809 |
| **Mean** | **3.2809** |
| **Std** | **±0.0009** |

**Statistics:**
- Mean: 3.2809, Target: 3.278, **Gap: +0.0029**
- Does NOT pass p < 0.01 for mean ≤ 3.278 at 1505 steps
- Need either more steps or further loss improvement to reach target

---

## === STEP COUNT SEARCH: wp=1.0, rl=sqrt(1.1) (Exps 89+) ===
## Reverted init to wp=1.0, rl=sqrt(1.1)≈1.0488 (pre-Exp 77 config)
## Goal: Find minimum sched that achieves mean val_loss ≤ 3.278 with p < 0.01
## Master baseline: sched=1515, ext=40 (1555 total)

---

### Exp 89: sched=1515 (1555 total) — master step count
**Config:** wp=1.0, rl=sqrt(1.1), sched=1515, ext=40, cd=0.55
**Result (2xH200, 2 runs before stopped):** 3.2737, 3.2746. Both well under 3.278. Overkill.

### Exp 90: sched=1470 (1510 total)
**Config:** wp=1.0, rl=sqrt(1.1), sched=1470, ext=40, cd=0.55
**Result (2xH200, 5 runs before stopped):** 3.2791, 3.2794, 3.2813, 3.2777, 3.2790
**Mean (5 runs): 3.2793** — too high, does not pass.

### Exp 91: sched=1475 (1515 total)
**Config:** wp=1.0, rl=sqrt(1.1), sched=1475, ext=40, cd=0.55
**Result (2xH200, 6 runs):**
| Run | val_loss |
|-----|----------|
| 1 | 3.2775 |
| 2 | 3.2776 |
| 3 | 3.2800 |
| 4 | 3.2780 |
| 5 | 3.2778 |
| 6 | 3.2781 |
| **Mean** | **3.2782** |
| **Std** | **±0.0009** |
**Analysis:** Mean 3.2782, gap +0.0002. Borderline — one outlier (3.2800) pulls mean up.
Likely does NOT pass p < 0.01. Need slightly more steps.

---

## === 8xH100 EXPERIMENTS ===
## First time testing on 8xH100 cluster.
## HC config: wp=1.0, rl=sqrt(1.1), sched=1475, ext=40 (1515 total steps)
## Master: sched=1515, ext=40 (1555 total steps)

---

### 8xH100 Baselines

| Variant | Steps | val_loss | train_time | step_avg |
|---------|-------|----------|------------|----------|
| **Master** | 1555 | 3.2807 | 92,775ms | **59.66ms** |
| **HC (full 2-lane)** | 1515 | 3.2804 | 93,809ms | **61.92ms** |

**HC overhead on 8xH100: +2.26ms/step (+3.8%).**
Lower than 4xH200's ~6% overhead, as predicted — communication (NCCL allreduce)
is a larger fraction of step time on 8 GPUs, reducing HC's relative overhead.

Peak memory: Master 30,676 MiB / HC 30,532 MiB (HC slightly lower — fewer scalar params).
Reserved: Master 47,738 MiB / HC 50,120 MiB (HC higher reserved — 2-lane activations).

---

### Profiling: Inductor Fusion Analysis

Used `TORCH_LOGS=output_code` + `torch.profiler` to inspect compiled Triton kernels.

**Key finding: The lane0 and lane1 updates ARE fused into single kernels by torch.compile.**

Post-attention kernel (steady-state, e.g. kernel_16):
- 17 loads, 3 stores, 1 reduction — **one kernel** handles:
  1. Deferred MLP lane1 update from previous layer
  2. Both lane0 and lane1 attention residual updates
  3. RMS norm of reading lane for next sublayer
- Both lanes updated via `in_out_ptr0` and `in_out_ptr1` (in-place mutation)

Post-MLP kernel (e.g. kernel_11):
- 4 loads, 1 store, 1 reduction — **one kernel** handles:
  1. Lane0 MLP residual update + norm for next attn input
  2. Lane1 MLP update is **deferred** (fused into next layer's post-attn kernel)

The compiler is quite clever: it defers lane1's MLP update since it's not needed until the
next layer's MLP reads lane1, avoiding a separate kernel launch. This means the overhead is
NOT from extra kernel launches — it's from the irreducible memory bandwidth of reading/writing
2 lane tensors instead of 1 through the fused kernels.

---

### Ablation 1: Single-Lane (no lane1 computation)

Removed all lane1 updates (attn, MLP, skip). MLP reads lane0 instead of lane1.
Final output = lane0 (no averaging). Tests pure memory bandwidth cost of lane1.

**Result (8xH100):** step_avg=**60.16ms**, val_loss=3.2830 (loss irrelevant, different model)

| Variant | step_avg | vs Master | vs Full HC |
|---------|----------|-----------|------------|
| Master | 59.66ms | — | — |
| Full 2-lane HC | 61.92ms | +2.26ms (+3.8%) | — |
| **Single-lane (no lane1)** | **60.16ms** | **+0.50ms (+0.84%)** | **-1.76ms** |

**Analysis:** Removing lane1 saves 1.76ms of the 2.26ms overhead (78%).
The remaining 0.50ms is from HC's scalar machinery on lane0 only (rl, wp, bias precompute).

---

### Ablation 2: Bandwidth-Only Lane1 (running)

Lane1 allocated and read/written each sublayer, but with minimal compute:
`lane1 = rl[si] * lane1` (just a scalar multiply — forces read+write without h/wp/bias ops).
Final merge: `(lane0 + lane1) * 0.5` to keep lane1 in the autograd graph.

Tests whether the overhead is from memory bandwidth (reading/writing lane1 data)
or from the extra compute (h*wp additions, bias injections, scalar gradient reductions).

**Interpretation grid:**
- If bandwidth-only ≈ full 2-lane (~61.9ms) → overhead is memory bandwidth
- If bandwidth-only ≈ no lane1 (~60.2ms) → overhead is the extra compute/scalar gradients

**Result (8xH100):** step_avg=**60.87ms**, val_loss=3.2930 (loss irrelevant, different model)

| Variant | step_avg | vs Master | vs Full HC |
|---------|----------|-----------|------------|
| Master | 59.66ms | — | — |
| Full 2-lane HC | 61.92ms | +2.26ms (+3.8%) | — |
| Single-lane (no lane1) | 60.16ms | +0.50ms (+0.84%) | -1.76ms |
| **Bandwidth-only lane1** | **60.87ms** | **+1.21ms (+2.0%)** | **-1.05ms** |

**Analysis:** Bandwidth-only is between single-lane and full HC, landing at 60.87ms.
This means the overhead is a **mix** of both memory bandwidth and extra compute:

| Source | Cost | % of total overhead |
|--------|------|---------------------|
| Lane0 scalar machinery (rl, wp, bias) | +0.50ms | 22% |
| Lane1 memory bandwidth (read+write per sublayer) | +0.71ms | 31% |
| Lane1 extra compute (h*wp, bias, scalar gradients) | +1.05ms | 47% |
| **Total HC overhead** | **+2.26ms** | **100%** |

**Conclusion:** Nearly half the overhead (47%) comes from the extra compute and scalar
gradient reductions on lane1, not just memory traffic. This suggests optimization paths:
- **Sharing scalars across lanes** (e.g., single wp, single rl) would reduce scalar gradient
  reductions and per-lane bias computations — targeting the 47% compute overhead.
- **Sharing biases across lanes** would eliminate duplicate x0_bias/bigram_bias per lane.
- Memory bandwidth (31%) is harder to reduce without fundamentally changing the 2-lane design.

---

### Optimization Attempts: Sharing Scalars/Biases

Three approaches tested to reduce the 47% compute overhead:

| Variant | step_avg | vs Master | vs Full HC | val_loss |
|---------|----------|-----------|------------|----------|
| Full 2-lane HC (baseline) | 61.92ms | +2.26ms | — | 3.2804 |
| Shared bias (same hc_bias for both lanes) | 62.24ms | +2.58ms | **+0.32ms worse** | 3.2798 |
| Bias-on-h (add bias to sublayer output before lane expansion) | 61.81ms | +2.15ms | **-0.11ms** | 3.2781 |
| Per-layer rl (rl once per layer, init 1.1, not per sublayer) | 61.77ms | +2.11ms | **-0.15ms** | 3.2818 |

**Shared bias** was slower — gradient fan-in from both lanes consuming the same
`hc_bias[i]` tensor changed inductor's fusion decisions for the worse.

**Bias-on-h** gave a small 0.11ms win by adding bias to `h` before `h*wp` lane expansion,
avoiding the fan-in. Single consumer of hc_bias, naturally distributed via wp.

**Per-layer rl** saved 0.15ms by halving resid_lambda multiplies (11 vs 22), but val_loss
regressed (+0.0014 vs full HC). The per-sublayer granularity may matter for quality.

**Conclusion:** These scalar-level optimizations yield diminishing returns (0.1-0.15ms).
The bulk of the overhead is fundamental: lane1 memory bandwidth (0.71ms) and the per-lane
`h * wp` multiplications + their backward gradients.

---

## === BIAS-ON-H + SKIP/BACKOUT EXPERIMENTS ===

### Bias-on-h implementation (permanent change)
Replaced per-lane x0_bias and bigram_bias `(n_sublayers, 2, 1)` with shared scalars `(n_sublayers,)`.
Single `hc_bias` added to `h` before wp multiply, distributed to lanes via `h * wp0` / `h * wp1`.
Previously measured at -0.11ms on 8xH100 with val_loss=3.2781 (beats master 3.2807).

### Backout removal test
Removed `x = x - backout_lambda * x_backout` and backout capture.
**Result:** Hurt val_loss slightly. **Reverted.**

### Full wp0/wp1 Analysis (learned weights, 2xH200)
```
Layer Type  si      rl     wp0     wp1  |diff|    x0_b      bb
-----------------------------------------------------------------
    0 attn   0  3.8514  1.4748  0.9514  0.5234  0.9592  0.0250
    0  mlp   1  4.9150  0.3074  0.1362  0.1711  0.0000  0.0500
    1 attn   2  0.6331  1.2106  1.3695  0.1589  0.7594  0.9640
    1  mlp   3  1.3601  0.6769  0.3691  0.3078  0.0000  0.0500
    2 attn   4  0.5172  1.5952  1.2749  0.3203  1.8705 -0.6549
    2  mlp   5  1.3252  0.4317  0.6774  0.2458  0.0000  0.0500
    3 attn   6  0.6356  0.6693  1.6723  1.0030  0.1970  0.5832
    3  mlp   7  0.9629  0.9285  0.6688  0.2597  0.0000  0.0500
    4 attn   8  0.4809  1.3223  1.3018  0.0205  1.2467  0.5896
    4  mlp   9  1.0102  0.9135  0.6806  0.2329  0.0000  0.0500
    5 attn  10  0.4667  1.2983  1.3155  0.0172  1.4795 -0.1873
    5  mlp  11  1.0595  0.5166  0.7446  0.2280  0.0000  0.0500
    6 attn  12  1.0488  1.0000  1.0000  0.0000  0.0000  0.0500
    6  mlp  13  0.9714  0.7552  0.8783  0.1232  0.0000  0.0500
    7 attn  14  0.5931  1.0198  1.5213  0.5015  0.7137  0.3025
    7  mlp  15  0.9608  0.5852  0.6740  0.0887  0.0000  0.0500
    8 attn  16  0.8994  0.6081  1.9730  1.3649 -0.0486 -0.1113
    8  mlp  17  1.0292  0.1367  0.6113  0.4746  0.0000  0.0500
    9 attn  18  0.8100  0.8111  1.8982  1.0871 -0.1497 -0.1259
    9  mlp  19  0.8962  0.0319  0.6349  0.6030  0.0000  0.0500
   10 attn  20  0.6604  1.3733  2.2150  0.8417 -0.8692  0.4248
   10  mlp  21  1.4194  0.4339  0.4339  0.0000  0.0000  0.0500
```

**Key patterns:**
- HC barely differentiates at layers 4-5 attn (|diff|=0.02), layer 6 (|diff|=0.00), layer 7/10 mlp (<0.09)
- HC differentiates heavily at layers 3,8,9,10 attn (|diff| > 0.84) — late attn routes heavily to lane1
- Layer 6 attn wp stuck at init (1.0, 1.0) — skip bypasses HC entirely

**Skip connection finding:** Skip currently bypasses HC lane machinery completely.
Injects `skip_gate_out * skip_val` directly into both lanes without going through wp/rl.
Layer 6 HC params are dead (stuck at init). Routing skip through HC could improve quality.

---

## Exp 92+: Skip Connection Experiments (2xH200)
### Baseline (bias-on-h, sched=1475, ext=40, 1515 total): val_loss=3.2777

### Exp 92: Route skip through HC (wp/rl)
**Config:** Treat skip output as sublayer h, route through `rl[si] * lane + h * wp[si]` + bias.
Layer 6 attn wp values now actually train instead of being stuck at init.
**Result:** val_loss=**3.2775**. Essentially identical to baseline. Layer 6 wp will now learn.
**Decision:** Keep — cleaner integration, no loss penalty.

### Exp 93: Skip injects to lane1 only (MLP stream)
**Config:** Exp 92 but skip only updates lane1 (`lane0 = rl * lane0` with no h addition).
**Hypothesis:** Skip replaces attn at layer 6; MLP reads lane1, so skip should target lane1.
**Result:** val_loss=**3.2808**. Worse by 0.003. Skip needs to reach both lanes. **Reverted.**

### Exp 94: Skip as MLP input only (not persisted to lanes)
**Config:** Skip value added to lane1 before MLP norm, but not to lane state.
**Hypothesis:** Skip might work as a temporary input augmentation without modifying residual stream.
**Result:** val_loss=**3.2822**. Worse by 0.0045. Skip must persist in residual stream. **Reverted.**

### Exp 95: Skip saves lane average instead of lane0
**Config:** Exp 92 but `skip_connections.append((lane0 + lane1) * 0.5)` instead of lane0.
**Hypothesis:** Average carries information from both streams for richer skip signal.
**Result:** val_loss=**3.2803**. Worse by 0.003. Lane0 (attn stream) is the right source. **Reverted.**

**Skip experiment summary:**
- **Best: Exp 92** (skip through HC wp/rl) — val_loss=3.2775 (matches baseline)
- Skip must: (1) persist to lane state, (2) go to both lanes, (3) save from lane0
- Routing through HC wp/rl added 1.2s wall time for no loss benefit. **Reverted to simple skip.**

---

## Selective HC Experiments (2xH200)
### Baseline: val_loss=3.2777, train_time=329,743ms @ 1515 steps

### Exp 96: Skip lane1 update entirely at layers 4-5 attn
**Config:** At layers 4-5 attn sublayers (|wp0-wp1|=0.02), don't update lane1 at all.
**Result:** val_loss=**3.2868** (+0.009), train_time=329,821ms (no time savings!)
**Analysis:** lane1 misses rl=0.48/0.47 decay → magnitude drift. No time saved because
torch.compile didn't actually eliminate the lane1 ops (still in the graph from other sublayers).

### Exp 97: Shared wp (wp0 for both lanes) at layers 4-5 attn
**Config:** `lane1 = rl * lane1 + h * wp0` (use wp0 instead of wp1) at layers 4-5 attn.
**Result:** val_loss=**3.2788** (+0.001), train_time=329,756ms (no time savings)
**Analysis:** Still does full read-modify-write on lane1. Only saves one scalar grad reduction.

### Exp 98: No cross-lane writes at layers 4-6 (rl only on cross lane)
**Config:** Attn: `lane1 = rl * lane1` (no h*wp1). MLP: `lane0 = rl * lane0` (no h*wp0).
**Result:** val_loss=**3.2909** (+0.013), train_time=330,126ms (no time savings!)
**Analysis:** Still does `rl * lane` read-modify-write. Must truly NOT TOUCH the cross lane.

### Exp 99: TRUE no-touch cross lane at layers 4-5
**Config:** At layers 4-5: attn doesn't touch lane1 AT ALL, MLP doesn't touch lane0 AT ALL.
**Result:** val_loss=**3.2883** (+0.011), train_time=**328,604ms** (-1,139ms!)
**Analysis:** First real time savings! Skipping 4 lane ops saves ~1.1s.
But loss too high — lane1 goes stale for 2 layers (misses rl decay + h injection).

### Exp 100: Single-stream layers 4-5 (MLP reads lane0)
**Config:** At layers 4-5: attn writes lane0 only, MLP reads lane0 (not stale lane1), writes lane0 only.
Lane1 completely frozen through these layers.
**Result:** val_loss=**3.2850** (+0.007), train_time=**328,489ms** (-1,254ms)

### Exp 101: Single-stream layers 4-7
**Config:** Expanded single-stream to layers 4-7.
**Result:** val_loss=**3.2936** (+0.016), train_time=**327,852ms** (-1,891ms)
Layer 7 has |diff|=0.50 — it needs HC. Too aggressive.

### Exp 102: Single-stream layers 4-6
**Config:** Single-stream layers 4-6 (layer 6 is skip layer with minimal HC differentiation).
**Result:** val_loss=**3.2861** (+0.008), train_time=**327,940ms** (-1,803ms)

### Selective HC Summary
| Exp | Layers | train_time | Δtime | val_loss | Δloss |
|-----|--------|------------|-------|----------|-------|
| baseline | — | 329,743ms | — | 3.2777 | — |
| 100 | 4-5 single | 328,489ms | -1,254ms | 3.2850 | +0.007 |
| 102 | 4-6 single | 327,940ms | -1,803ms | 3.2861 | +0.008 |
| 101 | 4-7 single | 327,852ms | -1,891ms | 3.2936 | +0.016 |

### Exp 103: MLP-only single-stream layers 4-7 (attn stays dual)
**Config:** Attn always dual-lane, MLP reads lane0 + writes lane0 only at layers 4-7.
**Result:** val_loss=**3.2878** (+0.010), train_time=**329,121ms** (-622ms)
Keeping attn dual costs time without helping loss enough.

### Exp 104: Single-stream layers 5-6 only
**Config:** Narrower range (just 2 layers).
**Result:** val_loss=**3.2850** (+0.007), train_time=**329,798ms** (-55ms only)
Too few layers — negligible time savings.

### Exp 105: Exp 102 + ext=48 (1523 total steps)
**Config:** Layers 4-6 single-stream + 8 extra extension steps to compensate for loss.
**Result:** val_loss=**3.2868** (+0.009), train_time=**330,820ms** (+1,077ms!)
Extra steps cost more time than we saved. Loss still not recovered.

### Selective HC Conclusion
| Exp | Config | train_time | Δtime | val_loss | Δloss |
|-----|--------|------------|-------|----------|-------|
| baseline | full HC | 329,743ms | — | 3.2777 | — |
| 100 | L4-5 single | 328,489ms | -1,254ms | 3.2850 | +0.007 |
| **102** | **L4-6 single** | **327,940ms** | **-1,803ms** | **3.2861** | **+0.008** |
| 101 | L4-7 single | 327,852ms | -1,891ms | 3.2936 | +0.016 |
| 103 | L4-7 MLP-only | 329,121ms | -622ms | 3.2878 | +0.010 |
| 104 | L5-6 single | 329,798ms | -55ms | 3.2850 | +0.007 |
| 105 | L4-6 + ext=48 | 330,820ms | +1,077ms | 3.2868 | +0.009 |

**Verdict:** Single-stream middle layers save ~1.8s wall time but cost ~0.008 loss.
Extra steps can't compensate — the loss degradation per removed lane op is too steep.
The 2-lane architecture is load-bearing at every layer. Even low-differentiation layers
(|wp0-wp1|=0.02) contribute meaningful gradient paths that aid convergence.

---

## Phase 7: Scheduled Steps Increase with Single-Stream (L4-6)

Testing whether increasing *scheduled* steps (which changes LR schedule) rather than extension
steps can recover the ~0.008 loss from single-stream L4-6.

### Exp 106: Single-stream L4-6, sched=1480 (+5), ext=40 (1520 total)
**Config:** Same as Exp 102 but with 5 more scheduled steps (affects LR cooldown schedule).
**Result:** val_loss=**3.2863** (+0.009), train_time=**329,222ms**
Loss essentially unchanged from Exp 102 (3.2861). Extra steps just cost wall time.

### Exp 107: Single-stream L4-6, sched=1485 (+10), ext=40 (1525 total)
**Config:** 10 more scheduled steps than baseline.
**Result:** val_loss=**3.2857** (+0.008), train_time=**330,148ms**
Tiny improvement vs Exp 106 but still same loss as Exp 102. +6.4s wall time wasted.

### Scheduled Steps Conclusion
| Exp | sched | total | train_time | val_loss | Δloss |
|-----|-------|-------|------------|----------|-------|
| 102 | 1475 | 1515 | 327,940ms | 3.2861 | +0.008 |
| 106 | 1480 | 1520 | 329,222ms | 3.2863 | +0.009 |
| 107 | 1485 | 1525 | 330,148ms | 3.2857 | +0.008 |

**Verdict:** Increasing scheduled steps (not extension) does NOT recover single-stream loss.
The ~0.008 loss degradation from removing lane1 at L4-6 is structural, not a training
duration issue. More LR budget can't fix information that simply isn't flowing through lane1.

---

## Phase 8: Late-Only HC (2-lane only at last few layers)

Key insight from weight analysis: late layers (L7-10) show massive cross-lane routing
(L9 attn: wp0=0.55, wp1=2.13). Early layers use HC mainly for rl amplification which
works fine on a single stream. Strategy: single-stream L0-6, full 2-lane L7-10.

### Exp 108: Late-only HC (L7-10), sched=1515, ext=40 (1555 total — matches master)
**Config:** Single-stream L0-6 (only lane0, MLP reads lane0). lane1 introduced at L7
by copying lane0. Full 2-lane L7-10. Same step count as master for direct comparison.
**Result:** val_loss=**3.2747** (master=3.2769, **-0.002 BETTER**), train_time=**333,815ms**
HC at just 4 layers (L7-10) beats master loss! Late-layer cross-lane routing is the
core convergence mechanism. wp1 at L0-6 stayed at init (1.0, never used).
Only 8 sublayers have dual lanes (vs 22 in full HC) — major per-step overhead reduction.

---

## Phase 9: w_post Init Tuning for Late-Only HC (8xH100)

### Architecture recap
Late-only HC: single-stream L0-6 (only lane0), full 2-lane L7-10.
`hc_start=7`. Lane1 introduced at L7 by copying lane0.
Master baseline (8xH100): val_loss=3.2807, train_time=92,775ms @ 1555 steps (step_avg=59.66ms)

---

### Exp 109: First record-beating run (uniform wp init=1.0)
**Config:** Late-only HC, sched=1475, ext=40 (1515 total), wp init=1.0 for all sublayers.
**Result (8xH100):** val_loss=**3.2794**, train_time=**92,426ms**, step_avg=61.01ms
**First record!** Beats master on both val_loss (-0.0013) and wall time (-349ms).

**Learned w_post weights (uniform init=1.0):**
```
Layer Type  si      rl     wp0     wp1  |diff|    x0_b      bb
-----------------------------------------------------------------
    7 attn  14  0.6877  0.9914  1.3937  0.4023  0.7402  0.3194
    7  mlp  15  1.1052  0.6850  0.6438  0.0412  0.0000  0.0500
    8 attn  16  0.9089  0.7224  1.9524  1.2300  0.0418 -0.2057
    8  mlp  17  1.0278  0.1540  0.6128  0.4588  0.0000  0.0500
    9 attn  18  0.8163  0.8758  1.9852  1.1094 -0.1988 -0.1152
    9  mlp  19  0.9320  0.0248  0.6245  0.5998  0.0000  0.0500
   10 attn  20  0.6625  1.4544  2.2998  0.8454 -0.8628  0.4887
   10  mlp  21  1.4320  0.4760  0.4760  0.0000  0.0000  0.0500
```

**Key patterns identified:**
- **Attn wp1 → ~2.0** (1.39, 1.95, 1.99, 2.30) — strong signal from attn into lane1 (MLP stream)
- **Attn wp0 ≈ 1.0** (0.99, 0.72, 0.88, 1.45) — noisy, stays near init
- **MLP wp0 → ~0.5 everywhere** (single-stream: 0.19-0.58, HC: 0.03-0.69) — MLP output weakly feeds lane0
- **MLP wp1 → ~0.5** (HC: 0.48-0.64) — moderate MLP self-feedback into lane1

---

### Exp 110: Differentiated wp init (attn wp1=2.0, MLP wp0=0.5)
**Config:** Exp 109 + differentiated w_post init based on learned patterns:
- Single-stream MLP wp0: 1.0 → 0.5
- HC attn wp0: 1.0 (unchanged), wp1: 1.0 → **2.0**
- HC MLP wp0: 1.0 → **0.5**, wp1: 1.0 → **0.5**
**Hypothesis:** Initing closer to learned values reduces optimizer work, faster convergence.
**Result (8xH100):** val_loss=**3.2796**, train_time=**92,197ms**, step_avg=61.06ms @ 1510 steps
Reduced step count by 5 (sched=1470). Essentially tied on loss, saved wall time.

**Learned weights shifted further from init (confirming trends):**
```
Layer Type  si      rl     wp0     wp1
-----------------------------------------------------------------
    7 attn  14  0.8322  0.9355  2.3795
    7  mlp  15  1.0811  0.0318  0.2572
    8 attn  16  0.9786  0.5248  2.5434
    8  mlp  17  1.0506  0.0884  0.2801
    9 attn  18  0.9030  0.4428  2.5978
    9  mlp  19  0.8965  0.0348  0.3135
   10 attn  20  0.7091  1.4004  2.7397
   10  mlp  21  1.4159  0.2283  0.2283
```

**Updated patterns (init 1.0→2.0 for attn wp1, 1.0→0.5 for MLP):**
- Attn wp1 still climbing: 2.38, 2.54, 2.60, 2.74 (wants even higher than 2.0 init)
- Attn wp0 dropping in L8-9: 0.52, 0.44 (wants ~0.5-0.75)
- MLP wp0 dropped further: 0.03, 0.09, 0.03, 0.23 (wants ~0, optimizer decays from 0.5)
- MLP wp1 dropped to: 0.26, 0.28, 0.31, 0.23 (near 0.25)

---

### Exp 111: MLP wp1 init = 0.25 (accidental, meant to test wp0)
**Config:** Exp 110 + HC MLP wp1: 0.5 → 0.25 (user meant to change wp0 instead)
**Result (8xH100):** val_loss=**3.2814**, train_time=91,899ms @ 1510 steps
**WORSE** (+0.0018 vs Exp 110). Despite 0.25 matching learned MLP wp1 values (~0.23-0.31),
initing closer to the learned value hurt performance.

**Key insight: Init closer to learned value ≠ better.**
The optimizer benefits from starting MLP weights higher (0.5) and decaying down.
The learning trajectory (exploration from 0.5→0.25) matters, not just the destination.
This constrains future init tuning — aggressive lowering of MLP inits is counterproductive.

---

### Exp 112: More aggressive attn init (wp0=0.75, wp1=2.5), MLP kept at 0.5
**Config:** Exp 110 + targeted changes informed by Exp 111 lesson:
- HC attn wp0: 1.0 → **0.75** (learned 0.52-0.94, mean ~0.7 in L8-9)
- HC attn wp1: 2.0 → **2.5** (learned 2.38-2.74, clear upward trend)
- MLP wp0: 0.5 (unchanged — optimizer needs decay room)
- MLP wp1: 0.5 (unchanged — 0.25 hurt in Exp 111)
**Hypothesis:** Attn weights trend upward from init, so pushing higher should help.
MLP weights trend downward, so keep init at 0.5 to preserve optimizer dynamics.
**Status:** Running.

---

### w_post Init Tuning Summary
| Exp | Attn wp0 | Attn wp1 | MLP wp0 | MLP wp1 | val_loss | Steps |
|-----|----------|----------|---------|---------|----------|-------|
| 109 | 1.0 | 1.0 | 1.0 | 1.0 | **3.2794** | 1515 |
| **110** | **1.0** | **2.0** | **0.5** | **0.5** | **3.2796** | **1510** |
| 111 | 1.0 | 2.0 | 0.5 | 0.25 | 3.2814 | 1510 |
| 112 | 0.75 | 2.5 | 0.5 | 0.5 | 3.2830 | 1510 |
| 113 | 0.75 | 3.0 | 0.5 | 0.5 | 3.2815 | 1510 |
| 114 | 0.5 | 3.0 | 0.5 | 0.5 | 3.2817 | 1510 |
| 115 | 1.0 | 3.0 | 0.5 | 0.5 | 3.2814 | 1510 |
| 116 | 1.0 | 2.0 | 0.25 | 0.5 | 3.2829 | 1510 |
| 117 | 1.0 | ramped 2→3 | 0.5 | 0.5 | 3.2824 | 1510 |
| 118 | 1.0 | 1.0 | 1.0 | 1.0 | 3.2804 | 1510 |
| 119 | 1.0 | 2.5 | 0.5 | 0.75 | 3.2815 | 1510 |
| 120 | 1.0 | 1.0 | 0.5 | 1.0 | 3.2811 | 1510 |
| 121 | 1.5 | 1.5 | 1.5 | 1.5 | 3.2806 | 1510 |
| 122 | 0.5 | 0.5 | 0.5 | 0.5 | 3.2879 | 1510 |
| 123 | 2.0 | 2.0 | 2.0 | 2.0 | 3.2818 | 1510 |
| **124** | **1.0** | **1.5** | **1.0** | **1.5** | **3.2800** | **1510** |
| 125 | 1.0 | 1.25 | 1.0 | 1.25 | 3.2815 | 1510 |
| 126 | 1.0 | 1.75 | 1.0 | 1.75 | 3.2812 | 1510 |
| **127** | **1.0** | **1.5(attn only)** | **1.0** | **1.0** | **3.2795** | **1510** |
| 128 | 1.0 | 1.5(attn only) | 1.0 | 1.0 | 3.2817 (repeat) | 1510 |
| 129 | 1.0 | 2.0(attn only) | 1.0 | 1.0 | 3.2809 | 1510 |
| 130 | 1.0 | 1.5(attn only) | 1.0 | 1.0 | 3.2804 (hc_start=6) | 1510 |
| **131** | **1.0** | **1.5(attn only)** | **1.0** | **1.0** | **3.2785** (hc_start=5) | **1510** |
| 132 | 1.0 | 1.5(attn only) | 1.0 | 1.0 | 3.2809 (hc_start=4) | 1510 |
| 133 | 1.0 | 1.5(attn only) | 1.0 | 1.0 | 3.2791 (hc_start=5, repeat) | 1510 |
| 134 | 1.0 | 1.0 | 1.0 | 1.0 | 3.2815 (hc_start=5, all-1.0) | 1510 |
| 135 | 1.0 | 1.5(attn only) | 0.5 | 1.0 | 3.2796 (hc_start=5) | 1510 |
| 136 | 1.0 | 1.75(attn only) | 1.0 | 1.0 | 3.2788 (hc_start=5) | 1510 |
| 137 | 1.25(SS) | 1.5(attn only) | 1.0 | 1.0 | 3.2785 (hc5, SS attn wp0=1.25) | 1510 |
| 138 | 1.5(SS) | 1.5(attn only) | 1.0 | 1.0 | 3.2803 (hc5, SS too high) | 1510 |
| 139 | 1.0 | 1.5(attn) | 1.0 | 1.25(HC MLP) | 3.2819 (hc5, MLP wp1 boost hurts) | 1510 |
| 140 | 1.0 | 1.5(attn only) | 1.0 | 1.0 | 3.2794 (hc5, 1475+40=1515 steps) | 1515 |
| 141 | 1.25(SS) | 1.75(attn only) | 1.0 | 1.0 | 3.2800 (hc5) | 1510 |
| 142 | 1.0 | 1.5(attn only) | 1.0 | 1.0 | *running* (hc5, 3rd run) | 1510 |

**Exp 131/133 mean: 3.2788 ± 0.0003** (attn-only wp1=1.5, hc_start=5)
**Exp 127/128 mean: 3.2806 ± 0.0011** (attn-only wp1=1.5, hc_start=7)

**hc_start sweep (with attn wp1=1.5):**
| hc_start | val_loss | Memory |
|----------|----------|--------|
| 4 | 3.2809 | 34304 MiB |
| **5** | **3.2788 (mean)** | **34160 MiB** |
| 6 | 3.2804 | 34016 MiB |
| 7 | 3.2806 (mean) | 33944 MiB |

**Key findings:**
1. **HC expansion helps!** hc_start=5 (3.2788 mean) > hc_start=6 (3.2804) ≈ hc_start=7 (3.2806)
2. **hc_start=4 is too early (3.2809)** — diminishing returns, more memory
3. **Exp 131/133 (hc_start=5, attn wp1=1.5): mean 3.2788** — approaching 3.278 target!
4. **Best init: attn-only wp1=1.5, all else 1.0** — mild asymmetry for lane1 injection
5. **Attn wp1=1.5 > 2.0 > 1.25**: sweet spot is 1.5 for HC attn asymmetry
6. **MLP should stay at 1.0**: adding MLP wp1=1.5 slightly worse than attn-only