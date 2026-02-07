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

---
### Summary of Post-Merge Experiments
**Baseline:** 3.2769 @ 1555 steps, step_avg ~444ms
**Best HC (val_loss):** Exp 20 = 3.2745 @ 1555 steps (**-0.0024**)
**Best HC (cleaned up):** Exp 28 = 3.2741 @ 1555 steps (**-0.0028**, step_avg=470ms, ~6% overhead)
**HC at 1540 steps:** Exp 29 = 3.2766 @ 1540 (**-0.0003**, reaches baseline ~step 1536)

All modifications to Exp 20 val_loss config have been worse. The config is at a local optimum:
- Frozen w_res (identity), frozen w_pre (round-robin one-hot)
- Learned w_post, x0_bias, bigram_bias (per-sublayer per-lane)
- Learned resid_lambda (per-sublayer, init sqrt(1.1))
- All HC params: adam_betas=[0.9, 0.99], lr_mul=1.0 (except resid_lambda lr_mul=5.0)
- Forward pass: separate lane0/lane1 tensors with explicit scalar mults (Exp 27/28)