# Hyper-Connections Experiment Log

**Baseline (master, no HC):** val_loss=3.2769 @ 1600 steps
**Previous best HC:** val_loss=3.2809 (4 lanes, post-apply, identity init, flat post=1.0)

---

# Hyper-Connections Experiment Log

**Baseline (master, no HC):** val_loss=3.2769 @ 1600 steps
**Previous best HC:** val_loss=3.2809 (4 lanes, post-apply, identity init, flat post=1.0)

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