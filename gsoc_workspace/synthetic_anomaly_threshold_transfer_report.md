# Synthetic Anomalies as a Threshold-Calibration Proxy for Unsupervised Anomaly Detection

*GSoC — Anomalib. Experiments run on anomalib 2.5.2-dev.0, 2× RTX 3090 (+ 8× RTX 3090
remote for phase 6). Report generated 2026-07-29; §6–7 (MVTec AD 2 extension) added
2026-08-05, phase 6 completed and finalized 2026-08-24.*

## Abstract

Unsupervised anomaly-detection models are trained on normal images only, yet deploying
them requires a **decision threshold** that is conventionally tuned on a labelled set of
*real* anomalies — precisely the data the unsupervised setting assumes is unavailable. We
ask whether **synthetically generated anomalies** can substitute for real anomalies when
calibrating this threshold. Using a confound-free protocol in which a single trained model
is re-thresholded from different label sources, we evaluate four synthetic-anomaly
generators across five detectors, two datasets (MVTec AD, VisA), and three seeds (580 jobs,
2845 evaluation rows). We find that self-supervised generators recover **89–91 %** of the
oracle (real-anomaly) F1 on average and preserve the model ranking almost perfectly
(Spearman ρ = 1.00 on MVTec, 0.90 on VisA), whereas texture-paste anomalies recover only
62 %. We further introduce an **area-routed hybrid blend** that applies alpha blending to
small mask components and Poisson blending to large ones; it significantly outperforms pure
alpha blending (Wilcoxon p = 0.002) but only ties with pure Poisson (p = 0.31), and a sweep
of the routing threshold shows performance degrading monotonically as area is diverted away
from Poisson — seamless blending alone is therefore the operative ingredient. Synthetic
calibration is a viable, label-free operating-point selector, with the caveat that the
*absolute* threshold value transfers weakly.

## 1. Motivation

A detector produces an anomaly score per image; turning scores into decisions needs a
threshold τ. In practice τ is chosen as the F1-maximising point on a validation set that
contains labelled anomalies. Unsupervised deployment has no such set. If a **synthetic**
anomaly set can stand in for real anomalies to pick τ, calibration becomes fully label-free.
We quantify the F1 lost by this substitution and identify which synthetic-generation
strategy transfers best.

## 2. Methodology

**Confound-free protocol.** Each job trains one model **once** on all train normals and then
derives every evaluation *arm* from that identical model, so arms differ only in the
*threshold source*, never in the weights. This eliminates the model-variance confound that
an earlier per-arm-retraining design introduced (verified: image-AUROC — a threshold-free
metric — varies by ≤ 0.05 across arms, i.e. float noise). Anomalib's `OneClassPostProcessor`
refits τ and score normalisation on any validation loop, so `Engine.validate` on a chosen
calibration set re-derives τ without touching the trained weights.

**Arms.**

- **A — oracle:** τ from the real test set (fit-time validation = `SAME_AS_TEST`) → F1-max upper bound.
- **B — proxy:** τ refit on {real-test normals + N synthetic anomalies} (N = real anomaly count), then applied to the full real test. Differs from A only by real → synthetic anomalies.
- **C — diagnostic (phase 0 only):** τ refit on *and* evaluated on the synthetic set, to gauge whether the synthetic set is too easy/hard relative to the real test.

**Synthetic generators (pipelines).**

| ID | Preset | Description |
|----|--------|-------------|
| P1 | `texture_alpha` | Alpha-blended DTD texture patches pasted onto normals |
| P2 | `self_alpha` | Alpha-blended patches cut from other normal images (self-supervised) |
| P3 | `self_poisson` | Poisson (seamless) blending of self-sourced patches |
| P4 | `self_hybrid` | Area-routed: alpha for mask components < 2500 px, Poisson for larger ones |

P4 is introduced in this work (§3.6). Its motivation is a measured failure mode of Poisson
blending: on a 193 px blob, Poisson leaves a visibility of 0.0020 against alpha's 0.2966 —
a **148× erasure** of small defects. A blob-area census over 8 categories shows 97–99 % of
mask components fall below 2500 px while holding only 15–36 % of total mask area, so the
routing threshold separates two genuinely distinct populations rather than degenerating
into P2 or P3.

**Models (5):** `patchcore`, `padim`, `efficient_ad`, `anomaly_dino`, `dinomaly` —
spanning memory-bank, distribution, student–teacher, and DINO-ViT feature families.

**Datasets & scale:** MVTec AD (15 categories) and VisA (12 categories), seeds {1, 2, 3}.
Main comparison = **phase 2**: 5 models × 27 categories × 3 seeds = **405 jobs → 1620 rows**
(train-normal counts 60–905). Hybrid-blend comparison = **phase 3**: 5 models × 27 categories
× {P2, P3, P4} × {A, B, C} × 1 seed = **135 jobs → 945 rows**. Metrics: image
AUROC/F1/AUPR/precision/recall and pixel AUROC/F1/AUPRO (torchmetrics-based `Evaluator`).

## 3. Results

### 3.1 Oracle vs. synthetic-transfer F1

Mean image-F1: **F1_A** = oracle, **F1_B** = synthetic-calibrated, **gap** = F1_A − F1_B
(lower is better), averaged over categories × 3 seeds.

**MVTec AD**

| model | F1_A | P1 gap | P2 gap | P3 gap |
|-------|------|--------|--------|--------|
| patchcore    | 0.971 | 0.215 | 0.068 | 0.060 |
| padim        | 0.914 | 0.312 | 0.065 | 0.045 |
| efficient_ad | 0.921 | 0.135 | 0.066 | 0.038 |
| anomaly_dino | 0.965 | 0.391 | 0.091 | 0.058 |
| dinomaly     | 0.972 | 0.279 | 0.065 | **0.020** |

**VisA**

| model | F1_A | P1 gap | P2 gap | P3 gap |
|-------|------|--------|--------|--------|
| patchcore    | 0.869 | 0.443 | 0.142 | 0.123 |
| padim        | 0.811 | 0.449 | 0.084 | **0.072** |
| efficient_ad | 0.854 | 0.159 | 0.120 | 0.117 |
| anomaly_dino | 0.866 | 0.526 | 0.160 | 0.126 |
| dinomaly     | 0.947 | 0.564 | 0.130 | 0.139 |

**Aggregate F1 recovery (F1_B / F1_A, both datasets):** P1 = **0.619**, P2 = **0.889**,
P3 = **0.914**. Self-supervised blending (P2/P3) recovers ~90 % of oracle F1; texture-paste
(P1) recovers only ~62 % and is consistently the worst proxy.

### 3.2 Model-ranking preservation (the key practical result)

Even when the absolute F1 is not fully recovered, synthetic calibration preserves the
*ordering* of models: Spearman ρ between oracle-F1 and synthetic-F1 rankings (P3) =
**1.00 (MVTec)** and **0.90 (VisA)**. A practitioner can thus select the best model on a new
dataset using only synthetic anomalies.

### 3.3 Threshold-value transfer is weak

Pearson r between synthetic τ_B and oracle τ_A across categories (P3) is only 0.16–0.48 on
MVTec and −0.43 to 0.36 on VisA. F1 is recovered despite poor τ correlation because the
score distribution near the operating point is flat; the *decision* is robust even when the
*number* is not. Distribution/memory-bank models (`padim`, `patchcore`) have wide,
un-normalised score scales (raw |τ gap| 7–30), so their thresholds are the most sensitive.

### 3.4 Stability and localization

Seed stability (P3, std of F1 gap): MVTec is tight (0.03–0.09); VisA is noisier
(up to 0.13), reflecting its harder, higher-variance categories. Localization quality
(oracle pixel-AUPRO) is strong overall — `dinomaly` leads (0.94 MVTec / 0.96 VisA),
`patchcore` next (0.93 / 0.86).

### 3.5 Difficulty diagnostic (Arm C)

Real-test AUROC ≈ 0.887. The synthetic set is **mis-calibrated in difficulty**: P1 is *too
easy* (patchcore 0.989, draem 0.917) and P3 is *too hard* (patchcore 0.671, draem 0.626).
Notably, P3 transfers thresholds best *despite* being the hardest synthetic set — matching
real-anomaly *difficulty* matters less than matching its *blending realism*.

### 3.6 Area-routed hybrid blending (P4)

P3's seamless blending is what makes it the best proxy, but it also erases small defects
(§2). P4 routes each connected mask component by area: alpha blend below 2500 px, Poisson
above. Phase 3 evaluates P2/P3/P4 on **identical trained weights** (verified: arm-A AUROC
spread across pipelines = 0.0000), making the comparison exactly paired.

**Mean F1 gap (oracle − proxy; lower is better), 945 rows, seed 1**

| dataset | P2 | P3 | P4 |
|---------|------|------|------|
| MVTec   | 0.0597 | 0.0446 | **0.0433** |
| VisA    | 0.1539 | 0.1128 | **0.1099** |

F1 recovery (F1_B / F1_A) rises P2 → P3 → P4 as 0.875 → 0.909 → **0.910**.

**Per-model F1 gap**

| dataset | model | P2 | P3 | P4 |
|---------|-------|------|------|------|
| MVTec | anomaly_dino | 0.0522 | **0.0313** | 0.0419 |
| MVTec | dinomaly     | 0.0757 | 0.0287 | **0.0269** |
| MVTec | efficient_ad | 0.0601 | **0.0402** | 0.0465 |
| MVTec | padim        | 0.0650 | 0.0522 | **0.0495** |
| MVTec | patchcore    | **0.0457** | 0.0706 | 0.0519 |
| VisA  | anomaly_dino | 0.2233 | **0.1295** | 0.1571 |
| VisA  | dinomaly     | 0.1322 | **0.1138** | 0.1140 |
| VisA  | efficient_ad | 0.1459 | 0.1106 | **0.0723** |
| VisA  | padim        | 0.1056 | **0.0645** | 0.0757 |
| VisA  | patchcore    | 0.1623 | 0.1454 | **0.1304** |

**Paired significance (135 cells, Wilcoxon signed-rank):** P4 beats P2 in 56/135 cells with
**p = 0.002**; P4 beats P3 in 56/135 cells with **p = 0.31**. Mean gaps are P2 = 0.1016,
P3 = 0.0749, P4 = 0.0729. P4 is the best of the three in 76/135 cells.

**Difficulty calibration (arm C).** The routing *did* work as designed — P4 lands between its
two parents and closer to real-test difficulty than P3 on both datasets:

| dataset | real (arm A) | P2 | P3 | P4 | \|Δ\| P3 | \|Δ\| P4 |
|---------|--------------|------|------|------|------|------|
| MVTec | 0.9479 | 0.8509 | 0.8121 | 0.8312 | 0.1408 | **0.1301** |
| VisA  | 0.8915 | 0.8038 | 0.7259 | 0.7808 | 0.2027 | **0.1604** |

**Interpretation.** P4 is a *statistically confirmed* improvement over alpha blending and a
*statistical tie* with Poisson: the 0.001–0.003 gap reduction over P3 is within noise. The
mechanism is validated (small components are rescued, synthetic difficulty moves toward
real) but the extra realism does **not** convert into better threshold transfer. Two
sub-results are still practically useful: P4 delivers the single largest improvement
anywhere (VisA `efficient_ad`, 0.1106 → 0.0723) and it repairs P3's worst regression
(MVTec `patchcore`, 0.0706 → 0.0519, where Poisson was *worse* than alpha).

### 3.7 Is the P3/P4 tie a mistuned knob? (threshold ablation)

The 2500 px cut-off was a default, so phase 4 sweeps it over {500, 1000, 2500, 6000,
12000} px with P3 (equivalent to a 0 px threshold, i.e. pure Poisson) re-run inside the same
jobs as a paired anchor. Subset: 5 models × 8 categories chosen to span the blob-census range
(7.5 %–68.7 % of mask area below threshold), 1 seed = **40 jobs → 280 rows**, again
confound-free (arm-A AUROC spread = 0.0000).

**Mean F1 gap vs. routing threshold**

| threshold (px) | 0 (=P3) | 500 | 1000 | 2500 | 6000 | 12000 |
|---|---|---|---|---|---|---|
| overall | 0.0814 | 0.0857 | **0.0735** | 0.0848 | 0.1114 | 0.1188 |
| MVTec | 0.0556 | 0.0637 | 0.0555 | **0.0432** | 0.0550 | 0.0538 |
| VisA | 0.1072 | 0.1076 | **0.0915** | 0.1264 | 0.1679 | 0.1838 |

**Wilcoxon vs. the P3 anchor (40 paired cells):** no threshold reaches significance.
The nominal best, 1000 px, gives −0.0078 at p = 0.43; 500 px +0.0043 (p = 0.95); 2500 px
+0.0035 (p = 0.85); 6000 px +0.0301 (p = 0.37); 12000 px +0.0374 (p = 0.085). P3 is the
single best pipeline in 17/40 cells, more than any hybrid variant.

**The trend is the real finding.** Beyond ~1000 px the gap degrades monotonically, and
steeply on VisA (0.0915 → 0.1264 → 0.1679 → 0.1838). Since a larger threshold routes *more*
mask area through alpha blending, the ablation says that **the more Poisson blending is
used, the better the threshold transfers** — the optimum sits at or near the pure-Poisson
end of the axis. The P3/P4 tie is therefore **fundamental, not an artefact of tuning**:
there is no setting of the routing threshold at which hybrid blending significantly beats
plain Poisson.

Dataset-dependence is worth noting: MVTec tolerates or mildly prefers routing (2500 px is
nominally best at 0.0432), whereas VisA is actively harmed by it. Because the effect
reverses across datasets and never clears significance, it does not support a
recommendation. (Minor artefact: MVTec `bottle` shows a marginally negative gap, −0.0017,
where the proxy threshold edges out the oracle; this is grid/normalisation noise in the
F1-max search, not a real inversion.)

## 4. Discussion

1. **Self-supervised, seamless blending (P3) is the recommended label-free calibrator**,
   giving the smallest oracle gap for 4/5 models on MVTec and the best aggregate recovery.
2. **Texture-paste (P1) should be avoided for threshold calibration** — its unrealistic
   appearance produces an over-easy set and inflates τ.
3. **Ranking is more transferable than absolute F1**, making synthetic anomalies immediately
   useful for label-free *model selection* even where exact operating points are not.
4. **VisA is the harder regime** (gaps ~2× MVTec); synthetic calibration degrades but the
   ordering is largely retained.
5. **Blend complexity has diminishing returns.** Area-routed hybrid blending (P4) is
   significantly better than alpha and indistinguishable from Poisson, so **seamless
   blending is the operative ingredient** and additional routing machinery is not warranted
   by default. This is a negative result worth recording: matching real-anomaly difficulty
   more closely (which P4 provably does) did not improve threshold transfer, reinforcing
   §3.5's finding that difficulty and transferability are largely decoupled. The threshold
   ablation (§3.7) closes this question: performance degrades monotonically as more area is
   routed away from Poisson, so the optimum lies at the pure-Poisson end and **P3 is the
   final recommendation**.

## 5. Remaining Work

The core matrix (5 models × 4 pipelines × 2 datasets) is complete and internally consistent
(0 NaN, confound-free), and it fully supports the conclusions above. Strictly optional
extensions, none required for the current claims:

- **`draem` at full scale.** DRAEM is wired and smoke-tested but currently only appears in the
  6-job phase-0 diagnostic. Since it is itself a synthetic-anomaly model, adding it to the
  phase-2 sweep (5 → 6 models) would round out the roster.
- **Generator-parameter ablation.** Anomaly count/size and blend probability are fixed
  (probability = 1.0); a small sweep would test sensitivity of the P3 recommendation.
- **Pixel-level threshold transfer.** The study analyses image-level operating points; the
  same arms already emit pixel metrics, so segmentation-threshold transfer could be reported
  without new runs.
- **Seed replication of §3.6–3.7.** The hybrid-blend phases use a single seed; the paired
  design keeps them internally valid, but multi-seed repetition would tighten the
  confidence intervals on the null results.

**Bottom line:** No further experiments are needed to support the reported findings on
MVTec AD 1 / VisA. The recommended pipeline is **P3 (`self_poisson`)**: it matches the best
hybrid configuration at every routing threshold tested while being the simpler generator.

## 6. Extension: MVTec AD 2 at Native Resolution (complete)

MVTec AD 1 and VisA are both near ceiling (mean oracle F1 0.93–0.97), so they cannot
stress-test whether the P3 recommendation holds on harder, non-saturated data. We extend to
**MVTec AD 2**, where current SOTA (RoBiS, SuperADD) reports only 51.0–57.4% SegF1 on
`test_private`, and add a sixth model (DRAEM).

### 6.1 448 px pilot (phase 5)

Running the same confound-free A/B/C protocol at 448 px (5 models × 8 categories × 3 seeds
where completed; 693 rows, 99/144 jobs done) confirms AD2 is far from saturated and shows
the same qualitative mechanism as AD1/VisA, with some notable differences. `draem` has not
yet completed a job at this phase (still OOM-prone at 448 px prior to the batch-size fix);
`anomaly_dino` completed only 3/24 (category, seed) combinations before the coreset-subsampling
fix landed, so its numbers below are preliminary (n = 3, not n = 24 like the other four models).

**Oracle (arm A) metrics, mean over completed categories/seeds**

| model | image_AUROC | image_F1 | pixel_F1 | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|
| patchcore | 0.727 | 0.826 | 0.260 | 0.540 | 0.288 |
| dinomaly | 0.681 | 0.818 | 0.306 | 0.598 | 0.327 |
| anomaly_dino (n=3) | 0.672 | 0.890 | 0.316 | 0.392 | 0.269 |
| padim | 0.628 | 0.806 | 0.075 | 0.418 | 0.098 |
| efficient_ad | 0.621 | 0.798 | 0.178 | 0.293 | 0.117 |

Oracle image-AUROC ranges **0.62–0.73**, 20–30 points below MVTec AD 1. AD2's native images
are 2.3–5.0 MP — 5–11× our 448 px input — making downsizing the leading suspect for this gap
and motivating the native-resolution extension in §6.2 rather than accepting 448 px as final.

**Image-F1 gap (A − B) by model × pipeline**

| model | P1 | P2 | P3 |
|---|---|---|---|
| patchcore | 0.481 | 0.111 | 0.142 |
| dinomaly | 0.421 | 0.111 | 0.177 |
| padim | 0.330 | 0.029 | 0.030 |
| efficient_ad | 0.074 | 0.016 | 0.028 |
| anomaly_dino (n=3) | 0.235 | 0.008 | 0.008 |

**Aggregate image-F1 recovery (F1_B / F1_A):** P1 = **0.60**, P2 = **0.92**, P3 = **0.89** —
the same P1 ≪ {P2, P3} pattern as AD1/VisA (§3.1). Unlike there, **P2 (self_alpha) edges out
P3 (self_poisson)** here rather than tying/losing to it; pixel-level recovery shows the same
reversal is small but consistent (P1 = 0.39, P2 = 0.75, P3 = **0.77**, so P3 still wins at the
pixel level). With only 1 seed's worth of B/C data completed per pipeline for most models, this
P2-vs-P3 flip at the image level should be treated as preliminary, not a revision of the P3
recommendation.

**Ranking preservation is markedly weaker than on AD1/VisA.** Spearman ρ between oracle and
synthetic-calibrated image-F1 across the 5 models is only **0.10 (P2)** and **0.20 (P3)**
(both n.s., n = 5, vs. ρ = 1.00/0.90 on MVTec/VisA) — and *negative* for P1 (ρ = −0.40). AD2's
harder categories appear to scramble the oracle-vs-synthetic model ordering more than AD1/VisA
did; this is a genuine open question for the extension rather than an artifact, though the
small model count (5, one at n = 3) limits how much weight to put on the exact ρ values.

### 6.2 Native-resolution tiled inference (phase 6, complete)


We built a tiled evaluation harness that trains each model with random-crop augmentation at
native resolution and scores test images by tiling through the (architecturally unmodified)
model and re-stitching per-tile anomaly maps with anomalib's `Tiler` (448 px tile, 336 px
stride, 25% overlap) — closing the resolution gap without altering any model.

Two measurement bugs surfaced during validation. Both are reported here because they are
generic to tiled-inference evaluation, not specific to our synthetic-anomaly framing, and
would silently corrupt any similarly-built pipeline:

- **Tile-border artifact.** The stitched anomaly map carries a strong, spurious peak at the
  *true* image border — a convolutional edge effect, not a tile-seam artifact (unaffected by
  `Tiler`'s `remove_border_count`, which only trims overlap between adjacent tiles, not the
  true outer edge). Verified directly: two unrelated normal images produced the identical
  score at the identical corner pixel. This constant border-max dominated every image-level
  max-score, collapsing image-AUROC to exactly 0.500 on every job. **Fix:** crop a 32 px
  margin from the stitched map before computing image- or pixel-level statistics.
- **Metric-library sigmoid saturation.** After the border fix, a completed job *still* showed
  AUROC = 0.500. Root cause: torchmetrics silently applies `sigmoid()` to any prediction
  tensor found outside `[0, 1]`, assuming it is a logit rather than a score. Our raw,
  tile-stitched scores (range ~30–120) saturate completely under sigmoid
  (`sigmoid(30) ≈ 1.0`), collapsing every score to one value regardless of content. Anomalib's
  standard `Engine.test` path is unaffected because `OneClassPostProcessor` always min-max
  normalizes `pred_score`/`anomaly_map` into `[0, 1]` before any metric sees them — our
  harness computes metrics manually and bypasses that post-processor. **Fix:** replicate the
  same min-max normalization (fit on the calibration set, applied before AUROC/AUPR only;
  `F1AdaptiveThreshold` is unaffected, since anomalib's own `BinaryPrecisionRecallCurve`
  override already disables this sigmoid step for threshold fitting). Verified against
  `sklearn.metrics.roc_auc_score` on real completed-job scores: 0.616 both ways.
- **Dinomaly's published crop breaks tile stitching.** Dinomaly's official recipe resizes to
  448 px then center-crops to 392 px *inside the model's own pre-processor*. Under tiling each
  tile is already exactly 448×448, so the crop shrinks every tile's output to 392×392,
  corrupting `Tiler.untile()`'s size assumptions — this crashed 100% (8/8) of dinomaly's
  tiled jobs with a tensor-size mismatch. **Fix:** disable the crop specifically when
  building a model for tiled inference (`crop_size == image_size`); confirmed the non-tiled
  448 px path is unaffected and still uses the published 392 px crop.
- **Pixel-threshold bin-range mismatch (the most consequential bug).** Even after the fixes
  above, patchcore/padim/efficient_ad showed a large *drop* in tiled pixel-F1 relative to
  448 px (e.g. patchcore 0.260 → 0.015) despite healthy, comparable pixel AUROC/AUPRO —
  a strong signal of a broken threshold rather than a real capability loss. Root cause: the
  pixel `F1AdaptiveThreshold` is fit in "binned" mode (`thresholds=200`, an int) to bound
  memory at native resolution, and torchmetrics silently reinterprets a bare int as
  `linspace(0, 1, 200)` — a fixed `[0, 1]` grid — regardless of the actual (raw, unbounded)
  score range. Since our pixel scores sit far above 1 almost everywhere, every candidate
  threshold ends up below virtually the entire score distribution, so the fitted threshold
  degenerates to "classify nearly all pixels positive". Reproduced synthetically: a clearly
  separable 1%-defect map gave threshold = 0.0 and F1 = 0.02 with the buggy bare-int
  thresholds, vs. threshold = 50.1 and F1 = 1.0 once `thresholds` is an explicit
  `linspace(pixel_min, pixel_max, 200)` spanning the true score range. **Fix:** compute the
  pixel score range first, then pass an explicit range-aware threshold tensor. This bug
  affected only the fitted pixel *threshold* (and therefore pixel-F1/pred_mask); pixel
  AUROC/AUPR were already correct because `_evaluate` separately min-max-normalizes the map
  into `[0, 1]` before those metrics, matching their bare-int bins. All pixel-F1 numbers from
  jobs completed before this fix were discarded and re-run.

With all four fixes applied, the full phase 6 sweep (6 models × 8 categories, 1 seed, 48
jobs — run split across two machines: 8 padim jobs locally, the remaining 40 across
dinomaly/anomaly_dino/draem/efficient_ad/patchcore on an 8-GPU remote machine) completed
cleanly: 0 errors, 0 rows with the exact-0.500 AUROC artifact. An earlier partial run
(38/48 jobs) surfaced the pixel-threshold bug via a suspicious tiled-vs-448 regression and
was discarded in full rather than patched selectively, since the bug corrupted pixel-F1 in
every completed row.

**Result: tiling does not close the gap to published SOTA, and on average is a net-negative.**

| model | n | image_AUROC 448 → tiled | pixel_F1 (SegF1) 448 → tiled | Δ SegF1 |
|---|---|---|---|---|
| anomaly_dino | 8 (448: n=3) | 0.672 → 0.703 | 0.316 → 0.320 | **+0.004** |
| dinomaly | 8 | 0.681 → 0.742 | 0.306 → 0.296 | −0.010 |
| efficient_ad | 8 | 0.621 → 0.655 | 0.178 → 0.134 | −0.044 |
| padim | 8 | 0.628 → 0.558 | 0.075 → 0.065 | −0.010 |
| patchcore | 8 | 0.727 → 0.737 | 0.260 → 0.188 | **−0.072** |
| draem | 8 | no 448 baseline | — → 0.112 | — |

Overall mean SegF1 across all completed rows: 0.208 (448 px) → 0.186 (tiled), a net
**−0.022** change — still far short of published SOTA (RoBiS 0.510, SuperADD 0.574).
Only `anomaly_dino` improved on both metrics, and its 448 px baseline is itself only n = 3
(pre-fix partial data, §6.1), so that improvement should be treated cautiously. Every other
model with a complete 448 px baseline (dinomaly, efficient_ad, padim, patchcore — all n = 8)
is flat or worse under tiling, with patchcore regressing the most (−0.072). Native-resolution
tiled inference therefore does **not** support the "downsizing is the dominant bottleneck"
hypothesis that motivated building it — the four measurement bugs documented above, and the
confirmed 448 px→native resolution SegF1 gap itself, are the load-bearing findings from this
phase rather than a tiling-driven accuracy gain.

## 7. Status and Next Steps

- Phase 5 (448 px, MVTec AD 2, §6.1) confirms a large, resolution-linked gap to SOTA and
  replicates the P1 ≪ {P2, P3} recovery pattern, but with a weaker/flipped P2-vs-P3 edge and
  much weaker ranking preservation than AD1/VisA — both need more seeds/models (`draem`,
  full `anomaly_dino`) before drawing firm conclusions.
- Phase 6 (native-resolution tiled, §6.2) is complete: tiling does **not** close the gap to
  published SOTA, and is a net regression on average (−0.022 mean SegF1) once measured
  correctly. Only `anomaly_dino` improved, on an unreliable (n = 3) 448 px baseline; every
  fully-populated model is flat or worse. This falsifies the "448 px downsizing is the
  dominant bottleneck" hypothesis that motivated §6.2 — closing the resolution gap did not
  translate into closing the accuracy gap for these six models on MVTec AD 2.
- Also complete: a SuperADD (anomalib-native, DINOv3-`large` backbone) pilot on the 4
  categories with semantic-defect-bank synthetic-anomaly coverage
  (`gsoc_workspace/superadd_mvtecad2_experiment.md`), replicating the P2/P3-style
  alpha-vs-Poisson calibration-gap finding with real-defect-bank patches instead of
  self-crops (Poisson ≤ alpha gap on every category).
- Remaining open items: re-run the difficulty-mechanism and score-coverage analyses (§3.5)
  on native-resolution AD2 data to check whether the same threshold-mismatch mechanism
  replicates; extend the hybrid-blend (P4) comparison (§3.6–3.7) to AD2 if motivated; decide
  whether SuperADD's own built-in percentile threshold ("arm D") is worth wiring in as a
  reference point against synthetic calibration.
- Given phase 6 did not close the gap, the four measurement bugs documented in §6.2 (tile-
  border artifact, metric-library sigmoid saturation, Dinomaly's crop breaking tile
  stitching, and the pixel-threshold bin-range mismatch) are now the primary reportable
  contribution from the tiling effort — the pixel-threshold bin-range mismatch in particular
  is a generic pitfall for anyone doing native-resolution/tiled anomaly segmentation with a
  binned adaptive threshold, independent of our specific pipeline.

---
*Reproducibility: aggregated results in `gsoc_workspace/experiments/results/results.csv`
(3415 rows for MVTec AD 1/VisA phase 0–4; phase 5–7 for MVTec AD 2, phase 6 complete at
48/48 jobs across two machines); statistics in `gsoc_workspace/experiments/report_stats.py`,
`analyze_p4.py`, `analyze_threshold_sweep.py`, and `compare_tiled_vs_448.py`; harness in
`gsoc_workspace/experiments/harness.py` (AD1/VisA, AD2 448 px, and SuperADD) and
`gsoc_workspace/experiments/tiled_harness.py` (AD2 native resolution).*
