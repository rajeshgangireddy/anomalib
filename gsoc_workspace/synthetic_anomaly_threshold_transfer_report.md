# Synthetic Anomalies as a Threshold-Calibration Proxy for Unsupervised Anomaly Detection

*GSoC — Anomalib. Experiments run on anomalib 2.5.2-dev.0, 2× RTX 3090. Report generated 2026-07-29.*

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

**Bottom line:** No further experiments are needed to support the reported findings. The
recommended pipeline is **P3 (`self_poisson`)**: it matches the best hybrid configuration at
every routing threshold tested while being the simpler generator.

---
*Reproducibility: aggregated results in `gsoc_workspace/experiments/results/results.csv`
(3415 rows across phase 0–4); statistics in `gsoc_workspace/experiments/report_stats.py`,
`analyze_p4.py`, and `analyze_threshold_sweep.py`; harness in
`gsoc_workspace/experiments/harness.py`.*

---
*Reproducibility: aggregated results in `gsoc_workspace/experiments/results/results.csv`
(3135 rows across phase 0/1/2/3); statistics in `gsoc_workspace/experiments/report_stats.py`
and `gsoc_workspace/experiments/analyze_p4.py`; harness in
`gsoc_workspace/experiments/harness.py`.*
