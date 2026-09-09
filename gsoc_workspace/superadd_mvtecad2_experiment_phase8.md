# SuperADD on MVTec AD 2 (huge_plus backbone, 8 categories) with Semantic-Defect-Bank Synthetic Calibration

*Experiments run on anomalib (branch merged with `feature/v2.6.0`, PR #3628 SuperADD). Phases:
`phase8_alpha`/`phase8_poisson` (SuperADD + semantic-bank, 2× RTX 3090),
`phase9_crossmodel` (`anomaly_dino`/`padim`/`patchcore` + semantic-bank, 2× RTX 3090 local
+ 8× RTX 3090 remote), and `phase10_superadd_perlin` (SuperADD + live Perlin pipelines,
2× RTX 3090). Covers 4 models total, both calibration sources for SuperADD; see §3 for
cross-model and cross-source generalization and the Appendix for full per-model detail.*

## 1. Methodology

### 1.1 Model

**SuperADD** (anomalib-native implementation, VAND 4.0 winning method): a training-free,
memory-bank anomaly detector built on a frozen **DINOv3** Vision Transformer backbone.
Multi-layer token features are extracted over overlapping image patches, a per-layer
memory bank is built from normal training images via distance-based coreset
subsampling, and anomalies are scored by nearest-neighbor distance to that bank.

| Setting | Value |
|---|---|
| Backbone | `vit_huge_plus_patch16_dinov3` (840M params) — the paper's default backbone (upgraded from the `vit_large_patch16_dinov3` 303M pilot used in the phase7 run) |
| Input resolution | 448×448 (resize, no center-crop) |
| Post-processor | Standard anomalib `PostProcessor` (F1-adaptive threshold), overriding SuperADD's own built-in percentile-based threshold, for arm parity with other models in the wider study |
| Training | Single pass: coreset-subsampled memory-bank construction (no gradient descent) |

### 1.2 Dataset

**MVTec AD 2**, all 8 categories: `can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`,
`vial`, `wallplugs`, `walnuts`. Train/val use the dataset's native `train/good` and
`validation/good` splits; test uses `test_public` (`good` + `bad` + pixel masks).

| category | n_train | n_val (heldout) | n_test |
|---|---|---|---|
| can | 412 | 162 | 162 |
| fabric | 387 | 156 | 156 |
| fruit_jelly | 263 | 80 | 80 |
| rice | 312 | 132 | 132 |
| sheet_metal | 137 | 114 | 114 |
| vial | 291 | 140 | 140 |
| wallplugs | 293 | 150 | 150 |
| walnuts | 432 | 150 | 150 |

One corrupted real-dataset image (`rice/train/good/265_regular.png`, pre-existing,
unrelated to synthetic generation) was found and quarantined during this run; `n_train`
for `rice` reflects the corrected count (312, one below the nominal 313).

### 1.3 Synthetic-anomaly calibration

Synthetic anomalies are **pre-generated** (not produced live during the sweep) by the
"semantic defect bank" pipeline, regenerated for this run with an improved notebook
(`flash-part1.ipynb`) that extends coverage from 4 to all 8 categories (3 donor-defect
pairs per category, vs. the earlier smaller donor bank): real defect patches are cut
from a donor bank of previously-generated anomalies, then replayed onto held-out
`train/good` host images (donors excluded) at MRSP/OBS-chosen sites, harmonized to the
host's local illumination, and composited with one of two blend operators:

- **P5 — alpha**: opacity blend, `out = host·(1−α) + patch·α`.
- **P6 — poisson**: seamless Poisson blend with a substrate "collar" around the
  defect so the solver's boundary condition lands on substrate, not the defect itself.

"P5/alpha" and "P6/poisson" name the corpus each sample was generated *for*, not a
guarantee of the blend actually used on every image: within P6, a sample whose defect
falls below a minimum equivalent radius is composited with alpha instead (Poisson would
otherwise dissolve it), so the P6 corpus is poisson-primary with a documented,
per-sample-traceable alpha fallback for small defects, rather than 100% Poisson.

Concretely, "alpha arm" and "poisson arm" each refer to one full dataset root written to
disk (`MVTec_AD_2_alpha_<seed>`, `MVTec_AD_2_poisson_<seed>`), one per category per seed.
The two arms are generated in lockstep, not independently: for a given (category, seed,
sample index) they share the identical host image, MRSP blob, bank entry, placement mode,
rotation, and flip — the arm only decides which compositing function runs at the very last
step (`alpha_composite` vs. `collar_poisson`, the latter falling back to alpha as above).
This pairing is deliberate: it means any difference measured between the two arms comes
from the blend operator alone, not from having sampled different hosts or defects.

Three fixed seeds (0, 1, 2) are available per category per arm; ground truth is the
warped defect alpha intersected with the object region.

### 1.4 Confound-free evaluation protocol

Each job trains SuperADD **once** per (category, seed); every evaluation arm reuses
that identical trained memory bank, differing only in the threshold-fitting data:

- **Arm A (oracle)**: threshold fit on the real `test_public` set itself → upper bound.
- **Arm B (proxy)**: threshold refit on `{heldout normals + N synthetic anomalies}`
  (N = real test anomaly count), then transferred to the real `test_public` set — the
  number that matters for label-free deployment.
- **Arm C (diagnostic)**: same synthetic threshold, evaluated on the synthetic set
  itself (tests whether the synthetic set is harder/easier than the real test).

Calibration negatives are drawn from the dataset's native `validation/good` split —
disjoint from `test_public` (leakage-free / `"heldout"` mode).

### 1.5 Sweep

8 categories × 3 seeds × 2 pipelines (P5, P6) = **48 jobs**, run sequentially in two
waves (all P6/poisson jobs first, then all P5/alpha jobs — poisson prioritized per the
phase7 result showing a slight edge), each producing arm A + B + C. Jobs ran on 2 GPUs
(1 job/GPU), ~2.2 GPU-hours of measured fit+test compute across all 48 jobs (real
per-job wall time is dominated by `huge_plus` backbone construction overhead, ~11–26
min/job depending on category size).

## 2. Results

### 2.1 Oracle (Arm A) — upper bound, mean ± std pooled over both phases' training runs (6 samples/category)

| category | image_AUROC | image_F1 | image_AUPR | image_Precision | image_Recall | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPR | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|---|---|---|---|
| can | 0.493 ± 0.006 | 0.713 ± 0.005 | 0.651 ± 0.032 | 0.564 ± 0.012 | 0.970 ± 0.018 | 0.659 ± 0.012 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.222 ± 0.019 | 0.011 ± 0.001 |
| fabric | 0.670 ± 0.032 | 0.748 ± 0.016 | 0.773 ± 0.016 | 0.631 ± 0.052 | 0.930 ± 0.065 | 0.996 ± 0.000 | 0.787 ± 0.004 | 0.729 ± 0.002 | 0.415 ± 0.014 | 0.251 ± 0.014 |
| fruit_jelly | 0.843 ± 0.010 | 0.858 ± 0.006 | 0.954 ± 0.003 | 0.778 ± 0.019 | 0.958 ± 0.023 | 0.920 ± 0.000 | 0.561 ± 0.002 | 0.440 ± 0.004 | 0.533 ± 0.004 | 0.173 ± 0.002 |
| rice | 0.868 ± 0.025 | 0.865 ± 0.017 | 0.940 ± 0.010 | 0.806 ± 0.044 | 0.937 ± 0.035 | 0.953 ± 0.002 | 0.588 ± 0.003 | 0.545 ± 0.001 | 0.669 ± 0.009 | 0.511 ± 0.007 |
| sheet_metal | 0.653 ± 0.012 | 0.895 ± 0.004 | 0.885 ± 0.009 | 0.822 ± 0.009 | 0.983 ± 0.014 | 0.880 ± 0.000 | 0.357 ± 0.001 | 0.229 ± 0.001 | 0.236 ± 0.000 | 0.049 ± 0.000 |
| vial | 1.000 ± 0.000 | 0.994 ± 0.002 | 0.999 ± 0.000 | 0.998 ± 0.004 | 0.990 ± 0.000 | 0.971 ± 0.000 | 0.571 ± 0.004 | 0.547 ± 0.002 | 0.961 ± 0.001 | 0.852 ± 0.003 |
| wallplugs | 0.535 ± 0.012 | 0.759 ± 0.008 | 0.652 ± 0.005 | 0.619 ± 0.011 | 0.981 ± 0.013 | 0.927 ± 0.003 | 0.543 ± 0.002 | 0.509 ± 0.003 | 0.454 ± 0.010 | 0.208 ± 0.004 |
| walnuts | 0.843 ± 0.008 | 0.840 ± 0.006 | 0.922 ± 0.003 | 0.908 ± 0.011 | 0.781 ± 0.011 | 0.979 ± 0.001 | 0.719 ± 0.001 | 0.742 ± 0.001 | 0.544 ± 0.021 | 0.274 ± 0.004 |

### 2.2 Arm B (synthetic-calibrated, on real test) — mean over 3 seeds

**Pipeline P5 (alpha)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.489 | 0.614 | 0.654 | 0.000 | 0.212 | 0.010 |
| fabric | 0.667 | 0.732 | 0.996 | 0.733 | 0.409 | 0.243 |
| fruit_jelly | 0.786 | 0.857 | 0.918 | 0.554 | 0.533 | 0.173 |
| rice | 0.865 | 0.811 | 0.953 | 0.525 | 0.671 | 0.511 |
| sheet_metal | 0.639 | 0.882 | 0.881 | 0.076 | 0.236 | 0.049 |
| vial | 1.000 | 0.857 | 0.970 | 0.479 | 0.960 | 0.850 |
| wallplugs | 0.525 | 0.750 | 0.926 | 0.177 | 0.450 | 0.206 |
| walnuts | 0.811 | 0.750 | 0.979 | 0.624 | 0.542 | 0.274 |

**Pipeline P6 (poisson)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.492 | 0.618 | 0.663 | 0.000 | 0.233 | 0.011 |
| fabric | 0.670 | 0.732 | 0.996 | 0.693 | 0.420 | 0.259 |
| fruit_jelly | 0.839 | 0.857 | 0.920 | 0.555 | 0.532 | 0.173 |
| rice | 0.871 | 0.811 | 0.952 | 0.525 | 0.668 | 0.511 |
| sheet_metal | 0.639 | 0.882 | 0.881 | 0.074 | 0.236 | 0.049 |
| vial | 1.000 | 0.857 | 0.971 | 0.393 | 0.961 | 0.853 |
| wallplugs | 0.535 | 0.750 | 0.929 | 0.180 | 0.458 | 0.210 |
| walnuts | 0.832 | 0.750 | 0.979 | 0.660 | 0.545 | 0.274 |

### 2.3 Arm C (synthetic-calibrated, on synthetic test) — mean over 3 seeds

**Pipeline P5 (alpha)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.967 | 0.946 | 0.997 | 0.678 | 0.961 | 0.819 |
| fabric | 0.552 | 0.802 | 0.924 | 0.463 | 0.603 | 0.521 |
| fruit_jelly | 0.508 | 0.756 | 0.988 | 0.314 | 0.935 | 0.732 |
| rice | 0.652 | 0.832 | 0.960 | 0.642 | 0.785 | 0.698 |
| sheet_metal | 0.310 | 0.899 | 0.891 | 0.260 | 0.461 | 0.262 |
| vial | 0.653 | 0.832 | 0.995 | 0.581 | 0.957 | 0.806 |
| wallplugs | 0.620 | 0.840 | 0.971 | 0.493 | 0.789 | 0.606 |
| walnuts | 0.626 | 0.784 | 0.984 | 0.489 | 0.810 | 0.585 |

**Pipeline P6 (poisson)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.961 | 0.942 | 0.996 | 0.615 | 0.956 | 0.830 |
| fabric | 0.515 | 0.802 | 0.915 | 0.427 | 0.587 | 0.486 |
| fruit_jelly | 0.531 | 0.756 | 0.987 | 0.335 | 0.933 | 0.726 |
| rice | 0.644 | 0.832 | 0.959 | 0.622 | 0.770 | 0.676 |
| sheet_metal | 0.291 | 0.899 | 0.857 | 0.236 | 0.414 | 0.231 |
| vial | 0.604 | 0.832 | 0.989 | 0.504 | 0.939 | 0.786 |
| wallplugs | 0.632 | 0.840 | 0.968 | 0.419 | 0.789 | 0.617 |
| walnuts | 0.646 | 0.784 | 0.974 | 0.460 | 0.780 | 0.588 |

### 2.4 Calibration gap (Arm A − Arm B), pixel_F1 (SegF1)

| category | P5 (alpha) gap | P6 (poisson) gap |
|---|---|---|
| can | 0.000 | 0.000 |
| fabric | 0.054 | 0.095 |
| fruit_jelly | 0.007 | 0.007 |
| rice | 0.063 | 0.063 |
| sheet_metal | 0.281 | 0.283 |
| vial | 0.093 | 0.179 |
| wallplugs | 0.366 | 0.362 |
| walnuts | 0.095 | 0.059 |

### 2.5 Overall means (across all 8 categories)

| phase | arm | image_F1 | pixel_F1 (SegF1) | image_AUROC | pixel_AUROC |
|---|---|---|---|---|---|
| phase8_alpha | A | 0.8346 | 0.5160 | 0.7385 | 0.9098 |
| phase8_alpha | B | 0.7817 | 0.3960 | 0.7227 | 0.9095 |
| phase8_alpha | C | 0.8364 | 0.4898 | 0.6108 | 0.9636 |
| phase8_poisson | A | 0.8338 | 0.5160 | 0.7379 | 0.9115 |
| phase8_poisson | B | 0.7821 | 0.3849 | 0.7346 | 0.9113 |
| phase8_poisson | C | 0.8358 | 0.4524 | 0.6032 | 0.9557 |

### 2.6 Backbone comparison on the 4 categories common to both runs (`fruit_jelly`, `rice`, `wallplugs`, `walnuts`), Arm B

| backbone | pipeline | image_F1 | pixel_F1 (SegF1) |
|---|---|---|---|
| `vit_large_patch16_dinov3` (303M, phase7) | alpha | 0.7997 | 0.4348 |
| `vit_large_patch16_dinov3` (303M, phase7) | poisson | 0.8017 | 0.4850 |
| `vit_huge_plus_patch16_dinov3` (840M, phase8) | alpha | 0.7920 | 0.4700 |
| `vit_huge_plus_patch16_dinov3` (840M, phase8) | poisson | 0.7920 | 0.4800 |

### 2.7 Note on `can`

`can` scores at chance level for image-level real-test detection (Arm A/B image_AUROC
≈ 0.49, pixel_F1 ≈ 0.000) while Arm C (in-distribution on the synthetic set) scores
near-perfect (pixel_AUROC ≈ 0.996–0.997). This pattern is not specific to
`huge_plus`/phase8: the same category is comparably hard for every model tested in
phase5/phase6 (`dinomaly`, `efficient_ad`, `padim`, `patchcore`; image_AUROC in the
0.47–0.63 range, Arm A across the board), so it reflects a genuine, cross-model
difficulty of `can`'s real defects rather than a pipeline or backbone-specific issue.

## 3. Cross-Model Generalization (Phase 9)

### 3.1 Motivation and methodology

Phase 8 shows the semantic-defect-bank calibration source (P5/P6) works well for
SuperADD. This phase tests whether that is a property of the **calibration source**
(real donor-patch defects, alpha/Poisson-blended) or specific to SuperADD's
training-free memory-bank architecture, by re-running the *identical* P5/P6 calibration
against three other model families already benchmarked in phase 5 with the **live**
Perlin-noise pipelines (P1 texture-paste, P2 self-alpha, P3 self-Poisson) on the same
dataset:

- **`anomaly_dino`** — training-free, DINO-feature memory bank (different backbone
  family than SuperADD's DINOv3).
- **`padim`** — Gaussian-per-patch parametric model, single-pass fit.
- **`patchcore`** — coreset-subsampled memory bank (closest architectural relative of
  SuperADD, but a different, smaller backbone).

Same dataset, same 8 categories, same 3 seeds, same P5/P6 pipelines, same heldout
calibration protocol, same arms A/B/C as §1.4 — only the model and backbone change.
`dinomaly` and `efficient_ad` are deferred (need an epoch-count decision, since they are
full gradient-trained rather than val-only/single-pass); `draem` is deferred separately
(trains on its own internal Perlin synthesis, a different mechanism).

**Compute:** 8 categories × 3 seeds × 3 models = 72 jobs (both pipelines evaluated per
job), split across a local 2-GPU machine (`anomaly_dino` + `padim`) and a remote 8-GPU
machine (`patchcore`, parallelized 8-way). All 72 jobs completed with 0 errors and 0
AUROC-degenerate rows. Per-job cost varies sharply by model: `padim` ≈ 68s, `anomaly_dino`
≈ 125s, `patchcore` ≈ 827s (mean fit+test seconds; patchcore's coreset subsampling scales
with training-set size, so the larger categories — `can`, `fabric`, `walnuts` — are
markedly slower than in phase5's original 4-category timing average).

### 3.2 Headline result: does the semantic-defect-bank source generalize?

Arm B (synthetic-calibrated, evaluated on the real test set), mean over all 8 categories
and both pipelines, comparing the **same model** against the **old live Perlin pipelines**
(phase 5, P1–P3) vs. the **new semantic-defect-bank pipelines** (phase 9, P5/P6):

| model | image_F1 (Perlin → semantic-bank) | Δ | pixel_F1 (Perlin → semantic-bank) | Δ |
|---|---|---|---|---|
| anomaly_dino | 0.807 → 0.770 | −0.036 | 0.223 → **0.260** | **+0.037** |
| padim | 0.677 → **0.793** | **+0.117** | 0.046 → 0.038 | −0.009 |
| patchcore | 0.582 → **0.735** | **+0.154** | 0.157 → **0.185** | **+0.028** |

**The semantic-defect-bank calibration source generalizes beyond SuperADD.** For 2 of 3
models (`padim`, `patchcore`) it lifts image-F1 substantially (+0.12, +0.15); for 2 of 3
(`anomaly_dino`, `patchcore`) it also lifts pixel-F1. Only `padim`'s pixel-F1 is
marginally lower, and both numbers there are small in absolute terms (0.046 vs. 0.038).
No model regresses on both axes simultaneously. This is evidence the improvement comes
from the **calibration source itself** (real donor-patch defects vs. procedural Perlin
masks), not merely from SuperADD's specific architecture.

### 3.3 Perlin-vs-semantic-bank comparison for SuperADD itself (Phase 10)

**Now run — this gap is closed.** SuperADD was originally only evaluated with the
pregenerated semantic-defect-bank pipelines (P5/P6, phases 7–8); it had never been
included in phase 5/6's live-Perlin-pipeline sweep. Phase 10 (`phase10_superadd_perlin`)
closes this: the same SuperADD (`huge_plus`) model, same dataset, same 8 categories,
same 3 seeds, same heldout calibration protocol as phase 8, but calibrated with the
**live** Perlin pipelines P1 (texture-paste), P2 (self-alpha), P3 (self-Poisson) instead
of the pregenerated semantic-bank blends — mirroring phase 5's setup for the other
models exactly. 24 jobs (8 categories × 3 seeds), each producing arm A + 3×(arm B, arm
C). All 24 jobs completed, 0 errors, 0 AUROC-degenerate rows.

Arm B (synthetic-calibrated, on real test), mean over all 8 categories, pooling all
three Perlin pipelines (P1–P3) vs. both semantic-bank pipelines (P5/P6):

| metric | perlin (phase 10, P1–P3) | semantic-bank (phase 8, P5/P6) | Δ |
|---|---|---|---|
| image_F1 | 0.7408 | 0.7819 | **+0.0411** |
| pixel_F1 (SegF1) | 0.3452 | 0.3905 | **+0.0453** |
| image_AUROC | 0.7432 | 0.7287 | −0.0145 |
| pixel_AUROC | 0.9093 | 0.9104 | +0.0012 |

**The semantic-defect-bank calibration source also beats Perlin for SuperADD itself** —
image-F1 and pixel-F1 both improve (+0.04–0.05), matching the direction (though not
always the magnitude) of the gains seen for `padim`/`patchcore`/`anomaly_dino` in §3.2.
Per-category breakdown:

| category | perlin image_F1 | semantic image_F1 | Δ | perlin pixel_F1 | semantic pixel_F1 | Δ |
|---|---|---|---|---|---|---|
| can | 0.577 | 0.616 | +0.039 | 0.000 | 0.000 | 0.000 |
| fabric | 0.601 | 0.732 | **+0.131** | 0.559 | 0.713 | **+0.154** |
| fruit_jelly | 0.794 | 0.857 | +0.064 | 0.493 | 0.555 | +0.061 |
| rice | 0.707 | 0.811 | **+0.104** | 0.417 | 0.525 | **+0.108** |
| sheet_metal | 0.859 | 0.882 | +0.024 | 0.150 | 0.075 | −0.075 |
| vial | **0.968** | 0.857 | −0.111 | 0.474 | 0.436 | −0.038 |
| wallplugs | 0.632 | 0.750 | +0.118 | 0.093 | 0.178 | +0.086 |
| walnuts | 0.790 | 0.750 | −0.040 | 0.576 | 0.642 | +0.066 |

Semantic-bank wins image-F1 on 6/8 categories and pixel-F1 on 6/8 categories; `vial` is
the clearest exception (Perlin calibration is notably better there for image-F1),
`sheet_metal` the clearest pixel-F1 exception. Per-pipeline breakdown (arm B, mean over
8 categories) shows the three Perlin pipelines have very different profiles: **P2
(self-alpha)** gives the best image-F1 (0.783) but weakest pixel-F1 (0.263); **P1
(texture-paste)** the opposite (0.696 image-F1, 0.410 pixel-F1); **P3 (self-Poisson)** is
intermediate (0.744, 0.362) — the same qualitative pattern documented for the other
models in phases 0–6.

As a sanity check, Arm A (oracle) is nearly identical between phase 10 and phase 8
(image_F1 0.8367 vs. 0.8342, pixel_F1 0.5157 vs. 0.516) — expected, since it is the same
model/backbone/dataset re-trained with a different seed of the same stochastic coreset
subsampling, confirming the two phases are otherwise comparable.

**Compute:** 24 jobs, mean ≈2.8 min fit+test compute/job, but real wall time is
dominated by CPU-bound live Perlin/Poisson generation across 3 pipelines per job
(20–55 min/job depending on category size) — markedly slower than phase 8's
pregenerated-pipeline path (11–26 min/job), because phase 8 just loads pre-rendered
PNGs while phase 10 generates and composites the calibration images live, three times
per job. Full per-category detail is in Appendix G.

### 3.4 Overall means by model and arm (phase 9, all 8 categories)

| model | arm | image_F1 | pixel_F1 (SegF1) | image_AUROC | pixel_AUROC |
|---|---|---|---|---|---|
| anomaly_dino | A | 0.8151 | 0.3347 | 0.7007 | 0.8729 |
| anomaly_dino | B | 0.7703 | 0.2604 | 0.7024 | 0.8730 |
| anomaly_dino | C | 0.8387 | 0.4356 | 0.5381 | 0.9540 |
| padim | A | 0.8076 | 0.0777 | 0.6190 | 0.7991 |
| padim | B | 0.7934 | 0.0377 | 0.6217 | 0.8066 |
| padim | C | 0.8260 | 0.2314 | 0.4156 | 0.9448 |
| patchcore | A | 0.8243 | 0.2610 | 0.7267 | 0.8726 |
| patchcore | B | 0.7354 | 0.1853 | 0.7121 | 0.8703 |
| patchcore | C | 0.8522 | 0.4042 | 0.5522 | 0.9649 |

Full per-category breakdowns for all three models (Arm A/B/C, both pipelines) are in
the Appendix.

## 4. Methodological Note: "Perlin baseline" is not vanilla anomalib Perlin

**This section documents an important caveat discovered while reviewing the results:
every "Perlin pipelines P1–P3" comparison in this document (§3.2, §3.3, Appendix E)
pools in two pipelines that already include our own enhancements over stock
anomalib/DRAEM-style Perlin-noise synthesis — the reported "Perlin baseline" is
therefore stronger than a genuinely untouched anomalib default, and the true
improvement margin of the semantic-defect-bank calibration is larger than reported
above.**

### 4.1 What "Perlin" actually means in this codebase

None of P1/P2/P3 call anomalib's stock `PerlinAnomalyGenerator` class. That class is
used internally by the DRAEM *model* for its own training loop (a separate mechanism,
unrelated to the calibration pipelines discussed here). Our calibration pipelines are
built from a custom compositional generator
(`SyntheticAnomalyGenerator`/`PipelineConfig`, added in this project's own commit
`aa61975c`) that reuses anomalib's stock Perlin *noise/mask* math as one of four
independent, swappable axes — mask shape, region constraint, content source, blend
strategy — and only P1 keeps all three non-mask axes at their vanilla-Perlin/DRAEM
settings:

| pipeline | region | source | blend | vs. vanilla Perlin/DRAEM |
|---|---|---|---|---|
| **P1** (`texture_alpha`) | whole image | external DTD texture | alpha | **matches** the classic Perlin/DRAEM recipe |
| P2 (`self_alpha`) | **foreground only** (our addition, Otsu-based) | **same-image self-patch** (our addition, NSA-style) | alpha | enhanced |
| P3 (`self_poisson`) | **foreground only** (our addition) | **same-image self-patch** (our addition) | **Poisson seamless** (our addition) | enhanced |

P2/P3's foreground-restriction and self-sourcing were built specifically by this
project (documented from the very earliest phases of this work) to produce a better
calibration signal than plain DRAEM/Perlin — they are prior contributions of this
project, not properties of unmodified anomalib.

### 4.2 The data confirms the effect, and it is large

Isolating **P1 alone** (the true vanilla-style baseline) vs. the **P1–P3 pooled mean**
that every prior comparison in this document used as "the Perlin baseline," as a
percentage of each model's own oracle (Arm A) performance:

| model | P1-only image_F1 (% of oracle) | P1–P3 pooled image_F1 (% of oracle) |
|---|---|---|
| padim | 59% | 84% |
| dinomaly | 49% | 71% |
| anomaly_dino | 74% | 91% |
| efficient_ad | 91% | 95% |
| patchcore | 42% | 70% |
| superadd | 83% | 89% |

P1 alone is dramatically worse than real detection for most models (42–83% of oracle)
— matching the intuition that naive Perlin-noise synthesis should perform much worse
than real defects. P2/P3 close most of that gap (70–95% of oracle) purely through their
own region/source enhancements, **before any real donor-patch content is introduced**.
Pooling all three into one number therefore substantially overstates what "Perlin"
alone can do.

### 4.3 Corrected improvement margin: semantic-bank vs. true vanilla-Perlin (P1-only)

| model | Δ image_F1 vs. **pooled P1–P3** (previously reported) | Δ image_F1 vs. **P1-only** (honest vanilla baseline) |
|---|---|---|
| padim | +0.117 | **+0.317** |
| patchcore | +0.154 | **+0.390** |
| anomaly_dino | −0.036 | **+0.116** (sign flips) |
| superadd | +0.041 | **+0.086** |

The semantic-defect-bank calibration source beats a genuine vanilla-Perlin baseline by
2–3× the margin reported against the pooled P1–P3 average, and for `anomaly_dino` the
comparison even flips from "roughly tied/slightly worse" to "clearly better." Pixel-F1
is more mixed: P1's whole-image alpha-pasted textures sometimes produce large, easily
segmented blobs that score *better* on pixel-F1 than the more subtle foreground-aware
pipelines despite being a much weaker calibration signal overall (visible in `padim`'s
and `anomaly_dino`'s pixel_F1 numbers above) — a real, separate effect, not an artifact
of pooling.

### 4.4 Recommendation for the paper

Report three tiers explicitly rather than one "Perlin baseline":

1. **P1 (vanilla-style Perlin/DRAEM)** — the true "no extra improvements" baseline,
   for any claim framed as "vs. plain anomalib Perlin synthesis."
2. **P2/P3 (foreground + self-source enhancement)** — an intermediate baseline that is
   itself a contribution of this project's earlier phases, not stock anomalib; useful
   for the separate claim "does real-defect content beat procedurally-enhanced
   synthetic content."
3. **P5/P6 (semantic-bank, real donor patches)** — the new method.

All P1-only, P2-only, and P3-only breakdowns needed to build these three-tier
comparisons are already available in Appendix G (SuperADD) and in `results.csv`
(`phase = "phase5"`, filter on `pipeline` for the other models).

---

## Appendix — Full Per-Category Results, All 4 Models

Detailed results for every model/category/arm/pipeline combination discussed above.
SuperADD (phase 8) is repeated here for a self-contained reference; `anomaly_dino`,
`padim`, and `patchcore` (phase 9) appear for the first time in full detail.

### A. SuperADD (`vit_huge_plus_patch16_dinov3`, phase 8)

**A.1 — Arm A (oracle), mean ± std pooled over both phase8_alpha/phase8_poisson training runs (6 samples/category)**

| category | image_AUROC | image_F1 | image_AUPR | image_Precision | image_Recall | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPR | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|---|---|---|---|
| can | 0.493 ± 0.006 | 0.713 ± 0.005 | 0.651 ± 0.032 | 0.564 ± 0.012 | 0.970 ± 0.018 | 0.659 ± 0.012 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.222 ± 0.019 | 0.011 ± 0.001 |
| fabric | 0.670 ± 0.032 | 0.748 ± 0.016 | 0.773 ± 0.016 | 0.631 ± 0.052 | 0.930 ± 0.065 | 0.996 ± 0.000 | 0.787 ± 0.004 | 0.729 ± 0.002 | 0.415 ± 0.014 | 0.251 ± 0.014 |
| fruit_jelly | 0.843 ± 0.010 | 0.858 ± 0.006 | 0.954 ± 0.003 | 0.778 ± 0.019 | 0.958 ± 0.023 | 0.920 ± 0.000 | 0.561 ± 0.002 | 0.440 ± 0.004 | 0.533 ± 0.004 | 0.173 ± 0.002 |
| rice | 0.868 ± 0.025 | 0.865 ± 0.017 | 0.940 ± 0.010 | 0.806 ± 0.044 | 0.937 ± 0.035 | 0.953 ± 0.002 | 0.588 ± 0.003 | 0.545 ± 0.001 | 0.669 ± 0.009 | 0.511 ± 0.007 |
| sheet_metal | 0.653 ± 0.012 | 0.895 ± 0.004 | 0.885 ± 0.009 | 0.822 ± 0.009 | 0.983 ± 0.014 | 0.880 ± 0.000 | 0.357 ± 0.001 | 0.229 ± 0.001 | 0.236 ± 0.000 | 0.049 ± 0.000 |
| vial | 1.000 ± 0.000 | 0.994 ± 0.002 | 0.999 ± 0.000 | 0.998 ± 0.004 | 0.990 ± 0.000 | 0.971 ± 0.000 | 0.571 ± 0.004 | 0.547 ± 0.002 | 0.961 ± 0.001 | 0.852 ± 0.003 |
| wallplugs | 0.535 ± 0.012 | 0.759 ± 0.008 | 0.652 ± 0.005 | 0.619 ± 0.011 | 0.981 ± 0.013 | 0.927 ± 0.003 | 0.543 ± 0.002 | 0.509 ± 0.003 | 0.454 ± 0.010 | 0.208 ± 0.004 |
| walnuts | 0.843 ± 0.008 | 0.840 ± 0.006 | 0.922 ± 0.003 | 0.908 ± 0.011 | 0.781 ± 0.011 | 0.979 ± 0.001 | 0.719 ± 0.001 | 0.742 ± 0.001 | 0.544 ± 0.021 | 0.274 ± 0.004 |

**A.2 — Arm B, Pipeline P5 (alpha)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.489 | 0.614 | 0.654 | 0.000 | 0.212 | 0.010 |
| fabric | 0.667 | 0.732 | 0.996 | 0.733 | 0.409 | 0.243 |
| fruit_jelly | 0.786 | 0.857 | 0.918 | 0.554 | 0.533 | 0.173 |
| rice | 0.865 | 0.811 | 0.953 | 0.525 | 0.671 | 0.511 |
| sheet_metal | 0.639 | 0.882 | 0.881 | 0.076 | 0.236 | 0.049 |
| vial | 1.000 | 0.857 | 0.970 | 0.479 | 0.960 | 0.850 |
| wallplugs | 0.525 | 0.750 | 0.926 | 0.177 | 0.450 | 0.206 |
| walnuts | 0.811 | 0.750 | 0.979 | 0.624 | 0.542 | 0.274 |

**A.3 — Arm B, Pipeline P6 (poisson)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.492 | 0.618 | 0.663 | 0.000 | 0.233 | 0.011 |
| fabric | 0.670 | 0.732 | 0.996 | 0.693 | 0.420 | 0.259 |
| fruit_jelly | 0.839 | 0.857 | 0.920 | 0.555 | 0.532 | 0.173 |
| rice | 0.871 | 0.811 | 0.952 | 0.525 | 0.668 | 0.511 |
| sheet_metal | 0.639 | 0.882 | 0.881 | 0.074 | 0.236 | 0.049 |
| vial | 1.000 | 0.857 | 0.971 | 0.393 | 0.961 | 0.853 |
| wallplugs | 0.535 | 0.750 | 0.929 | 0.180 | 0.458 | 0.210 |
| walnuts | 0.832 | 0.750 | 0.979 | 0.660 | 0.545 | 0.274 |

**A.4 — Arm C, Pipeline P5 (alpha)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.967 | 0.946 | 0.997 | 0.678 | 0.961 | 0.819 |
| fabric | 0.552 | 0.802 | 0.924 | 0.463 | 0.603 | 0.521 |
| fruit_jelly | 0.508 | 0.756 | 0.988 | 0.314 | 0.935 | 0.732 |
| rice | 0.652 | 0.832 | 0.960 | 0.642 | 0.785 | 0.698 |
| sheet_metal | 0.310 | 0.899 | 0.891 | 0.260 | 0.461 | 0.262 |
| vial | 0.653 | 0.832 | 0.995 | 0.581 | 0.957 | 0.806 |
| wallplugs | 0.620 | 0.840 | 0.971 | 0.493 | 0.789 | 0.606 |
| walnuts | 0.626 | 0.784 | 0.984 | 0.489 | 0.810 | 0.585 |

**A.5 — Arm C, Pipeline P6 (poisson)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.961 | 0.942 | 0.996 | 0.615 | 0.956 | 0.830 |
| fabric | 0.515 | 0.802 | 0.915 | 0.427 | 0.587 | 0.486 |
| fruit_jelly | 0.531 | 0.756 | 0.987 | 0.335 | 0.933 | 0.726 |
| rice | 0.644 | 0.832 | 0.959 | 0.622 | 0.770 | 0.676 |
| sheet_metal | 0.291 | 0.899 | 0.857 | 0.236 | 0.414 | 0.231 |
| vial | 0.604 | 0.832 | 0.989 | 0.504 | 0.939 | 0.786 |
| wallplugs | 0.632 | 0.840 | 0.968 | 0.419 | 0.789 | 0.617 |
| walnuts | 0.646 | 0.784 | 0.974 | 0.460 | 0.780 | 0.588 |

### B. `anomaly_dino` (phase 9)

**B.1 — Arm A (oracle), mean ± std over 3 seeds**

| category | image_AUROC | image_F1 | image_AUPR | image_Precision | image_Recall | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPR | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|---|---|---|---|
| can | 0.502 ± 0.004 | 0.714 ± 0.000 | 0.619 ± 0.002 | 0.556 ± 0.000 | 1.000 ± 0.000 | 0.590 ± 0.068 | 0.001 ± 0.000 | 0.000 ± 0.000 | 0.226 ± 0.014 | 0.058 ± 0.006 |
| fabric | 0.618 ± 0.014 | 0.761 ± 0.004 | 0.698 ± 0.005 | 0.618 ± 0.003 | 0.989 ± 0.019 | 0.980 ± 0.001 | 0.462 ± 0.004 | 0.356 ± 0.004 | 0.447 ± 0.019 | 0.202 ± 0.005 |
| fruit_jelly | 0.868 ± 0.006 | 0.873 ± 0.005 | 0.962 ± 0.002 | 0.962 ± 0.037 | 0.800 ± 0.033 | 0.922 ± 0.002 | 0.401 ± 0.003 | 0.257 ± 0.001 | 0.659 ± 0.006 | 0.375 ± 0.001 |
| rice | 0.810 ± 0.011 | 0.833 ± 0.012 | 0.907 ± 0.005 | 0.760 ± 0.028 | 0.922 ± 0.033 | 0.953 ± 0.002 | 0.585 ± 0.000 | 0.555 ± 0.001 | 0.688 ± 0.009 | 0.502 ± 0.004 |
| sheet_metal | 0.667 ± 0.007 | 0.886 ± 0.006 | 0.897 ± 0.003 | 0.808 ± 0.002 | 0.981 ± 0.013 | 0.884 ± 0.003 | 0.318 ± 0.003 | 0.206 ± 0.002 | 0.378 ± 0.004 | 0.258 ± 0.001 |
| vial | 0.916 ± 0.006 | 0.909 ± 0.007 | 0.972 ± 0.002 | 0.847 ± 0.008 | 0.981 ± 0.010 | 0.919 ± 0.002 | 0.322 ± 0.002 | 0.315 ± 0.001 | 0.901 ± 0.001 | 0.718 ± 0.004 |
| wallplugs | 0.434 ± 0.004 | 0.747 ± 0.003 | 0.604 ± 0.003 | 0.606 ± 0.002 | 0.974 ± 0.006 | 0.799 ± 0.007 | 0.024 ± 0.001 | 0.008 ± 0.000 | 0.262 ± 0.021 | 0.038 ± 0.001 |
| walnuts | 0.791 ± 0.003 | 0.798 ± 0.006 | 0.867 ± 0.002 | 0.772 ± 0.008 | 0.826 ± 0.006 | 0.936 ± 0.003 | 0.565 ± 0.001 | 0.486 ± 0.004 | 0.582 ± 0.017 | 0.266 ± 0.004 |

**B.2 — Arm B, Pipeline P5 (alpha)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.509 | 0.522 | 0.581 | 0.000 | 0.227 | 0.058 |
| fabric | 0.643 | 0.732 | 0.978 | 0.371 | 0.446 | 0.202 |
| fruit_jelly | 0.865 | 0.857 | 0.923 | 0.239 | 0.660 | 0.378 |
| rice | 0.796 | 0.811 | 0.951 | 0.497 | 0.680 | 0.494 |
| sheet_metal | 0.667 | 0.882 | 0.889 | 0.154 | 0.380 | 0.258 |
| vial | 0.908 | 0.857 | 0.918 | 0.263 | 0.898 | 0.711 |
| wallplugs | 0.446 | 0.750 | 0.804 | 0.023 | 0.266 | 0.040 |
| walnuts | 0.780 | 0.750 | 0.938 | 0.522 | 0.587 | 0.269 |

**B.3 — Arm B, Pipeline P6 (poisson)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.509 | 0.524 | 0.586 | 0.000 | 0.227 | 0.058 |
| fabric | 0.643 | 0.732 | 0.978 | 0.376 | 0.446 | 0.202 |
| fruit_jelly | 0.877 | 0.857 | 0.923 | 0.212 | 0.660 | 0.378 |
| rice | 0.796 | 0.811 | 0.951 | 0.568 | 0.680 | 0.494 |
| sheet_metal | 0.654 | 0.882 | 0.889 | 0.090 | 0.380 | 0.258 |
| vial | 0.908 | 0.857 | 0.918 | 0.274 | 0.898 | 0.711 |
| wallplugs | 0.450 | 0.750 | 0.804 | 0.018 | 0.266 | 0.040 |
| walnuts | 0.788 | 0.750 | 0.938 | 0.559 | 0.587 | 0.269 |

**B.4 — Arm C, Pipeline P5 (alpha)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.979 | 0.970 | 0.997 | 0.588 | 0.966 | 0.884 |
| fabric | 0.449 | 0.802 | 0.946 | 0.466 | 0.699 | 0.587 |
| fruit_jelly | 0.441 | 0.756 | 0.997 | 0.452 | 0.982 | 0.939 |
| rice | 0.657 | 0.832 | 0.957 | 0.546 | 0.831 | 0.760 |
| sheet_metal | 0.237 | 0.899 | 0.816 | 0.248 | 0.412 | 0.254 |
| vial | 0.566 | 0.832 | 0.988 | 0.500 | 0.943 | 0.865 |
| wallplugs | 0.495 | 0.840 | 0.974 | 0.434 | 0.865 | 0.742 |
| walnuts | 0.583 | 0.784 | 0.994 | 0.415 | 0.923 | 0.785 |

**B.5 — Arm C, Pipeline P6 (poisson)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.965 | 0.960 | 0.995 | 0.541 | 0.956 | 0.870 |
| fabric | 0.394 | 0.802 | 0.929 | 0.409 | 0.676 | 0.565 |
| fruit_jelly | 0.452 | 0.756 | 0.997 | 0.457 | 0.978 | 0.926 |
| rice | 0.610 | 0.832 | 0.952 | 0.501 | 0.800 | 0.700 |
| sheet_metal | 0.213 | 0.899 | 0.796 | 0.239 | 0.398 | 0.248 |
| vial | 0.517 | 0.832 | 0.971 | 0.442 | 0.906 | 0.833 |
| wallplugs | 0.484 | 0.840 | 0.969 | 0.344 | 0.848 | 0.724 |
| walnuts | 0.567 | 0.784 | 0.985 | 0.387 | 0.870 | 0.734 |

### C. `padim` (phase 9)

**C.1 — Arm A (oracle), mean ± std over 3 seeds**

| category | image_AUROC | image_F1 | image_AUPR | image_Precision | image_Recall | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPR | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|---|---|---|---|
| can | 0.528 ± 0.019 | 0.715 ± 0.000 | 0.681 ± 0.009 | 0.563 ± 0.003 | 0.981 ± 0.006 | 0.748 ± 0.097 | 0.013 ± 0.020 | 0.001 ± 0.001 | 0.300 ± 0.062 | 0.069 ± 0.020 |
| fabric | 0.609 ± 0.064 | 0.741 ± 0.017 | 0.690 ± 0.050 | 0.595 ± 0.027 | 0.981 ± 0.013 | 0.713 ± 0.034 | 0.030 ± 0.002 | 0.014 ± 0.001 | 0.441 ± 0.012 | 0.182 ± 0.050 |
| fruit_jelly | 0.894 ± 0.014 | 0.898 ± 0.009 | 0.965 ± 0.003 | 0.899 ± 0.055 | 0.900 ± 0.050 | 0.893 ± 0.007 | 0.115 ± 0.009 | 0.051 ± 0.003 | 0.567 ± 0.030 | 0.099 ± 0.037 |
| rice | 0.567 ± 0.146 | 0.814 ± 0.014 | 0.750 ± 0.082 | 0.704 ± 0.043 | 0.967 ± 0.038 | 0.781 ± 0.013 | 0.079 ± 0.032 | 0.028 ± 0.016 | 0.374 ± 0.022 | 0.192 ± 0.007 |
| sheet_metal | 0.694 ± 0.114 | 0.899 ± 0.013 | 0.892 ± 0.052 | 0.835 ± 0.017 | 0.974 ± 0.017 | 0.804 ± 0.019 | 0.085 ± 0.015 | 0.032 ± 0.006 | 0.198 ± 0.014 | 0.024 ± 0.013 |
| vial | 0.572 ± 0.027 | 0.867 ± 0.010 | 0.812 ± 0.003 | 0.777 ± 0.012 | 0.981 ± 0.016 | 0.800 ± 0.008 | 0.109 ± 0.005 | 0.049 ± 0.002 | 0.537 ± 0.013 | 0.013 ± 0.002 |
| wallplugs | 0.443 ± 0.128 | 0.759 ± 0.012 | 0.577 ± 0.091 | 0.621 ± 0.023 | 0.978 ± 0.019 | 0.748 ± 0.024 | 0.009 ± 0.002 | 0.004 ± 0.000 | 0.301 ± 0.013 | 0.060 ± 0.026 |
| walnuts | 0.645 ± 0.062 | 0.769 ± 0.016 | 0.744 ± 0.043 | 0.639 ± 0.019 | 0.967 ± 0.038 | 0.907 ± 0.005 | 0.181 ± 0.041 | 0.104 ± 0.036 | 0.504 ± 0.020 | 0.147 ± 0.014 |

**C.2 — Arm B, Pipeline P5 (alpha)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.553 | 0.678 | 0.808 | 0.000 | 0.330 | 0.072 |
| fabric | 0.610 | 0.732 | 0.713 | 0.011 | 0.441 | 0.182 |
| fruit_jelly | 0.894 | 0.870 | 0.893 | 0.009 | 0.567 | 0.099 |
| rice | 0.565 | 0.811 | 0.781 | 0.018 | 0.374 | 0.192 |
| sheet_metal | 0.684 | 0.882 | 0.804 | 0.070 | 0.198 | 0.024 |
| vial | 0.573 | 0.857 | 0.800 | 0.001 | 0.537 | 0.013 |
| wallplugs | 0.471 | 0.750 | 0.748 | 0.006 | 0.301 | 0.060 |
| walnuts | 0.645 | 0.750 | 0.907 | 0.172 | 0.504 | 0.147 |

**C.3 — Arm B, Pipeline P6 (poisson)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.549 | 0.701 | 0.808 | 0.000 | 0.330 | 0.072 |
| fabric | 0.610 | 0.732 | 0.713 | 0.012 | 0.441 | 0.182 |
| fruit_jelly | 0.894 | 0.882 | 0.893 | 0.013 | 0.567 | 0.099 |
| rice | 0.565 | 0.811 | 0.781 | 0.041 | 0.374 | 0.192 |
| sheet_metal | 0.649 | 0.882 | 0.804 | 0.068 | 0.198 | 0.024 |
| vial | 0.573 | 0.857 | 0.800 | 0.003 | 0.537 | 0.013 |
| wallplugs | 0.468 | 0.750 | 0.748 | 0.007 | 0.301 | 0.060 |
| walnuts | 0.645 | 0.750 | 0.907 | 0.171 | 0.504 | 0.147 |

**C.4 — Arm C, Pipeline P5 (alpha)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.846 | 0.851 | 0.994 | 0.446 | 0.942 | 0.777 |
| fabric | 0.228 | 0.802 | 0.925 | 0.215 | 0.625 | 0.422 |
| fruit_jelly | 0.697 | 0.782 | 0.995 | 0.192 | 0.981 | 0.896 |
| rice | 0.593 | 0.832 | 0.963 | 0.568 | 0.845 | 0.713 |
| sheet_metal | 0.205 | 0.899 | 0.861 | 0.175 | 0.415 | 0.190 |
| vial | 0.614 | 0.832 | 0.986 | 0.222 | 0.929 | 0.714 |
| wallplugs | 0.051 | 0.840 | 0.918 | 0.050 | 0.649 | 0.241 |
| walnuts | 0.316 | 0.786 | 0.981 | 0.140 | 0.864 | 0.554 |

**C.5 — Arm C, Pipeline P6 (poisson)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.818 | 0.830 | 0.992 | 0.414 | 0.936 | 0.766 |
| fabric | 0.138 | 0.802 | 0.902 | 0.131 | 0.555 | 0.345 |
| fruit_jelly | 0.662 | 0.772 | 0.994 | 0.165 | 0.979 | 0.879 |
| rice | 0.494 | 0.832 | 0.954 | 0.506 | 0.752 | 0.584 |
| sheet_metal | 0.163 | 0.899 | 0.812 | 0.127 | 0.332 | 0.131 |
| vial | 0.547 | 0.832 | 0.983 | 0.219 | 0.915 | 0.702 |
| wallplugs | 0.019 | 0.840 | 0.886 | 0.024 | 0.580 | 0.152 |
| walnuts | 0.258 | 0.786 | 0.970 | 0.108 | 0.820 | 0.480 |

### D. `patchcore` (phase 9)

**D.1 — Arm A (oracle), mean ± std over 3 seeds**

| category | image_AUROC | image_F1 | image_AUPR | image_Precision | image_Recall | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPR | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|---|---|---|---|
| can | 0.476 ± 0.005 | 0.709 ± 0.000 | 0.616 ± 0.004 | 0.553 ± 0.000 | 0.989 ± 0.000 | 0.758 ± 0.013 | 0.001 ± 0.000 | 0.000 ± 0.000 | 0.310 ± 0.004 | 0.064 ± 0.004 |
| fabric | 0.748 ± 0.021 | 0.798 ± 0.004 | 0.802 ± 0.014 | 0.703 ± 0.006 | 0.922 ± 0.000 | 0.776 ± 0.001 | 0.153 ± 0.004 | 0.086 ± 0.002 | 0.499 ± 0.003 | 0.186 ± 0.004 |
| fruit_jelly | 0.938 ± 0.018 | 0.923 ± 0.009 | 0.979 ± 0.007 | 0.942 ± 0.009 | 0.906 ± 0.019 | 0.942 ± 0.000 | 0.402 ± 0.002 | 0.330 ± 0.003 | 0.669 ± 0.001 | 0.398 ± 0.004 |
| rice | 0.563 ± 0.032 | 0.810 ± 0.002 | 0.776 ± 0.015 | 0.686 ± 0.003 | 0.989 ± 0.000 | 0.848 ± 0.002 | 0.232 ± 0.004 | 0.157 ± 0.002 | 0.415 ± 0.005 | 0.198 ± 0.002 |
| sheet_metal | 0.795 ± 0.026 | 0.890 ± 0.000 | 0.939 ± 0.010 | 0.809 ± 0.000 | 0.989 ± 0.000 | 0.849 ± 0.002 | 0.306 ± 0.003 | 0.158 ± 0.005 | 0.328 ± 0.004 | 0.131 ± 0.003 |
| vial | 0.916 ± 0.010 | 0.905 ± 0.003 | 0.971 ± 0.004 | 0.842 ± 0.009 | 0.978 ± 0.015 | 0.939 ± 0.000 | 0.323 ± 0.001 | 0.333 ± 0.000 | 0.925 ± 0.000 | 0.741 ± 0.001 |
| wallplugs | 0.533 ± 0.020 | 0.748 ± 0.006 | 0.625 ± 0.007 | 0.611 ± 0.024 | 0.967 ± 0.038 | 0.905 ± 0.003 | 0.160 ± 0.006 | 0.083 ± 0.004 | 0.483 ± 0.001 | 0.245 ± 0.000 |
| walnuts | 0.846 ± 0.005 | 0.811 ± 0.011 | 0.907 ± 0.001 | 0.755 ± 0.046 | 0.878 ± 0.038 | 0.963 ± 0.001 | 0.512 ± 0.002 | 0.500 ± 0.001 | 0.682 ± 0.009 | 0.351 ± 0.003 |

**D.2 — Arm B, Pipeline P5 (alpha)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.474 | 0.401 | 0.735 | 0.000 | 0.310 | 0.064 |
| fabric | 0.722 | 0.732 | 0.776 | 0.150 | 0.499 | 0.186 |
| fruit_jelly | 0.938 | 0.678 | 0.942 | 0.131 | 0.669 | 0.398 |
| rice | 0.563 | 0.811 | 0.848 | 0.033 | 0.415 | 0.198 |
| sheet_metal | 0.795 | 0.882 | 0.849 | 0.283 | 0.328 | 0.131 |
| vial | 0.908 | 0.857 | 0.939 | 0.299 | 0.925 | 0.741 |
| wallplugs | 0.488 | 0.750 | 0.905 | 0.088 | 0.483 | 0.245 |
| walnuts | 0.836 | 0.750 | 0.963 | 0.511 | 0.682 | 0.351 |

**D.3 — Arm B, Pipeline P6 (poisson)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.474 | 0.392 | 0.746 | 0.000 | 0.310 | 0.064 |
| fabric | 0.714 | 0.732 | 0.776 | 0.151 | 0.499 | 0.186 |
| fruit_jelly | 0.938 | 0.732 | 0.942 | 0.154 | 0.669 | 0.398 |
| rice | 0.563 | 0.811 | 0.848 | 0.037 | 0.415 | 0.198 |
| sheet_metal | 0.795 | 0.882 | 0.849 | 0.259 | 0.328 | 0.131 |
| vial | 0.851 | 0.857 | 0.939 | 0.283 | 0.925 | 0.741 |
| wallplugs | 0.502 | 0.750 | 0.905 | 0.090 | 0.483 | 0.245 |
| walnuts | 0.835 | 0.750 | 0.963 | 0.496 | 0.682 | 0.351 |

**D.4 — Arm C, Pipeline P5 (alpha)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.984 | 0.962 | 0.997 | 0.560 | 0.968 | 0.819 |
| fabric | 0.419 | 0.802 | 0.966 | 0.312 | 0.735 | 0.511 |
| fruit_jelly | 0.898 | 0.898 | 0.998 | 0.548 | 0.986 | 0.947 |
| rice | 0.611 | 0.832 | 0.960 | 0.566 | 0.777 | 0.684 |
| sheet_metal | 0.267 | 0.899 | 0.927 | 0.287 | 0.563 | 0.319 |
| vial | 0.635 | 0.832 | 0.991 | 0.433 | 0.926 | 0.748 |
| wallplugs | 0.258 | 0.840 | 0.958 | 0.206 | 0.728 | 0.429 |
| walnuts | 0.630 | 0.786 | 0.987 | 0.457 | 0.841 | 0.592 |

**D.5 — Arm C, Pipeline P6 (poisson)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.963 | 0.940 | 0.995 | 0.541 | 0.959 | 0.815 |
| fabric | 0.286 | 0.802 | 0.944 | 0.248 | 0.660 | 0.386 |
| fruit_jelly | 0.848 | 0.854 | 0.997 | 0.526 | 0.980 | 0.909 |
| rice | 0.547 | 0.832 | 0.951 | 0.518 | 0.704 | 0.591 |
| sheet_metal | 0.178 | 0.899 | 0.884 | 0.229 | 0.463 | 0.224 |
| vial | 0.499 | 0.832 | 0.972 | 0.432 | 0.890 | 0.719 |
| wallplugs | 0.239 | 0.840 | 0.941 | 0.197 | 0.698 | 0.421 |
| walnuts | 0.572 | 0.786 | 0.971 | 0.408 | 0.787 | 0.534 |

### E. Perlin (phase 5) vs. semantic-defect-bank (phase 9) — full metric comparison

Arm B, mean over all 8 categories and both pipelines (P1–P3 pooled for phase 5; P5/P6
pooled for phase 9):

| model | metric | perlin (phase 5) | semantic-bank (phase 9) | Δ |
|---|---|---|---|---|
| anomaly_dino | image_F1 | 0.8065 | 0.7703 | −0.0362 |
| anomaly_dino | pixel_F1 | 0.2230 | 0.2604 | +0.0374 |
| anomaly_dino | image_AUROC | 0.6769 | 0.7024 | +0.0255 |
| anomaly_dino | pixel_AUROC | 0.8931 | 0.8730 | −0.0201 |
| padim | image_F1 | 0.6766 | 0.7934 | +0.1168 |
| padim | pixel_F1 | 0.0462 | 0.0377 | −0.0085 |
| padim | image_AUROC | 0.6276 | 0.6217 | −0.0059 |
| padim | pixel_AUROC | 0.7989 | 0.8066 | +0.0077 |
| patchcore | image_F1 | 0.5817 | 0.7354 | +0.1538 |
| patchcore | pixel_F1 | 0.1573 | 0.1853 | +0.0279 |
| patchcore | image_AUROC | 0.7243 | 0.7121 | −0.0122 |
| patchcore | pixel_AUROC | 0.8719 | 0.8703 | −0.0016 |

### F. Timing (phase 9, mean fit+test seconds per job)

| model | mean seconds/job | n jobs |
|---|---|---|
| padim | 67.6 | 24 |
| anomaly_dino | 124.6 | 24 |
| patchcore | 827.3 | 24 |

### G. SuperADD with live Perlin pipelines (phase 10)

**G.1 — Arm A (oracle), mean ± std over 3 seeds**

| category | image_AUROC | image_F1 | image_AUPR | image_Precision | image_Recall | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPR | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|---|---|---|---|
| can | 0.493 ± 0.007 | 0.715 ± 0.003 | 0.649 ± 0.042 | 0.566 ± 0.009 | 0.970 ± 0.023 | 0.647 ± 0.019 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.215 ± 0.003 | 0.010 ± 0.000 |
| fabric | 0.691 ± 0.035 | 0.760 ± 0.022 | 0.781 ± 0.017 | 0.671 ± 0.065 | 0.885 ± 0.063 | 0.996 ± 0.000 | 0.785 ± 0.003 | 0.727 ± 0.003 | 0.410 ± 0.009 | 0.250 ± 0.013 |
| fruit_jelly | 0.846 ± 0.006 | 0.856 ± 0.001 | 0.954 ± 0.002 | 0.848 ± 0.132 | 0.889 ± 0.123 | 0.920 ± 0.000 | 0.562 ± 0.002 | 0.442 ± 0.001 | 0.533 ± 0.004 | 0.174 ± 0.003 |
| rice | 0.877 ± 0.009 | 0.862 ± 0.007 | 0.944 ± 0.005 | 0.791 ± 0.026 | 0.948 ± 0.042 | 0.952 ± 0.002 | 0.587 ± 0.004 | 0.545 ± 0.002 | 0.673 ± 0.007 | 0.515 ± 0.011 |
| sheet_metal | 0.647 ± 0.022 | 0.895 ± 0.004 | 0.879 ± 0.019 | 0.825 ± 0.009 | 0.978 ± 0.019 | 0.881 ± 0.000 | 0.359 ± 0.001 | 0.229 ± 0.002 | 0.237 ± 0.000 | 0.049 ± 0.000 |
| vial | 1.000 ± 0.000 | 0.995 ± 0.000 | 0.999 ± 0.000 | 1.000 ± 0.000 | 0.990 ± 0.000 | 0.971 ± 0.000 | 0.572 ± 0.002 | 0.547 ± 0.003 | 0.961 ± 0.000 | 0.852 ± 0.002 |
| wallplugs | 0.558 ± 0.012 | 0.765 ± 0.010 | 0.663 ± 0.006 | 0.629 ± 0.006 | 0.978 ± 0.019 | 0.930 ± 0.001 | 0.541 ± 0.005 | 0.508 ± 0.003 | 0.458 ± 0.017 | 0.214 ± 0.003 |
| walnuts | 0.854 ± 0.006 | 0.845 ± 0.009 | 0.926 ± 0.003 | 0.911 ± 0.020 | 0.789 ± 0.000 | 0.978 ± 0.001 | 0.719 ± 0.001 | 0.742 ± 0.002 | 0.540 ± 0.013 | 0.274 ± 0.004 |

**G.2 — Arm B, Pipeline P1 (texture-paste)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.487 | 0.503 | 0.647 | 0.000 | 0.215 | 0.010 |
| fabric | 0.691 | 0.514 | 0.996 | 0.744 | 0.410 | 0.250 |
| fruit_jelly | 0.846 | 0.754 | 0.920 | 0.561 | 0.533 | 0.174 |
| rice | 0.877 | 0.618 | 0.952 | 0.567 | 0.673 | 0.515 |
| sheet_metal | 0.637 | 0.812 | 0.881 | 0.283 | 0.237 | 0.049 |
| vial | 1.000 | 0.919 | 0.971 | 0.396 | 0.961 | 0.852 |
| wallplugs | 0.559 | 0.621 | 0.930 | 0.122 | 0.458 | 0.214 |
| walnuts | 0.854 | 0.824 | 0.978 | 0.609 | 0.540 | 0.274 |

**G.3 — Arm B, Pipeline P2 (self-alpha)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.487 | 0.612 | 0.647 | 0.000 | 0.215 | 0.010 |
| fabric | 0.691 | 0.732 | 0.996 | 0.347 | 0.410 | 0.250 |
| fruit_jelly | 0.846 | 0.847 | 0.920 | 0.390 | 0.533 | 0.174 |
| rice | 0.877 | 0.786 | 0.952 | 0.211 | 0.673 | 0.515 |
| sheet_metal | 0.633 | 0.882 | 0.881 | 0.046 | 0.237 | 0.049 |
| vial | 1.000 | 0.995 | 0.971 | 0.572 | 0.961 | 0.852 |
| wallplugs | 0.559 | 0.657 | 0.930 | 0.042 | 0.458 | 0.214 |
| walnuts | 0.844 | 0.750 | 0.978 | 0.495 | 0.540 | 0.274 |

**G.4 — Arm B, Pipeline P3 (self-Poisson)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.487 | 0.615 | 0.647 | 0.000 | 0.215 | 0.010 |
| fabric | 0.691 | 0.556 | 0.996 | 0.585 | 0.410 | 0.250 |
| fruit_jelly | 0.846 | 0.779 | 0.920 | 0.529 | 0.533 | 0.174 |
| rice | 0.877 | 0.716 | 0.952 | 0.473 | 0.673 | 0.515 |
| sheet_metal | 0.633 | 0.882 | 0.881 | 0.120 | 0.237 | 0.049 |
| vial | 1.000 | 0.989 | 0.971 | 0.455 | 0.961 | 0.852 |
| wallplugs | 0.559 | 0.618 | 0.930 | 0.113 | 0.458 | 0.214 |
| walnuts | 0.853 | 0.797 | 0.978 | 0.622 | 0.540 | 0.274 |

**G.5 — Arm C, Pipeline P1 (texture-paste)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.999 | 0.991 | 0.962 | 0.637 | 0.775 | 0.425 |
| fabric | 0.984 | 0.970 | 0.946 | 0.667 | 0.534 | 0.289 |
| fruit_jelly | 0.995 | 0.972 | 0.987 | 0.782 | 0.857 | 0.559 |
| rice | 0.974 | 0.964 | 0.962 | 0.648 | 0.634 | 0.302 |
| sheet_metal | 0.927 | 0.949 | 0.953 | 0.591 | 0.618 | 0.293 |
| vial | 1.000 | 0.994 | 0.980 | 0.691 | 0.775 | 0.385 |
| wallplugs | 0.985 | 0.972 | 0.944 | 0.656 | 0.618 | 0.356 |
| walnuts | 0.984 | 0.968 | 0.950 | 0.652 | 0.574 | 0.322 |

**G.6 — Arm C, Pipeline P2 (self-alpha)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.977 | 0.955 | 0.974 | 0.620 | 0.771 | 0.404 |
| fabric | 0.753 | 0.802 | 0.785 | 0.367 | 0.304 | 0.190 |
| fruit_jelly | 0.788 | 0.796 | 0.942 | 0.431 | 0.604 | 0.308 |
| rice | 0.799 | 0.850 | 0.835 | 0.461 | 0.425 | 0.270 |
| sheet_metal | 0.253 | 0.899 | 0.827 | 0.207 | 0.323 | 0.142 |
| vial | 0.882 | 0.880 | 0.974 | 0.457 | 0.831 | 0.440 |
| wallplugs | 0.885 | 0.890 | 0.911 | 0.521 | 0.481 | 0.288 |
| walnuts | 0.612 | 0.784 | 0.899 | 0.434 | 0.419 | 0.221 |

**G.7 — Arm C, Pipeline P3 (self-Poisson)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| can | 0.979 | 0.966 | 0.918 | 0.522 | 0.503 | 0.220 |
| fabric | 0.946 | 0.943 | 0.893 | 0.413 | 0.568 | 0.360 |
| fruit_jelly | 0.860 | 0.867 | 0.961 | 0.614 | 0.587 | 0.319 |
| rice | 0.925 | 0.927 | 0.914 | 0.527 | 0.607 | 0.371 |
| sheet_metal | 0.307 | 0.899 | 0.858 | 0.259 | 0.455 | 0.211 |
| vial | 0.921 | 0.929 | 0.954 | 0.472 | 0.714 | 0.319 |
| wallplugs | 0.971 | 0.968 | 0.912 | 0.470 | 0.556 | 0.310 |
| walnuts | 0.760 | 0.814 | 0.902 | 0.471 | 0.527 | 0.336 |

---
*Reproducibility: aggregated results in `gsoc_workspace/experiments/results/results.csv`
(rows with `phase` in `{phase8_alpha, phase8_poisson, phase9_crossmodel,
phase10_superadd_perlin}`, plus `phase5` for the Perlin-pipeline comparison in
§3.2/§4/Appendix E); harness in `gsoc_workspace/experiments/harness.py`
(`_pregenerated_eval_set`, `_synthetic_eval_set`, `PREGENERATED_PIPELINES`,
`SUPERADD_BACKBONE`, `PREGENERATED_CATEGORIES`); pipeline definitions in
`src/anomalib/data/utils/generators/` (`synthetic_anomaly.py`, `region.py`,
`source.py`, `blend.py`, `mask.py`); phase specs in
`gsoc_workspace/experiments/sweep.py` (`phase8_alpha`, `phase8_poisson`,
`phase9_crossmodel`, `phase10_superadd_perlin`).*
