# SuperADD on MVTec AD 2 (huge_plus backbone, 8 categories) with Semantic-Defect-Bank Synthetic Calibration

*Experiments run on anomalib (branch merged with `feature/v2.6.0`, PR #3628 SuperADD), 2× RTX 3090. Phases: `phase8_alpha`, `phase8_poisson`.*

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

---
*Reproducibility: aggregated results in `gsoc_workspace/experiments/results/results.csv`
(rows with `phase` in `{phase8_alpha, phase8_poisson}`); harness in
`gsoc_workspace/experiments/harness.py` (`_pregenerated_eval_set`, `PREGENERATED_PIPELINES`,
`SUPERADD_BACKBONE`, `PREGENERATED_CATEGORIES`); phase specs in
`gsoc_workspace/experiments/sweep.py`.*
