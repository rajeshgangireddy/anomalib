# SuperADD on MVTec AD 2 with Semantic-Defect-Bank Synthetic Calibration

*Experiments run on anomalib (branch merged with `feature/v2.6.0`, PR #3628 SuperADD), 1× RTX 3090. Phases: `phase7_alpha`, `phase7_poisson`.*

## 1. Methodology

### 1.1 Model

**SuperADD** (anomalib-native implementation, VAND 4.0 winning method): a training-free,
memory-bank anomaly detector built on a frozen **DINOv3** Vision Transformer backbone.
Multi-layer token features are extracted over overlapping image patches, a per-layer
memory bank is built from normal training images via distance-based coreset
subsampling, and anomalies are scored by nearest-neighbor distance to that bank.

| Setting | Value |
|---|---|
| Backbone | `vit_large_patch16_dinov3` (303M params) — a lighter pilot vs. the paper's default `vit_huge_plus_patch16_dinov3` (840M) |
| Input resolution | 448×448 (resize, no center-crop) |
| Post-processor | Standard anomalib `PostProcessor` (F1-adaptive threshold), overriding SuperADD's own built-in percentile-based threshold, for arm parity with other models in the wider study |
| Training | Single pass: coreset-subsampled memory-bank construction (no gradient descent) |

### 1.2 Dataset

**MVTec AD 2**, restricted to the 4 categories with donor-defect-bank coverage:
`rice`, `walnuts`, `wallplugs`, `fruit_jelly`. Train/val use the dataset's native
`train/good` and `validation/good` splits; test uses `test_public` (`good` + `bad` +
pixel masks).

| category | n_train | n_val (heldout) | n_test |
|---|---|---|---|
| rice | 313 | 132 | 132 |
| walnuts | 432 | 150 | 150 |
| wallplugs | 293 | 150 | 150 |
| fruit_jelly | 263 | 80 | 80 |

### 1.3 Synthetic-anomaly calibration

Synthetic anomalies are **pre-generated** (not produced live during the sweep) by a
"semantic defect bank" pipeline (`semantic_bank_blend.ipynb`): real defect patches are
cut from a small donor bank of previously-generated anomalies, then replayed onto
held-out `train/good` host images (donors excluded) at MRSP/OBS-chosen sites, harmonized
to the host's local illumination, and composited with one of two blend operators:

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

4 categories × 3 seeds × 2 pipelines (P5, P6) = **24 jobs**, run sequentially in two
waves (all P5/alpha jobs first, then all P6/poisson jobs), each producing arm A + B + C.

## 2. Results

### 2.1 Oracle (Arm A) — upper bound, mean ± std over 3 seeds

| category | image_AUROC | image_F1 | image_AUPR | image_Precision | image_Recall | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPR | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|---|---|---|---|
| rice | 0.832 ± 0.011 | 0.860 ± 0.014 | 0.922 ± 0.004 | 0.795 ± 0.015 | 0.937 ± 0.039 | 0.969 ± 0.003 | **0.603 ± 0.001** | 0.563 ± 0.001 | 0.719 ± 0.019 | 0.539 ± 0.005 |
| walnuts | 0.868 ± 0.007 | 0.823 ± 0.006 | 0.927 ± 0.003 | 0.909 ± 0.105 | 0.769 ± 0.095 | 0.976 ± 0.002 | **0.724 ± 0.001** | 0.753 ± 0.001 | 0.584 ± 0.012 | 0.316 ± 0.004 |
| wallplugs | 0.507 ± 0.008 | 0.753 ± 0.003 | 0.638 ± 0.005 | 0.610 ± 0.002 | 0.983 ± 0.014 | 0.913 ± 0.005 | **0.415 ± 0.004** | 0.332 ± 0.004 | 0.448 ± 0.022 | 0.202 ± 0.004 |
| fruit_jelly | 0.820 ± 0.010 | 0.855 ± 0.006 | 0.945 ± 0.003 | 0.787 ± 0.068 | 0.947 ± 0.073 | 0.911 ± 0.002 | **0.524 ± 0.004** | 0.514 ± 0.006 | 0.538 ± 0.006 | 0.156 ± 0.002 |

### 2.2 Arm B (synthetic-calibrated, on real test) — mean over 3 seeds

**Pipeline P5 (alpha)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| rice | 0.830 | 0.811 | 0.971 | 0.565 | 0.733 | 0.543 |
| walnuts | 0.869 | 0.781 | 0.977 | 0.609 | 0.590 | 0.316 |
| wallplugs | 0.494 | 0.750 | 0.913 | 0.082 | 0.437 | 0.201 |
| fruit_jelly | 0.573 | 0.857 | 0.911 | 0.484 | 0.539 | 0.158 |

**Pipeline P6 (poisson)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 (SegF1) | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| rice | 0.834 | 0.811 | 0.968 | 0.568 | 0.704 | 0.536 |
| walnuts | 0.867 | 0.789 | 0.976 | 0.666 | 0.578 | 0.316 |
| wallplugs | 0.509 | 0.750 | 0.913 | 0.201 | 0.459 | 0.203 |
| fruit_jelly | 0.635 | 0.857 | 0.914 | 0.505 | 0.537 | 0.155 |

### 2.3 Arm C (synthetic-calibrated, on synthetic test) — mean over 3 seeds

**Pipeline P5 (alpha)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| rice | 0.709 | 0.832 | 0.971 | 0.723 | 0.753 | 0.677 |
| walnuts | 0.961 | 0.952 | 0.990 | 0.600 | 0.895 | 0.753 |
| wallplugs | 0.607 | 0.840 | 0.975 | 0.538 | 0.773 | 0.627 |
| fruit_jelly | 0.389 | 0.756 | 0.982 | 0.140 | 0.926 | 0.674 |

**Pipeline P6 (poisson)**

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 | pixel_AUPRO | pixel₀.₀₅_AUPRO |
|---|---|---|---|---|---|---|
| rice | 0.653 | 0.832 | 0.967 | 0.691 | 0.702 | 0.621 |
| walnuts | 0.947 | 0.952 | 0.992 | 0.566 | 0.896 | 0.791 |
| wallplugs | 0.769 | 0.840 | 0.992 | 0.453 | 0.940 | 0.836 |
| fruit_jelly | 0.404 | 0.756 | 0.981 | 0.130 | 0.915 | 0.668 |

### 2.4 Calibration gap (Arm A − Arm B), pixel_F1 (SegF1)

| category | P5 (alpha) gap | P6 (poisson) gap |
|---|---|---|
| rice | 0.038 | 0.035 |
| walnuts | 0.115 | 0.058 |
| wallplugs | 0.333 | 0.214 |
| fruit_jelly | 0.040 | 0.020 |

---
*Reproducibility: aggregated results in `gsoc_workspace/experiments/results/results.csv`
(rows with `phase` in `{phase7_alpha, phase7_poisson}`); harness in
`gsoc_workspace/experiments/harness.py` (`_pregenerated_eval_set`, `PREGENERATED_PIPELINES`);
phase specs in `gsoc_workspace/experiments/sweep.py`.*
