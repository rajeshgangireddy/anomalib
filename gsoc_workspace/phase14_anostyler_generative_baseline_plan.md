# Phase 14 — AnoStyler (AAAI 2026) Generative Baseline

**Status:** ✅ complete — 96/96 jobs done (24 local + 72 remote), 480 rows in
`results.csv`. Results aggregated in §Results below.

## Why we switched from AnomalyAny (P7) to AnoStyler (P8)

AnomalyAny (CVPR 2025) is a diffusion-based method: ~115 s/image, image-only (no defect
mask), torch 1.13.1. AnoStyler (AAAI 2026, MIT) is a **lightweight per-image style
transfer**: a tiny U-Net (`ngf=16`) optimized for 75 epochs × 64 crops under VGG
perceptual + CLIP directional losses, masked to a procedurally-generated (SAM-foreground
∩ meta-shape) anomaly region.

| | AnomalyAny (P7) | AnoStyler (P8) |
|---|---|---|
| time/image | ~115 s | **~15 s** (~8×) |
| 705 images (2 GPUs) | ~11 h | **~1.5 h** |
| emits mask | no | **yes** |
| stack | torch 1.13.1 | torch 2.5.1 |

AnoStyler's **mask output** removes the limitation we had to cite for P7 (image-only →
classification fallback). P8 is full segmentation, so arm B fits both image and pixel
thresholds from the synthetic set, and arm C reports pixel metrics on the synthetic set.

## Setup (done)

- Cloned `yulimso/AnoStyler`, uv venv on **Python 3.12** (repo's "3.10 + numpy 2.3.2" pin
  is contradictory; 2.3.2 needs ≥3.11). torch 2.5.1 + torchvision + CLIP (git) +
  segment-anything + SAM checkpoint (`sam_vit_b_01ec64.pth`, 375 MB).
- Fixed two bugs in `src/def_train.py`: CLIP re-loaded per image; `img_normalize`/
  `clip_normalize` hardcoded `cuda:0` (crashed on `--gpu 1`).
- Fixed a third bug in `src/utils.py:load_image2`: it did not convert to RGB, so
  grayscale MVTec AD 2 categories (`vial`, `wallplugs`, `sheet_metal` are `L`-mode)
  crashed `StyleNet`'s `conv_init(3→16)` with "expected 3 channels, got 1". Added
  `.convert("RGB")` and verified all grayscale categories load as 3-channel.
- Wrote `generate_mvtec2_anostyler.py` (resumable batch, image+mask, harness naming).
- Harness: `PREGENERATED_PIPELINES["P8"]="anostyler"`; `phase14_anostyler` spec
  (superadd huge_plus / padim / patchcore / anomaly_dino, `P0`+`P8`, 3 seeds, heldout).

## Recipe

- Defects per category (labels only — AnoStyler prompts are category-level
  "damaged {category}", the defect string names the folder): can/vial/wallplugs/walnuts
  = crack, fabric = hole, fruit_jelly = crack, rice = broken, sheet_metal = scratch.
- Output 256×256 (image + mask); the harness resizes both to 448×448 uniformly.
- Settings: `max_step=75`, `num_crops=64`, `crop_size=128`, `fg_mask=true` (SAM),
  `m_max=5`, `alpha=0.7`, losses `lambda_tv=0.002 / pdir=9000 / gdir=500 / c=150 /
  mclip=1000` (repo defaults, config.yaml).

## Speed levers (if needed)

`max_step`, `num_crops`, `crop_size` are all configurable in `generate_mvtec2_anostyler.py`
(`--steps`, `--crops`), so the already-fast ~15 s/image can be pushed further by trading
optimization epochs.

## Combined-machine execution (local + remote)

Once generation completes, the sweep runs on **local 2 GPUs + remote 6 GPUs** to finish
~4× faster than local alone.

- **Orchestration:** `gsoc_workspace/experiments/run_phase14_combined.sh` (running
  detached). Flow: wait for 4230 files → `scp` the 3 `MVTec_AD_2_anostyler_*` dirs to
  remote → launch local sweep + remote sweep in parallel → wait for both → `scp` remote
  JSONs back → `--aggregate-only`.
- **Partition** (mirrors phase 12): local = `{can, vial}` (24 jobs), remote =
  `{fabric, fruit_jelly, rice, sheet_metal, wallplugs, walnuts}` (72 jobs).
- **Code sync:** harness.py + sweep.py (P8 registry, `_reset_post_processor` leak fix,
  maskless fallback, backbone column) already `scp`-ed to remote; remote enumeration
  verified (72 jobs, correct models/seeds).
- **Remote detach:** the remote sweep is launched via
  `ssh 'nohup sh -c "...sweep..." >/dev/null 2>&1 </dev/null &' </dev/null >/dev/null 2>&1`
  (ssh must redirect its own streams or it blocks until the remote job finishes); the
  completion poll uses the `pgrep -f '[p]hase14_anostyler'` bracket trick to avoid
  self-matching the poll's own ssh-spawned shell.
- **Remote env:** `/srv/research/experiments/anomalib_synthetic_anomaly/code/.venv/bin/python`
  (anomalib 2.5.2-dev), GPUs 0–5. Local: repo `.venv` (Python 3.13, anomalib 2.5.2-dev),
  GPUs 0–1.
- **CUDA OOM fix:** the first launch at `--procs-per-gpu 2` OOM'd (patchcore/anomaly_dino
  coreset spikes ~7 GiB each, two jobs/GPU exceed 24 GiB). Restarted both sweeps at
  `--procs-per-gpu 1` (1 job/GPU) — 0 OOMs thereafter; ~2× slower but reliable.

## Generation parallelization (2 procs/GPU)

AnoStyler is memory-light (~5.6 GiB/process: SAM 0.75 + VGG19 0.55 + CLIP 0.35 GiB +
U-Net + activations), so we run **2 processes per GPU** (4 total) to fill idle GPU
cycles (util 50% → ~90%). Measured ~1.4× throughput (11 vs 8 image/min).

- Split (disjoint, resume-safe; `can`/`fruit_jelly` already done):
  - GPU 0 → `fabric rice` (P1), `sheet_metal` (P2)
  - GPU 1 → `vial walnuts` (P3), `wallplugs` (P4)
- **Bug fixed:** `init_sam_model` hardcodes `sam.to("cuda")` (GPU 0), so a `--gpu 1`
  process put its SAM on GPU 0 (3.4 GiB waste + cross-GPU compute). `generate_mvtec2_
  anostyler.py` now calls `predictor.model.to(device)` after init. `SamPredictor.device`
  is a read-only property (`return self.model.device`), so the model must be moved, not
  the attribute. A ~658 MiB CUDA-context residual remains on GPU 0 (the initial
  `sam.to("cuda")` load) — negligible, but a clean `device` param on `init_sam_model`
  would remove it for future runs.

## Results

Means over 8 categories × 3 seeds (24 jobs/model). Arm A = oracle (real-test threshold);
arm B = synthetic-calibrated threshold transferred to the real test; arm C =
synthetic-on-synthetic diagnostic. `image_F1Score`/`pixel_F1Score` are threshold-based
(headline); `*_AUROC`/`*_AUPR`/`*_AUPRO` are threshold-free (leakage sanity). Note
`pixel_AUROC` is identical across arms A/B for a model because arm B evaluates the same
anomaly maps on the real test — only the threshold differs.

### superadd (huge_plus)

| arm | pipe | imgAUROC | imgF1 | imgAUPR | imgPrec | imgRec | pxAUROC | pxF1 | pxAUPR | pxAUPRO | pxAUPRO@.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | - | 0.745 | 0.836 | 0.852 | 0.776 | 0.934 | 0.909 | 0.515 | 0.467 | 0.507 | 0.293 |
| B | P0 | 0.741 | 0.791 | 0.839 | 0.726 | 0.916 | 0.909 | 0.191 | 0.468 | 0.507 | 0.293 |
| B | P8 | 0.742 | 0.644 | 0.836 | 0.848 | 0.584 | 0.909 | 0.370 | 0.467 | 0.507 | 0.293 |
| C | P0 | 0.696 | 0.880 | 0.888 | 0.817 | 0.972 | 0.776 | 0.406 | 0.371 | 0.383 | 0.203 |
| C | P8 | 0.978 | 0.963 | 0.990 | 0.982 | 0.946 | 0.993 | 0.733 | 0.776 | 0.910 | 0.730 |

### padim

| arm | pipe | imgAUROC | imgF1 | imgAUPR | imgPrec | imgRec | pxAUROC | pxF1 | pxAUPR | pxAUPRO | pxAUPRO@.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | - | 0.619 | 0.805 | 0.749 | 0.700 | 0.963 | 0.812 | 0.077 | 0.035 | 0.425 | 0.097 |
| B | P0 | 0.619 | 0.677 | 0.752 | 0.702 | 0.775 | 0.812 | 0.030 | 0.035 | 0.425 | 0.097 |
| B | P8 | 0.618 | 0.624 | 0.739 | 0.681 | 0.676 | 0.812 | 0.054 | 0.034 | 0.425 | 0.097 |
| C | P0 | 0.485 | 0.871 | 0.747 | 0.808 | 0.964 | 0.729 | 0.237 | 0.175 | 0.406 | 0.190 |
| C | P8 | 0.839 | 0.892 | 0.898 | 0.848 | 0.953 | 0.973 | 0.463 | 0.394 | 0.851 | 0.594 |

### patchcore

| arm | pipe | imgAUROC | imgF1 | imgAUPR | imgPrec | imgRec | pxAUROC | pxF1 | pxAUPR | pxAUPRO | pxAUPRO@.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | - | 0.725 | 0.825 | 0.827 | 0.735 | 0.959 | 0.873 | 0.262 | 0.206 | 0.540 | 0.286 |
| B | P0 | 0.725 | 0.669 | 0.819 | 0.756 | 0.711 | 0.873 | 0.151 | 0.206 | 0.540 | 0.286 |
| B | P8 | 0.725 | 0.430 | 0.818 | 0.828 | 0.339 | 0.873 | 0.161 | 0.206 | 0.540 | 0.286 |
| C | P0 | 0.649 | 0.890 | 0.865 | 0.859 | 0.944 | 0.741 | 0.336 | 0.292 | 0.387 | 0.203 |
| C | P8 | 0.964 | 0.958 | 0.986 | 0.981 | 0.937 | 0.988 | 0.656 | 0.669 | 0.863 | 0.650 |

### anomaly_dino

| arm | pipe | imgAUROC | imgF1 | imgAUPR | imgPrec | imgRec | pxAUROC | pxF1 | pxAUPR | pxAUPRO | pxAUPRO@.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | - | 0.699 | 0.815 | 0.814 | 0.730 | 0.949 | 0.872 | 0.335 | 0.274 | 0.515 | 0.304 |
| B | P0 | 0.701 | 0.634 | 0.808 | 0.656 | 0.668 | 0.874 | 0.173 | 0.272 | 0.512 | 0.302 |
| B | P8 | 0.701 | 0.499 | 0.808 | 0.827 | 0.419 | 0.873 | 0.140 | 0.271 | 0.509 | 0.302 |
| C | P0 | 0.794 | 0.918 | 0.929 | 0.895 | 0.958 | 0.780 | 0.423 | 0.345 | 0.508 | 0.311 |
| C | P8 | 0.988 | 0.977 | 0.994 | 0.989 | 0.966 | 0.991 | 0.617 | 0.583 | 0.939 | 0.774 |

### Headline: AnoStyler (P8) vs stock Perlin (P0), arm B

| model | P0 imgF1 | P8 imgF1 | **P8−P0 imgF1** | P0 pxF1 | P8 pxF1 | **P8−P0 pxF1** |
|---|---|---|---|---|---|---|
| superadd | 0.791 | 0.644 | **−0.148** | 0.191 | 0.370 | **+0.179** |
| padim | 0.677 | 0.624 | −0.053 | 0.030 | 0.054 | +0.024 |
| patchcore | 0.669 | 0.430 | −0.239 | 0.151 | 0.161 | +0.010 |
| anomaly_dino | 0.634 | 0.499 | −0.135 | 0.173 | 0.140 | −0.033 |

**Interpretation.** AnoStyler's threshold **beats Perlin on pixel localization** (biggest
win +0.179 pxF1 on superadd, where localization matters most) but **loses on image-level
classification for every model** (its generated anomalies produce an image-score
distribution whose optimum under-flags real images: recall collapses, e.g. superadd
imgRec 0.916→0.584, patchcore 0.711→0.339). This is the opposite trade-off to our
hybrid (P6), which won image-level for small backbones and pixel-level for large
backbones. Arm C confirms the detectors fit AnoStyler's own images easily (superadd
C/P8 pxF1 0.733 vs C/P0 0.406), so the gap is threshold *transfer*, not detection
capacity — a clean three-way story: copy-paste (P0) vs generation (P8) vs our
semantic-bank hybrid (P6).
