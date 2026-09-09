# Phase 12 — SuperADD Backbone Ablation (First Ablation Study)

**Status:** ✅ **completed** (2026-08-27) — all 72 jobs done (local + remote), results
aggregated into §8.

- **Local** (2× RTX 3090): `can`, `vial` — 18 jobs, GPUs 0–1, `--procs-per-gpu 2`.
  Log: `/tmp/phase12_local.log`.
- **Remote** (Cluster2): `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `wallplugs`,
  `walnuts` — 54 jobs, GPUs 0–5, `--procs-per-gpu 2`. Log: `/tmp/phase12_remote.log`.
- 3 smaller backbone dirs `scp`-ed to remote cache before launch (verified present).
- Code plumbing (`backbone` through `JobConfig`/`build_model`/`run_job`/`run_one.py`/
  `sweep.py` + `phase12_superadd_backbone_ablation` spec) implemented and synced to
  remote; enumeration verified at 72 jobs on both machines; CPU smoke test confirmed
  `forward_intermediates` (NLC) returns 4 layers with dims 384/768/1024 for
  small/base/large.

## 1. Goal

Ablate SuperADD's **backbone size** while holding the entire phase-11 protocol fixed, to
answer: *does the hybrid (P6) over Perlin (P0) threshold-calibration advantage scale
with backbone capacity, or is it size-agnostic?* Combined with phase 11's `huge_plus`,
this becomes a clean 4-point size scaling.

## 2. Backbones

| backbone (timm name) | params | status |
|---|---|---|
| `vit_small_patch16_dinov3` | ~22M | **new** |
| `vit_base_patch16_dinov3` | ~86M | **new** |
| `vit_large_patch16_dinov3` | ~300M | **new** |
| `vit_huge_plus_patch16_dinov3` | ~840M | ✅ done (phase 11) |

All four are `patch16` DINOv3 and all four are **already cached locally**
(`~/.cache/huggingface/hub/models--timm--vit_*_patch16_dinov3.lvd1689m`). SuperADD reads
feature dims dynamically from timm (`forward_intermediates`, `NLC`), so no model code
change is required — only backbone-name plumbing through the harness.

## 3. Design (identical to phase 11 except backbone)

- **Models:** `superadd` only.
- **3 cases** (per backbone, same as phase 11):
  1. **Arm A** — real MVTec AD 2 anomalies (oracle threshold).
  2. **Arm B / P0** — anomalib's stock, unmodified `PerlinAnomalyGenerator`.
  3. **Arm B / P6** — our semantic-defect-bank hybrid (poisson+alpha).
- Arm C (synthetic-on-synthetic diagnostic) also emitted for free, as in phase 11, so
  every backbone's rows are directly comparable to phase 11's `huge_plus` rows.
- **Categories:** all 8 MVTec AD 2. **Seeds:** 1, 2, 3. **Calibration:** `heldout`.
  **Pipelines:** `P0`, `P6`.

## 4. Job count & timing

- 3 backbones × 8 categories × 3 seeds = **72 jobs** → 72 × 5 = **360 rows**.
- Smaller backbones are ~3–10× faster than `huge_plus` (which was 13–22 min/job):
  est. ~2–4 min (small), ~3–5 min (base), ~6–10 min (large). Total wall-clock ≈
  45–90 min across 8 GPUs.

## 5. Machine allocation & weight sync

Remote only has `huge_plus` cached, so the 3 smaller cached model dirs must be `scp`-ed
to `/home/rajesh/.cache/huggingface/hub/` **before** launching remote jobs (avoids the
phase-10 "download-while-running made the box unreachable" issue).

Partition by category (reuse `--categories`; 9 jobs/GPU):

| machine | GPUs | categories | jobs |
|---|---|---|---|
| local | 2 | `can`, `vial` | 18 |
| remote (Cluster2) | 6 (GPUs 0–5) | `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `wallplugs`, `walnuts` | 54 |

## 6. Code changes needed (prep only)

- `harness.py`: add `backbone: str | None = None` to `JobConfig` (+ include in `.key` so
  different backbones are distinct jobs); add `backbone` param to `build_model` and
  thread `job.backbone` through `run_job`.
- `run_one.py`: add `--backbone` argument, pass to `JobConfig`.
- `sweep.py`: add `backbones` to the phase spec, iterate it in `enumerate_jobs`, add
  `--backbone` to the subprocess `cmd` in `run_job`, and (optionally) a `--backbones`
  CLI filter for symmetry with `--models`.
- Add `phase12_superadd_backbone_ablation` phase spec.

## 7. Analysis

- Per-backbone table: arm A vs B/P0 vs B/P6 for image-F1, pixel-F1, plus threshold-free
  AUROC/AUPRO as a leakage sanity check.
- Headline: **P6 − P0 pixel-F1 gain vs backbone capacity** (does the gain scale with
  size?), and P6's gap-recovery toward the oracle (arm A) as a function of capacity.
- Compare with phase 11's `huge_plus` to complete the 4-point curve.

## 8. Results (completed)

Means over 8 categories × 3 seeds, expressed as percentages (0–100). Rows are the four
DINOv3 backbones (ascending capacity); columns are the four anomaly sources — **Real**
(oracle, arm A), **Perlin** (stock, arm B/P0), **AnoStyler** (arm B/P8), and **Flash
(ours)** (the semantic-defect-bank hybrid, arm B/P6). **AnoStyler was only run on
`huge_plus`**, so its column is `—` for the small/base/large backbones.

### Pixel F1 (SegF1, %)

| backbone | Real | Perlin | AnoStyler | Flash (ours) |
|---|---|---|---|---|
| `vit_small_patch16_dinov3` (22M) | 30.06 | 17.78 | — | 23.70 |
| `vit_base_patch16_dinov3` (86M) | 36.44 | 18.43 | — | 26.94 |
| `vit_large_patch16_dinov3` (300M) | 49.21 | 19.98 | — | 39.75 |
| `vit_huge_plus_patch16_dinov3` (840M) | 51.53 | 19.44 | 36.99 | 38.38 |

### Image F1 (%)

| backbone | Real | Perlin | AnoStyler | Flash (ours) |
|---|---|---|---|---|
| `vit_small_patch16_dinov3` (22M) | 82.27 | 64.20 | — | 76.72 |
| `vit_base_patch16_dinov3` (86M) | 81.50 | 62.56 | — | 75.91 |
| `vit_large_patch16_dinov3` (300M) | 82.78 | 75.62 | — | 76.57 |
| `vit_huge_plus_patch16_dinov3` (840M) | 83.64 | 79.12 | 64.36 | 78.13 |

### Key findings

1. **Oracle image-F1 is size-agnostic** (81.50→83.64): image-level anomaly detection is
   saturated even at 22M. Oracle **pixel-F1 scales with capacity** (30.06→36.44→49.21→
   51.53): finer localization needs a bigger backbone.
2. **Flash (ours) over Perlin advantage is *pixel-level* and *scales with capacity***: the
   Flash−Perlin pixel-F1 gap rises 5.92→8.51→19.77 pp, then 18.94 pp for `huge_plus`.
3. **At image level the Flash−Perlin advantage *vanishes* for big backbones** (+12.52→
   +13.35→+0.95→−0.99 pp): once the backbone is large enough, Perlin's image-level signal
   is already sufficient, so the hybrid's realistic-defect benefit is confined to pixel
   localization.
4. Threshold-free pixel AUROC on the oracle is flat (~0.906–0.918), so the pixel-F1 scaling
   is driven by threshold calibration/localization, not raw discriminative power — consistent
   with SuperADD's oracle being bottlenecked by the 448 px downscale of MVTec AD 2's fine defects.
