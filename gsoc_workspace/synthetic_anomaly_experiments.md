# Synthetic Anomaly Experiments — Checklist

Goal: finalize 2–3 synthetic-anomaly generation pipelines and benchmark anomalib
models trained/validated on **real** vs **synthetic** anomalies. Best pipeline =
the one whose synthetic-validated performance is closest to real-validated
performance on the untouched real test set.

## Dimensions

- **Models (val-only):** PaDiM, PatchCore, EfficientAD, AnomalyDINO, Dinomaly — synthetic anomalies used only for threshold / model selection.
- **Models (train):** DRAEM — synthetic anomalies used as training signal.
- **Datasets:** MVTec AD (primary), VisA (expansion).
- **Pipelines (mask × region × source × blend):**
  - `P1 texture_alpha` — perlin · whole · DTD texture · alpha (DRAEM/OOD floor)
  - `P2 self_alpha` — perlin · foreground · self · alpha (subtle, object-aware)
  - `P3 self_poisson` — perlin · foreground · self · poisson (realistic, seamless)
  - `P4 cross/CutPaste` *(optional)* — perlin · foreground · cross-image · alpha/poisson
- **Arms:** A = real-val reference (upper bound), B = synthetic-val on real test (the number that matters), C = synthetic-val on synthetic test (optimistic diagnostic).
- **Metrics:** image AUROC (quality), image F1/precision/recall @ transferred threshold, pixel AUPRO + pixel F1, calibration gap = metric(A) − metric(B).

## Results So Far (as of 2026-08-05)

**Phases 0–4 (MVTec AD 1 + VisA — complete, 3415 rows).** Full write-up in
`synthetic_anomaly_threshold_transfer_report.md`.
- Self-supervised, seamlessly-blended synthetic anomalies (**P3 `self_poisson`**) recover
  **~91%** of oracle F1 on average (P1 texture-paste only ~62%).
- Model ranking is preserved almost perfectly (Spearman ρ = 1.00 MVTec, 0.90 VisA) —
  synthetic anomalies are already reliable for label-free model *selection*, even where the
  absolute threshold *value* transfers weakly (r = 0.16–0.48).
- Introduced a novel area-routed hybrid blend (P4): beats alpha blending (Wilcoxon
  p = 0.002) but ties with pure Poisson (p = 0.31); a routing-threshold ablation (280 rows)
  confirms this is fundamental, not a tuning artifact.
- **Conclusion: P3 (`self_poisson`) is the recommended label-free calibrator** on these two
  (saturated) benchmarks.

**Phase 5 (MVTec AD 2, 448 px — 693 rows, 99/144 jobs).**
- Moved to MVTec AD 2 because AD1/VisA are near-ceiling and don't stress-test the method;
  AD2 is genuinely hard with a large published gap to close.
- Oracle image-AUROC at 448 px: patchcore 0.727, dinomaly 0.681, anomaly_dino 0.672,
  padim 0.628, efficient_ad 0.621 — all well below published SOTA, confirming AD2 needs
  native-resolution handling (its native images are 2.3–5.0 MP vs. our 448×448 input).
- Fixed DRAEM/anomaly_dino OOMs (batch-size reduction, coreset subsampling) that had
  dropped those two models from most/every category.

**Phase 6 (MVTec AD 2, native-resolution tiled inference — running now).**
- Built a custom tiled train/eval harness (`tiled_harness.py`): trains with random-crop
  augmentation, scores full-resolution images by tiling through the 448 px model and
  re-stitching — closing the resolution gap without modifying any model internals.
- Found and fixed **two measurement bugs** during shakeout (both are genuine, paper-worthy
  findings, not just plumbing issues):
  1. **Tile-border artifact** — a convolutional edge effect at the true image border
     dominated every image-level max-score, collapsing AUROC to exactly 0.500 on every
     job. Fixed by cropping a 32 px margin before scoring.
  2. **Sigmoid-saturation bug** (found because AUROC was *still* 0.500 after fix #1) —
     torchmetrics silently applies `sigmoid()` to any prediction tensor outside `[0, 1]`,
     assuming it's a logit. Our raw, unbounded tile-stitched scores (range ~30–120)
     saturate completely (`sigmoid(30) ≈ 1.0`), collapsing every score to one value.
     Anomalib's standard pipeline avoids this because `OneClassPostProcessor` always
     min-max normalizes scores before metrics — our tiled harness bypassed that
     post-processor entirely. Fixed by mirroring the same normalization; verified against
     sklearn ground truth (0.616 both ways) on a real completed job.
- Sweep relaunched clean with both fixes: 48 jobs (6 models × 8 categories, 1 seed),
  in progress.

## Current Experiment
Phase 6 tiled sweep, relaunched with both fixes above. Tiered by speed: padim first, then
dinomaly/anomaly_dino/draem, then efficient_ad/patchcore. ETA is multi-day (native-resolution
tiled inference is ~1 hr/job for the fastest model).

## Planned Next
1. Finish phase 6; compare tiled vs. 448 px vs. published SOTA (SuperADD 57.4%, RoBiS 51.0%
   SegF1) to see whether native resolution closes the gap.
2. Re-run the difficulty-mechanism / score-coverage analyses on AD2 tiled data.
3. Write up native-resolution findings for the paper regardless of outcome (the two bugs
   above are themselves useful negative-space results for anyone else building tiled
   inference on top of anomalib).
4. Consider upstreaming the AD2 test-split fix (already applied in this checkout) and
   filing the AUPRO native-resolution memory crash as anomalib issues.


## Phase 0 — Harness / tracer bullet
Prove the A/B/C plumbing end-to-end before scaling.

- [x] MVTec (3 cats) × {PatchCore, DRAEM} × {P1, P3} × {A, B, C} × 1 seed (~30 runs)
- [x] Confirm A > B > chance, masks/metrics sane, DRAEM trains on our generator.

## Phase 1 — Pipeline shortlisting (MVTec, val-only)
Pick the 2–3 pipelines whose B is closest to A.

- [x] MVTec (15 cats) × {PatchCore, PaDiM, EfficientAD} × {P1, P2, P3, P4} × {A, B} × 3 seeds (~585 runs)
- [x] Select shortlist of 2–3 pipelines for Phase 2.

## Phase 2 — Grand comparison (val-only regime)
All val-only models × shortlisted pipelines, MVTec then VisA.

- [x] MVTec (15) × {PaDiM, PatchCore, EfficientAD, AnomalyDINO, Dinomaly} × {A + 3 shortlisted} × 3 seeds (~750 runs)
- [x] VisA (12) × same models × {A + 3 shortlisted} × 3 seeds (~600 runs)
- [x] Report per-model B-vs-A calibration gap.

## Phase 3 — DRAEM synth-as-training regime
Here the pipeline is training data — the biggest lever. Report separately.

- [x] MVTec (15) + VisA (12) × DRAEM × {P1 baseline + shortlisted} × {train, train+val} × 3 seeds (~324 runs)

## Phase 4 — Optional expansion
Only if earlier phases warrant it.

- [ ] Extra models: GLASS / SimpleNet / CutPaste (train regime)
- [x] Extra datasets: **MVTec AD 2** (see Phase 5/6 below); MPDD/BTAD not pursued
- [ ] Axis ablations (region / blend / source) isolated on 1 model

## Phase 5 — MVTec AD 2, 448 px (sweep.py `phase5`)
Moved to AD2 because AD1/VisA are saturated; AD2 has a large gap to published SOTA.

- [x] MVTec AD 2 (8 cats) × {padim, patchcore, efficient_ad, anomaly_dino, dinomaly} ×
      {A, B×P1/P2/P3, C} × 3 seeds, heldout calibration (693/~864 rows so far)
- [x] Fixed DRAEM/anomaly_dino OOMs (batch size, coreset subsampling)

## Phase 6 — MVTec AD 2, native-resolution tiled inference (sweep.py `phase6`, running)
Closes the 448 px → native-resolution (2.3–5.0 MP) gap without changing model internals.

- [x] Built `tiled_harness.py` (random-crop training + Tiler-based tiled inference)
- [x] Found + fixed tile-border scoring artifact (32 px margin crop)
- [x] Found + fixed torchmetrics sigmoid-saturation bug (min-max normalize before AUROC/AUPR)
- [ ] 48 jobs (6 models × 8 categories × 1 seed) — in progress
- [ ] Compare tiled vs. 448 px vs. published SOTA (SuperADD 57.4%, RoBiS 51.0% SegF1)

## Notes

- Arm A computed once per (model, dataset, category) and reused across pipelines.
- AUROC won't rank pipelines for val-only models (threshold-free) — select on thresholded + pixel + calibration gap.
- Confirm AnomalyDINO / Dinomaly availability in the installed anomalib version before Phase 2.
