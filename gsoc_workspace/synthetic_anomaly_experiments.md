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

**Phase 6 (MVTec AD 2, native-resolution tiled inference — complete).**
- Built a custom tiled train/eval harness (`tiled_harness.py`): trains with random-crop
  augmentation, scores full-resolution images by tiling through the 448 px model and
  re-stitching — closing the resolution gap without modifying any model internals.
- Found and fixed **four measurement bugs** during shakeout (all genuine, paper-worthy
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
  3. **Dinomaly's published 392px crop breaks tile stitching** — every tile came back a
     different size than the tiler expected, crashing 100% of dinomaly's jobs. Fixed by
     disabling the crop specifically under tiled inference.
  4. **Pixel-threshold bin-range mismatch** (the most consequential) — the binned pixel
     F1AdaptiveThreshold silently defaulted to a `[0,1]` grid regardless of our raw
     unbounded scores, collapsing the fitted threshold to "classify almost everything
     positive." This corrupted pixel-F1 in an entire earlier 38/48-job partial run, which
     was discarded in full and re-run clean after the fix.
- Sweep completed: 48/48 jobs (6 models × 8 categories, 1 seed) — split across this
  machine (padim, 8 jobs) and an 8-GPU remote machine (dinomaly/anomaly_dino/draem/
  efficient_ad/patchcore, 40 jobs); results merged back and re-aggregated. 0 errors,
  0 rows with the 0.500-AUROC artifact.
- **Result: tiling does not close the gap to published SOTA and is a net regression on
  average** (mean SegF1 448px→tiled: 0.208→0.186, Δ −0.022; only `anomaly_dino` improved,
  on an unreliable n=3 448px baseline — every fully-populated model (dinomaly,
  efficient_ad, padim, patchcore) is flat or worse, patchcore worst at Δ −0.072). Full
  breakdown in `synthetic_anomaly_threshold_transfer_report.md` §6.2.

**Phase 7 (SuperADD on MVTec AD 2 — complete).**
- Merged anomalib's native SuperADD (PR #3628, `feature/v2.6.0`) into our branch; verified
  via a 4-backbone smoke test on MVTec AD toothbrush (small/base/large/huge_plus, all
  clean, all weights now cached).
- Built a new pre-generated synthetic-anomaly loader (`_pregenerated_eval_set`) reading
  the "semantic defect bank" pipeline's output (`SynthetciGenMVAD2/`, from
  `semantic_bank_blend.ipynb`) — real defect patches cut from a donor bank and replayed
  onto held-out hosts, alpha (P5) vs. Poisson (P6) blend.
  Only 4/8 categories have donor-bank coverage (rice, walnuts, wallplugs, fruit_jelly).
- Ran 24 jobs (4 categories × 3 seeds × {P5, P6}) with DINOv3-`large` backbone (pilot vs.
  the paper's `huge_plus`), forcing the standard F1AdaptiveThreshold post-processor
  (overriding SuperADD's own percentile-based one) for arm parity with the other models.
- Complete, 0 errors. Poisson beats alpha on calibration gap for every category (replicates
  the P2/P3 finding with real-defect-bank patches); `wallplugs` is a clear outlier with
  near-chance oracle image-AUROC. Full results + methodology in
  `superadd_mvtecad2_experiment.md`.

## Current Experiment
None running. Phases 0–7 are all complete as of 2026-08-24.

## Planned Next
1. Re-run the difficulty-mechanism / score-coverage analyses on AD2 tiled data.
2. Decide whether to wire in SuperADD's own built-in percentile threshold as a bonus
   "arm D" reference point.
3. Consider a `huge_plus`-backbone SuperADD re-run (the paper's actual config) now that
   the pipeline is validated with `large`.
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

## Phase 6 — MVTec AD 2, native-resolution tiled inference (sweep.py `phase6`, complete)
Closes the 448 px → native-resolution (2.3–5.0 MP) gap without changing model internals.

- [x] Built `tiled_harness.py` (random-crop training + Tiler-based tiled inference)
- [x] Found + fixed tile-border scoring artifact (32 px margin crop)
- [x] Found + fixed torchmetrics sigmoid-saturation bug (min-max normalize before AUROC/AUPR)
- [x] Found + fixed Dinomaly's published crop breaking tile stitching
- [x] Found + fixed pixel-threshold bin-range mismatch (most consequential; corrupted an
      earlier 38/48-job partial run, discarded and re-run clean)
- [x] 48 jobs (6 models × 8 categories × 1 seed) — complete (8 local + 40 remote, merged)
- [x] Compare tiled vs. 448 px vs. published SOTA (SuperADD 57.4%, RoBiS 51.0% SegF1) —
      **tiling does not close the gap; net regression on average (Δ mean SegF1 −0.022)**

## Phase 7 — SuperADD on MVTec AD 2 (sweep.py `phase7_alpha`/`phase7_poisson`, complete)
Anomalib-native SuperADD (PR #3628), calibrated with the pre-generated semantic-defect-bank
synthetic pipelines (P5=alpha, P6=poisson) instead of the live P1/P2/P3 generators.

- [x] Merged `feature/v2.6.0` (brings in SuperADD + AutoVI); verified via 4-backbone smoke
      test on MVTec AD toothbrush
- [x] Built `_pregenerated_eval_set()` loader for `SynthetciGenMVAD2/` pre-rendered pairs
- [x] 24 jobs (4 categories × 3 seeds × {P5, P6}, DINOv3-`large` backbone) — complete,
      0 errors. Results + methodology in `superadd_mvtecad2_experiment.md`.

## Notes

- Arm A computed once per (model, dataset, category) and reused across pipelines.
- AUROC won't rank pipelines for val-only models (threshold-free) — select on thresholded + pixel + calibration gap.
- Confirm AnomalyDINO / Dinomaly availability in the installed anomalib version before Phase 2.
