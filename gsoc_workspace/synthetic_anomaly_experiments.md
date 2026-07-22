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

## Phase 0 — Harness / tracer bullet
Prove the A/B/C plumbing end-to-end before scaling.

- [ ] MVTec (3 cats) × {PatchCore, DRAEM} × {P1, P3} × {A, B, C} × 1 seed (~30 runs)
- [ ] Confirm A > B > chance, masks/metrics sane, DRAEM trains on our generator.

## Phase 1 — Pipeline shortlisting (MVTec, val-only)
Pick the 2–3 pipelines whose B is closest to A.

- [ ] MVTec (15 cats) × {PatchCore, PaDiM, EfficientAD} × {P1, P2, P3, P4} × {A, B} × 3 seeds (~585 runs)
- [ ] Select shortlist of 2–3 pipelines for Phase 2.

## Phase 2 — Grand comparison (val-only regime)
All val-only models × shortlisted pipelines, MVTec then VisA.

- [ ] MVTec (15) × {PaDiM, PatchCore, EfficientAD, AnomalyDINO, Dinomaly} × {A + 3 shortlisted} × 3 seeds (~750 runs)
- [ ] VisA (12) × same models × {A + 3 shortlisted} × 3 seeds (~600 runs)
- [ ] Report per-model B-vs-A calibration gap.

## Phase 3 — DRAEM synth-as-training regime
Here the pipeline is training data — the biggest lever. Report separately.

- [ ] MVTec (15) + VisA (12) × DRAEM × {P1 baseline + shortlisted} × {train, train+val} × 3 seeds (~324 runs)

## Phase 4 — Optional expansion
Only if earlier phases warrant it.

- [ ] Extra models: GLASS / SimpleNet / CutPaste (train regime)
- [ ] Extra datasets: MVTec AD 2, MPDD, BTAD
- [ ] Axis ablations (region / blend / source) isolated on 1 model

## Notes

- Arm A computed once per (model, dataset, category) and reused across pipelines.
- AUROC won't rank pipelines for val-only models (threshold-free) — select on thresholded + pixel + calibration gap.
- Confirm AnomalyDINO / Dinomaly availability in the installed anomalib version before Phase 2.
