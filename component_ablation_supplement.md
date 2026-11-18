# FLASH supplementary ablation tables

SuperADD on MVTec AD 2; three seeds per category, then an equal-weight mean over eight categories. F1 values are percentages; deltas are percentage points relative to the paired full FLASH rerun. The plus/minus term is the sample SD of three seed-level category macros, not a confidence interval or a significance test. Independent image/pixel thresholds maximize calibration F1 on native validation normals plus synthetic positives, then transfer unchanged to real public-test data. These are not test-optimized F1max scores. The rerun control (78.23 image / 38.61 pixel) is distinct from the published 78.13 / 38.38. Unresolved archived-pixel replay differences remain documented in the ablation README. Combinations are post-hoc exploratory configurations chosen after viewing Phase 1, including heldout category results; they are not an untouched confirmatory evaluation or a new method default. All six approved combinations are reported, including negative results; pending cells are unmeasured. Budget sensitivity changes positive coverage and calibration class balance and is shown separately. The simplified control retains the original donor bank, host eligibility, resizing and augmentation; it is not an independent external CutPaste method. H denotes CIELAB harmonisation. Forced Poisson with versus without collar isolates the collar effect; either versus the hybrid baseline also changes routing.

**Interpretation caveat:** An always-anomalous image predictor scores 79.42% macro image F1 on this split. The baseline has all-positive image predictions in 21/24 category/seed runs. Higher image F1 alone must not be interpreted as improved discrimination or accuracy. Observed AUROC variation alone is not a quantified detector-refit noise floor.

## Table-design rationale

Use absolute F1 and signed deltas together, with pixel results first but image results retained. Do not switch to delta-only reporting or omit image results because they are less favourable. The settings matrix uses categorical replacements rather than misleading on/off ticks. This follows the settings-plus-absolute-results structure of [Dinomaly Table 3](https://arxiv.org/html/2405.14325v3#S4.T3), which reports both image and pixel metrics. Its F1-max is not the same metric as our synthetic-calibrated real-test F1.

## Component counterfactuals and exploratory combinations

| Configuration | Pixel F1 (%) | Delta pixel (pp) | Image F1 (%) | Delta image (pp) |
| --- | ---: | ---: | ---: | ---: |
| Full FLASH (paired rerun) | 38.61 +/- 1.46 | +0.00 | 78.23 +/- 0.11 | +0.00 |
| OBS: colour only | 41.86 +/- 1.22 | +3.24 | 78.15 +/- 0.15 | -0.07 |
| OBS: texture only | 36.51 +/- 3.06 | -2.11 | 79.43 +/- 0.02 | +1.20 |
| OBS: full frame | 36.91 +/- 1.71 | -1.71 | 73.61 +/- 1.83 | -4.62 |
| MRSP: frame-relative | 38.69 +/- 0.67 | +0.07 | 78.16 +/- 0.12 | -0.07 |
| MRSP: uniform blob | 37.88 +/- 1.39 | -0.74 | 75.19 +/- 1.79 | -3.04 |
| Placement: original only | 37.54 +/- 0.78 | -1.07 | 78.34 +/- 0.79 | +0.11 |
| Placement: adaptive only | 39.05 +/- 1.84 | +0.43 | 78.24 +/- 0.11 | +0.02 |
| Harmonisation off | 39.64 +/- 0.76 | +1.03 | 78.23 +/- 0.11 | +0.00 |
| Blend: alpha only | 39.44 +/- 1.16 | +0.82 | 78.23 +/- 0.11 | +0.00 |
| Blend: Poisson with collar | 38.57 +/- 1.22 | -0.04 | 78.21 +/- 0.09 | -0.01 |
| Blend: Poisson without collar | 38.17 +/- 0.76 | -0.44 | 79.42 +/- 0.00 | +1.19 |
| Bank: one donor | 37.31 +/- 5.76 | -1.30 | 77.63 +/- 0.81 | -0.60 |
| Bank: two donors | 39.95 +/- 1.27 | +1.33 | 77.69 +/- 0.85 | -0.54 |
| Exploratory: Colour OBS + no harmonisation | 41.71 +/- 0.91 | +3.10 | 78.20 +/- 0.11 | -0.03 |
| Exploratory: Colour OBS + two donors | 42.31 +/- 1.98 | +3.70 | 78.24 +/- 0.11 | +0.02 |
| Exploratory: No harmonisation + alpha | 38.01 +/- 0.76 | -0.61 | 78.21 +/- 0.18 | -0.01 |
| Exploratory: Colour OBS + no H + adaptive + two donors | 42.18 +/- 1.11 | +3.57 | 78.56 +/- 0.56 | +0.33 |
| Exploratory: Full frame + uniform blob | 35.67 +/- 1.09 | -2.94 | 74.36 +/- 1.47 | -3.87 |
| Exploratory: Simplified cut-paste control | 34.07 +/- 3.28 | -4.54 | 72.88 +/- 0.89 | -5.34 |

## Calibration-budget sensitivity (not a pure reuse ablation)

| Configuration | Pixel F1 (%) | Delta pixel (pp) | Image F1 (%) | Delta image (pp) |
| --- | ---: | ---: | ---: | ---: |
| Full FLASH (paired rerun) | 38.61 +/- 1.46 | +0.00 | 78.23 +/- 0.11 | +0.00 |
| Eight unique positives | 35.92 +/- 4.31 | -2.70 | 65.13 +/- 3.22 | -13.10 |
| Sixteen unique positives | 39.73 +/- 0.85 | +1.11 | 70.01 +/- 4.28 | -8.22 |

## Image-threshold diagnostic

| Configuration | Precision (%) | Recall (%) | All-positive runs |
| --- | ---: | ---: | ---: |
| Full FLASH (paired rerun) | 66.38 | 96.11 | 21/24 |
| OBS: colour only | 66.35 | 95.97 | 21/24 |
| OBS: texture only | 66.31 | 100.00 | 23/24 |
| OBS: full frame | 76.81 | 74.93 | 6/24 |
| MRSP: frame-relative | 66.32 | 96.02 | 21/24 |
| MRSP: uniform blob | 74.14 | 82.92 | 11/24 |
| Placement: original only | 68.99 | 92.41 | 18/24 |
| Placement: adaptive only | 66.37 | 96.16 | 21/24 |
| Harmonisation off | 66.38 | 96.11 | 21/24 |
| Blend: alpha only | 66.38 | 96.11 | 21/24 |
| Blend: Poisson with collar | 66.36 | 96.11 | 21/24 |
| Blend: Poisson without collar | 66.30 | 100.00 | 24/24 |
| Bank: one donor | 70.37 | 90.37 | 17/24 |
| Bank: two donors | 67.59 | 94.17 | 20/24 |
| Exploratory: Colour OBS + no harmonisation | 66.36 | 96.06 | 21/24 |
| Exploratory: Colour OBS + two donors | 66.37 | 96.16 | 21/24 |
| Exploratory: No harmonisation + alpha | 66.35 | 96.11 | 21/24 |
| Exploratory: Colour OBS + no H + adaptive + two donors | 66.36 | 97.04 | 21/24 |
| Exploratory: Full frame + uniform blob | 80.66 | 73.96 | 2/24 |
| Exploratory: Simplified cut-paste control | 79.28 | 75.56 | 3/24 |

## Descriptive category-group pixel effects

| Configuration | Development delta (pp) | Heldout delta (pp) |
| --- | ---: | ---: |
| Full FLASH (paired rerun) | +0.00 | +0.00 |
| OBS: colour only | -1.02 | +7.51 |
| OBS: texture only | -4.73 | +0.52 |
| OBS: full frame | +0.86 | -4.28 |
| MRSP: frame-relative | -1.14 | +1.28 |
| MRSP: uniform blob | -0.19 | -1.28 |
| Placement: original only | -1.37 | -0.77 |
| Placement: adaptive only | +0.51 | +0.36 |
| Harmonisation off | +1.27 | +0.79 |
| Blend: alpha only | -1.65 | +3.29 |
| Blend: Poisson with collar | +0.04 | -0.13 |
| Blend: Poisson without collar | -4.24 | +3.36 |
| Bank: one donor | -0.52 | -2.07 |
| Bank: two donors | -0.01 | +2.68 |
| Exploratory: Colour OBS + no harmonisation | -0.92 | +7.12 |
| Exploratory: Colour OBS + two donors | -1.39 | +8.79 |
| Exploratory: No harmonisation + alpha | -3.38 | +2.16 |
| Exploratory: Colour OBS + no H + adaptive + two donors | -1.59 | +8.73 |
| Exploratory: Full frame + uniform blob | +1.85 | -7.73 |
| Exploratory: Simplified cut-paste control | -3.55 | -5.53 |

## Settings matrix (C: colour, T: texture, P: Poisson)

| Configuration | OBS | Mask | Placement | H | Blend | Donors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full FLASH (paired rerun) | C+T | MRSP/obj | mixed | yes | hybrid | 3 |
| OBS: colour only | C | MRSP/obj | mixed | yes | hybrid | 3 |
| OBS: texture only | T | MRSP/obj | mixed | yes | hybrid | 3 |
| OBS: full frame | frame | MRSP/obj | mixed | yes | hybrid | 3 |
| MRSP: frame-relative | C+T | MRSP/frame | mixed | yes | hybrid | 3 |
| MRSP: uniform blob | C+T | blob | mixed | yes | hybrid | 3 |
| Placement: original only | C+T | MRSP/obj | original | yes | hybrid | 3 |
| Placement: adaptive only | C+T | MRSP/obj | adaptive | yes | hybrid | 3 |
| Harmonisation off | C+T | MRSP/obj | mixed | no | hybrid | 3 |
| Blend: alpha only | C+T | MRSP/obj | mixed | yes | alpha | 3 |
| Blend: Poisson with collar | C+T | MRSP/obj | mixed | yes | P+collar | 3 |
| Blend: Poisson without collar | C+T | MRSP/obj | mixed | yes | P | 3 |
| Bank: one donor | C+T | MRSP/obj | mixed | yes | hybrid | 1 |
| Bank: two donors | C+T | MRSP/obj | mixed | yes | hybrid | 2 |
| Exploratory: Colour OBS + no harmonisation | C | MRSP/obj | mixed | no | hybrid | 3 |
| Exploratory: Colour OBS + two donors | C | MRSP/obj | mixed | yes | hybrid | 2 |
| Exploratory: No harmonisation + alpha | C+T | MRSP/obj | mixed | no | alpha | 3 |
| Exploratory: Colour OBS + no H + adaptive + two donors | C | MRSP/obj | adaptive | no | hybrid | 2 |
| Exploratory: Full frame + uniform blob | frame | blob | mixed | yes | hybrid | 3 |
| Exploratory: Simplified cut-paste control | frame | blob | original | no | alpha | 3 |

## Provenance

- `/home/devuser/.copilot/session-state/8a2f0e48-4fe2-4d47-ba4d-4bb694881422/files/flash/evaluation/ablations/results/launches/20261022_phase1_paper_components_v1/phase1_results.csv SHA-256=c2468fc11cb86d345fbc013b6acf3b6f36fed768a1755e6946e1c25f8bd1a52b`
- `/home/devuser/.copilot/session-state/8a2f0e48-4fe2-4d47-ba4d-4bb694881422/files/flash/evaluation/ablations/results/launches/20261027_phase2_paper_combinations_v2/phase1_results.csv SHA-256=ad9e864949c1c95ea72f3633f3a864289f331a05e7f382ad49ee0fec9d8de304`

All six approved combinations are included.
