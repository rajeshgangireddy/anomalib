# FLASH Dinomaly defect-bank-size ablation: observations

Analysis date: 2026-10-29.

## Main conclusion

Larger defect banks substantially improve Dinomaly's transferred **pixel-level
F1** on these three categories. The first expansion, B=3 to B=5, improves all
nine category/seed pairs. The predefined largest bank, B=14, improves the real
pixel macro by **10.85 percentage points (pp)**. The best observed macro is
at B=9, **44.36% versus 32.77% at B=3**, a gain of **11.59 pp**.

This is stronger bank-size sensitivity than observed for SuperADD, but not a
universal or monotonic benefit: rice and fruit jelly drive the Dinomaly gains,
walnuts changes little, and image-level F1 does not consistently improve.
These are changes in threshold transfer with a fixed detector, not improvements
in the learned representation or evidence that Dinomaly outperforms SuperADD.

## Sources and protocol

- [Individual results](dinomaly_bank_size_results.csv): 108 primary evaluation rows.
- [Three-seed summaries](dinomaly_bank_size_f1_summary.csv): 36 summary rows.
- [SuperADD analysis](FLASH_defect_bank_size_analysis.md): the companion experiment.
- Categories: rice, fruit_jelly and walnuts; generation seeds 0, 1, 2,
  mapped to detector seeds 1, 2, 3.
- Source-image bank sizes: B=3,5,7,9,11,14, evaluated in the requested order
  3,5,14,7,9,11. Within each seed, source groups are strictly nested and
  retain the original three paper source images.
- B counts source images, not extracted regions or independent backgrounds.
  Rice retains all four original regions at B=3 and has 15 regions at B=14.
- Detector: Dinomaly, `vit_base_patch14_reg4_dinov2`, float32, batch size eight.
  Preprocessing resizes to 448x448 and then center-crops to 392x392.
- The frozen paper training budget is **1,000 optimizer steps**. The
  `max_epochs=1000` setting is a safety cap, not 1,000 completed epochs.
  Every one of the nine fits reached exactly 1,000 optimizer steps.
- Each category/seed fits once on normal training data. That exact fit is
  shared across all six bank sizes and a final B=3 repeat.
- Calibration positives remain 90/60/90 for rice/fruit jelly/walnuts;
  calibration negatives are 35/37/48 native validation normals.
- Real-test anomaly/normal counts are 90/42, 60/20 and 90/60 respectively.
  Image and pixel thresholds are selected independently on calibration data
  and transferred unchanged to real and fixed synthetic evaluation data.
- The completed experiment has 54 primary bank conditions, each with two
  evaluation sets, plus 18 diagnostic repeat rows. Repeats are not extra seeds.
- Means first average the three seeds within category, then equally weight
  the three categories. These are not eight-category paper macros.
- The approved reference policy is `paired_runtime_with_paper_audit`:
  historical differences are retained, not used to replace the current control.
  No Oracle was recomputed.

Run evidence is under
`SESSION/files/flash/evaluation/ablations/results/launches/20261029_dinomaly_bank_size_v1/`,
where `SESSION` is the current Copilot session workspace. It includes
`completion_verified.json`, `export_verification.json`, per-fit recipe/checkpoint
hashes, input provenance, independent confusion counts, image scores, paper audits
and repeated-control audits.

Source SHA-256:

```text
dinomaly_bank_size_results.csv
51e9a3b8d52487a04c989e4471748c0b60ca4a21e9866df5a994343c2afdba68

dinomaly_bank_size_f1_summary.csv
7953508a712330b25f4fb8bc7737c61b7bb9962e8504ccc607ec54e2517cc916
```

## 1. Pixel F1: complete size comparison

All F1 values are percentages at transferred calibration thresholds, not
real-test F1-max.

| Bank size B | Rice: real | Fruit jelly: real | Walnuts: real | Real macro | Synthetic macro |
|---:|---:|---:|---:|---:|---:|
| 3 | 17.93 | 34.38 | 46.01 | 32.77 | 37.40 |
| 5 | 28.56 | 42.31 | **46.37** | 39.08 | 40.49 |
| 7 | 37.81 | 44.50 | 45.81 | 42.71 | 41.41 |
| 9 | 43.25 | **44.83** | 45.00 | **44.36** | **41.75** |
| 11 | 40.79 | 41.55 | 45.26 | 42.53 | 41.31 |
| 14 | **44.43** | 40.14 | 46.30 | 43.62 | 41.37 |

Every larger bank improves the real pixel macro over B=3. However, B=11 loses
1.83 pp relative to B=9, and B=14 remains 0.74 pp below B=9. Six measured
points do not establish a monotonic curve or a universal saturation size.

### Category-specific effects

| Category | B=3 | B=9 | B=9 change | B=14 | B=14 change |
|---|---:|---:|---:|---:|---:|
| Rice | 17.93 | 43.25 | +25.32 pp | 44.43 | +26.49 pp |
| Fruit jelly | 34.38 | 44.83 | +10.45 pp | 40.14 | +5.76 pp |
| Walnuts | 46.01 | 45.00 | -1.01 pp | 46.30 | +0.29 pp |

**Rice:** every larger bank improves over B=3 in all three seeds. Its largest
mean is at B=14, 44.43 +/- 1.90%, versus 17.93 +/- 0.76% at B=3.

**Fruit jelly:** B=5,7,9,11 each improve in all three seeds. B=9 gives the
largest mean, 44.83 +/- 1.19%, versus 34.38 +/- 4.39% at B=3.
B=14 still improves the mean but loses in one seed.

**Walnuts:** B=5 gives a small positive change in all seeds, averaging
only +0.36 pp. B=7,9,11 lose on average. B=14 gains +0.29 pp in the mean
and improves all three seeds, but this is small compared with the rice gains.

The best sizes are descriptive, selected after inspecting these outcomes,
not independently validated category-specific settings. The data do not
establish a causal explanation in terms of particular defect families.

## 2. Seed consistency

| B | Real pixel macro, mean +/- seed SD | Change from B=3 | Improved category/seed pairs |
|---:|---:|---:|---:|
| 3 | 32.77 +/- 1.71 | 0.00 pp | Reference |
| 5 | 39.08 +/- 0.09 | +6.31 pp | 9/9 |
| 7 | 42.71 +/- 1.64 | +9.93 pp | 7/9 |
| 9 | 44.36 +/- 1.62 | +11.59 pp | 8/9 |
| 11 | 42.53 +/- 0.55 | +9.76 pp | 7/9 |
| 14 | 43.62 +/- 0.46 | +10.85 pp | 8/9 |

Macro SD is the sample standard deviation across three seed-level category
macros, not across categories. Category SDs above use the three category-specific
seed results. Neither is a confidence interval or a significance test.

| Generation seed | B=5 macro change | B=9 macro change | B=14 macro change |
|---:|---:|---:|---:|
| 0 | +4.87 pp | +11.48 pp | +8.78 pp |
| 1 | +8.13 pp | +13.76 pp | +12.97 pp |
| 2 | +5.92 pp | +9.52 pp | +10.80 pp |

The positive macro effect is not confined to one seed. Nevertheless, donor
ordering, synthesis randomness and detector initialization vary together.
Three seeds do not exhaust donor-subset variability or establish significance.

## 3. Image F1 and specificity

| B | Rice: image F1 | Fruit jelly: image F1 | Walnuts: image F1 | Image macro | Macro image FPR |
|---:|---:|---:|---:|---:|---:|
| 3 | 81.08 | 87.87 | 79.50 | 82.82 | 80.93 |
| 5 | 81.08 | 87.01 | 78.82 | 82.30 | 82.59 |
| 7 | 81.08 | 86.86 | 79.22 | 82.39 | 87.22 |
| 9 | 81.08 | 86.99 | 79.76 | 82.61 | 85.56 |
| 11 | 80.32 | 87.90 | 77.70 | 81.97 | 76.27 |
| 14 | 79.28 | 87.66 | 79.29 | 82.08 | 74.02 |

FPR is the percentage of normal test images predicted anomalous, averaged
equally across categories and seeds. These high values remain a serious
image-threshold limitation; image F1 is not classification accuracy.

The largest banks reduce macro FPR, but macro anomaly recall also declines:
98.27% at B=3 versus 94.75% at B=14. The image F1 mean is lower than B=3
at every larger bank. At B=3, four of nine conditions predict every real
image anomalous; the corresponding counts for B=5,7,9,11,14 are 4,4,4,2,3.
Unlike SuperADD, not all Dinomaly conditions collapse to all-positive predictions,
but larger banks do not establish a consistent image-level benefit.

## 4. Synthetic evaluation and transfer

| B | Rice: synthetic pixel F1 | Fruit jelly: synthetic pixel F1 | Walnuts: synthetic pixel F1 | Synthetic image macro |
|---:|---:|---:|---:|---:|
| 3 | 43.74 | 46.04 | 22.42 | 75.74 |
| 5 | 49.54 | 49.33 | 22.61 | 77.14 |
| 7 | 52.03 | 49.90 | 22.31 | 77.56 |
| 9 | 53.05 | 50.05 | 22.15 | 76.82 |
| 11 | 52.39 | 49.33 | 22.21 | 76.16 |
| 14 | 52.95 | 48.57 | 22.58 | 75.56 |

The synthetic and real pixel macros both peak at B=9 in this experiment.
This alignment is descriptive, not evidence that synthetic F1 can reliably
select the best size on unseen real data. The companion SuperADD experiment
has different synthetic and real best sizes.

Each category/seed uses one fixed B=14-pool synthetic evaluation set, with
calibration-disjoint host identities and distinct synthesis seeds. The
synthetic positives match the real anomaly count, and evaluation negatives
are the same real-test normals, so image FPR agrees across evaluation sets.
Synthetic backgrounds come from normal training images, not an independent
held-out training-background pool. Defect donors overlap calibration donors
increasingly as the calibration bank grows.

Therefore, these are auxiliary within-generator diagnostics, not unseen-donor
or unseen-background generalization. The real/synthetic F1 gap is not a
standalone measure of image realism or generation quality.

## 5. What changes mechanistically

The frozen representation is unchanged throughout each comparison. Real pixel
AUROC remains approximately **96.0572%** at all sizes. Raw image AUROC is
unchanged at approximately **74.8761%**. Reported post-processed image AUROC
does change slightly because normalization/clamping changes with calibration;
that does not demonstrate improved raw ranking.

Independent pixel confusion counts clarify the rice gains:

| Rice bank size | Mean raw pixel threshold | Mean pixel precision | Mean pixel recall |
|---:|---:|---:|---:|
| 3 | 0.1550 | 50.99% | 10.88% |
| 9 | 0.1212 | 50.12% | 38.47% |
| 14 | 0.1185 | 49.56% | 40.49% |

Lower transferred thresholds recover many more real anomalous pixels while
mean precision stays near 50%. Fruit jelly also gains recall, from 22.89%
at B=3 to 33.75% at B=9, while precision falls from 70.10% to 66.84%.
This supports an operating-point explanation, not a better learned detector.
These precision/recall values are seed means; their harmonic mean need not
equal the separately averaged F1.

All 108 primary evaluations passed independent raw/processed confusion-F1
agreement, decision-agreement and unchanged-threshold checks. All 18 repeated
B=3 real/synthetic passes have identical raw prediction hashes. The fitted
representation hash is unchanged after each bank. These checks support this
run's integrity, not a universal fix for earlier unrelated metric failures.

## 6. Historical paper comparison

The new B=3 fit is the paired control. It does not exactly reproduce the
published Dinomaly results:

| Category | Published B=3 pixel F1 | Current B=3 pixel F1 | Difference |
|---|---:|---:|---:|
| Rice | 21.21 | 17.93 | -3.27 pp |
| Fruit jelly | 37.18 | 34.38 | -2.81 pp |
| Walnuts | 45.35 | 46.01 | +0.66 pp |
| Three-category macro | 34.58 | 32.77 | -1.80 pp |

Published/current image macros are 82.17%/82.82%. Current fits use explicit
model seeds and native-normal fit validation, rather than the historical
harness's real-test oracle-first validation. The discrepancies are retained
in per-fit audits; their full cause is not established. Do not mix the
published B=3 row into this same-fit curve or claim exact paper reproduction.

## 7. Comparison with SuperADD

| B | SuperADD real pixel macro | Dinomaly real pixel macro | SuperADD change from own B=3 | Dinomaly change from own B=3 |
|---:|---:|---:|---:|---:|
| 3 | 58.20 | 32.77 | 0.00 pp | 0.00 pp |
| 5 | 59.12 | 39.08 | +0.92 pp | +6.31 pp |
| 7 | 59.83 | 42.71 | +1.63 pp | +9.93 pp |
| 9 | 59.58 | 44.36 | +1.37 pp | +11.59 pp |
| 11 | 60.52 | 42.53 | +2.32 pp | +9.76 pp |
| 14 | 59.48 | 43.62 | +1.28 pp | +10.85 pp |

Both experiments support category- and detector-dependent pixel calibration
sensitivity. Dinomaly shows the larger improvement relative to its own control,
but SuperADD retains higher absolute pixel F1 at every measured size.
Their best observed sizes differ, so neither B=9 nor B=11 is universally optimal.

The studies reuse the same synthetic corpora but have different detector
preprocessing: SuperADD uses 448x448, Dinomaly a 392x392 center crop. SuperADD's
B=3 and larger-bank sweep also use separate seeded fits, whereas Dinomaly
shares one fit across all sizes. Do not describe this as a perfectly matched
detector-ranking experiment or compare it directly with eight-category macros.

## 8. Interpretation limits and recommended presentation

- Quantity changes together with source identities, quality, diversity and
  reuse frequencies. Nested banks do not isolate quantity independently of
  donor composition or exhaust all subsets.
- Provenance-qualified sources can share an underlying normal scene; B is
  not the number of independent backgrounds.
- Positive calibration budgets match known real-test anomaly counts, as in
  the paper protocol. This is test-informed budgeting, not a demonstration
  of an entirely test-agnostic deployment procedure.
- The benchmark has already been inspected. Best-size selection and
  category-specific explanations are post-hoc, not independent validation.
- Evidence covers three categories and three coupled seeds. Preserve all
  negative outcomes and do not claim statistical significance.

Use the complete six-point real-pixel curves for both detectors, category
panels, seed variability, and unchanged original B=3 controls. Report B=14
as the predefined largest-bank comparison and B=9 as the best observed
Dinomaly point, without silently replacing the paper's B=3 default.
Show image specificity and synthetic evaluation separately rather than hiding
the image-level limitation.

Suggested wording:

> On Dinomaly, expanding the source-image bank from three to fourteen
> improved transferred pixel-level F1 by 10.85 percentage points on the
> three-category benchmark. The highest observed mean occurred at nine
> sources, improving by 11.59 points. Gains were concentrated in rice and
> fruit jelly and did not translate into consistent image-level gains.
> Together with SuperADD, these results indicate detector- and
> category-dependent calibration sensitivity rather than monotonic benefits
> from increasing bank size.
