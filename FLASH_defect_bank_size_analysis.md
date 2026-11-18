# FLASH defect-bank-size ablation: observations

Analysis date: 2026-10-29.

## Main conclusion

Increasing the defect bank can improve transferred pixel-level F1, especially
for rice, but the improvement is neither universal nor monotonic. The highest
observed three-category real-test mean is at **B=11**, not B=14. No bank size
improves real-test image F1 or resolves the all-anomalous image predictions.

This is useful evidence of category-dependent calibration sensitivity, not
evidence that more defect images always improve image and pixel accuracy.

## Sources and protocol

- [Individual results](bank_size_results.csv): 108 evaluation rows.
- [Three-seed summaries](bank_size_f1_summary.csv): 36 summary rows.
- Detector: SuperADD, `vit_huge_plus_patch16_dinov3`, 448x448.
- Categories: rice, fruit_jelly and walnuts.
- Generation seeds: 0, 1, 2; detector seeds: 1, 2, 3 respectively.
- Source-image bank sizes: 3, 5, 7, 9, 11, 14.
- There are 54 category/seed/bank configurations, each evaluated on both
  real `test_public` data and a separate synthetic evaluation set.
- Each seed uses strictly nested source-image groups, retaining the original
  three paper source images. B counts source images, not extracted regions:
  rice has four original regions at B=3 and 15 regions at B=14.
- Calibration positives stay fixed at 90 for rice, 60 for fruit_jelly and
  90 for walnuts. Native validation normals supply calibration negatives.
- Image and pixel thresholds are selected on calibration data and transferred
  unchanged to each evaluation set. No oracle was recomputed.
- All three seeds are averaged within each category. Macro results below
  equally weight these three categories, not all eight paper categories.
- B=3 was evaluated in a separate seeded fit; B=5,14,7,9,11 share one subsequent
  fit per category/seed. There were 18 successful fit invocations in total.

Source SHA-256:

```text
bank_size_results.csv
8c86beda33297008d47ada2e820f5b432d038abc8476c36df9381867a43aa803

bank_size_f1_summary.csv
6845e64c8e6ea62ea5274f3f4a8b118dbed220fd3f810add49e84d64e5e974bd
```

## 1. Pixel F1: complete size comparison

All values are percentages. These are F1 values at transferred calibration
thresholds, not real-test F1-max.

| Bank size B | Rice: real | Fruit jelly: real | Walnuts: real | Real macro | Synthetic macro |
|---:|---:|---:|---:|---:|---:|
| 3 | 52.55 | 55.39 | 66.67 | 58.20 | 49.61 |
| 5 | 55.24 | 55.60 | 66.52 | 59.12 | 51.08 |
| 7 | 57.71 | **55.64** | 66.14 | 59.83 | **53.09** |
| 9 | 57.24 | 55.25 | 66.25 | 59.58 | 52.29 |
| 11 | **58.33** | 55.18 | **68.05** | **60.52** | 53.07 |
| 14 | 56.64 | 54.67 | 67.13 | 59.48 | 52.95 |

Real pixel macro F1 rises from 58.20 at B=3 to 60.52 at B=11:
**+2.32 percentage points (pp)**. B=14 falls to 59.48, a **1.04 pp loss**
relative to B=11. B=7 to B=9 also declines. Do not describe the curve as
monotonically increasing or infer an asymptotic plateau from six points.

### Category-specific effects

| Category | B=3 real pixel F1 | Best observed size | Best observed real pixel F1 | Change from B=3 | Seed direction at that size |
|---|---:|---:|---:|---:|---|
| Rice | 52.55 | 11 | 58.33 | +5.78 pp | Positive in 3/3 |
| Fruit jelly | 55.39 | 7 | 55.64 | +0.25 pp | Positive in 2/3 |
| Walnuts | 66.67 | 11 | 68.05 | +1.38 pp | Positive in 3/3 |

The best-size column is descriptive, selected after inspecting these results;
it is not a validated category-specific configuration recommendation.

**Rice:** every larger bank outperforms its corresponding B=3 seed. Rice
contributes approximately 83% of the B=11 macro gain. Its B=11 mean and
sample seed standard deviation are 58.33 +/- 0.34%, compared with
52.55 +/- 2.60% at B=3.

**Fruit jelly:** gains are small and inconsistent. B=14 loses 0.72 pp on
average and is worse than B=3 in all three seeds. More sources are not
automatically more useful calibration sources.

**Walnuts:** B=11 gives a modest positive change in all three seeds.
B=5,7,9 have negative mean changes; B=14 has a +0.46 pp mean change but
improves in only one of three seeds.

These observations do not identify a causal defect-type mechanism. The
study lacks a controlled cross-category defect-family comparison.

## 2. Seed consistency

| Bank size B | Real pixel macro F1, mean +/- seed SD | Mean change from B=3 |
|---:|---:|---:|
| 3 | 58.20 +/- 0.34 | 0.00 pp |
| 5 | 59.12 +/- 2.27 | +0.92 pp |
| 7 | 59.83 +/- 0.96 | +1.63 pp |
| 9 | 59.58 +/- 1.17 | +1.37 pp |
| 11 | 60.52 +/- 0.50 | +2.32 pp |
| 14 | 59.48 +/- 1.61 | +1.28 pp |

SD is computed across the three seed-level category macros. It is not a
confidence interval, a standard error, or variation across categories.

| Generation seed | B=11 real pixel macro change | B=14 real pixel macro change |
|---:|---:|---:|
| 0 | +1.53 pp | -0.83 pp |
| 1 | +2.22 pp | +2.48 pp |
| 2 | +3.20 pp | +2.18 pp |

B=11 has encouraging directional consistency. Three seeds do not establish
statistical significance. Donor order, synthesis randomness and detector
seed vary together; the study does not independently estimate each source
of variability or exhaustively average over donor subsets.

## 3. Image F1 and specificity

At every bank size, real-test image F1 is:

| Category | Image F1 |
|---|---:|
| Rice | 81.08% |
| Fruit jelly | 85.71% |
| Walnuts | 75.00% |
| Three-category macro | 80.60% |

Recorded precision and recall imply **every real-test image is predicted
anomalous in all 54 configurations**:

- Anomaly recall is 100%.
- Image precision equals anomaly prevalence.
- Normal-image false-positive rate is 100%; specificity is 0%.

The always-anomalous comparator is `2P / (2P + N)`, where P and N are the
numbers of anomalous and normal test images. It reproduces these F1 values.
These scores must not be presented as evidence of useful normal/anomaly
discrimination or called classification accuracy.

Synthetic image evaluations also imply 100% false-positive rates. Their
small F1 differences reflect changes in anomalous-image recall, not improved
normal rejection. Increasing the source bank does not fix image calibration.

## 4. Synthetic evaluation and transfer

Synthetic pixel macro F1 rises from 49.61 at B=3 to 53.09 at B=7, 53.07 at
B=11 and 52.95 at B=14. Its best observed size differs from the real-test
best size.

Rice makes this mismatch particularly clear:

- Synthetic pixel F1: 46.70 at B=3, 56.93 at B=11, 58.08 at B=14.
- Real pixel F1: 52.55 at B=3, 58.33 at B=11, 56.64 at B=14.

Thus, increasing from B=11 to B=14 helps synthetic rice evaluation while
hurting real rice evaluation. Synthetic improvement cannot substitute for
evidence of real-world improvement.

The synthetic evaluation set is fixed per category/seed across bank sizes,
with a B=14 donor pool, disjoint calibration/evaluation host identities and
different synthesis seeds. It uses normal training-image backgrounds and
overlaps the calibration donor pool increasingly as B grows. It is held out
from calibration composites, **not an unseen-donor or unseen-background
generalization benchmark**. Real and synthetic F1 levels also reflect
different anomaly/mask distributions; their gap is not a realism score.

## 5. What changes mechanistically

Real pixel AUROC remains approximately 95.086% across bank sizes. Together
with shared fitted models within the larger-bank sweep, this is consistent
with changes in the transferred pixel operating threshold rather than a
demonstrated improvement in detector representation.

For rice, the mean pixel threshold changes from approximately 0.7205 at B=3
to 0.6162 at B=11 and 0.5768 at B=14. Lowering it further does not
monotonically improve real F1. The CSVs do not contain pixel precision and
recall, so the exact pixel false-positive/false-negative trade-off cannot
be established from these tables alone.

Anomalib's post-processing normalizes and clamps scores. Small differences
in reported ranking metrics therefore do not, on their own, establish
better raw anomaly-score ordering.

## 6. Reproduction and interpretation limits

- B=3 passed the approved faithful Phase 1 reference checks. This is not a
  claim of exact equality with the older published paper CSV.
- The known older-archive rice/seed2 image discrepancy and the original
  failed attempt remain preserved; no pixels or thresholds were edited to
  force agreement.
- B=3 and the larger-bank sweep use separate fits. Comparisons are paired
  by category and seed, not all from one identical fitted instance.
- Bank size changes source identities and reuse frequencies as well as the
  nominal source count. It does not isolate a universal effect of quantity
  independent of donor quality or composition.
- Provenance-qualified old/new sources can share an underlying normal scene.
  B does not count independent normal backgrounds.
- This study covers one detector and three categories. Its macro must not
  be compared directly with the earlier eight-category component macro.
- The real test set has already been inspected. Selecting B=11, or a
  different best B per category, is post-hoc configuration selection.

## 7. Recommended paper presentation

Use the complete six-point real-pixel sensitivity table/plot. Retain the
fruit-jelly losses and B=14 decline. Show synthetic evaluation separately
as an auxiliary diagnostic, and disclose the image specificity failure.
Keep the original B=3 method default unless a separately justified,
independently evaluated protocol change is made.

Suggested wording:

> Increasing defect-bank size produced category-dependent, non-monotonic
> improvements in transferred pixel-level F1. The highest observed
> three-category mean occurred at 11 source images, improving by 2.32
> percentage points over the three-image baseline, primarily through gains
> on rice. Further expansion did not consistently improve performance.

Avoid claims that more sources always help, that B=11 is universally optimal,
that image discrimination improved, or that three seeds prove significance.
