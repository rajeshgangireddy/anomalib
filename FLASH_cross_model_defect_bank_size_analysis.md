# FLASH cross-model defect-bank-size ablation: paper assessment

Analysis date: 2026-10-30.

## Main conclusion

The completed studies provide useful supporting evidence for the paper:
**larger defect banks substantially improve transferred pixel F1 for
Dinomaly and PatchCore, but the benefit depends on detector and category.**
The evidence does not support a universal, monotonic benefit or a broad
image-level improvement claim.

The strongest predefined comparison is B=3 versus B=14. Real-test pixel
macro F1 improves by **10.85 percentage points (pp) for Dinomaly** and
**7.79 pp for PatchCore**, with positive seed-level macro changes in all
three seeds. SuperADD improves modestly; PaDiM remains weak in absolute
pixel F1 and adds a useful negative result rather than a strong headline.

This is a calibration-sensitivity ablation, not a new generator contribution
or evidence that the underlying detector representation improved.

## 1. Measured sources and protocol

| Detector | Individual results | Three-seed summaries | Companion analysis |
|---|---|---|---|
| SuperADD | [Raw CSV](bank_size_results.csv) | [Summary CSV](bank_size_f1_summary.csv) | [SuperADD analysis](FLASH_defect_bank_size_analysis.md) |
| Dinomaly | [Raw CSV](dinomaly_bank_size_results.csv) | [Summary CSV](dinomaly_bank_size_f1_summary.csv) | [Dinomaly analysis](FLASH_dinomaly_defect_bank_size_analysis.md) |
| PatchCore | [Raw CSV](patchcore_bank_size_results.csv) | [Summary CSV](patchcore_bank_size_f1_summary.csv) | This report |
| PaDiM | [Raw CSV](padim_bank_size_results.csv) | [Summary CSV](padim_bank_size_f1_summary.csv) | This report |

Each detector has 54 primary bank conditions and 108 primary evaluation
rows: three categories, three seeds, six bank sizes, and real/synthetic
evaluation sets. Categories are rice, fruit_jelly and walnuts. These are
**three-category means, not the paper's eight-category benchmark means**.

Bank sizes are B=3,5,7,9,11,14, executed in the requested order
3,5,14,7,9,11. Banks contain nested source-image groups retaining the
original three paper sources. B counts source images, not extracted
regions or independent backgrounds; rice has four regions at B=3.

The calibration-positive budgets remain fixed at 90/60/90, with 35/37/48
native validation normals. Bank expansion changes the available sources,
not the number of calibration-positive images. Thresholds are selected
on calibration data and transferred unchanged to evaluation data.
The positive budgets use known benchmark anomaly counts; disclose this
test-informed budgeting rather than describing the procedure as entirely
test-agnostic deployment.

Dinomaly, PatchCore and PaDiM share one fitted detector across all six bank
sizes within each category/seed. Dinomaly uses the paper's 1,000 optimizer
steps. PatchCore and PaDiM use one complete normal-training feature pass,
without gradient optimization. SuperADD's B=3 and larger-bank sweep use
separate seeded fits; the larger-bank points share the subsequent fit.

SuperADD, PatchCore and PaDiM use 448x448 inputs. Dinomaly resizes to
448x448 and center-crops to 392x392. Comparisons below assess sensitivity
within each detector, not a perfectly matched detector-ranking benchmark.
No new Oracle evaluation was performed.

## 2. Cleanest headline: the predefined B=14 endpoint

All F1 values are percentages at transferred calibration thresholds,
not real-test F1-max. Means equally weight categories and seeds.

| Detector | B=3 real pixel F1 | B=14 real pixel F1 | Change |
|---|---:|---:|---:|
| SuperADD | 58.20 | 59.48 | +1.28 pp |
| Dinomaly | 32.77 | 43.62 | **+10.85 pp** |
| PatchCore | 22.86 | 30.65 | **+7.79 pp** |
| PaDiM | 5.88 | 6.31 | +0.44 pp |

B=14 was specified before these results were inspected. Therefore, this
comparison is a cleaner headline than choosing each detector's best bank
using real-test outcomes.

| Detector | B=14 minus B=3: seed0 | Seed1 | Seed2 | Positive seed macros |
|---|---:|---:|---:|---:|
| SuperADD | -0.83 pp | +2.48 pp | +2.18 pp | 2/3 |
| Dinomaly | +8.78 pp | +12.97 pp | +10.80 pp | 3/3 |
| PatchCore | +8.64 pp | +5.93 pp | +8.79 pp | 3/3 |
| PaDiM | +0.13 pp | +0.36 pp | +0.83 pp | 3/3 |

Three seeds support descriptive consistency, not a statistical-significance
claim. They reuse the same real benchmark images and do not establish
independence across categories, source composition and synthesis randomness.
The small PaDiM effect should not be made impressive using relative
percentage improvements from its very low baseline.

## 3. Complete real-pixel sensitivity curves

| B | SuperADD | Dinomaly | PatchCore | PaDiM |
|---:|---:|---:|---:|---:|
| 3 | 58.20 | 32.77 | 22.86 | 5.88 |
| 5 | 59.12 | 39.08 | 25.75 | 6.33 |
| 7 | 59.83 | 42.71 | 29.30 | **6.42** |
| 9 | 59.58 | **44.36** | **31.36** | 5.99 |
| 11 | **60.52** | 42.53 | 30.03 | 6.04 |
| 14 | 59.48 | 43.62 | 30.65 | 6.31 |

Bold marks the highest observed value within each detector, not an
independently validated optimal setting.

SuperADD peaks at B=11, Dinomaly and PatchCore at B=9, and PaDiM at B=7.
The corresponding best-observed changes from B=3 are +2.32, +11.59,
+8.50 and +0.55 pp. Every curve contains a decline between adjacent sizes.
Do not claim monotonic improvement, a universal optimum, or a demonstrated
asymptotic saturation point.

Preserve the original B=3 method default unless a separately justified,
independently evaluated selection protocol supports changing it. The
best observed sizes are useful sensitivity findings, not new defaults.

## 4. Category-dependent gains are a useful paper observation

Real pixel F1 changes from B=3 to B=14:

| Category | SuperADD | Dinomaly | PatchCore | PaDiM |
|---|---:|---:|---:|---:|
| Rice | +4.09 pp | **+26.49 pp** | **+12.78 pp** | +0.74 pp |
| Fruit jelly | -0.72 pp | +5.76 pp | **+10.56 pp** | +0.74 pp |
| Walnuts | +0.46 pp | +0.29 pp | +0.04 pp | -0.17 pp |

Rice and fruit jelly account for almost all of the Dinomaly and PatchCore
endpoint improvements; walnuts changes little. For PatchCore, both rice
and fruit jelly improve in all three seeds at B=14.

This is a repeatable pattern across these two detectors, but it does not
identify a causal texture or defect-family mechanism. Source identities,
quality, diversity and reuse frequencies change along with bank size.
Nested banks do not isolate quantity independently of source composition
or exhaust alternative source subsets.

For the three same-fit studies, raw real-test prediction hashes are
identical across all bank sizes within each category/seed. The improvements
therefore reflect transferred operating thresholds and post-processing,
not better learned representations or improved raw-score ordering.

## 5. Image-level findings: a positive example, not a broad claim

Real-test image F1:

| B | SuperADD | Dinomaly | PatchCore | PaDiM |
|---:|---:|---:|---:|---:|
| 3 | 80.60 | 82.82 | 76.73 | 80.77 |
| 5 | 80.60 | 82.30 | 78.07 | 80.60 |
| 7 | 80.60 | 82.39 | 74.42 | 80.60 |
| 9 | 80.60 | 82.61 | 81.87 | 80.60 |
| 11 | 80.60 | 81.97 | 79.55 | 80.60 |
| 14 | 80.60 | 82.08 | 77.74 | 80.60 |

### PatchCore fruit jelly improves at both levels

| B | Real pixel F1 | Real image F1 |
|---:|---:|---:|
| 3 | 13.92 | 74.12 |
| 5 | 21.15 | 78.13 |
| 7 | 24.51 | 67.19 |
| 9 | 22.69 | 89.52 |
| 11 | 19.37 | 82.58 |
| 14 | 24.48 | 77.13 |

At B=9, image F1 improves by **15.40 pp** and pixel F1 by **8.77 pp**,
with both improving in all three seeds. This is a useful illustrative
example of simultaneous image/pixel benefits.

However, fruit-jelly image anomaly recall rises from 58.89% to 87.22%,
while normal-image FPR rises from 0% to 21.67%. It is a better F1 trade-off,
not improved normal rejection. B=9 is also a post-hoc peak; preserve the
other sizes, including the image-F1 decline at B=7.

### Specificity remains a major limitation

Macro real-image FPR, the fraction of normal images predicted anomalous:

| Detector | B=3 FPR | B=14 FPR |
|---|---:|---:|
| SuperADD | 100.00% | 100.00% |
| Dinomaly | 80.93% | 74.02% |
| PatchCore | 66.67% | 66.67% |
| PaDiM | 97.22% | 100.00% |

SuperADD predicts every real-test image anomalous at every bank size.
PaDiM does so at every size above B=3. PatchCore does so for rice and
walnuts at every size; its image-level changes come from fruit jelly.
Dinomaly's image macro is below its B=3 value at every larger size.

Thus, image F1 around 80% must not be presented as classification accuracy
or sufficient evidence of useful normal/anomaly discrimination. Include
specificity/FPR when reporting these image-level results.

## 6. Synthetic evaluation is auxiliary, not a selection guarantee

Synthetic pixel macro F1:

| B | SuperADD | Dinomaly | PatchCore | PaDiM |
|---:|---:|---:|---:|---:|
| 3 | 49.61 | 37.40 | 42.00 | 21.02 |
| 5 | 51.08 | 40.49 | 43.17 | 22.50 |
| 7 | 53.09 | 41.41 | 43.40 | 22.20 |
| 9 | 52.29 | 41.75 | 42.79 | 22.82 |
| 11 | 53.07 | 41.31 | 42.96 | 22.88 |
| 14 | 52.95 | 41.37 | 42.97 | 22.66 |

The best synthetic and real sizes disagree for SuperADD, PatchCore and
PaDiM. Their agreement for Dinomaly does not establish a reliable
synthetic-only bank-selection procedure.

The synthetic evaluation set is fixed per category/seed, uses a B=14
source pool, and has calibration-disjoint host identities and distinct
synthesis seeds. Its normal negatives are the real-test normals.
Synthetic backgrounds come from normal training images, and source donors
overlap calibration donors increasingly as B grows. This is not unseen-donor
or unseen-background generalization. The real/synthetic F1 gap is not a
standalone realism or generation-quality metric.

## 7. Baselines, numerical policy and interpretation boundaries

Use the current study's B=3 control for each curve. Historical paper
references are retained as non-gating audits, not substituted baselines.
For example, historical/current B=3 pixel macros are 22.31%/22.86% for
PatchCore and 6.64%/5.88% for PaDiM. The Dinomaly companion report also
documents historical/current differences.

PaDiM's original walnuts attempts stopped at a strict raw-versus-normalized
decision audit. The explicitly approved follow-up reused the same three
saved fitted checkpoints and accepted only individually verified float32
normalization-boundary ties. Ordinary normalization, threshold selection,
prediction rules and framework metrics were not changed; unexplained
mismatches remained fatal. No additional fitting occurred.

Across 42 resumed evaluation rows, including repeats and both evaluation
sets, 13 pixel disagreements and zero image disagreements were recorded.
Every disagreement was explained by the approved predicate. The largest
raw/processed pixel F1 difference was 0.00075444 pp, below 0.001 pp; it
does not explain PaDiM's poor absolute results. Original failed attempts
remain preserved. Both feature-detector studies have identical raw hashes
for all 18 B=3 repeated real/synthetic evaluations per detector.

Post-processed AUROC can change when normalization/clipping changes even
with identical raw predictions. In PaDiM, reported real pixel AUROC changes
from approximately 81.46% at B=3 to 85.03% at larger banks. Do not frame
that as improved raw ranking or a better learned detector.

Completed verification evidence is under
`SESSION/files/flash/evaluation/ablations/results/launches/`,
where `SESSION` is the current Copilot session workspace. The feature run
`20261029_feature_bank_size_v1` contains `patchcore_export_verification.json`,
`padim_export_verification.json`, the approved-policy records and preserved
original failures. Archive this evidence with the paper artifacts rather
than relying on session-local storage alone.

## 8. Suggested paper wording

> Expanding nested defect banks from three to fourteen source images
> improved real-test pixel F1 by 10.85 percentage points for Dinomaly
> and 7.79 points for PatchCore without retraining the detectors.
> Benefits were category-dependent and non-monotonic, with substantially
> smaller effects for SuperADD and PaDiM.

Accompany this with the three-category/three-seed scope, complete curves,
same-fit qualifications and image-specificity limitations. Do not claim
statistical significance, universal improvement, a test-selected optimal
bank, or a new generator contribution from this ablation alone.

## 9. What remains to finalize

The measurements needed for these limited bank-size claims are complete.
No further detector experiments are necessary to support this assessment.
Publication preparation remains:

1. Produce the final F1-versus-B figure using all six sizes. Use category
   panels, consistent source-count labels and seed variability. Compute
   macro error bars across the three seed-level category means, not by
   averaging category standard deviations; label SD rather than CI.
2. Transfer the verified tables/figure into the manuscript and supplement.
   Lead with the predefined B=14 comparison; label best-observed points
   post-hoc and keep the original B=3 default.
3. Finalize protocol/caption disclosures: three categories, nested sources,
   fixed composite budgets, same-fit exceptions, fresh paired B=3 controls,
   synthetic-evaluation overlap, high image FPR and the PaDiM audit policy.
4. Archive/version the raw and summary CSVs, source-group manifests,
   detector recipes, code snapshots, provenance and audit evidence.
   The cwd research reports and CSVs are currently untracked; saving them
   locally does not make the paper artifacts independently reproducible.
5. Keep older failed or partial component-ablation/reproduction cells out
   of accepted tables unless separately resolved. Completion of these
   bank-size studies does not clear the older blocked PatchCore component
   reproduction or other historical research tasks.

The final figure remains pending in task tracking. This report does not
certify that the whole manuscript or every older component experiment is
camera-ready; it records the completed bank-size evidence and its limits.
No new runs, commits, pushes or changes to existing measurements were made
when writing this report.

## Source checksums

SHA-256 of the primary measured CSVs used in this report:

```text
bank_size_results.csv
8c86beda33297008d47ada2e820f5b432d038abc8476c36df9381867a43aa803

dinomaly_bank_size_results.csv
51e9a3b8d52487a04c989e4471748c0b60ca4a21e9866df5a994343c2afdba68

patchcore_bank_size_results.csv
49491d742af25793717fe9ddca061c5fe71f18a81d8952258d5bcf3dfc9c3384

padim_bank_size_results.csv
933e643b4d8717bdc9fe1326eadd20077085a030ad08fbc1a43a45a1d9e88dc5
```
