## G.2. Defect-bank-size ablation

**Experimental protocol.** We assess how the number of synthetic source images
in the defect bank affects threshold transfer on three MVTec AD 2 categories:
rice, fruit jelly, and walnuts. Using the construction described in
Section G.1, we evaluate bank sizes B = 3, 5, 7, 9, 11, and 14 across three
seeds. Each seed defines a nested expansion retaining the original three
sources. B counts source images, not extracted regions or independent
backgrounds; all regions from each selected source are retained.

The number of synthetic calibration anomalies is fixed at 90, 60, and 90
for rice, fruit jelly, and walnuts, respectively, with 35, 37, and 48 native
validation normals. Thus, bank expansion does not increase calibration-set
size. Image and pixel thresholds independently maximize calibration F1 and
are transferred unchanged to the real public-test set. Thresholds are not
optimized on real-test labels; however, the positive budgets match the
benchmark anomaly counts and are therefore test-informed. Reported F1
averages seeds within categories, then equally weights the three categories.

We evaluate SuperADD, Dinomaly, PatchCore, and PaDiM. Within each
category/seed, Dinomaly, PatchCore, and PaDiM share one normal-trained fit
across all bank sizes. Dinomaly uses 1,000 optimizer steps; PatchCore and
PaDiM use one complete normal-training feature pass. SuperADD uses a
separate B=3 fit and one shared fit for the larger banks. Dinomaly uses a
392 x 392 center crop after resizing to 448 x 448; the other detectors use
448 x 448 inputs. Comparisons therefore concern within-detector sensitivity,
not matched cross-detector ranking.

**Results.** Figure G.X shows category-wise pixel F1. For the predefined
B=3-to-14 comparison, macro pixel F1 increases most for Dinomaly (+10.85
percentage points, pp) and PatchCore (+7.79 pp), with positive macro changes
in all three seeds for both detectors. SuperADD and PaDiM improve by
1.28 and 0.44 pp, respectively.

| Detector | Pixel F1, B=3 (%) | Pixel F1, B=14 (%) | Change (pp) | Image F1, B=3 / B=14 (%) |
|---|---:|---:|---:|---:|
| SuperADD | 58.20 | 59.48 | +1.28 | 80.60 / 80.60 |
| Dinomaly | 32.77 | 43.62 | +10.85 | 82.82 / 82.08 |
| PatchCore | 22.86 | 30.65 | +7.79 | 76.73 / 77.74 |
| PaDiM | 5.88 | 6.31 | +0.44 | 80.77 / 80.60 |

Gains are category-dependent: rice improves by 26.49 pp for Dinomaly and
12.78 pp for PatchCore, whereas walnuts changes by only 0.29 and 0.04 pp.
The curves are non-monotonic. The highest observed macro pixel F1 occurs at
B=9 for Dinomaly (44.36%) and PatchCore (31.36%), B=11 for SuperADD
(60.52%), and B=7 for PaDiM (6.42%). These test-observed peaks do not
establish optimal bank sizes or asymptotic saturation. Image-level benefits
are inconsistent; SuperADD predicts every test image as anomalous at every
size, as does PaDiM above B=3. Their approximately 80% image F1 therefore
does not imply useful normal-image rejection.

**Interpretation.** Identical raw real-test predictions across sizes for
the three same-fit detectors localize the changes to calibration and
post-processing, rather than representation learning. We hypothesize that
additional sources can improve synthetic coverage relevant to threshold
transfer, with diminishing or adverse effects when their score distributions
are less representative. Bank size changes source composition and reuse
jointly, so this mechanism is not isolated. Three seeds provide descriptive
consistency, not evidence of statistical significance.

[Figure G.X placeholder: insert `bank_size_pixel_f1_v4.pdf` (V4, full-width).]

**Figure G.X.** Defect-bank-size sensitivity of transferred real-test pixel
F1. Lines connect category-wise means over three seeds; variance shading is
omitted. Panels use detector-specific y scales. Diagonal marks indicate
omitted intervals containing no category-mean line.
