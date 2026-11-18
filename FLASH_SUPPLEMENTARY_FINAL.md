# FLASH: final supplementary selection guide and file index

Prepared 2026-10-30. Existing measurements only; no new detector experiments.

## Recommended selection, in priority order

**If space is tight, choose Tables S1, S2 and S3, plus the protocol/limitations
note below.** Add S4 or S5 for combination effects. S6 and S7 are useful
category-specific bank-size examples. S8-S10 are optional supporting analyses.

The ranking reflects relevance to the paper, not statistical significance.
It is acceptable to select informative examples, but label post-hoc examples
illustrative, keep complete results accessible, and do not imply uniform
benefits from a favourable subset. Repeated settings across tables are not
independent replications.

| Priority | Candidate | Why include it | Suggested conclusion |
|---:|---|---|---|
| 1 | S1: single-axis component counterfactuals | Directly answers whether the stages affect F1 | OBS and MRSP affect both macro F1s; other stages exhibit image/pixel trade-offs. |
| 2 | S2: complete bank-size curves and predefined endpoint | Strong new evidence across detectors, without selecting only a peak | Expanding B=3 to B=14 improves pixel F1 substantially for Dinomaly and PatchCore, but not uniformly across detectors. |
| 3 | S3: category-wise component effects | Explains why modest macro changes can hide large effects | Category-dependent gains and losses cancel in the macro average. |
| 4 | S4: localisation combinations and simplified control | Shows complete settings, interactions and a simpler internal comparator | Full FLASH exceeds the internal simplified control on mean F1, without every component being independently beneficial. |
| 5 | S5: harmonisation x blending | A complete factorial example, rather than a favourable staircase | Harmonisation's pixel-F1 effect reverses with the blending configuration. |
| 6 | S6: category-wise bank expansion | Localises the cross-model gains | Rice and fruit jelly drive the Dinomaly/PatchCore gains; walnuts changes little. |
| 7 | S7: PatchCore fruit-jelly image/pixel example | A concrete case improving both evaluation levels | B=9 improves both F1s, through an anomaly-recall/FPR trade-off, not better normal rejection. |
| 8 | S8: five-seed exploratory shortlist | Adds seed context without mixing cohorts | Colour-only OBS retains its SuperADD pixel gain, whereas the two-donor addition is not an established extra benefit. |
| 9 | S9: calibration-positive budget | Relevant if explaining the chosen positive budget | Reducing calibration positives changes image and pixel F1 differently; this is not a pure reuse ablation. |
| 10 | S10: synthetic versus real evaluation | Useful transfer diagnostic, not primary evidence | Synthetic F1 does not consistently identify the best bank on real data. |

Use the conclusion sentences as measured observations. A texture/defect-family
mechanism, universal optimal bank size, or causal explanation of component
interactions would be an **untested hypothesis**, not established by these tables.

## Shared reading rules

F1 values are percentages at synthetic-calibrated thresholds transferred to
real public-test data, not real-test-optimized F1-max or classification accuracy.
Differences are percentage points (pp), computed before rounding.
`+/-` denotes sample seed SD, not CI or significance.

Component tables use SuperADD, all eight MVTec AD 2 categories and the original
three seeds unless S8 explicitly says five. Their paired rerun baseline is
78.226401 image / 38.614184 pixel, not the historical paper's 78.127359 /
38.378047. P2 combinations are exploratory and use separate launch fits;
their baseline F1 pairs match P1, but this does not prove bit-identical
predictions between launches. Duplicate P2 baselines are not extra seeds.

Bank-size tables use rice, fruit jelly and walnuts, three seeds, and each
detector's current B=3 control. Do not mix their three-category macros with
the eight-category component macros. Dinomaly uses a 392x392 center crop
after a 448x448 resize; the other detectors use 448x448. Dinomaly, PatchCore
and PaDiM share one fit across sizes. SuperADD's B=3 uses a separate fit.
Bank size counts source images, not regions or independent backgrounds.
Source ordering, synthesis and detector RNG seeds vary together, so the
reported SD does not isolate bank-subset uncertainty from other seed effects.

## S1. Single-axis component counterfactuals

**Recommended core table.** All rows are P1, eight categories, three seeds.
OBS off removes its placement-region restriction, not host eligibility.
The blob replaces MRSP with the existing matched-area uniform-blob control;
it does not remove all mask information.

| Configuration | Pixel F1 (%) +/- SD | Delta P (pp) | Image F1 (%) +/- SD | Delta I (pp) |
|---|---:|---:|---:|---:|
| Full FLASH | 38.61 +/- 1.46 | +0.00 | 78.23 +/- 0.11 | +0.00 |
| OBS off | 36.91 +/- 1.71 | -1.71 | 73.61 +/- 1.83 | -4.62 |
| MRSP replaced by uniform blob | 37.88 +/- 1.39 | -0.74 | 75.19 +/- 1.79 | -3.04 |
| Original placement only | 37.54 +/- 0.78 | -1.07 | 78.34 +/- 0.79 | +0.11 |
| Harmonisation off | 39.64 +/- 0.76 | +1.03 | 78.23 +/- 0.11 | +0.00 |
| Alpha-only blending | 39.44 +/- 1.16 | +0.82 | 78.23 +/- 0.11 | +0.00 |

**Conclusion sentence:** "Single-axis counterfactuals show that OBS and MRSP
affect both image and pixel F1, while placement, harmonisation and blending
exhibit distinct operating-point trade-offs."

Do not claim every component is necessary or every removal lowers F1.
Image-F1 differences do not establish better normal rejection; retain the
image-threshold diagnostic in the limitations note.

**Manual source:** [all component macros](FLASH_supplementary_final/component_settings_macro.csv).
Filter `arm` to `baseline`, `obs_off`, `mrsp_uniform_blob`,
`placement_original`, `harmonisation_off`, `blend_alpha_only`.

## S2. Defect-bank-size sensitivity across four detectors

**Recommended core table/figure.** Complete real-test pixel macro curves:

| B | SuperADD | Dinomaly | PatchCore | PaDiM |
|---:|---:|---:|---:|---:|
| 3 | 58.20 | 32.77 | 22.86 | 5.88 |
| 5 | 59.12 | 39.08 | 25.75 | 6.33 |
| 7 | 59.83 | 42.71 | 29.30 | 6.42 |
| 9 | 59.58 | 44.36 | 31.36 | 5.99 |
| 11 | 60.52 | 42.53 | 30.03 | 6.04 |
| 14 | 59.48 | 43.62 | 30.65 | 6.31 |

The predefined B=14 minus B=3 changes are +1.28 pp for SuperADD,
**+10.85 pp for Dinomaly**, **+7.79 pp for PatchCore**, and +0.44 pp for
PaDiM. Dinomaly and PatchCore improve their macro in every seed.
The best observed sizes differ and all curves contain an adjacent decline.

**Conclusion sentence:** "Expanding nested banks from three to fourteen
source images improves real-test pixel F1 by 10.85 points for Dinomaly
and 7.79 points for PatchCore without retraining, with detector-dependent
and non-monotonic effects."

**V5 figure:** [full-width PDF](FLASH_supplementary_final/bank_size_pixel_f1_v5.pdf) /
[PNG preview](FLASH_supplementary_final/bank_size_pixel_f1_v5.png);
[single-column PDF](FLASH_supplementary_final/bank_size_pixel_f1_single_column_v5.pdf) /
[PNG preview](FLASH_supplementary_final/bank_size_pixel_f1_single_column_v5.png).
V5 retains the V4 means, colours, selective axis breaks, and absence of SD
shading. PaDiM's y-axis now ends at 16, every x-axis reads "Bank size", and
vertical gaps between rows and above the legend are slightly reduced.
All model panels remain equal-size with horizontally aligned titles.
V1–V4 files are preserved unchanged. The caption below also applies to V5.

**Previous V4 figure (V1–V3 are preserved):** [full-width PDF](FLASH_supplementary_final/bank_size_pixel_f1_v4.pdf) /
[PNG preview](FLASH_supplementary_final/bank_size_pixel_f1_v4.png);
[single-column PDF](FLASH_supplementary_final/bank_size_pixel_f1_single_column_v4.pdf) /
[PNG preview](FLASH_supplementary_final/bank_size_pixel_f1_single_column_v4.png).
V4 removes the shaded sample-SD bands; category lines and markers still show
the mean across three seeds. It retains equal-size panels, aligned model-name
titles, labels on both axes, and one shared legend. The larger breaks use
category-mean gaps: SuperADD 58.85–65.60, PatchCore 25.05–48.30, and PaDiM
3.25–13.80. Each break leaves at least 0.5 F1 points from the closest mean,
and no connected category-mean line crosses the omitted interval. Dinomaly
remains continuous because its category means overlap.

The full-width layout is the recommended paper candidate. The single-column
version is included to assess legibility at final print size. Do not compare
absolute category/detector heights as if all panels shared one y scale; the
axis ranges are independent. Previous V3 figures remain available unchanged:
[full-width PDF](FLASH_supplementary_final/bank_size_pixel_f1_v3.pdf) /
[PNG](FLASH_supplementary_final/bank_size_pixel_f1_v3.png). Previous V2 figures
remain available unchanged:
[full-width PDF](FLASH_supplementary_final/bank_size_pixel_f1_v2.pdf) /
[PNG](FLASH_supplementary_final/bank_size_pixel_f1_v2.png). Previous V1 figures
also remain available unchanged:
[full-width PDF](FLASH_supplementary_final/bank_size_pixel_f1.pdf) /
[PNG](FLASH_supplementary_final/bank_size_pixel_f1.png).

**Ready V4 caption:** "Defect-bank-size sensitivity on three MVTec AD 2
categories. Markers show mean transferred-threshold real-test pixel F1
over three seeds, connected within category across bank sizes; variance
shading is omitted. Intel brand colors identify categories. Y-axis ranges
are detector-specific; diagonal marks denote omitted, mean-free score
intervals in SuperADD, PatchCore and PaDiM. Dinomaly has no break because
its category means overlap. Each bank contains the smaller banks' sources; calibration-positive
budgets remain fixed. Dinomaly, PatchCore and PaDiM share one fitted detector
across sizes per category/seed; SuperADD uses a separate B=3 fit. Detector
preprocessing differs, so panels assess within-detector sensitivity rather
than a matched detector-ranking benchmark."

Historical Oracle/reference lines are intentionally omitted: they are not
current same-fit controls. No Oracle was recomputed. Keep B=3 as the original
default; do not promote a test-selected peak to a validated setting.

**Manual sources:** [macro](FLASH_supplementary_final/bank_size_all_models_macro.csv) /
[category](FLASH_supplementary_final/bank_size_all_models_by_category.csv).
Filter `eval_set=real_test`; synthetic rows are separate.

## S3. Component effects hidden by the macro average

**Recommended companion to S1.** All eight categories, three seeds.
Cells show absolute pixel F1 followed by change from that category's full
baseline in parentheses. This selected four-row view retains categories
that reverse the favourable trend.

| Configuration | can | fabric | fruit_jelly | rice | sheet_metal | vial | wallplugs | walnuts | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full FLASH | 0.00 | 69.12 | 55.39 | 52.55 | 7.76 | 38.91 | 18.51 | 66.67 | 38.61 |
| OBS off | 0.00 (+0.00) | 52.68 (-16.44) | 30.51 (-24.88) | 49.78 (-2.78) | 30.26 (+22.50) | 15.74 (-23.17) | 46.12 (+27.60) | 70.17 (+3.51) | 36.91 |
| Uniform blob | 0.00 (+0.00) | 55.39 (-13.73) | 55.00 (-0.39) | 45.30 (-7.26) | 13.74 (+5.98) | 41.52 (+2.60) | 24.11 (+5.60) | 67.96 (+1.29) | 37.88 |
| Colour-only OBS | 0.00 (+0.00) | 77.95 (+8.83) | 55.80 (+0.40) | 58.43 (+5.88) | 25.80 (+18.04) | 42.08 (+3.17) | 10.21 (-8.31) | 64.60 (-2.06) | 41.86 |

**Conclusion sentence:** "Category-wise analysis reveals substantial
component sensitivity concealed by macro averaging, including opposite
pixel-F1 effects across categories."

For example, OBS removal loses 24.88 pp on fruit jelly but gains 27.60 pp
on wallplugs, producing only a -1.71 pp macro change. Colour-only OBS
gains 18.04 pp on sheet metal but loses 8.31 pp on wallplugs.

**Final heatmap:** [PDF](FLASH_supplementary_final/component_category_pixel_deltas.pdf) /
[PNG preview](FLASH_supplementary_final/component_category_pixel_deltas.png).
It includes these three changes plus harmonisation-off and alpha-only,
across all eight categories.

**Ready caption:** "Pixel-F1 changes from the paired full-FLASH baseline
for five selected single-axis counterfactuals, averaged over three seeds.
All eight categories are retained. Opposing category effects explain why
small macro changes do not imply inactive components. Image outcomes and
the complete settings remain available in the accompanying tables."

Do not turn this into an established texture/object taxonomy. Fabric is
one textile-category observation, not replication across textile families.

**Manual source:** [all category means](FLASH_supplementary_final/component_settings_by_category.csv).
Absolute image F1 and both signed deltas are included there.

## S4. Localisation-prior combinations and an internal simplified control

**Useful alternative to S1 if settings/combination structure is the priority.**
OBS `Frame` means no OBS region restriction; `C+T` means full colour/texture.
`H` is CIELAB harmonisation. The middle four rows form a complete
OBS-region x MRSP/blob comparison; the first and last provide context.

| Configuration | OBS | Mask | Placement | H | Blend | Pixel F1 | Image F1 |
|---|---|---|---|---|---|---:|---:|
| Internal simplified control | Frame | Blob | Original | Off | Alpha | 34.07 | 72.88 |
| OBS off + blob | Frame | Blob | Mixed | On | Hybrid | 35.67 | 74.36 |
| OBS off | Frame | MRSP | Mixed | On | Hybrid | 36.91 | 73.61 |
| Uniform blob | C+T | Blob | Mixed | On | Hybrid | 37.88 | 75.19 |
| Full FLASH | C+T | MRSP | Mixed | On | Hybrid | 38.61 | 78.23 |
| Colour-only OBS | C | MRSP | Mixed | On | Hybrid | 41.86 | 78.15 |

**Conclusion sentence:** "The full configuration exceeds the internal
simplified control by 4.54 points in pixel F1 and 5.34 points in image F1,
while the localisation and cue comparisons reveal non-uniform component effects."

This is not a cumulative one-component staircase. The simplified control
changes several stages together and retains the donor bank, host policy,
resizing and augmentation; it is **not a reproduced external CutPaste method**.
P2 rows are exploratory. Use the original P2 control 34.07, not its separate
audit replay 34.03. Do not count S1/S4 repeated rows as extra evidence.

**Manual sources:** component macro/category CSVs; filter the six `arm`
identifiers shown in [the complete settings alternatives](FLASH_ablation_table_options.md).

## S5. Harmonisation interacts with blending

**Compact complete factorial table**, not only individually favourable rows.
Three seeds, eight categories; the no-H/alpha combination is exploratory.

| Harmonisation | Blending | Pixel F1 | Delta P (pp) | Image F1 |
|---|---|---:|---:|---:|
| On | Hybrid | 38.61 | +0.00 | 78.23 |
| Off | Hybrid | 39.64 | +1.03 | 78.23 |
| On | Alpha | 39.44 | +0.82 | 78.23 |
| Off | Alpha | 38.01 | -0.61 | 78.21 |

Adding harmonisation changes pixel F1 by -1.03 pp under hybrid blending
but +1.43 pp under alpha blending. The descriptive interaction between
the two modifications is -2.46 pp; this is not a significance test.

**Conclusion sentence:** "The pixel-F1 effect of harmonisation reverses
with the blending configuration, so individually favourable modifications
do not combine additively."

Do not claim this proves a particular visual mechanism or that the original
hybrid setting maximizes F1. The source rows and their SDs are in
`component_settings_macro.csv`: `baseline`, `harmonisation_off`,
`blend_alpha_only`, `combo_harmonisation_off_alpha_only`.

## S6. Where the bank-size improvements come from

Three categories, three seeds; predefined B=3 versus B=14.

| Category | Dinomaly B=3 | Dinomaly B=14 | Delta P | PatchCore B=3 | PatchCore B=14 | Delta P |
|---|---:|---:|---:|---:|---:|---:|
| Rice | 17.93 | 44.43 | +26.49 | 5.16 | 17.93 | +12.78 |
| Fruit jelly | 34.38 | 40.14 | +5.76 | 13.92 | 24.48 | +10.56 |
| Walnuts | 46.01 | 46.30 | +0.29 | 49.51 | 49.54 | +0.04 |

**Conclusion sentence:** "The larger-bank benefit is concentrated in rice
and fruit jelly for both Dinomaly and PatchCore, while walnuts is nearly
unchanged."

Keep all three categories, not only the two large improvements. Source
quantity changes together with source composition, so this is not a pure
quantity-independent-of-quality experiment. The S2 figure already shows
the complete category curves; this table is a compact numerical explanation.

**Manual source:** [bank category means](FLASH_supplementary_final/bank_size_all_models_by_category.csv),
filtered to the two models, `eval_set=real_test` and B=3/14.

## S7. An illustrative simultaneous image/pixel improvement

**Post-hoc example:** PatchCore on fruit jelly, complete size curve.

| B | Pixel F1 | Image F1 | Normal-image FPR (%) |
|---:|---:|---:|---:|
| 3 | 13.92 | 74.12 | 0.00 |
| 5 | 21.15 | 78.13 | 33.33 |
| 7 | 24.51 | 67.19 | 0.00 |
| 9 | 22.69 | 89.52 | 21.67 |
| 11 | 19.37 | 82.58 | 1.67 |
| 14 | 24.48 | 77.13 | 0.00 |

**Conclusion sentence:** "For the illustrative fruit-jelly/PatchCore case,
B=9 improves pixel F1 by 8.77 points and image F1 by 15.40 points in
three-seed means, with both directions positive in every seed."

Immediately qualify this: anomaly recall increases from 58.89% to 87.22%,
but normal FPR increases from 0% to 21.67%. This is an F1 trade-off,
not better normal rejection, and B=9 is not an independently validated optimum.

**Final image figure:** [PDF](FLASH_supplementary_final/bank_size_image_f1.pdf) /
[PNG preview](FLASH_supplementary_final/bank_size_image_f1.png).
It preserves the other categories/detectors and all measured sizes.

**Ready caption:** "Real-test image F1 versus source-bank size, shown for
all measured categories and detectors. Curves are three-seed means with
one sample SD. F1 is not classification accuracy: several flat curves
are all-anomalous predictions. See the accompanying FPR/protocol note
before interpreting apparent image-level performance."

**Manual sources:** bank category CSV for mean/SD; exact FPR/recall counts
are in [the PatchCore raw results](patchcore_bank_size_results.csv).

## S8. Optional: five-seed exploratory stability

All rows use the same five physical generation seeds, all eight categories.
Three decimals are retained to avoid double-rounding the colour-only result.
Audit replays and repeated controls are not extra seeds.

| Configuration | Pixel F1 +/- SD | Delta P (pp) | Image F1 +/- SD | Delta I (pp) |
|---|---:|---:|---:|---:|
| Full FLASH | 38.848 +/- 1.080 | +0.000 | 78.217 +/- 0.123 | +0.000 |
| Colour-only OBS | 41.995 +/- 0.945 | +3.146 | 78.131 +/- 0.111 | -0.085 |
| Colour OBS + two sources | 41.644 +/- 1.703 | +2.795 | 78.210 +/- 0.091 | -0.007 |
| Internal simplified control | 33.468 +/- 2.588 | -5.380 | 72.928 +/- 0.667 | -5.289 |

**Conclusion sentence:** "The exploratory five-seed shortlist retains the
SuperADD pixel gain from colour-only OBS, but does not establish an
additional benefit from reducing the source bank to two images."

This is not untouched confirmation: the shortlist was selected after
inspection. The legacy follow-up also retains documented raw-fingerprint
drift, tiny repeated-F1 differences and permitted unchanged metric-error
retries. Its four-arm audit is not an independent audit of every P1/P2
setting. Do not present this as exact numerical reproducibility or universal
cross-detector superiority. The older PatchCore component comparison is a
separate exploratory record, not a passed later paper-reproduction gate.

**Manual sources:** [five-seed macro](FLASH_supplementary_final/five_seed_shortlist_macro.csv) /
[five-seed category](FLASH_supplementary_final/five_seed_shortlist_by_category.csv).
The historical audit and limitations are in
[the consolidated component record, sections 6-8](FLASH_ablation_results_report.md).

## S9. Optional: calibration-positive budget

This changes positive coverage and calibration class balance, not just
reuse frequency or the number of source images.

| Positive-image budget per category | Pixel F1 | Image F1 |
|---|---:|---:|
| Original category budget, 60-105 | 38.61 | 78.23 |
| Eight unique positives | 35.92 | 65.13 |
| Sixteen unique positives | 39.73 | 70.01 |

**Conclusion sentence:** "Calibration-positive budget affects image and
pixel F1 differently; the sixteen-positive setting improves pixel F1
slightly but reduces image F1 relative to the original budget."

Do not call this a pure reuse ablation or use it to establish a universal
minimum budget. Source: component CSVs, `baseline`, `reuse_8`, `reuse_16`.

## S10. Optional: synthetic/real transfer diagnostic

Pixel macro F1; both evaluation sets are kept separate.

| B | Dinomaly real | Dinomaly synthetic | PatchCore real | PatchCore synthetic |
|---:|---:|---:|---:|---:|
| 3 | 32.77 | 37.40 | 22.86 | 42.00 |
| 5 | 39.08 | 40.49 | 25.75 | 43.17 |
| 7 | 42.71 | 41.41 | 29.30 | 43.40 |
| 9 | 44.36 | 41.75 | 31.36 | 42.79 |
| 11 | 42.53 | 41.31 | 30.03 | 42.96 |
| 14 | 43.62 | 41.37 | 30.65 | 42.97 |

**Conclusion sentence:** "Synthetic evaluation and real transfer need not
prefer the same bank size, so auxiliary synthetic F1 is not by itself a
validated real-data bank-selection criterion."

Synthetic source donors increasingly overlap calibration as B grows,
and backgrounds come from normal training images. This is not unseen-donor
or unseen-background generalization. Show complete real/synthetic curves
for all models in the full CSV; the two displayed detectors are selected
examples, not the complete experiment.

## Mandatory protocol and limitations note

Include this information even when selecting the positive examples:

- Image F1 is not accuracy. The three-seed component audit baseline has
  95.89% macro normal-image FPR and all-positive predictions in 21/24 runs;
  an always-anomalous predictor reaches 79.42% macro image F1 on that split.
  Full FLASH's 78.23% image F1 is not evidence that image calibration is solved.
- For bank-size runs, SuperADD is all-positive at every size and PaDiM at
  every size above B=3. PatchCore is all-positive on rice and walnuts.
  Include FPR/recall or point to the full diagnostic, not only high F1.
- Thresholds use native-normal/synthetic-positive calibration, not test-label
  F1 optimization. Positive budgets nevertheless match known public-test
  anomaly counts: disclose the test-informed budgeting.
- Components and selected combinations use eight categories; expanded-bank
  studies use three. Do not substitute historical published baselines,
  mix seed cohorts, pool category pixels as if they were macro means,
  or relabel already-inspected categories as untouched validation.
- The P2 colour-only + no-harmonisation / vial / seed1 pixel AUROC is invalid
  (7563479.0). It is preserved in the numerical source and flagged in the
  browseable summaries. Do not publish that AUROC, cap it, repair it
  silently, or claim every stored metric is validated because F1 is finite.
- Legacy component audits retain six raw-fingerprint differences and three
  non-identical repeated baselines. Their independent confusion checks apply
  to the four audited arms, not automatically to all 22 configurations.
- PaDiM's completed bank-size follow-up uses an explicitly approved
  rounding-only audit exception with ordinary prediction rules unchanged,
  original fitted checkpoints reused and zero additional fits. Across
  42 resumed real/synthetic/repeat rows, 13 pixel ties were explained and
  no image mismatch occurred; maximum raw/processed pixel-F1 difference
  was below 0.001 pp. Original failed attempts remain preserved.
- None of these tables clears older failed paper-reproduction gates.
  The legacy PatchCore component CSV and the new PatchCore bank-size
  results are different studies with different scopes and baselines.
- Raw predictions remain identical across bank sizes in the new shared-fit
  studies. Changes in processed AUROC can reflect score normalization and
  clipping; do not claim improved learned representations or raw-score ranking.

## Final files to inspect manually

All paths below are relative to the anomalib cwd. Start with the generated
six spreadsheets; the complete original sources are archived alongside them.
For bank CSVs, filter `eval_set=real_test` before comparing real performance.

### A. Most useful final spreadsheets

| File | Rows | What to inspect / hand-pick |
|---|---:|---|
| [component_settings_macro.csv](FLASH_supplementary_final/component_settings_macro.csv) | 22 | All original component and combination settings; both F1s, SDs, deltas and exact settings. Best place to choose S1/S4/S5/S9 rows. |
| [component_settings_by_category.csv](FLASH_supplementary_final/component_settings_by_category.csv) | 176 | All 22 settings x eight categories; absolute F1s and paired deltas. Best place for category examples and S3. |
| [bank_size_all_models_macro.csv](FLASH_supplementary_final/bank_size_all_models_macro.csv) | 48 | Four models x six sizes x two evaluation sets; seed-level macro SD and changes from each model's B=3. |
| [bank_size_all_models_by_category.csv](FLASH_supplementary_final/bank_size_all_models_by_category.csv) | 144 | Four models x three categories x six sizes x two sets; both F1s, seed SD and B=3 deltas. Best place for S6/S7. |
| [five_seed_shortlist_macro.csv](FLASH_supplementary_final/five_seed_shortlist_macro.csv) | 4 | The four exploratory settings using all five seeds consistently; do not mix these means with three-seed baselines. |
| [five_seed_shortlist_by_category.csv](FLASH_supplementary_final/five_seed_shortlist_by_category.csv) | 32 | The same four settings across all eight categories, with five-seed means/SD/deltas. |

Each derived component row identifies its cohort and exploratory/budget
status. An `invalid_source_pixel_auroc` flag marks the affected source
without changing its F1 measurements. The frozen original CSVs remain intact.

### B. Final figures

| Figure | Publication/vector file | Quick preview |
|---|---|---|
| Bank-size pixel F1 V5, full-width 2x2, compact spacing and PaDiM capped at 16 | [bank_size_pixel_f1_v5.pdf](FLASH_supplementary_final/bank_size_pixel_f1_v5.pdf) | [PNG](FLASH_supplementary_final/bank_size_pixel_f1_v5.png) |
| Bank-size pixel F1 V5, single-column 2x2 | [bank_size_pixel_f1_single_column_v5.pdf](FLASH_supplementary_final/bank_size_pixel_f1_single_column_v5.pdf) | [PNG](FLASH_supplementary_final/bank_size_pixel_f1_single_column_v5.png) |
| Bank-size pixel F1 V4, full-width 2x2, no SD shading, enlarged selective breaks | [bank_size_pixel_f1_v4.pdf](FLASH_supplementary_final/bank_size_pixel_f1_v4.pdf) | [PNG](FLASH_supplementary_final/bank_size_pixel_f1_v4.png) |
| Bank-size pixel F1 V4, single-column 2x2 | [bank_size_pixel_f1_single_column_v4.pdf](FLASH_supplementary_final/bank_size_pixel_f1_single_column_v4.pdf) | [PNG](FLASH_supplementary_final/bank_size_pixel_f1_single_column_v4.png) |
| Previous V3 full-width pixel figure, preserved unchanged | [bank_size_pixel_f1_v3.pdf](FLASH_supplementary_final/bank_size_pixel_f1_v3.pdf) | [PNG](FLASH_supplementary_final/bank_size_pixel_f1_v3.png) |
| Previous V3 single-column pixel figure, preserved unchanged | [bank_size_pixel_f1_single_column_v3.pdf](FLASH_supplementary_final/bank_size_pixel_f1_single_column_v3.pdf) | [PNG](FLASH_supplementary_final/bank_size_pixel_f1_single_column_v3.png) |
| Previous V2 full-width pixel figure, preserved unchanged | [bank_size_pixel_f1_v2.pdf](FLASH_supplementary_final/bank_size_pixel_f1_v2.pdf) | [PNG](FLASH_supplementary_final/bank_size_pixel_f1_v2.png) |
| Previous V2 single-column pixel figure, preserved unchanged | [bank_size_pixel_f1_single_column_v2.pdf](FLASH_supplementary_final/bank_size_pixel_f1_single_column_v2.pdf) | [PNG](FLASH_supplementary_final/bank_size_pixel_f1_single_column_v2.png) |
| Previous V1 full-width pixel figure, preserved unchanged | [bank_size_pixel_f1.pdf](FLASH_supplementary_final/bank_size_pixel_f1.pdf) | [PNG](FLASH_supplementary_final/bank_size_pixel_f1.png) |
| Previous V1 single-column pixel figure, preserved unchanged | [bank_size_pixel_f1_single_column.pdf](FLASH_supplementary_final/bank_size_pixel_f1_single_column.pdf) | [PNG](FLASH_supplementary_final/bank_size_pixel_f1_single_column.png) |
| Bank-size image F1, same complete layout | [bank_size_image_f1.pdf](FLASH_supplementary_final/bank_size_image_f1.pdf) | [PNG](FLASH_supplementary_final/bank_size_image_f1.png) |
| Component category-effect heatmap | [component_category_pixel_deltas.pdf](FLASH_supplementary_final/component_category_pixel_deltas.pdf) | [PNG](FLASH_supplementary_final/component_category_pixel_deltas.png) |

### C. Detailed readable analyses

| File | Purpose / authority |
|---|---|
| [FLASH_SUPPLEMENTARY_FINAL.md](FLASH_SUPPLEMENTARY_FINAL.md) | This ranked selection, ready tables, conclusion sentences, captions and final file index. |
| [FLASH_cross_model_defect_bank_size_analysis.md](FLASH_cross_model_defect_bank_size_analysis.md) | Full bank-size interpretation, image FPR, synthetic transfer, historical/current baseline differences and PaDiM numerical policy. |
| [FLASH_defect_bank_size_analysis.md](FLASH_defect_bank_size_analysis.md) | Detailed SuperADD bank-size analysis and seed/category effects. |
| [FLASH_dinomaly_defect_bank_size_analysis.md](FLASH_dinomaly_defect_bank_size_analysis.md) | Detailed Dinomaly bank-size analysis, threshold/precision/recall mechanism and paper-reference audit. |
| [FLASH_ablation_table_options.md](FLASH_ablation_table_options.md) | Five earlier six-row table alternatives, including handpicked paired category examples and settings matrices. |
| [FLASH_ablation_results_report.md](FLASH_ablation_results_report.md) | Frozen 28 October component/combination/follow-up record, including all 832 original CSV rows. Its statements that bank-size work had not started are historical, not current status. |
| [component_ablation_supplement.md](component_ablation_supplement.md) | Existing complete three-seed component/combination tables and image diagnostics; not the five-seed or new bank-size package. |
| [component_ablation_supplement.tex](component_ablation_supplement.tex) | Existing components-only LaTeX starting point. It does not contain the new bank-size figures/tables. |
| [PATCHCORE_ABLATION_PLAN.md](PATCHCORE_ABLATION_PLAN.md) | Historical strict-gate component plan, not evidence that that gate passed or a source for the completed bank-study results. |

### D. Original cwd result files

| Original file pair | Content |
|---|---|
| [phase1_component_f1_macro.csv](phase1_component_f1_macro.csv) / [phase1_component_f1_by_category.csv](phase1_component_f1_by_category.csv) | P1 only: 16 macro / 128 category rows, without the six P2 combinations. |
| [bank_size_results.csv](bank_size_results.csv) / [bank_size_f1_summary.csv](bank_size_f1_summary.csv) | SuperADD: 108 primary raw rows / 36 category-summary rows. |
| [dinomaly_bank_size_results.csv](dinomaly_bank_size_results.csv) / [dinomaly_bank_size_f1_summary.csv](dinomaly_bank_size_f1_summary.csv) | Dinomaly: 108 primary raw rows / 36 category-summary rows. |
| [patchcore_bank_size_results.csv](patchcore_bank_size_results.csv) / [patchcore_bank_size_f1_summary.csv](patchcore_bank_size_f1_summary.csv) | PatchCore bank study: 108 primary raw rows / 36 category-summary rows. |
| [padim_bank_size_results.csv](padim_bank_size_results.csv) / [padim_bank_size_f1_summary.csv](padim_bank_size_f1_summary.csv) | PaDiM bank study: 108 primary raw rows / 36 category-summary rows, with audit-policy metadata. |

### E. Frozen source data and reproduction support

Under `FLASH_supplementary_final/sources/`:

- `component_p1.csv`: 384 original single-axis/budget records.
- `component_p2.csv`: 168 original combination/control records, including
  24 duplicate baseline checks that are not double-counted in final summaries.
- `component_extra_seeds.csv`: 64 physical seed3/4 records, used only
  for the four five-seed shortlist settings.
- `component_audit.csv`: 120 diagnostic passes; not extra seeds.
- `legacy_patchcore_components.csv`: 96 older exploratory component
  records; not the new bank study or a passed strict reproduction gate.
- `bank_superadd.csv`, `bank_dinomaly.csv`, `bank_patchcore.csv`,
  `bank_padim.csv`: exact copies of the four 108-row bank result sources.
- `component_specs.py`: archived component-settings definitions.
- `protocol_checks/`: archived numerical verification/protocol records,
  including the intentionally preserved original feature-launch failure
  and the approved PaDiM completion evidence.

The small archived protocol records are:

| Record under `FLASH_supplementary_final/sources/protocol_checks/` | Purpose |
|---|---|
| [padim_export_verification.json](FLASH_supplementary_final/sources/protocol_checks/padim_export_verification.json) | Complete PaDiM matrix/export checks and rounding-disagreement counts. |
| [padim_approved_protocol.json](FLASH_supplementary_final/sources/protocol_checks/padim_approved_protocol.json) | Explicitly approved checkpoint-only, rounding-boundary exception. |
| [padim_checkpoint_evaluation_provenance.json](FLASH_supplementary_final/sources/protocol_checks/padim_checkpoint_evaluation_provenance.json) | Original fit/checkpoint identities and successor evaluation provenance. |
| [patchcore_export_verification.json](FLASH_supplementary_final/sources/protocol_checks/patchcore_export_verification.json) | Complete new PatchCore bank-study checks; not the historical component gate. |
| [dinomaly_export_verification.json](FLASH_supplementary_final/sources/protocol_checks/dinomaly_export_verification.json) | Dinomaly matrix/export and repeat-control checks. |
| [original_feature_launch_failed_status.json](FLASH_supplementary_final/sources/protocol_checks/original_feature_launch_failed_status.json) | Preserved initial failure; later approved completions do not rewrite it. |
| [legacy_followup_completion.json](FLASH_supplementary_final/sources/protocol_checks/legacy_followup_completion.json) | Older four-arm audit, extra-seed and exploratory PatchCore follow-up coverage. |

[manifest.json](FLASH_supplementary_final/manifest.json) records source/output
SHA-256 hashes, row counts, source warnings, audit context and plotting
versions. [verification.json](FLASH_supplementary_final/verification.json)
records independent raw-data recomputation of the 426 derived rows, all ten
candidate tables, unchanged numerical sources and file links, with the
guide/manifest hashes. This verifies the presentation package, not every
historical experiment or ranking metric.

[build_assets.py](FLASH_supplementary_final/build_assets.py)
rebuilds the six spreadsheets and three PDF/PNG figure pairs from the
archived numerical sources:

```bash
uv run --no-sync python FLASH_supplementary_final/build_assets.py
```

This rebuilds presentation assets only; it does not fit detectors or
regenerate synthetic images. Model checkpoints, generated corpora and the
complete experiment code histories are not bundled in this small package.
Preserve the original research archive separately if publishing full
end-to-end reproduction materials.

## What remains after this package

The result selection guide, complete browseable tables and final figure
assets are now prepared. No new experiments are needed for the limited
claims above. Remaining author work is to choose the subset that fits the
supplement, insert tables/PDFs and their qualifications into the manuscript,
and version/archive the selected materials. Do not interpret this package
as a review or completion certificate for the whole manuscript.

The original paper/default configuration is not changed. No measurements
were replaced, failed gates cleared, commits created or pushes performed.
