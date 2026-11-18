# FLASH: five compact ablation-table options

**For camera-ready review, 29 October 2026. Existing measurements only; no new experiments.**

## What we can already show

**The components do change F1. Some dataset averages are modest because large,
opposite category effects cancel.** For example, disabling OBS changes pixel F1
by -16.44 pp on fabric, -24.88 pp on fruit_jelly and +27.60 pp on wallplugs,
but only -1.71 pp in the eight-category average. That is evidence of sensitivity
and trade-offs, not evidence that OBS does nothing.

Below are five **deliberately selected, post-hoc** alternatives, each with six
rows. Selection is explicit: A covers single-axis changes, B a complete
harmonisation/blending interaction, C a complete localisation-prior interaction
with pipeline controls, D category heterogeneity, and E illustrative large paired
effects. They are not five independent studies, and repeated rows are not extra
evidence. The full results, including unselected configurations, remain in
[the consolidated report](FLASH_ablation_results_report.md).

**My preferred choice:** C for one compact settings/results table, with D for
category dependence. Add B if the supplementary material needs a specific
combination-interaction example. A is the clearest alternative when the reviewer
wants one-at-a-time component changes. E is a narrower illustrative table, not a
replacement for the all-category evidence in D.

## Relation to Dinomaly Table 3

[Dinomaly Table 3 and Section 4.4][dinomaly] use component-setting columns followed
by image- and pixel-level results, covering individual elements and combinations.
Importantly, their discussion also distinguishes standalone from
combination-dependent benefits: an element need not improve every setting alone
to be worth analysing.

We follow that **settings plus measured outcomes** structure, not its metric
protocol or a requirement to construct a monotonically improving staircase.
Dinomaly reports F1-max on MVTec-AD; these FLASH tables report
**synthetic-calibrated, transferred-threshold F1 on MVTec AD 2**. The values are
not directly comparable across the two papers.

## Shared protocol and notation

- Detector: **SuperADD**, `vit_huge_plus_patch16_dinov3`, 448 x 448 inputs.
- Every table below uses the **original three generation seeds, 0/1/2**.
  Model/data RNG uses generation seed + 1. Within each launch, arms for a
  category/seed share one fitted detector. P1 and P2 use separate fits with the
  same settings/seeds and exactly matching baseline F1 pairs; that agreement
  does not establish bitwise-identical predictions across launches. The
  interventions change calibration data, not detector training.
- Each category is averaged over its three seeds; dataset means weight all eight
  categories equally. Category cells are not pooled-pixel or pooled-image F1.
- Separate image/pixel thresholds maximize F1 on native validation normals plus
  synthetic positives and transfer unchanged to real public-test data.
- F1 is in **percent**. `dP` and `dI` are signed **percentage-point** differences
  from the paired full-FLASH baseline. Differences are computed before rounding.
  `+/-` is the sample SD of three seed-level dataset macros, not a confidence
  interval or a significance test.
- `C+T` means full colour-and-texture OBS; `C` means colour-only OBS;
  `Frame` means the OBS placement-region restriction is disabled. The original
  host-eligibility policy is retained across arms. `MRSP` means the original object-relative MRSP;
  `Blob` is the existing matched-area uniform-blob replacement, not no mask.
  `H` means CIELAB harmonisation. `Mixed` means the original mixed placement
  policy. Every displayed arm uses three source donors per category.
- These are curated diagnostic comparisons, not a new selected method default.
  All six original combination outcomes remain in the full report.

## Option A: one-at-a-time component counterfactuals

**Question:** Does changing one component affect image F1, pixel F1, or both?

**Why these rows:** full FLASH plus five single-axis replacements/removals:
OBS, MRSP, placement, harmonisation and blending. This deliberately includes
changes that improve F1, rather than only removals that favour the full method.

| Configuration | OBS | Mask | Placement | H | Blend | Pixel F1 (%) +/- SD | dP (pp) | Image F1 (%) +/- SD | dI (pp) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| Full FLASH (`baseline`) | C+T | MRSP | Mixed | On | Hybrid | 38.61 +/- 1.46 | +0.00 | 78.23 +/- 0.11 | +0.00 |
| OBS off (`obs_off`) | Frame | MRSP | Mixed | On | Hybrid | 36.91 +/- 1.71 | -1.71 | 73.61 +/- 1.83 | -4.62 |
| Uniform blob (`mrsp_uniform_blob`) | C+T | Blob | Mixed | On | Hybrid | 37.88 +/- 1.39 | -0.74 | 75.19 +/- 1.79 | -3.04 |
| Original placement only (`placement_original`) | C+T | MRSP | Original | On | Hybrid | 37.54 +/- 0.78 | -1.07 | 78.34 +/- 0.79 | +0.11 |
| Harmonisation off (`harmonisation_off`) | C+T | MRSP | Mixed | Off | Hybrid | 39.64 +/- 0.76 | +1.03 | 78.23 +/- 0.11 | +0.00 |
| Alpha only (`blend_alpha_only`) | C+T | MRSP | Mixed | On | Alpha | 39.44 +/- 1.16 | +0.82 | 78.23 +/- 0.11 | +0.00 |

**Supported reading:** disabling OBS lowers both macro F1 scores; replacing MRSP
lowers both, especially image F1. Removing adaptive placement lowers pixel F1
while slightly increasing image F1. Harmonisation removal and alpha-only improve
pixel F1 in this detector/dataset setting.

**Possible paper wording:** "Single-axis counterfactuals reveal different
image- and pixel-level effects. OBS and MRSP improve the measured macro F1
relative to their controls, whereas harmonisation and blending exhibit
configuration-dependent trade-offs."

**Do not claim:** every stage is individually necessary, that changes smaller
than the displayed SD are necessarily zero, or that a nonzero mean difference
establishes statistical significance. Source: P1.

## Option B: harmonisation depends on the blending configuration

**Question:** Is one combination better than another, and are individual effects
additive?

**Why these rows:** the first four form the complete H-on/off x hybrid/alpha
2x2. The last two isolate the collar under forced Poisson blending. OBS, MRSP,
placement and donor count stay fixed.

| Configuration | H | Blend | Pixel F1 (%) +/- SD | dP (pp) | Image F1 (%) +/- SD | dI (pp) |
|---|---|---|---:|---:|---:|---:|
| Full FLASH (`baseline`) | On | Hybrid | 38.61 +/- 1.46 | +0.00 | 78.23 +/- 0.11 | +0.00 |
| Harmonisation off (`harmonisation_off`) | Off | Hybrid | 39.64 +/- 0.76 | +1.03 | 78.23 +/- 0.11 | +0.00 |
| Alpha only (`blend_alpha_only`) | On | Alpha | 39.44 +/- 1.16 | +0.82 | 78.23 +/- 0.11 | +0.00 |
| Harmonisation off + alpha (`combo_harmonisation_off_alpha_only`) | Off | Alpha | 38.01 +/- 0.76 | -0.61 | 78.21 +/- 0.18 | -0.01 |
| Forced Poisson + collar (`blend_poisson_collar`) | On | Poisson+collar | 38.57 +/- 1.22 | -0.04 | 78.21 +/- 0.09 | -0.01 |
| Forced Poisson, no collar (`blend_poisson_no_collar`) | On | Poisson | 38.17 +/- 0.76 | -0.44 | 79.42 +/- 0.00 | +1.19 |

**The clearest interaction:** adding H changes pixel F1 by **-1.03 pp under
hybrid blending**, but **+1.43 pp under alpha blending**. Thus, the conclusion
"harmonisation helps/hurts" depends on the other component.

For the two removal/replacement factors, the descriptive interaction is:

`F(no H, alpha) - F(no H, hybrid) - F(H, alpha) + F(H, hybrid) = -2.46 pp`.

Its three seed-level values are -0.86, -4.46 and -2.06 pp. This is a consistent
observed interaction direction across these seeds, not a statistical test or a
mechanistic explanation.

The matched forced-Poisson comparison gives **+0.40 pp pixel / -1.20 pp image**
with the collar. The no-collar configuration predicts all images anomalous in
24/24 runs; its higher image F1 is not evidence of better image discrimination.
Comparing either forced-Poisson row to hybrid also changes routing and does not
isolate the collar.

**Possible paper wording:** "The effect of harmonisation reverses with the
blending rule, and the two individually improving modifications do not combine
additively."

**Do not claim:** hybrid is best in this table, or that the small collar pixel
effect is significant. Sources: P1 and P2; the 2x2 combination was exploratory.

## Option C: localisation-prior combinations and pipeline controls

**Question:** How do OBS and MRSP work together, and how does the full pipeline
compare with a simpler combination?

**Why these rows:** the middle four form a complete full/frame OBS x MRSP/blob
2x2. A simplified pipeline and colour-only variant provide explicit context.
This is the closest compact settings-matrix option to the requested Dinomaly
layout.

| Configuration | OBS | Mask | Placement | H | Blend | Pixel F1 (%) +/- SD | dP (pp) | Image F1 (%) +/- SD | dI (pp) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| Simplified control (`combo_naive_cutpaste`) | Frame | Blob | Original | Off | Alpha | 34.07 +/- 3.28 | -4.54 | 72.88 +/- 0.89 | -5.34 |
| OBS off + uniform blob (`combo_obs_off_uniform_blob`) | Frame | Blob | Mixed | On | Hybrid | 35.67 +/- 1.09 | -2.94 | 74.36 +/- 1.47 | -3.87 |
| OBS off (`obs_off`) | Frame | MRSP | Mixed | On | Hybrid | 36.91 +/- 1.71 | -1.71 | 73.61 +/- 1.83 | -4.62 |
| Uniform blob (`mrsp_uniform_blob`) | C+T | Blob | Mixed | On | Hybrid | 37.88 +/- 1.39 | -0.74 | 75.19 +/- 1.79 | -3.04 |
| Full FLASH (`baseline`) | C+T | MRSP | Mixed | On | Hybrid | 38.61 +/- 1.46 | +0.00 | 78.23 +/- 0.11 | +0.00 |
| Colour-only OBS (`obs_color_only`) | C | MRSP | Mixed | On | Hybrid | 41.86 +/- 1.22 | +3.24 | 78.15 +/- 0.15 | -0.07 |

**Supported reading:** full FLASH exceeds the internal simplified control by
**4.54 pp pixel / 5.34 pp image**, and exceeds the configuration with both
localisation priors removed by **2.94 pp pixel / 3.87 pp image**. Colour-only
then improves SuperADD pixel F1, without improving image F1.

This is **not a cumulative one-component-at-a-time construction**: the first
two rows change placement, harmonisation and blending together, and the middle
rows are alternative branches of a factorial comparison. In particular, adding
MRSP with OBS disabled raises pixel F1 but lowers image F1. Do not draw a
monotonic improvement arrow through the table.

The simplified control retains the original donor bank, host eligibility,
resizing and augmentation; it is not a reproduced external CutPaste method.

**Robustness context, not extra rows in this three-seed table:** with five seeds,
full FLASH versus simplified control is 38.85 versus 33.47 pixel F1, and
colour-only reaches 41.99. The colour-only gain does not transfer to the earlier
PatchCore comparison: 18.25 full versus 17.37 colour-only. Those separate cohorts
limit a universal-component claim.

**Possible paper wording:** "The complete localisation-prior configuration
outperforms its simplified controls on macro F1, while cue simplification
reveals a detector-dependent trade-off."

Sources: P1 and P2; combinations are exploratory. Use the original P2 control
34.07 here, not the separate audit control 34.03.

## Option D: category effects hidden by the dataset average

**Question:** Are small macro changes hiding substantial category-specific effects?

**Why these rows:** baseline plus OBS removal, MRSP replacement, cue
simplification and the two blending-related single-axis changes. **All eight
categories are shown**, including zero-score can and categories that reverse the
average trend; no favourable category subset defines the macro.

Cells are **absolute pixel F1 (%)**, followed by **delta from that category's
baseline in parentheses (pp)**. The last column is the eight-category macro.
This localisation diagnostic accompanies image results in A/C; it is not a
reason to remove image results from the paper.

| Configuration | can | fabric | fruit_jelly | rice | sheet_metal | vial | wallplugs | walnuts | Macro pixel F1 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full FLASH (`baseline`) | 0.00 | 69.12 | 55.39 | 52.55 | 7.76 | 38.91 | 18.51 | 66.67 | 38.61 |
| OBS off (`obs_off`) | 0.00 (+0.00) | 52.68 (-16.44) | 30.51 (-24.88) | 49.78 (-2.78) | 30.26 (+22.50) | 15.74 (-23.17) | 46.12 (+27.60) | 70.17 (+3.51) | 36.91 |
| Uniform blob (`mrsp_uniform_blob`) | 0.00 (+0.00) | 55.39 (-13.73) | 55.00 (-0.39) | 45.30 (-7.26) | 13.74 (+5.98) | 41.52 (+2.60) | 24.11 (+5.60) | 67.96 (+1.29) | 37.88 |
| Colour-only OBS (`obs_color_only`) | 0.00 (+0.00) | 77.95 (+8.83) | 55.80 (+0.40) | 58.43 (+5.88) | 25.80 (+18.04) | 42.08 (+3.17) | 10.21 (-8.31) | 64.60 (-2.06) | 41.86 |
| Harmonisation off (`harmonisation_off`) | 0.00 (+0.00) | 68.18 (-0.94) | 55.86 (+0.47) | 56.93 (+4.38) | 7.62 (-0.14) | 43.13 (+4.22) | 18.14 (-0.37) | 67.28 (+0.61) | 39.64 |
| Alpha only (`blend_alpha_only`) | 0.00 (+0.00) | 74.50 (+5.38) | 55.05 (-0.34) | 52.42 (-0.14) | 7.29 (-0.47) | 47.17 (+8.26) | 16.97 (-1.54) | 62.10 (-4.57) | 39.44 |

**Patterns actually present:**

- **fabric:** full OBS/MRSP outperform their frame/blob alternatives for pixel
  F1, while colour-only OBS and alpha-only blending improve over the full
  configuration.
- **sheet_metal versus wallplugs:** colour-only gains 18.04 pp on sheet_metal but
  loses 8.31 pp on wallplugs. OBS removal, however, improves both pixel scores.
  There is no single category-independent ranking of OBS variants.
- **vial versus walnuts:** alpha-only improves vial by 8.26 pp but reduces
  walnuts by 4.57 pp.
- **Macro cancellation:** OBS removal has large gains and losses of opposing
  signs; its -1.71 pp macro is not a bound on its category-level influence.

**Possible paper wording:** "Category-wise analysis reveals substantial
heterogeneity that is obscured by macro averaging; the preferred synthetic
calibration configuration depends on the evaluated category."

**Do not generalise this to a proven texture/object taxonomy.** fabric provides
one textile-category observation, not replicated evidence across textile
categories. Nor do these aggregates identify which defect types improved. The
detector is fixed across arms: the measurements show changes in threshold
transfer, not that a synthesis component learned to detect a new defect class.
Source: P1.

## Option E: six handpicked, paired category contrasts

**Question:** Can we show concrete examples where one change affects both F1s,
only pixel F1, or trades one against the other?

**Why these rows:** six illustrative category/contrast pairs from D, chosen
after inspecting results to demonstrate those three behaviours. They are not a
representative random sample, a category-selection policy, or six independent
replications of one effect. The unselected can and walnuts remain visible in D.

Every reference below is **full FLASH on that same category**. `P seeds + / - / =`
counts positive, negative and exactly zero paired pixel-F1 changes across the
three original seeds. It is not a significance test.

| Category | Single change from full FLASH | Reference P-F1 (%) | Changed P-F1 (%) | dP (pp) | Reference I-F1 (%) | Changed I-F1 (%) | dI (pp) | P seeds + / - / = |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fabric | OBS off (`obs_off`) | 69.12 | 52.68 | -16.44 | 73.17 | 62.62 | -10.55 | +0 / -3 / =0 |
| fruit_jelly | OBS off (`obs_off`) | 55.39 | 30.51 | -24.88 | 85.71 | 83.59 | -2.12 | +0 / -3 / =0 |
| rice | Uniform blob (`mrsp_uniform_blob`) | 52.55 | 45.30 | -7.26 | 81.08 | 57.51 | -23.57 | +0 / -3 / =0 |
| sheet_metal | Colour-only OBS (`obs_color_only`) | 7.76 | 25.80 | +18.04 | 88.24 | 88.24 | +0.00 | +3 / -0 / =0 |
| vial | Alpha only (`blend_alpha_only`) | 38.91 | 47.17 | +8.26 | 85.71 | 85.71 | +0.00 | +3 / -0 / =0 |
| wallplugs | OBS off (`obs_off`) | 18.51 | 46.12 | +27.60 | 75.00 | 61.23 | -13.77 | +3 / -0 / =0 |

**Supported reading:** the first three examples decrease both image and pixel
F1; the next two improve pixel F1 without changing image F1; wallplugs exhibits
a large image/pixel trade-off. Pixel-change directions agree across all three
seeds for these selected examples, although effect magnitudes vary.

**Possible paper wording:** "Illustrative paired contrasts show that component
changes can affect both evaluation levels, pixel F1 alone, or the two in
opposite directions."

**Publication condition:** label this selection illustrative and provide D or
the complete category table alongside it. Do not present only these selected
examples as evidence that each component generally benefits a named defect
family. Source: P1.

## What every final caption or nearby protocol note must preserve

These are **descriptive measured differences**, not proof of significance,
causality of a particular visual mechanism, or universal component necessity.
Use original three-seed cohorts consistently; do not combine them with extra
seeds only for favourable configurations or substitute audit reruns selectively.

The full-FLASH baseline here is the accepted current-runtime paired rerun,
78.226401 image / 38.614184 pixel, not the paper's original 78.127359 / 38.378047.
Image interpretation needs the existing diagnostic: the audited baseline has
95.89% macro FPR and all-positive predictions in 21/24 runs; an always-anomalous
predictor scores 79.42% macro image F1. High image F1 alone is not good image
discrimination.

The category-specific synthetic-positive budgets match public-test anomaly
counts, so they are test-informed. Combination and table selections are post-hoc;
nominal heldout categories are no longer untouched. Three-seed SD mixes generation
and detector/data RNG variation.

These tables intentionally contain no ranking-metric claims. The saved
colour-only + no-harmonisation / vial / seed1 pixel AUROC is invalid; that
configuration is not part of these thematic selections, but its full F1 row and
invalid metric remain in the consolidated report. Calibration image AUROC
sigmoid saturation and intermittent metric-index failures also remain unresolved.
The four-arm audit independently checks F1 on its own cohort; it is not a blanket
independent audit of every P1/P2 row.

## Source trail and reproducibility

All table values were recomputed from full-precision CSVs, not copied from rounded
tables. P1 contributes the single-axis arms and the baseline. P2 contributes the
combination arms; its duplicate baseline is checked against P1 and not counted as
an additional seed. The document uses 24 category/seed records per configuration.

The persistent FLASH checkout is:

`/home/devuser/.copilot/session-state/8a2f0e48-4fe2-4d47-ba4d-4bb694881422/files/flash`

Paths below are relative to that checkout:

- **P1:** `evaluation/ablations/results/launches/20261022_phase1_paper_components_v1/phase1_results.csv`
  SHA-256: `c2468fc11cb86d345fbc013b6acf3b6f36fed768a1755e6946e1c25f8bd1a52b`.
- **P2:** `evaluation/ablations/results/launches/20261027_phase2_paper_combinations_v2/phase1_results.csv`
  SHA-256: `ad9e864949c1c95ea72f3633f3a864289f331a05e7f382ad49ee0fec9d8de304`.
- **Extra-seed context:** `evaluation/ablations/results/launches/20261027_threshold_audit_stability_patchcore_v2/stability_results.csv`
  SHA-256: `77fb7a9d6995cbddfeae0fbb12f89813857a35df7d83613018a7eb34f0ef5dc2`.
- **Existing PatchCore context:** `evaluation/ablations/results/launches/20261027_threshold_audit_stability_patchcore_v2/patchcore_results.csv`
  SHA-256: `0b0988159707753286e51c2f992665dcbb865a06593e1873139462ca355a1e5c`.
- **Audit context:** `evaluation/ablations/results/launches/20261027_threshold_audit_stability_patchcore_v2/audit_results.csv`
  SHA-256: `f9295fe4a0a1cc9d09b4545572895bec0e433242b2bd98f0cb76db18081a1b64`.
- **Exact component settings:** `evaluation/ablations/specs.py`, `ARM_BY_NAME`.
- **Complete accessible results:** [FLASH_ablation_results_report.md](FLASH_ablation_results_report.md),
  including all original configurations and category/seed records.
- **Existing exports:** [component_ablation_supplement.md](component_ablation_supplement.md),
  [phase1_component_f1_macro.csv](phase1_component_f1_macro.csv),
  [phase1_component_f1_by_category.csv](phase1_component_f1_by_category.csv).

[dinomaly]: https://arxiv.org/html/2405.14325v3#S4.T3
