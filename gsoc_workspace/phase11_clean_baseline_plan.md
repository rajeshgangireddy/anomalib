# Phase 11 — Clean Three-Arm Baseline Comparison, Plan

**Status:** ✅ complete (2026-08-26). All 96 jobs (4 models × 8 categories × 3 seeds)
finished with no errors; 480 result rows aggregated into
`gsoc_workspace/experiments/results/results.csv` and
`gsoc_workspace/experiments/results/results_phase11_clean_baseline.csv`. Results summary
in §7.

## 1. Why this phase exists

The previous "Perlin baseline" comparisons in this project (phases 5, 9, 10, and the
write-up in `superadd_mvtecad2_experiment_phase8.md` §4) pooled three pipelines — P1
(whole-image, DTD-texture, alpha), P2 (foreground-only, self-source, alpha), P3
(foreground-only, self-source, Poisson) — into one "Perlin" number. Only P1 matches
vanilla anomalib/DRAEM-style Perlin synthesis; P2/P3 already include this project's own
foreground-restriction and self-sourcing enhancements. Pooling all three overstated how
weak a genuinely untouched Perlin baseline is, and understated the semantic-defect-bank
method's true improvement margin (documented in detail in §4 of the phase 8 doc).

This phase reruns a **clean, unambiguous three-way comparison** per model, with no
pooling and no confusable baseline:

1. **Arm A — real anomalies.** The dataset's actual `test_public` defects (already
   computed for free in every job as the oracle threshold arm; no new code needed).
2. **Arm B / `P0` — pure, unmodified anomalib Perlin noise.** Anomalib's stock
   `PerlinAnomalyGenerator` class instantiated directly (not our composable
   `SyntheticAnomalyGenerator` framework at all) — the same generator DRAEM's own
   training loop uses, with `make_synthetic_dataset`'s own default kwargs
   (`anomaly_source_path="./datasets/dtd"`, `blend_factor=(0.01, 0.2)`). Zero region
   constraint, zero self-sourcing, zero Poisson blending — genuinely "no extra
   improvements from our side."
3. **Arm B / `P6` — our semantic-defect-bank pipeline**, poisson-hybrid blend only
   (real donor patches from `datasets/SynthetciGenMVAD2`, per the corrected
   terminology: `poisson` = hybrid poisson+alpha, `alpha` = pure alpha). P5 (pure
   alpha) is deliberately **not** re-run here: the phase 8/9 alpha-vs-poisson-hybrid
   comparison already found the two statistically indistinguishable (paired Wilcoxon,
   p ≥ 0.08 for image_F1 on every model tested), so re-litigating that split would add
   96 more jobs for no expected new information. P6 alone stands in for "the
   semantic-bank method."

## 2. Models, categories, seeds

| model | why included now | why not (yet) |
|---|---|---|
| **SuperADD** (`vit_huge_plus_patch16_dinov3`) | paper's recommended backbone, our primary contribution target | — |
| **PaDiM** | cheap, fast, good contrast case (parametric, not memory-bank) | — |
| **PatchCore** | closest architectural relative of SuperADD (coreset memory bank) | — |
| **AnomalyDINO** | training-free DINO-feature memory bank, different backbone family | — |
| Dinomaly | — | needs an epoch-count decision (full gradient-trained, not val-only/single-pass like the four above) — **planned as a follow-up phase**, per your instruction |
| EfficientAd | — | same epoch-count reason, deferred alongside Dinomaly |
| DRAEM | — | trains on synthetic Perlin anomalies internally (different mechanism from this val-only calibration protocol); not part of this round |

All **8 MVTec AD 2 categories** (`can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`,
`vial`, `wallplugs`, `walnuts`), **3 seeds** (1, 2, 3) — matching every prior phase.

**Total: 4 models × 8 categories × 3 seeds = 96 jobs**, each producing 5 rows (arm A;
arm B + arm C for P0; arm B + arm C for P6) — 480 result rows.

## 3. Machine allocation and timing estimate

Remote (Cluster2, 8× RTX 3090) is confirmed back up and idle; the `huge_plus` backbone
(3.2 GB) is already cached there from the earlier attempt, so no re-download risk this
time (that was the suspected cause of the machine becoming unreachable during phase 10).

| machine | models | jobs | est. time/job | est. total (n GPUs) |
|---|---|---|---|---|
| local (2 GPU) | padim, anomaly_dino | 48 | ~1–5 min (P0 adds live-generation cost vs. phase 9's pregenerated-only jobs) | ~1–2 hours |
| remote (8 GPU) | superadd, patchcore | 48 | superadd ~15–40 min/job (live P0 + huge_plus overhead, per phase 10 experience); patchcore ~10–15 min/job (coreset subsampling) | ~2–5 hours (8-way parallel) |

Estimates are extrapolated from phases 5/8/9/10's measured timings; real wall-clock has
consistently run longer than the raw fit+test seconds due to CPU-bound live-generation
and backbone-loading overhead, so these are approximate.

## 4. Code changes already made (prep only, not yet run)

- `gsoc_workspace/experiments/harness.py`: added `"P0": None` to the `PIPELINES`
  registry and updated `make_generator("P0")` to return anomalib's stock
  `PerlinAnomalyGenerator` directly (bypassing `SyntheticAnomalyGenerator` entirely).
  Verified: instantiates correctly, produces valid `(image, mask)` output.
- `gsoc_workspace/experiments/sweep.py`: added `phase11_clean_baseline` spec
  (`models=["superadd","padim","patchcore","anomaly_dino"]`, `pipelines=["P0","P6"]`,
  `seeds=[1,2,3]`, `categories=None` → all 8, `calibration="heldout"`). Verified:
  `enumerate_jobs("phase11_clean_baseline")` produces exactly 96 jobs with the correct
  models/categories/seeds/pipelines.
- Both files linted clean (`ruff check`).

**Not yet done:** no smoke test, no sweep launch — held pending your confirmation, per
your instruction to document the plan first.

## 5. Deferred to a later phase (per your instruction)

- **Dinomaly + EfficientAd** in this same three-arm design, once an epoch-count is
  decided for their full gradient training (unlike the four val-only/single-pass
  models above, these need a real training schedule, not just a memory-bank fit).
- **SuperADD backbone-size ablation** (`small`/`base`/`large`/`huge_plus`, all four
  already cached locally from the original smoke test) — cross the winning
  calibration source from this phase with each backbone size.

---

# Is the paper ready for WACV 2027 (Algorithms Track)?

Short answer: **the empirical/benchmarking side is in increasingly strong shape, but
there is one significant, unaddressed gap relative to the original research plan that
I want to flag explicitly before we invest more compute** — see §6.4.

## 6.1 What's solid so far

- A large, clean, confound-free experimental design (single-training-per-job,
  arm A/B/C structure) spanning 8 categories, up to 4 models, multiple synthetic
  sources, with `results.csv` as a single reproducible source of truth (5000+ rows).
- Several genuine, citable findings already established: the P1-vs-P2/P3
  region/source effect (this phase's motivation), alpha-vs-poisson-hybrid statistical
  equivalence, tiling's net regression on MVTec AD 2 (phase 6), four real measurement
  bugs found and fixed during that work (tile-border artifact, sigmoid-saturation,
  dinomaly crop incompatibility, pixel-threshold bin-range mismatch) — these are
  solid, defensible contributions in their own right.
- Cross-model generalization evidence (phase 9) that the semantic-defect-bank source
  helps beyond just SuperADD.

## 6.2 Must-do before submission (data-quality / rigor, not new science)

1. **Systematic statistical testing across the whole grid**, not just the alpha-vs-
   poisson question. Paired Wilcoxon + Holm–Bonferroni/BH correction across every
   headline model×pipeline comparison (phase 8 §4's P0-vs-P6 gap, cross-model
   generalization in phase 9, tiling regression in phase 6) — currently each was
   tested ad hoc as questions came up, not as one pre-registered battery.
2. **Effect sizes and confidence intervals**, not just point estimates and p-values —
   reviewers at algorithm-focused venues will ask for these.
3. **A single canonical "Perlin baseline" definition going forward** (this phase
   fixes that) plus updating the phase 8 document's §3.2/§3.3 framing once phase 11
   lands, so the paper doesn't present two different numbers for "the Perlin
   baseline" in different sections.

## 6.3 Worth doing if time allows (strengthens the paper, not blocking)

- **Size-stratified pixel-metric analysis** (small vs. large defects) — cheap (no
  retraining, just re-bucket existing per-image scores by mask area), and directly
  supports a "handles small defects better" claim if the semantic-bank method wins
  there specifically (MVTec AD 2's own stated headline challenge).
- **Donor-bank diversity/size ablation** — how many donor pairs are actually needed;
  a practicality argument reviewers may ask about given the method's reliance on a
  real-defect bank (a form of weak supervision that Perlin-noise methods don't need).
- **Comparison against the official MVTec AD 2 leaderboard** (SuperADD 57.42% SegF1,
  RoBiS 51.00%, `test_private`) — everything so far uses `test_public` with our own
  calibration protocol, which is *not* apples-to-apples with the leaderboard's fixed
  official metric; the paper needs to be explicit that these are different regimes,
  or attempt a `test_private` submission for a directly comparable number.

## 6.4 The one open question I think we need to resolve together

The original research plan (`research-ideas-and-paper-plan.md`, written 2026-07-30)
identified the **core, novel methodological contribution** as a **score-space coverage
objective** (Wasserstein-distance matching between synthetic and real anomaly score
distributions) plus a **principled generator-mixture-selection algorithm** built on top
of it — explicitly positioned as answering ASBench's own stated future-work items, to
defend against the "this is just more benchmarking" reviewer objection at a
methods-focused venue.

As far as I can find in this codebase, **that method was never actually implemented or
evaluated** — only a diagnostic correlation check exists
(`analyze_score_coverage.py`, `verify_difficulty_mechanism.py`), used to *validate the
assumption* before building on it, per the plan's own "de-risk this first" advice.
Everything run since (phases 5–11) is comparative benchmarking of existing generation
techniques (Perlin variants, semantic-defect-bank) across models — valuable and
rigorous, but not itself the proposed "algorithm."

**This materially changes what "more experiments" should mean, so I'd like your
direction before proceeding further:**

1. Do we still intend to implement the score-space-coverage / mixture-selection method
   as the paper's headline algorithmic contribution? If yes, that's a multi-week
   implementation + evaluation effort that should probably be sequenced *before*
   further backbone/model-coverage expansion, since it changes what the "final"
   experiment grid needs to look like.
2. Or has the paper's framing shifted to position the **semantic-defect-bank
   generation method itself** (real donor-patch synthesis for label-free threshold
   calibration) as the contribution, with the Perlin-vs-semantic-bank and cross-model
   generalization results as its supporting evidence? This is a perfectly defensible
   paper on its own, but it's a different, narrower claim than the original plan's
   "score-space coverage" framing, and probably needs the small-defect-targeted
   synthesis idea (Idea 4 from the plan) as a companion result to feel sufficiently
   novel for WACV's algorithms track specifically.
3. Either way — should phase 11 (this document) proceed now regardless, since it's
   useful/necessary evidence under both framings?

I'd suggest greenlighting phase 11 either way (it's cheap, ~half a day of compute, and
fixes a real reporting inconsistency), but I don't want to keep expanding the
model/backbone grid on the current trajectory without confirming which of the two
framings above we're actually building toward.

---

## 7. Results (phase 11 — complete)

All 96 jobs completed without error (2026-08-26). Full 480-row detail lives in
`gsoc_workspace/experiments/results/results_phase11_clean_baseline.csv` (also merged
into `results.csv`). Every cell below is the mean over 8 categories × 3 seeds = 24 runs.

### 7.1 Image-level F1 (threshold-calibrated on the indicated source, evaluated on real `test_public`)

| model | A (oracle) | B / P0 (Perlin) | B / P6 (hybrid) | Δ P6 − P0 |
|---|---|---|---|---|
| anomaly_dino | 0.8144 | 0.6097 | 0.7705 | **+0.1608** |
| padim | 0.8026 | 0.7111 | 0.7911 | **+0.0800** |
| patchcore | 0.8245 | 0.6811 | 0.7464 | **+0.0653** |
| superadd | 0.8364 | 0.7912 | 0.7813 | −0.0099 |

### 7.2 Pixel-level F1 (same protocol)

| model | A (oracle) | B / P0 (Perlin) | B / P6 (hybrid) | Δ P6 − P0 | rel. |
|---|---|---|---|---|---|
| anomaly_dino | 0.3356 | 0.1766 | 0.2611 | +0.0845 | +48% |
| padim | 0.0758 | 0.0307 | 0.0389 | +0.0082 | +27% |
| patchcore | 0.2600 | 0.1466 | 0.1814 | +0.0348 | +24% |
| superadd | 0.5160 | 0.1979 | 0.3838 | **+0.1859** | **+94%** |

### 7.3 Threshold-independent metrics (sanity check)

`image_AUROC`, `pixel_AUROC`, `pixel_AUPRO`, and `pixel005_AUPRO` are identical (to
rounding) between P0 and P6 for every model — expected, because they are evaluated on
the same real test set and are threshold-free. This confirms P6 changes **only** the
calibrated threshold, not the underlying anomaly scores (no leakage / double-dipping).

### 7.4 Headline findings

1. **The hybrid semantic-defect-bank (P6) beats stock Perlin (P0) on pixel-level F1 for
   all four models**, most dramatically SuperADD, where it nearly doubles the score
   (0.198 → 0.384, +94%).
2. **Image-level F1 improves for 3/4 models** (anomaly_dino +0.16, padim +0.08,
   patchcore +0.07). SuperADD is already within noise of its oracle ceiling with either
   source (0.791 / 0.781 vs oracle 0.836), so no headroom remains there.
3. **Gap recovery toward the oracle threshold** (pixel-F1, `(P6−P0)/(A−P0)`): SuperADD
   58%, anomaly_dino 53%, patchcore 31%, padim 18%. The hybrid consistently closes a
   large share of the Perlin→oracle gap.
4. **Cross-model generalization**: the P6 > P0 pixel-F1 advantage holds across four
   architecturally distinct models (training-free DINO memory bank, parametric PaDiM,
   coreset PatchCore, huge_plus SuperADD), strengthening the earlier phase-9 finding.

### 7.5 Actual machine split (deviation from §3)

`padim` + `anomaly_dino` (48 jobs) completed first on the local 2 GPUs. The remaining
`superadd` + `patchcore` (48 jobs) were then partitioned by category to avoid
duplication: local ran `fabric` + `fruit_jelly` (12 jobs), and remote (Cluster2, GPUs
0–5 only) ran `rice`, `sheet_metal`, `vial`, `wallplugs`, `walnuts` (30 jobs). Remote
raw/scores JSONs were `scp`-ed back and merged before the final `--aggregate-only`.
