# FLASH: reproduction-gated PatchCore component study

**Decision: use PatchCore, not PaDiM. Execution is blocked by a metric failure; no sweep is running.**

## Execution status (28 October 2026)

The user has requested continuing with this plan. Final offline checks passed
(105 tests and Ruff), including a CPU check of scalar BatchNorm-buffer hashing.
The final prepare-only audit,
`evaluation/ablations/results/launches/20261028_patchcore_preflight_v3`,
verified all 24 paper reference pairs, 248 input groups and 476 matching source
snapshots, with zero GPU fits. Its runner SHA-256 is
`77573407eac0fb638217e773674e16cd4bd48002bd6f46fd537f7df30fedecfc`.

Smoke v1 failed in both workers before model construction or fitting: Lightning's
hyperparameter deep copy cannot pickle the audit observer's SHA-256 state. The
failure is preserved in `patchcore_components_smoke_v1`. A regression test through
the real Lightning constructor reproduced it. The separate runner now attaches
the unchanged audit observer after construction, matching the existing follow-up
runner's pattern. No model settings, RNG seeds, calibration rules or tolerances
changed; this is a code repair, not a result-based retry.

The fresh two-fit resource smoke stopped in
`evaluation/ablations/results/launches/patchcore_components_smoke_v2`, with a
sibling `.coordinator.log`. Both detectors finished fitting and archived-corpus
calibration, but can/seed0 failed during real-test AUROC computation:

```text
torchmetrics._binary_clf_curve -> preds[torch.argsort(preds, descending=True)]
IndexError: index 281474980245205 is out of bounds for dimension 0 with size 32514048
```

The companion fabric test was terminated by the coordinator after can failed;
its interruption is not a second independent metric failure. There are **zero
accepted result rows**, no paper-reproduction run and no component-panel run.
Both GPUs are idle. Source, input and runtime fingerprints were independently
rechecked after failure and still match the launch.

Six standalone exact-size probes (continuous, clipped and heavily tied scores
on each GPU) all passed; maximum CPU/GPU AUROC difference was
`5.960464477539063e-08` on those synthetic inputs. This does **not** reproduce,
explain or fix the real-score failure. The failed real metric arrays were not
retained, so an exact replay is not currently possible. Evidence is preserved in
the smoke's `diagnosis.json`, both failure/fit records and
`metric_sort_probe_gpu{0,1}.json`; the standalone probe script is
`../probe_patchcore_metric_sort.py`, relative to the FLASH checkout.

### Authorized diagnostic-only follow-up

The user selected **diagnostic capture**. One can/physical-seed0 case completed
on GPU 0 in `evaluation/ablations/results/launches/patchcore_metric_capture_can0_v1`.
It uses the same detector, data, seeds and threshold rules, with observation-only
capture added in the separate `evaluation/patchcore_metric_capture.py`. The gated
runner and profile remain unchanged. The combined offline suite now has 110
passing tests; Ruff passed for the diagnostic and gated-runner files.

The original evaluator defaults to `compute_on_cpu=True`, which offloads list
states after updates. The previous failure must therefore not be attributed to
GPU sorting without recording the actual computation/state devices. Capture
records those devices; replay compares identical states on CPU at the original
12 threads and at one thread, and on GPU. Persistence and CPU copies change
allocation/timing, so a passing capture would not itself establish a permanent fix.

**Verified diagnostic outcome:** all 59 batch files cover 136 calibration and
162 real-test images. The fitted checkpoint loads with `weights_only=True`;
its memory-bank hash equals the failed smoke's can/seed0 bank. Backbone weights
and bank stayed fixed across calibration/testing, raw/processed decisions agreed,
and the full saved raw-prediction hash was independently reconstructed.

All 16 pre-compute metric states were replayed on each of the three backend/thread
settings (48 computations). Thresholds and image/pixel F1 were identical; maximum
absolute numerical difference across other outputs was
`5.960464477539063e-08`. The test pixel-AUROC list states were **CPU tensors**,
although the metric module's device was `cuda:0`. Independent sklearn pixel
AUROC was `0.770651723262633`, versus framework `0.7706517577171326`.
Another 50 computations of the fixed pixel-AUROC state on the original 12 CPU
threads all returned the same valid value without an indexing error.
These checks do **not** reproduce or fix the intermittent failure.

The diagnostic also exposed a separate existing calibration-metric caveat:
raw image distances range from 24.835890 to 71.986847, and BinaryAUROC treats
values outside `[0,1]` as logits. Float32 sigmoid maps all these image scores to
one, producing logged calibration image AUROC **0.5**, whereas independent AUROC
on the raw distances is **0.9765700483091787**. The saved metric states confirm
this saturation. This is not a change to the raw F1 threshold-selection rule;
neither the historical metric outputs nor thresholds were silently corrected.

The diagnostic can/seed0 transferred image F1 was **61.971831%**, versus the exact
paper pair's **57.286429%**: **+4.685402 pp**, outside the planned 0.01 pp gate.
Pixel F1 was zero in both. This single diagnostic pair is not an accepted
reference run or a macro result, but it does not support a paper-match claim.
Do not tune seeds or relax the tolerance to erase that discrepancy.

Evidence: `capture_verified.json`, `diagnostic_results.json`,
`replay_{cpu_threads12,cpu_threads1,cuda_threads12}.json` and
`stress_cpu12_pixel_auroc.json`. No accepted ablation rows were produced; no
paper-reference or component run was launched. Both GPUs are idle. The immediate
blockers remain the unexplained intermittent metric error and the observed
single-pair paper mismatch, not a lack of saved diagnostic inputs.

The diagnostic saves `fitted_state.pt` (tensor state and RNG state), full raw and
processed batch inputs, and `threshold_states.pt`, `validation_states.pt` and
`test_states.pt` before metric computation. Replays use tensor-only loading and
never fit a model or select new thresholds:

```bash
uv run --no-project --active python -m evaluation.patchcore_metric_capture \
  --mode capture --data-root "$MAIN/datasets" \
  --run-dir evaluation/ablations/results/launches/patchcore_metric_capture_can0_v1

uv run --no-project --active python -m evaluation.patchcore_metric_capture \
  --mode replay --device cpu --threads 12 --data-root "$MAIN/datasets" \
  --run-dir evaluation/ablations/results/launches/patchcore_metric_capture_can0_v1
```

The capture command is historical once launched: do not overwrite or rerun that
directory. Capture requires `CUDA_VISIBLE_DEVICES=0`; CPU replay needs no GPU.
Use `--device cpu --threads 1` for the CPU thread control, and
`CUDA_VISIBLE_DEVICES=0 ... --device cuda --threads 12` for GPU replay.

Do not count this as an accepted ablation result or retry until a favourable
score appears. Simply switching to CPU is not an established fix: the relevant
states were already on CPU. Further numerical-runtime stabilization or a
runtime-reference revision needs an explicit, documented protocol decision
before another accepted smoke/reproduction run. The nine-arm panel and all
paper/bridge tolerances remain unchanged.

## Why this detector and which metric?

Lead the localization component table with **transferred-threshold pixel F1**. It is
the more informative endpoint for these synthesis components, not a universally
better metric or permission to omit image-level failures. Show absolute pixel F1
and its signed delta in percentage points; retain image F1 and image false-positive
rate (FPR), with both levels' precision/recall and category results alongside them.
Do not replace absolute scores with delta-only reporting.

The paper's PaDiM pixel F1 is only **3.8926649708931413%**, versus PatchCore's
**18.137765454594046%**. PatchCore offers a less floor-dominated localization study
and an existing, operationally demonstrated evaluation path. This choice is not
based on which detector supports the desired component story: the completed
PatchCore shortlist already reverses the SuperADD colour-only gain, and that
negative result must remain visible. PaDiM is not included in this run.

PatchCore adds a different scoring mechanism, but this two-detector component
study would not establish universality across all anomaly-detector families.
The panel is post-hoc, chosen after inspecting the current benchmark results.

## Exact paper configuration and targets

Authoritative reference, relative to the FLASH checkout:
`evaluation/results/paper_table1_results.csv`.
Select `experiment=cross_model_comparison`, `dataset=mvtec2`, `model=patchcore`,
`calibration_source=FLASH`, `evaluation_set=Real test set`, sweep seeds 1/2/3.
There are exactly 24 category/seed rows.

| Setting | Frozen value |
|---|---|
| Anomalib source commit | `0e6107dc51bf803ae6ae522fa2fb37baf463b6ac` |
| Model / pretrained backbone | `Patchcore` / `wide_resnet50_2`, pretrained |
| Features | `layer2`, `layer3` |
| Coreset sampling ratio | `0.1` |
| Neighbours for image-score weighting | `9` |
| Precision | `float32` |
| Resize / center crop | 448 x 448 / none |
| Normalization | Standard model ImageNet normalization |
| Train / evaluation batch size | 8 / 8 |
| DataLoader workers / fitting epochs | 4 / 1 |
| Calibration negatives | Native validation normals, disjoint from train/test paths |
| Generation seeds | 0, 1, 2 |
| New-run data and global RNG seeds | 1, 2, 3 respectively |
| Thresholds | Separate image/pixel F1 optima on calibration only |
| Evaluation | Transfer those thresholds unchanged to real public-test data |

These settings were traced through the frozen `evaluation/harness/harness.py`,
`sweep.py`, and `run_one.py`, and the pinned
`src/anomalib/models/image/patchcore/{lightning_model,torch_model}.py`.
The original paper row's blank backbone field means no constructor override:
the verified default is WRN-50-2, not ResNet18.

| Category | Paper image F1 (%) | Paper pixel F1 (%) | Synthetic positives | Native validation normals |
|---|---:|---:|---:|---:|
| can | 45.329851 | 0.000000 | 90 | 46 |
| fabric | 73.170733 | 15.186644 | 90 | 43 |
| fruit_jelly | 73.560963 | 14.278106 | 60 | 37 |
| rice | 81.081080 | 3.261196 | 90 | 35 |
| sheet_metal | 88.235295 | 26.324027 | 90 | 19 |
| vial | 85.714287 | 28.141728 | 105 | 41 |
| wallplugs | 75.000000 | 8.530717 | 90 | 33 |
| walnuts | 75.000000 | 49.379706 | 90 | 48 |
| **Category macro** | **74.636526** | **18.137765** | **705 per seed** | **302** |

Full-precision macro gate targets are **74.63652615745862% image** and
**18.137765454594046% pixel**. The gate reads the original unrounded 24 pairs,
not the rounded table above. Average seeds within category, then categories equally.

Paper CSV SHA-256:
`194c34516690bed01a1f839d5fbc576ab1c369775e2d62ca4565b84f208f69d4`.

### Reproduction caveats: do not promise an exact match

The paper harness records data seeds, but does **not** explicitly seed all model
RNGs. PatchCore's sparse random projection uses Torch and NumPy/scikit-learn
randomness, and its initial coreset center uses `torch.randint`. Original RNG
state or a PatchCore checkpoint was not found in the inspected harness
`results/raw` and `results/scores` directories. This is not a claim that every
possible archive has been searched.

The new runner explicitly seeds RNGs and records pretrained-weight and fitted
memory-bank fingerprints. That makes the new protocol better specified; it does
not reconstruct missing historical RNG state. No seed search is allowed.

The current Anomalib source equals the pinned commit. Its imported version is
`2.5.2-dev.0`, while installed distribution metadata says `2.6.3.dev0`; both are
recorded separately. Other current versions include Torch 2.14.0, torchvision
0.29.0, timm 1.0.29, Lightning 2.6.6 and Torchmetrics 1.9.0. The original dependency
requirements are not a complete runtime lock.

The new fit uses native validation normals instead of the original harness's
real-test oracle-first validation. PatchCore fits its memory bank from training
normals; the ordinary postprocessor is reset before each synthetic calibration.
We intentionally do not recreate real-test threshold selection as a paper result.
Call this a paper-configuration/archived-corpus reproduction check, not a
bit-for-bit replay of the entire original oracle/Perlin/FLASH lifecycle.

The already completed **restored-corpus** PatchCore baseline is
**75.20886721710363% image / 18.254198968255274% pixel**, respectively
**+0.5723410596450123 / +0.116433513661228 pp** above the paper.
It is not a successful paper-baseline reproduction. The new same-fit
archived/restored bridge separates corpus differences from refitting variation.

## Fixed component panel

| Arm | Purpose |
|---|---|
| `baseline` | Full FLASH, paired reference |
| `obs_off` | Remove foreground suppression |
| `obs_color_only` | Check whether the SuperADD cue simplification transfers |
| `mrsp_uniform_blob` | Replace MRSP with the existing uniform-blob control |
| `placement_original` | Remove adaptive placements |
| `harmonisation_off` | Remove colour harmonisation |
| `blend_alpha_only` | Replace hybrid routing with alpha-only blending |
| `combo_harmonisation_off_alpha_only` | Complete the harmonisation x blending 2x2 |
| `combo_naive_cutpaste` | Existing multi-component simplified-pipeline control |

All nine settings already have three-seed corpora. **No synthesis, new
combination sweep, PaDiM run, VLM rerun or bank-size experiment is included.**
The simplified control is internal, not an external CutPaste implementation.
This panel does not isolate a collar effect, texture-cue complementarity, or a
bank-size effect; do not make those claims from it.

## Stages and hard gates

| Stage | Fits | Evaluation passes | What it establishes |
|---|---:|---:|---|
| Preparation | 0 | 0 | Full input/config/source provenance, no GPU experiment |
| Resource smoke | 2 | 4 | can/fabric generation seed 0, archived + restored baseline |
| Paper reproduction + bridge | 24 | 48 | All 8 categories x 3 seeds, two corpora per shared fit |
| Component panel | 24 | 240 | 216 primary rows + 24 end-of-fit baseline repeats |
| **Planned gated path** | **50** | **292** | **216 primary component-table rows** |

Failed attempts and the separately authorized diagnostic capture are not extra
independent seeds or accepted ablation rows.

Each fit is isolated in a fresh process. Two GPU slots dynamically take jobs,
one visible GPU per process. Calibration arms within a job share one fitted
detector. Checkpoints are disabled to avoid redundant large model files.

**Paper gate:** every one of the 24 image F1 pairs AND 24 pixel F1 pairs,
and both macro values, must differ from the original CSV by at most
**0.01 percentage point**. Opposite errors cancelling in the macro cannot pass.

**Generation bridge gate:** on the same fit, every restored-baseline image/pixel
F1 must agree with the archived corpus within **0.01 pp**. Raw real-test prediction
fingerprints must also remain identical between those passes.

**Component controls:** fresh baseline and repeated-baseline F1 must agree with
their references within **0.01 pp** for every category/seed. All arms in a fit
must have identical raw prediction hashes. Pretrained weights must remain fixed
within/across fits and across stages; the memory bank must not change across arms.
Hash drift blocks acceptance even when F1 remains close. Hashes alone cannot
quantify score drift; preserve the evidence for investigation.

The runner rejects missing/duplicate conditions, wrong physical seeds or budgets,
missing/extra masks, split overlaps, invalid metadata, nonfinite/out-of-range
calibration or test ranking metrics, and disagreement between independently
accumulated confusion-count F1 and framework F1. It records both raw thresholds,
raw/processed confusion counts, image/pixel precision/recall/FPR/specificity,
image scores, clipping, and raw prediction hashes.

**No automatic retries, discrepancy-acceptance switch, tolerance widening or
fallback model is implemented.** A failure preserves candidate metrics, partial
JSONL, logs and traceback. Failed candidates are not accepted result rows.
Invalid candidate JSON can contain NaN/Infinity for forensic preservation;
accepted rows cannot. Child-process groups are stopped on failure/interruption.

If the paper gate fails, stop and diagnose the missing RNG/runtime/corpus
relationship. Any decision to adopt a new runtime reference needs explicit user
approval and a separately documented protocol revision; retain the failed paper
comparison. Do not select a favourable seed or change the paper targets.

## Sol execution handoff

The prepared code lives in the existing persistent FLASH checkout, **not** in
Anomalib's public model implementation:

```bash
export MAIN=/home/devuser/workspace/code/Anomalib/anomalib
export FLASH=/home/devuser/.copilot/session-state/8a2f0e48-4fe2-4d47-ba4d-4bb694881422/files/flash
export VIRTUAL_ENV="$MAIN/.venv"
export FLASH_EVAL_DATA_ROOT="$MAIN/datasets"
export PYTHONPATH="$FLASH/src:$FLASH"
cd "$FLASH"
```

Profile: `evaluation/ablations/patchcore_study.json`.
Runner: `evaluation/ablations/run_patchcore_study.py`.
Tests: `tests/test_patchcore_study.py`.
The frozen harness, generator and completed results were not edited.

First run the offline checks and use a **new output directory** for each stage.
The runner never overwrites an existing run. A prepare-only output is not an
executable run to resume and cannot authorize the component stage.

```bash
uv run --no-project --active ruff check \
  evaluation/ablations/run_patchcore_study.py tests/test_patchcore_study.py
uv run --no-project --active pytest -q \
  tests/test_patchcore_study.py tests/test_followups.py \
  tests/test_ablations.py tests/test_supplement_tables.py

uv run --no-project --active python -m evaluation.ablations.run_patchcore_study \
  --stage smoke --gpus 0 1 --data-root "$MAIN/datasets" \
  --run-dir evaluation/ablations/results/launches/patchcore_components_smoke_v2
```

Inspect the smoke's four rows, logs, GPU memory, fit fingerprints and both
calibration paths before proceeding. A smoke pass does not reproduce the paper.
Do not call an active coreset calculation a completed successful smoke.

```bash
uv run --no-project --active python -m evaluation.ablations.run_patchcore_study \
  --stage reproduce --gpus 0 1 --data-root "$MAIN/datasets" \
  --run-dir evaluation/ablations/results/launches/patchcore_paper_reference_v1
```

Inspect `gate.json`, including every paired delta. **Only if it passes:**

```bash
uv run --no-project --active python -m evaluation.ablations.run_patchcore_study \
  --stage components --gpus 0 1 --data-root "$MAIN/datasets" \
  --reference-run evaluation/ablations/results/launches/patchcore_paper_reference_v1 \
  --run-dir evaluation/ablations/results/launches/patchcore_components_panel_v1
```

The component command recomputes the reference gate and checks the exact CSV
hash plus matching source/input/runtime fingerprints; a saved `passed: true`
alone is insufficient. Every stage fingerprints inputs before and after execution.
Do not edit numerical code/profile between these stages.

When later authorized to leave a stage running, use the execution tool's detached
mode and a persistent coordinator log. Check coordinator/child PIDs, each assigned
GPU, per-fit logs and durable rows during startup. Coreset fitting can take many
minutes before the first row; allow hours for the complete matrix, not minutes.
Finish smoke/gate inspection before advancing, then stop active monitoring as
requested. Do not launch any stage merely by reading this handoff.

Artifacts: `launch.json`, source snapshots, `status.json`, per-fit logs,
`<category>_seed<seed>/fit.json`, partial `rows.jsonl`, per-pass image-score CSVs,
candidate/calibration JSON, final `results.csv`, and `gate.json`.
Only primary component rows enter the 216-row table; archived controls and
baseline repeats are diagnostics, not extra independent seeds.

## Reporting and unresolved limitations

For each arm report seed-averaged category results, equal-category macro F1,
paired deltas and the sample SD across three seed-level macros. Do not call that
SD a confidence interval or treat all 24 rows as independent categories.
The harmonisation/blending interaction is the descriptive contrast
`F(noH,alpha) - F(noH,hybrid) - F(H,alpha) + F(H,hybrid)`, not a significance claim.

Preserve the previous four-arm PatchCore results separately rather than merging
unmatched fits or counting repeats as extra seeds. Retain the colour-only reversal
and the full pipeline's advantage over its simplified control, regardless of the
new panel's outcome.

More detectors do not fix image-threshold collapse, test-informed positive budgets,
post-hoc exposure of nominal heldout categories, three unresolved archived PNG replay
differences, or the unexplained historical Torchmetrics indexing failures. The old
vial/seed1 colour+no-harmonisation pixel AUROC of 7563479.0 remains invalid; it is
not repaired by the new runner's range guard. These limitations stay disclosed.

After accepted completion, export macro and category CSVs and a compact
pixel-first (not pixel-only) supplementary settings table in the main repository.
Keep the full existing `FLASH_ablation_results_report.md` unchanged as historical
evidence. Defect-bank-size experiments remain the next separately scoped study.
