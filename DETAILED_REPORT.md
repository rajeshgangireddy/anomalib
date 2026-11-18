# Anomalib Workshop Readiness — detailed report

## 1. Scope and environment

- Source: `open-edge-platform/anomalib` frozen `main` SHA `1684b8ab316d098beb997e4f2d340262b4c21566`; the remote `main` still matched this SHA during the audit.
- Host: Linux x86_64, Python 3.12.3, `uv`; Intel Graphics `0xe216` exposed as PyTorch `xpu:0` and OpenVINO `GPU.0`. NVIDIA RTX 3090 was `GPU.1`, so all OpenVINO claims below explicitly use Intel `GPU.0`.
- Main-source environment: Anomalib `2.7.1.dev0`, PyTorch `2.14.1+xpu`, torchvision `0.29.1+xpu`, Lightning `2.6.6`, OpenVINO `2026.4.1`, NNCF `3.4.0`, ONNX `1.23.2`.
- The isolated source install (`uv sync --frozen --extra xpu`, with audit-required extras) imported Anomalib, passed source CLI checks, performed a synchronized XPU operation, and ran OpenVINO Intel GPU inference.
- The audit used the 60-normal-image MVTec AD `toothbrush` category for the broad model matrix. Each model was fitted within the four-epoch ceiling. PTQ used at most 20 unique training images, disjoint from the 20 held-out parity images.
- Included scope: 31 RGB image models. CFM was excluded (requires RGB plus a 3D point cloud); VlmAd was excluded (external/local VLM backend scope). AnomalyVFM was included in training but its Torch/OpenVINO exports are unsupported.
- No comparative accuracy benchmark was run. Short fits establish API/device compatibility only.

## 2. Installation and examples

### README/PyPI route

The README install command `uv pip install "anomalib[xpu]"` in a fresh environment resolved published Anomalib `2.0.0`, emitted a warning that this release does not provide the `xpu` extra, and installed no Torch, torchvision, Lightning, OpenVINO, or NNCF. The CLI then failed during `anomalib --help` with:

```text
ImportError: cannot import name '_ActionSubCommands' from 'jsonargparse._actions'
```

This tests the literal published-PyPI route, not the frozen source tree. In contrast, the isolated source install using `uv sync --frozen --extra xpu` installed `2.7.1.dev0`, PyTorch XPU, and the working stack. The workshop setup guide should say which Anomalib version/distribution to install and verify that the `xpu` extra is available from that distribution.

### Example workflows

The source API training example, CLI training, CLI OpenVINO export, and a dynamic-shape OpenVINO Intel inference example were exercised in bounded adaptations. The source CLI and Intel XPU/OpenVINO smoke tests passed. Dataset caveats below still apply: MVTec must be staged manually, and tiny normal-only Folder data is not a reliable prediction demo.

## 3. Model training, checkpoint restore, and Torch single-frame inference

All 31 in-scope models fitted on Intel XPU within the four-epoch limit. `TorchInferencer` single-frame inference passed for 28 models. GeneralAD failed with `AttributeError: 'Attention' object has no attribute '_forward'`; WinClip failed with `AttributeError: 'WinClipModel' object has no attribute 'masks'`. AnomalyVFM is unsupported by `TorchInferencer`.

Checkpoint restore was retried using narrowly scoped safe-global allowlists; all 31 models loaded. Thirty reproduced exact saved tensor state. DFKDE loaded and inferred, but `PCA.num_components` changed from saved `int64 [16]` to restored `float32 [16.]`; the numeric value is unchanged. This is a dtype/state-match quirk, not an observed DFKDE inference failure. The detailed original and refined evidence is in `training-matrix-progress.json` and the `checkpoint-reload-*` logs.

### Combined model matrix

`PASS` in a profile means both export and single-frame Intel `GPU.0` inference passed. `INFER-FAIL` means export passed but inference failed; `EXPORT-FAIL` means conversion failed and inference could not run; `BLOCKED` is a non-run inference stage; `UNSUPPORTED` is intentional.

| Model               | XPU fit | TorchInferencer | FP32        | FP16        | INT8        | INT8_PTQ    |
| ------------------- | ------- | --------------- | ----------- | ----------- | ----------- | ----------- |
| AnomalyVFM          | PASS    | UNSUPPORTED     | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| AnomalyDINO         | PASS    | PASS            | EXPORT-FAIL | EXPORT-FAIL | EXPORT-FAIL | EXPORT-FAIL |
| Cfa                 | PASS    | PASS            | PASS        | PASS        | PASS        | PASS        |
| Cflow               | PASS    | PASS            | PASS        | PASS        | PASS        | INFER-FAIL  |
| Csflow              | PASS    | PASS            | PASS        | PASS        | PASS        | INFER-FAIL  |
| Dfkde               | PASS    | PASS            | INFER-FAIL  | INFER-FAIL  | INFER-FAIL  | EXPORT-FAIL |
| Dfm                 | PASS    | PASS            | INFER-FAIL  | INFER-FAIL  | INFER-FAIL  | EXPORT-FAIL |
| Dinomaly            | PASS    | PASS            | PASS        | PASS        | PASS        | PASS        |
| Draem               | PASS    | PASS            | PASS        | PASS        | PASS        | PASS        |
| Dsr                 | PASS    | PASS            | PASS        | PASS        | PASS        | PASS        |
| EfficientAd         | PASS    | PASS            | PASS        | PASS        | PASS        | PASS        |
| Fastflow            | PASS    | PASS            | PASS        | PASS        | PASS        | PASS        |
| FoundAD             | PASS    | PASS            | EXPORT-FAIL | EXPORT-FAIL | EXPORT-FAIL | EXPORT-FAIL |
| Fre                 | PASS    | PASS            | INFER-FAIL  | INFER-FAIL  | INFER-FAIL  | EXPORT-FAIL |
| Ganomaly            | PASS    | PASS            | PASS        | PASS        | PASS        | PASS        |
| GeneralAD           | PASS    | FAIL            | PASS        | PASS        | PASS        | PASS        |
| Glass               | PASS    | PASS            | INFER-FAIL  | INFER-FAIL  | INFER-FAIL  | EXPORT-FAIL |
| InpFormer           | PASS    | PASS            | PASS        | PASS        | PASS        | PASS        |
| L2BT                | PASS    | PASS            | PASS        | PASS        | PASS        | PASS        |
| MHPatchcore         | PASS    | PASS            | INFER-FAIL  | INFER-FAIL  | EXPORT-FAIL | EXPORT-FAIL |
| Padim               | PASS    | PASS            | PASS        | PASS        | PASS        | PASS        |
| Patchcore           | PASS    | PASS            | PASS        | PASS        | PASS        | PASS        |
| Patchflow           | PASS    | PASS            | PASS        | PASS        | PASS        | PASS        |
| Rad                 | PASS    | PASS            | INFER-FAIL  | INFER-FAIL  | INFER-FAIL  | EXPORT-FAIL |
| ReverseDistillation | PASS    | PASS            | PASS        | PASS        | PASS        | PASS        |
| Stfpm               | PASS    | PASS            | PASS        | PASS        | PASS        | PASS        |
| Supersimplenet      | PASS    | PASS            | PASS        | PASS        | PASS        | PASS        |
| SuperADD            | PASS    | PASS            | PASS        | PASS        | PASS        | INFER-FAIL  |
| Uflow               | PASS    | PASS            | PASS        | PASS        | PASS        | INFER-FAIL  |
| UniNet              | PASS    | PASS            | INFER-FAIL  | INFER-FAIL  | INFER-FAIL  | EXPORT-FAIL |
| WinClip             | PASS    | FAIL            | EXPORT-FAIL | EXPORT-FAIL | EXPORT-FAIL | EXPORT-FAIL |

## 4. OpenVINO export and Intel GPU inference

Thirty models were export-applicable. All exports were converted on CPU; inference was compiled explicitly for OpenVINO `GPU.0`.

| Profile  | Exports passed | Intel GPU inference passed | Inference failures after export | Export failures |
| -------- | -------------: | -------------------------: | ------------------------------: | --------------: |
| FP32     |          27/30 |                      20/30 |                               7 |               3 |
| FP16     |          27/30 |                      20/30 |                               7 |               3 |
| INT8     |          26/30 |                      20/30 |                               6 |               4 |
| INT8_PTQ |          20/30 |                      16/30 |                               4 |              10 |

Sixteen models passed export and Intel inference in **all four** profiles: Cfa, Dinomaly, Draem, Dsr, EfficientAd, Fastflow, Ganomaly, GeneralAD, InpFormer, L2BT, Padim, Patchcore, Patchflow, ReverseDistillation, Stfpm, Supersimplenet.

### Failure groups

- **Fixed batch dimension (seven models):** DFKDE, DFM, FRE, Glass, MHPatchcore, RAD, and UniNet exported FP32/FP16/INT8 IR with input shape `[2,3,1024,1024]`. The single-frame inferencer supplied `[1,3,1024,1024]`, which the IR rejects. This is not an Intel GPU-only failure: PTQ calibration also encountered the incompatible batch dimension. The export should preserve a dynamic batch dimension or explicitly produce a single-frame batch shape, then be retested through `OpenVINOInferencer`.
- **ONNX conversion:** AnomalyDINO, FoundAD, and WinClip failed all four export profiles with PyTorch ONNX FX graph decomposition `ConversionError`; inference was consequently blocked.
- **Additional INT8 export:** MHPatchcore's INT8 weight compression rejected a `float64` weight type.
- **INT8_PTQ inference:** Cflow failed Intel GPU program build with OpenCL `CL_OUT_OF_RESOURCES`; Csflow returned a non-finite anomaly map; SuperADD failed GPU program build with `CL_OUT_OF_RESOURCES`; Uflow failed graph validation because a subtraction mixed `f16` and `f32`.
- Remaining INT8_PTQ exports were blocked by the earlier ONNX or calibration/batch-shape failures. The full exact status for every model is in `openvino-matrix-progress.json`.

## 5. Torch/OpenVINO parity

Parity used the same 20 held-out RGB inputs per profile, comparing end-to-end single-image scores/maps, predicted labels, and masks where available. There is no universal tolerance across different model score scales; medians below are descriptive summaries, not pass thresholds or quality rankings.

| Profile  | Median score MAE (models) | Median map MAE (models) | Median scale-aware map relative MAE (models) | Median label agreement (models) | Median mask IoU (models) |
| -------- | ------------------------: | ----------------------: | -------------------------------------------: | ------------------------------: | -----------------------: |
| FP32     |           0.00178785 (19) |         0.00336788 (18) |                               0.0059606 (18) |                          1 (19) |            0.931951 (18) |
| FP16     |           0.00194282 (19) |         0.00337182 (18) |                              0.00596803 (18) |                          1 (19) |            0.929591 (18) |
| INT8     |            0.0191032 (19) |          0.0108558 (18) |                               0.0204235 (18) |                          1 (19) |            0.828181 (18) |
| INT8_PTQ |             0.117899 (15) |          0.0583264 (14) |                                0.151281 (14) |                        0.9 (15) |            0.603264 (14) |

### Intel GPU discrepancies

For these four FP32 exports, CPU inference from the same IR was nearly identical to Torch, while Intel `GPU.0` produced the following Torch-relative mean errors. This isolates the observed discrepancy to Intel GPU execution of the exported graph/runtime path rather than the common IR conversion or preprocessing, although the specific operation or plugin defect remains unidentified.

| Model               | GPU score MAE | GPU anomaly-map MAE | CPU score MAE | CPU anomaly-map MAE |
| ------------------- | ------------: | ------------------: | ------------: | ------------------: |
| ReverseDistillation |        0.4637 |              0.5878 |     0.0000149 |          0.00000032 |
| Cfa                 |        0.1444 |              0.0130 |     0.0000190 |           0.0000043 |
| Csflow              |       0.00102 |              0.0965 |     0.0000012 |           0.0000001 |
| SuperADD            |        0.0569 |              0.1646 |     0.0000009 |           0.0000151 |

PaDiM FP32 was a strong parity canary on this sample: score MAE `0.00312`, anomaly-map MAE `0.00116`, predicted-label agreement `100%`, and mean mask IoU `0.9913`. PatchCore FP32 had map MAE `0.00357` and mean mask IoU `0.9807`. Quantized profiles can have larger differences even when inference returns successfully; see the per-model rows in `parity-summary.csv`.

The four large GPU discrepancies are likely an OpenVINO Intel GPU runtime/plugin or graph-execution defect; this audit did not identify a root cause or change product code. The CPU cross-check was performed on FP32; the parity CSV shows the profile-by-profile GPU deltas. Avoid these models for Intel GPU demonstrations until triaged.

## 6. Datasets

### MVTec AD

The staged toothbrush category had 60 normal training images, 12 normal test images, 30 anomalous test images, and 30 masks; dataloaders opened successfully. Preparing a missing category in a fresh directory reproduced `FileNotFoundError`: main reports that MVTec's upstream download links are broken (tracked in anomalib issue #3892). Workshop users must download and extract MVTec AD manually before using the datamodule.

### VisA

Fresh preparation/download and a second cached preparation passed. Cashew, the smallest selected category, contained 450 normal training images, 50 normal test images, and 100 anomalous test images/masks. PaDiM fitted one epoch on `xpu:0`, the checkpoint safely reloaded, Torch export passed, and the single-frame inferencer passed on 20 held-out inputs. File path, RGB NumPy, PIL, and float CHW input variants were exercised.

### Folder

- Folder API training/testing and deployment passed for segmentation with normal images, anomalous images, and masks. Checkpoint reload and XPU `TorchInferencer` inference on an anomalous frame passed.
- Folder CLI training/testing passed for image-level classification without masks.
- Separate normal test input and normal/abnormal setups with and without masks were exercised.
- Empty and missing normal directories return actionable errors. A one-image split raises a generic `ValueError` (`not enough values to unpack`).
- `normal_split_ratio` is accepted but does not control the observed split. The base `test_split_ratio` and `val_split_ratio` do. With the tested defaults, normal-only sets of 5, 10, and 20 images split 4/0/1, 8/1/1, and 16/2/2 (train/validation/test). The five-image run had an empty validation set and NaN thresholds; exported inference produced an all-positive label/mask. The ten-image run had a NaN pixel threshold and degenerate zero outputs. These tiny normal-only outputs are not useful anomaly predictions; provide a realistically sized normal set and, if available, `abnormal_dir` for held-out anomaly examples.

## 7. Recommended workshop tracks

- **Beginner baselines:** PaDiM and PatchCore; both passed XPU fit, checkpoint/Torch inference, and all four OpenVINO profiles. Use FP32/FP16 for the clearest parity.
- **Additional contrast:** STFPM and FastFlow provide gradient-trained alternatives; both passed the tested workflow. No comparative timing or detection-quality claim is made.
- **Lightweight/teacher-student option:** EfficientAD passed the tested workflows; ensure its required auxiliary/pretrained assets are available in the workshop environment.
- **Advanced elective:** Dinomaly passed the tested workflows but should be treated as a more advanced/pretrained model.
- **SuperADD:** keep optional; its checkpoint is about 4.1 GB, Intel GPU parity is poor in the tested sample, and INT8_PTQ inference failed.

## 8. Reproducibility and worktree status

The detailed matrix, parity values, source/environment manifests, dataset manifests, logs, model checkpoints, exports, and predictions are under this audit directory. Copies of `WORKSHOP_REPORT.md`, `DETAILED_REPORT.md`, `model-matrix.csv`, `parity-summary.csv`, `openvino-matrix-progress.json`, and `training-matrix-progress.json` are in the repository root. The focused offline unit-test slice passed 97 tests, and audit-runner pass/fail/timeout/spawn-failure checks passed. No Anomalib product files were changed. The detached audit source worktree and audit-only runner changes, isolated environments, and datasets are all under the session audit directory. The original checkout stayed at the frozen SHA. Final `git status --short` showed untracked `docs/juntion.md`; the audit did not create or modify it.
