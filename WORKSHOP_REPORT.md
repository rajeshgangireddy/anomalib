# Anomalib Workshop Readiness — concise report

**Scope:** frozen `main` at `1684b8ab316d098beb997e4f2d340262b4c21566`, Linux, Intel Graphics `0xe216` (`xpu:0` / OpenVINO `GPU.0`). Training was capped at four epochs. This is a workflow compatibility check, not a model-quality benchmark.

## Verdict

- **Installation:** The README-style `uv pip install "anomalib[xpu]"` resolved the older PyPI Anomalib `2.0.0`; its `xpu` extra was unavailable, Torch/OpenVINO were not installed, and `anomalib --help` failed importing `jsonargparse._actions._ActionSubCommands`. A clean environment installed from the frozen source with `uv sync --frozen --extra xpu`; XPU tensor computation, source CLI, and OpenVINO Intel GPU smoke inference passed.
- **Training and Torch inference:** All **31** in-scope image models fit on Intel XPU within the epoch cap. Single-frame `TorchInferencer` passed for **28**; **GeneralAD** (`Attention._forward` missing) and **WinClip** (`WinClipModel.masks` missing) failed; **AnomalyVFM** is unsupported by this inferencer. Checkpoint reload succeeded for all 31 after scoped safe-load retries; DFKDE has one numeric-equal buffer dtype cast.
- **OpenVINO:** Of the 30 export-applicable models, **16** passed export and explicit `GPU.0` inference in all four profiles. Seven exported fixed batch size 2 and therefore failed single-frame inference. AnomalyDINO, FoundAD, and WinClip failed ONNX conversion. PTQ adds four inference failures and other export blockers; see the model-by-model matrix in `DETAILED_REPORT.md` and `model-matrix.csv`.
- **Parity:** Do **not** treat every successful inference as equivalent. FP32 CPU IR outputs closely matched Torch for the tested canaries, while Intel GPU outputs diverged substantially for **ReverseDistillation, Cfa, Csflow, and SuperADD**. This points to an Intel GPU execution/runtime issue, but the offending operation has not been isolated.
- **Data:** VisA downloaded fresh and reused its cache; PaDiM trained on cashew and passed checkpoint reload and single-frame inference. Folder classification and segmentation workflows passed with usable data. MVTec AD automatic download failed because upstream links are broken; stage MVTec manually. Very small normal-only Folder data produced invalid thresholds/unusable predictions.

## Workshop recommendation

Start with **PaDiM** and **PatchCore** as reliable unsupervised baselines. Add **STFPM** and **FastFlow** for contrasting gradient-trained approaches; **EfficientAD** is a useful lightweight/teacher-student example. **Dinomaly** is an optional advanced model. Make **SuperADD** elective: its checkpoint is about 4.1 GB, its Intel GPU outputs disagreed with Torch, and INT8_PTQ inference failed. Do not rank these by detection quality; the runs were short smoke tests.

For OpenVINO demonstrations, use **FP32 or FP16 PaDiM/PatchCore** and avoid the four known GPU-parity outliers. Do not use the seven batch-2 exports for single-frame inference or the three ONNX-conversion failures until resolved.

## Priority issues before the workshop

1. Replace or clarify the README installation route so it installs the intended Anomalib source/version and a working Intel XPU stack.
2. Provide/stage MVTec AD locally; its automatic download cannot currently bootstrap a fresh category.
3. Triage OpenVINO single-frame batch-2 exports and the four GPU parity outliers; pin Intel execution to `GPU.0`.
4. For Folder, use a realistic normal set and, when available, an `abnormal_dir` for held-out defects; do not use the five/ten-image normal-only probe as a working prediction demo.

## Artifacts

Detailed findings: [`DETAILED_REPORT.md`](DETAILED_REPORT.md). Full per-model/per-profile outcomes: [`model-matrix.csv`](model-matrix.csv) and [`openvino-matrix-progress.json`](openvino-matrix-progress.json). Per-model quantitative parity: [`parity-summary.csv`](parity-summary.csv). The reports and selected summary matrices are copied into the repository root for convenience; full logs, scripts, environments, datasets, and test outputs remain under this audit directory, outside the checkout. No product source files were changed. The original checkout remained on the frozen SHA; the final status showed untracked `docs/juntion.md`, which the audit did not create or modify.
