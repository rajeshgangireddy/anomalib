# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Patchcore model on custom dataset."""

from pathlib import Path

import torch

from anomalib.data import Folder
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import Patchcore

CUSTOM_DATASETS_ROOT = "datasets/Custom/MVTecCustom/screw"


dataset_dir = Path(CUSTOM_DATASETS_ROOT) / "MVTeC_screw_50"
datamodule = Folder(
    name="screw_50",
    root=dataset_dir,
    normal_dir="train/good",
    normal_test_dir="test_good",
    abnormal_dir="test",
    mask_dir="ground_truth",
)
model = Patchcore()
cuda_available = torch.cuda.is_available()
if cuda_available:
    print("CUDA is available. Using default strategy.")
    engine = Engine()
else:
    print("CUDA is not available. Using XPU strategy.")
    xpu_avialable = torch.xpu.is_available()
    if not xpu_avialable:
        msg = "XPU is not available. Cannot proceed with XPU strategy."
        raise RuntimeError(msg)
    print(f"XPU Available: {xpu_avialable}")
    print(f"XPU Device Count: {torch.xpu.device_count()}")
    print(f"XPU Current Device: {torch.xpu.current_device()}")
    print(f"XPU Device Name: {torch.xpu.get_device_name(0)}")
    print(f"XPU Device Properties: {torch.xpu.get_device_properties(0)}")

    engine = Engine(strategy=SingleXPUStrategy(), accelerator=XPUAccelerator())

engine.fit(datamodule=datamodule, model=model)
results = engine.test(datamodule=datamodule, model=model)
