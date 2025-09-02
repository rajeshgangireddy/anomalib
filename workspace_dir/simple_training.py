"""Simple training script for anomaly detection using Dinomaly model."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import Dinomaly as ModelClass

cuda_available = torch.cuda.is_available()
# Initialize components
datamodule = MVTecAD(category="toothbrush")

model = ModelClass()
MAX_EPOCHS = 2
if not cuda_available:
    # If CUDA is not available, use XPU strategy
    print("CUDA is not available. Using XPU strategy.")
    print(f"XPU Avialable: {torch.xpu.is_available()}")

    engine = Engine(
        strategy=SingleXPUStrategy(),
        accelerator=XPUAccelerator(),
    )

else:
    print("CUDA is available. Using default strategy.")
    # If CUDA is available, use the default strategy
    engine = Engine(max_epochs=MAX_EPOCHS)
# Train the model
engine.fit(datamodule=datamodule, model=model)
results = engine.test(datamodule=datamodule, model=model)
