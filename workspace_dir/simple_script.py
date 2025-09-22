"""Simple training script to test models on xpu."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import Patchcore as ModelClass

# working Models Patchcore, Padim, Dfm, Fastflow, ReverseDistillation
# Mostly the models that required training in epochs fail.
# The solution of moving the tensors to CPU is not really useful for
# such models, as the training will then be very slow.

cuda_available = torch.cuda.is_available()
# Initialize components

# Ensure INFO-level logs from library modules are printed to stdout. By
# default the root logger level is WARNING so INFO messages from module
# loggers (even if they call logger.setLevel(logging.INFO)) won't be
# emitted unless a handler is configured. basicConfig attaches a
# StreamHandler to the root logger.
logging.basicConfig(level=logging.INFO)


model = ModelClass()
CATEGORY = "transistor"

# if model is EfficientAd, use batchsize of 1
if model.name == "EfficientAd":
    datamodule = MVTecAD(category=CATEGORY, train_batch_size=1)
else:
    datamodule = MVTecAD(category=CATEGORY)

MAX_EPOCHS = 20
if not cuda_available:
    # If CUDA is not available, use XPU strategy
    print("CUDA is not available. Using XPU strategy.")
    print(f"XPU Available: {torch.xpu.is_available()}")

    engine = Engine(
        strategy=SingleXPUStrategy(),
        accelerator=XPUAccelerator(),
        max_epochs=MAX_EPOCHS,
    )

else:
    print("CUDA is available. Using default strategy.")
    # If CUDA is available, use the default strategy
    engine = Engine(max_epochs=MAX_EPOCHS)
# Train the model
engine.fit(datamodule=datamodule, model=model)
results = engine.test(datamodule=datamodule, model=model)
