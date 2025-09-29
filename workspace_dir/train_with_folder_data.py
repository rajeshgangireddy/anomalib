"""Simple Script.

Just a simple script to run the training and testing of a
model for debugging.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Supersimplenet as ModelClass

# Initialize components
datamodule = Folder(
    name="toothbrush_custom_50",
    root="datasets/Custom/MVTecCustom/screw/01_MVTeC_screw_50",
    normal_dir="train/good",
    anomaly_dir="test",
    mask_dir="ground_truth",
    normal_test_dir="test_good",
    num_workers=0,
)

model = ModelClass()
engine = Engine(max_epochs=3)

# Train the model
engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
