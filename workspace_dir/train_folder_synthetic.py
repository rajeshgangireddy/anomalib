"""Simple Script.

Just a simple script to run the training and testing of a
model for debugging.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
from anomalib.engine import Engine
from anomalib.models import Padim

# Initialize components
datamodule = Folder(
    name="screw_dataset",
    root="datasets/Custom/MVTecCustom/screw/03_MVTeC_screw_30",  # Path to your dataset
    normal_dir="train/good",  # And folder of the "good/normal" images
    test_split_mode=TestSplitMode.SYNTHETIC,
)
model = Padim()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
