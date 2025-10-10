"""Simple Script.

Just a simple script to run the training and testing of a
model for debugging.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from lightning import seed_everything

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore as ModelClass

seed_everything(42)
torch.cuda.deterministic = True
torch.backends.cudnn.benchmark = False


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# datamodule = Folder(
#     name="toothbrush_custom_20",
#     root="datasets/Custom/MVTecCustom/screw/02_MVTeC_screw_20",
#     normal_dir="train/good",
#     # num_workers=0,
#     abnormal_dir="test",
#     mask_dir="ground_truth",
#     normal_test_dir="test_good",
# )


datamodule = Folder(
    name="toothbrush_custom_20",
    root="datasets/Custom/MVTecCustom/screw/02_MVTeC_screw_20",
    normal_dir="train/good",
    test_split_mode=TestSplitMode.SYNTHETIC,
)

model = ModelClass()
engine = Engine(max_epochs=3)

# Train the model
engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
