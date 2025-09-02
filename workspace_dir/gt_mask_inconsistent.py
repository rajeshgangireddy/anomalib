"""Simple Script.

Just a simple script to run the training and testing of a
model for debugging.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from torchvision import transforms as v2

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore
from anomalib.visualization import ImageVisualizer

path_to_dataset = Path("./datasets/Custom/toothbrush")


def create_datamodule(dataset_folder: Path) -> Folder:
    """Create data module for training."""
    augmentations = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally with 50% probability
        v2.RandomVerticalFlip(p=0.2),  # Randomly flip images vertically with 20% probability
        v2.RandomRotation(degrees=30),  # Randomly rotate images within a range of ±30 degrees
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Randomly adjust colors
        v2.RandomGrayscale(p=0.1),  # Convert images to grayscale with 10% probability
    ])

    datamodule = Folder(
        name=dataset_folder.name,
        root=str(dataset_folder),
        normal_dir="train/good",
        abnormal_dir="test/defective",
        normal_test_dir="test/good",
        mask_dir="ground_truth/defective",
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.FROM_TEST,
        train_augmentations=augmentations,
        val_augmentations=None,
        test_augmentations=None,
        num_workers=1,
    )
    datamodule.setup()
    return datamodule


model_type = Patchcore
datamodule = create_datamodule(path_to_dataset)

# Calling code:
for model_name in ["Patchcore", "Padim"]:
    if model_name == "Patchcore":
        model_type = Patchcore
    elif model_name == "Padim":
        model_type = Padim
    else:
        msg = f"Unknown model type: {model_name}"
        raise ValueError(msg)
    visualizer = ImageVisualizer()
    model = model_type(visualizer=visualizer)
    engine = Engine(max_epochs=10)
    dataset_folder = datamodule.root
    engine.fit(datamodule=datamodule, model=model)
    results = engine.test(datamodule=datamodule, model=model)
