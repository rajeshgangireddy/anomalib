"""Simple Script.

Just a simple script to run the training and testing of a
model for debugging.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from torchvision import transforms as v2
from torchvision.transforms import functional

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore
from anomalib.visualization import ImageVisualizer

path_to_dataset = Path("./datasets/Custom/toothbrush-BoxedMasks")


class DualInputAugmentations:
    """Custom augmentations that handle both image and mask."""

    def __init__(self) -> None:
        # Define transforms that work with both image and mask
        self.horizontal_flip_prob = 0.5
        self.vertical_flip_prob = 0.2
        self.rotation_prob = 0.5  # 50% chance to apply rotation
        self.rotation_degrees = 30

        # Color transforms only apply to images, not masks
        self.color_transform = v2.Compose([
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            v2.RandomGrayscale(p=0.1),
        ])

    def __call__(self, image: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply augmentations to both image and mask."""
        # Apply geometric transforms to both image and mask
        if torch.rand(1) < self.horizontal_flip_prob:
            image = functional.hflip(image)
            if mask is not None:
                mask = functional.hflip(mask)

        if torch.rand(1) < self.vertical_flip_prob:
            image = functional.vflip(image)
            if mask is not None:
                mask = functional.vflip(mask)

        # Apply rotation with careful mask handling
        if torch.rand(1) < self.rotation_prob:
            angle = (torch.rand(1) * 2 * self.rotation_degrees - self.rotation_degrees).item()
            image = functional.rotate(image, angle)
            if mask is not None:
                # Ensure mask has the right shape for rotation
                if mask.dim() == 2:  # Add channel dimension if needed
                    mask = mask.unsqueeze(0)

                # Convert to float for rotation, then back to preserve binary nature
                original_dtype = mask.dtype
                mask_float = mask.float()
                mask_rotated = functional.rotate(mask_float, angle, interpolation=v2.InterpolationMode.NEAREST)
                mask = (mask_rotated > 0.5).to(original_dtype)  # Re-binarize

        # Apply color transforms only to image
        image = self.color_transform(image)

        return image, mask


def create_datamodule(dataset_folder: Path) -> Folder:
    """Create data module for training."""
    # Use custom augmentation class that properly handles both image and mask
    augmentations = DualInputAugmentations()

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
