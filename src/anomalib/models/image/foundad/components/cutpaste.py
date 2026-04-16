# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CutPaste augmentation for FoundAD model.

Implements the CutPaste synthetic anomaly generation used during FoundAD training.
CutPaste creates off-manifold examples by cutting patches from images, optionally
applying color jitter and rotation, then pasting them elsewhere in the same image.

The augmentation provides supervision signal for the manifold projector: the
projector learns to map these synthetic anomalies back to the normal manifold,
while normal features are already on the manifold.

This is a simplified version of the original FoundAD CutPaste that drops
per-category foreground masking (which is MVTec/VisA-specific). Instead, patches
are placed randomly, making this augmentation dataset-agnostic.

Reference:
    Zhai et al., "Foundation Visual Encoders Are Secretly Few-Shot Anomaly
    Detectors", ICLR 2026. https://arxiv.org/abs/2510.01934
    Li et al., "CutPaste: Self-Supervised Learning for Anomaly Detection and
    Localization", CVPR 2021.
"""

import math
import random

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms


class CutPasteNormal:
    """Cut a rectangular patch and paste it at a random location.

    Args:
        area_ratio: Range of patch area as fraction of image area.
        aspect_ratio: Controls patch aspect ratio range (aspect_ratio to 1/aspect_ratio).
        color_jitter: Strength of color jitter applied to cut patch. None to disable.
    """

    def __init__(
        self,
        area_ratio: tuple[float, float] = (0.02, 0.25),
        aspect_ratio: float = 0.3,
        color_jitter: float | None = 0.5,
    ) -> None:
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.jitter = (
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter,
            )
            if color_jitter is not None
            else None
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply CutPaste to a single image.

        Args:
            img: Image tensor of shape (C, H, W).

        Returns:
            Augmented image tensor of shape (C, H, W).
        """
        _, h, w = img.shape
        area = h * w

        target_area = random.uniform(*self.area_ratio) * area
        ar = random.uniform(self.aspect_ratio, 1.0 / self.aspect_ratio)
        cut_w = int(round(math.sqrt(target_area * ar)))
        cut_h = int(round(math.sqrt(target_area / ar)))

        if cut_w <= 0 or cut_h <= 0 or cut_w >= w or cut_h >= h:
            return img

        # Cut
        from_x = random.randint(0, w - cut_w)
        from_y = random.randint(0, h - cut_h)
        patch = img[:, from_y : from_y + cut_h, from_x : from_x + cut_w].clone()

        # Optional color jitter
        if self.jitter is not None:
            patch = self.jitter(patch)

        # Paste at random location
        to_x = random.randint(0, w - cut_w)
        to_y = random.randint(0, h - cut_h)
        result = img.clone()
        result[:, to_y : to_y + cut_h, to_x : to_x + cut_w] = patch
        return result


class CutPasteScar:
    """Cut a thin scar-like patch, rotate it, and paste at a random location.

    Args:
        width: Range of scar width in pixels.
        height: Range of scar height in pixels.
        rotation: Range of rotation angles in degrees.
        color_jitter: Strength of color jitter. None to disable.
    """

    def __init__(
        self,
        width: tuple[int, int] = (2, 16),
        height: tuple[int, int] = (10, 25),
        rotation: tuple[int, int] = (-45, 45),
        color_jitter: float | None = 0.5,
    ) -> None:
        self.width = width
        self.height = height
        self.rotation = rotation
        self.jitter = (
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter,
            )
            if color_jitter is not None
            else None
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply CutPaste scar to a single image.

        Args:
            img: Image tensor of shape (C, H, W).

        Returns:
            Augmented image tensor of shape (C, H, W).
        """
        _, h, w = img.shape

        cut_w = random.randint(*self.width)
        cut_h = random.randint(*self.height)
        if cut_w <= 0 or cut_h <= 0 or cut_w >= w or cut_h >= h:
            return img

        # Cut
        from_x = random.randint(0, w - cut_w)
        from_y = random.randint(0, h - cut_h)
        patch = img[:, from_y : from_y + cut_h, from_x : from_x + cut_w].clone()

        # Color jitter
        if self.jitter is not None:
            patch = self.jitter(patch)

        # Rotate
        rot_deg = random.uniform(*self.rotation)
        patch = TF.rotate(patch, angle=rot_deg, interpolation=TF.InterpolationMode.BILINEAR, expand=True)

        _, patch_h, patch_w = patch.shape
        if patch_w >= w or patch_h >= h:
            return img

        # Paste with blending mask (rotated patch may have black borders)
        to_x = random.randint(0, w - patch_w)
        to_y = random.randint(0, h - patch_h)
        result = img.clone()
        # Create a mask for the rotated patch (non-zero pixels)
        mask = (patch.sum(dim=0, keepdim=True) != 0).float()
        region = result[:, to_y : to_y + patch_h, to_x : to_x + patch_w]
        result[:, to_y : to_y + patch_h, to_x : to_x + patch_w] = region * (1 - mask) + patch * mask
        return result


class CutPasteUnion:
    """Union of CutPasteNormal and CutPasteScar augmentations.

    Randomly applies one of the two augmentations with equal probability to
    each image in a batch. This provides diverse off-manifold examples for
    FoundAD training.

    Args:
        color_jitter: Strength of color jitter for both augmentations.

    Example:
        >>> augment = CutPasteUnion(color_jitter=0.5)
        >>> images = torch.randn(4, 3, 224, 224)
        >>> augmented = augment(images)
        >>> augmented.shape
        torch.Size([4, 3, 224, 224])
    """

    def __init__(self, color_jitter: float = 0.5) -> None:
        self.cutpaste_normal = CutPasteNormal(color_jitter=color_jitter)
        self.cutpaste_scar = CutPasteScar(color_jitter=color_jitter)

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """Apply random CutPaste augmentation to a batch of images.

        Args:
            imgs: Batch of images of shape (B, C, H, W).

        Returns:
            Augmented batch of shape (B, C, H, W).
        """
        augmented = imgs.clone()
        for i in range(imgs.shape[0]):
            if random.random() < 0.5:
                augmented[i] = self.cutpaste_normal(imgs[i])
            else:
                augmented[i] = self.cutpaste_scar(imgs[i])
        return augmented
