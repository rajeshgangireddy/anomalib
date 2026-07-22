# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomaly mask generators.

A mask generator decides the *shape* of a synthetic anomaly. It returns a binary
mask that is later combined with a region constraint, an anomaly source and a
blending strategy to produce the final synthetic anomaly.
"""

from abc import ABC, abstractmethod

import torch
from torchvision.transforms import v2

from .perlin import generate_perlin_noise


class MaskGenerator(ABC):
    """Base class for anomaly mask generators."""

    @abstractmethod
    def generate(self, height: int, width: int, device: torch.device | None = None) -> torch.Tensor:
        """Return a binary anomaly mask of shape ``[1, H, W]``."""


class PerlinMaskGenerator(MaskGenerator):
    """Perlin-noise mask generator.

    Args:
        threshold (float): Noise value above which a pixel is marked anomalous.
        rotation_range (tuple[float, float]): Random rotation applied to the mask.
    """

    def __init__(self, threshold: float = 0.5, rotation_range: tuple[float, float] = (-90, 90)) -> None:
        self.threshold = threshold
        self.rotate = v2.RandomAffine(degrees=rotation_range, interpolation=v2.InterpolationMode.BILINEAR, fill=0)

    def generate(self, height: int, width: int, device: torch.device | None = None) -> torch.Tensor:
        """Return a thresholded, randomly rotated Perlin-noise mask."""
        noise = generate_perlin_noise(height, width, device=device)
        # Rescale to span the threshold when the raw noise never exceeds it.
        if not (noise > self.threshold).any():
            noise_range = noise.max() - noise.min()
            noise = (noise - noise.min()) / noise_range * 2 - 1 if noise_range > 0 else torch.zeros_like(noise)
        noise = self.rotate(noise.unsqueeze(0)).squeeze(0)
        return (noise > self.threshold).float().unsqueeze(0)
