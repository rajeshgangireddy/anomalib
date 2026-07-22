# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomaly content sources.

A source decides *what* fills the anomaly region. ``SelfSource`` samples content from
the same image (self-blend / NSA style), producing subtle same-material defects.
``TextureSource`` pastes content from an external texture bank (DRAEM style), producing
stronger out-of-distribution defects.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torchvision import io
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import v2


def _augment() -> v2.Transform:
    """Return a light photometric/geometric augmentation used by sources."""
    return v2.Compose([
        v2.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.3, hue=0.1),
        v2.RandomAffine(degrees=(-45, 45), interpolation=v2.InterpolationMode.BILINEAR, fill=0),
    ])


class AnomalySource(ABC):
    """Base class for anomaly content sources."""

    @abstractmethod
    def get(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return source content of shape ``[C, H, W]`` in ``[0, 1]``."""


class SelfSource(AnomalySource):
    """Content sampled from the same image via a random spatial shift and augmentation."""

    def __init__(self, augment: bool = True) -> None:
        self.augment = _augment() if augment else None

    def get(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        """Return a spatially shifted, augmented copy of the image."""
        _, height, width = image.shape
        shift_h = int(torch.randint(height // 4, height, (1,)).item())
        shift_w = int(torch.randint(width // 4, width, (1,)).item())
        source = torch.roll(image, shifts=(shift_h, shift_w), dims=(1, 2))
        if self.augment is not None:
            source = self.augment(source)
        return source.clamp(0, 1)


class TextureSource(AnomalySource):
    """Content sampled from an external texture bank (e.g. DTD).

    Args:
        texture_path (Path | str): Directory of texture images searched recursively.
        augment (bool): Whether to apply photometric/geometric augmentation.

    Raises:
        FileNotFoundError: If no images are found under ``texture_path``.
    """

    def __init__(self, texture_path: Path | str, augment: bool = True) -> None:
        self.paths: list[Path] = []
        for extension in IMG_EXTENSIONS:
            self.paths.extend(Path(texture_path).rglob("*" + extension))
        if not self.paths:
            msg = f"No texture images found under {texture_path}."
            raise FileNotFoundError(msg)
        self.augment = _augment() if augment else None

    def get(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        """Return a random texture image resized to the input and augmented."""
        _, height, width = image.shape
        path = self.paths[int(torch.randint(len(self.paths), (1,)).item())]
        texture = io.read_image(str(path), mode=io.ImageReadMode.RGB).float().to(image) / 255.0
        texture = v2.functional.resize(texture, [height, width], antialias=True)
        if self.augment is not None:
            texture = self.augment(texture)
        return texture.clamp(0, 1)
