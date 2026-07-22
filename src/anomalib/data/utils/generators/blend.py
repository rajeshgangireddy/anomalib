# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Blending strategies for synthetic anomalies.

A blend decides *how* the anomaly source is composited into the image inside the mask.
``AlphaBlend`` performs a simple convex combination, while ``PoissonBlend`` uses seamless
cloning for smoother, more photorealistic transitions.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch


def _sample_beta(blend_factor: float | tuple[float, float], device: torch.device) -> torch.Tensor:
    """Sample a blend weight from a fixed value or a ``(min, max)`` range."""
    if isinstance(blend_factor, tuple):
        low, high = blend_factor
        return torch.rand(1, device=device) * (high - low) + low
    return torch.tensor(blend_factor, device=device)


class BlendStrategy(ABC):
    """Base class for blending strategies."""

    @abstractmethod
    def blend(self, image: torch.Tensor, source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Composite ``source`` into ``image`` within ``mask``, returning ``[C, H, W]``."""


class AlphaBlend(BlendStrategy):
    """Convex combination of image and source inside the mask.

    Args:
        blend_factor (float | tuple[float, float]): Opacity of the source, or a range
            to sample from per call.
    """

    def __init__(self, blend_factor: float | tuple[float, float] = (0.2, 1.0)) -> None:
        self.blend_factor = blend_factor

    def blend(self, image: torch.Tensor, source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Blend the source into the image with a sampled opacity inside the mask."""
        beta = _sample_beta(self.blend_factor, image.device)
        return image * (1 - mask) + mask * (beta * source + (1 - beta) * image)


class PoissonBlend(BlendStrategy):
    """Seamless (Poisson) cloning of the source into the image.

    Falls back to a full-opacity paste if seamless cloning fails (e.g. a mask that
    touches the image border).
    """

    def blend(self, image: torch.Tensor, source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Seamlessly clone the source into the image within the mask."""
        image_np = self._to_uint8(image)
        source_np = self._to_uint8(source)
        mask_np = (mask.squeeze(0).cpu().numpy() > 0).astype(np.uint8) * 255

        ys, xs = np.nonzero(mask_np)
        if len(xs) == 0:
            return image
        center = (int(xs.mean()), int(ys.mean()))
        try:
            blended = cv2.seamlessClone(source_np, image_np, mask_np, center, cv2.NORMAL_CLONE)
        except cv2.error:
            binary = torch.from_numpy(mask_np > 0).to(image)
            return image * (1 - binary) + source * binary
        return torch.from_numpy(blended).to(image).permute(2, 0, 1) / 255.0

    @staticmethod
    def _to_uint8(image: torch.Tensor) -> np.ndarray:
        return (image.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
