# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Blending strategies for synthetic anomalies.

A blend decides *how* the anomaly source is composited into the image inside the mask.
``AlphaBlend`` performs a simple convex combination, ``PoissonBlend`` uses seamless
cloning for smoother, more photorealistic transitions, and ``HybridBlend`` routes each
connected component to one of the two based on its area.
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


class HybridBlend(BlendStrategy):
    """Area-routed combination of alpha and Poisson blending.

    Seamless cloning reconstructs the interior of a region from its boundary, so on a
    small component the boundary condition dominates and the source is washed out --
    the anomaly becomes nearly invisible. Alpha blending always shows the source but
    leaves a visible seam on large regions. This strategy routes each connected
    component by area: components smaller than ``area_threshold`` are alpha blended so
    they stay visible, larger ones are Poisson blended so they stay seamless.

    Args:
        area_threshold (int): Component area in pixels at or above which Poisson
            blending is used instead of alpha blending.
        blend_factor (float | tuple[float, float]): Opacity passed to the alpha blend.

    Example:
        >>> blend = HybridBlend(area_threshold=2500)
        >>> blended = blend.blend(image, source, mask)  # doctest: +SKIP
    """

    def __init__(
        self,
        area_threshold: int = 2500,
        blend_factor: float | tuple[float, float] = (0.2, 1.0),
    ) -> None:
        self.area_threshold = area_threshold
        self.alpha = AlphaBlend(blend_factor)
        self.poisson = PoissonBlend()

    def blend(self, image: torch.Tensor, source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Blend each connected component of the mask according to its area."""
        binary = (mask.squeeze(0).cpu().numpy() > 0).astype(np.uint8)
        count, labels = cv2.connectedComponents(binary)
        small = np.zeros_like(binary, dtype=bool)
        large = np.zeros_like(binary, dtype=bool)
        for label in range(1, count):
            component = labels == label
            if component.sum() >= self.area_threshold:
                large |= component
            else:
                small |= component

        blended = image
        if small.any():
            blended = self.alpha.blend(blended, source, self._as_mask(small, mask))
        if large.any():
            blended = self.poisson.blend(blended, source, self._as_mask(large, mask))
        return blended

    @staticmethod
    def _as_mask(component: np.ndarray, reference: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(component).to(reference).unsqueeze(0)
