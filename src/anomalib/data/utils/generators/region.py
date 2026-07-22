# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Region selectors for synthetic anomalies.

A region selector constrains *where* an anomaly may appear. ``WholeImageRegion``
allows anomalies anywhere, while ``ForegroundRegion`` restricts them to the object
(object-aware synthesis), which better matches real defects that live on the part
rather than the background.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch


class RegionSelector(ABC):
    """Base class for region selectors."""

    @abstractmethod
    def constrain(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return ``mask`` restricted to the allowed region, shape ``[1, H, W]``."""


class WholeImageRegion(RegionSelector):
    """Allow anomalies anywhere in the image."""

    def constrain(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG002, PLR6301
        """Return the mask unchanged."""
        return mask


class ForegroundRegion(RegionSelector):
    """Restrict anomalies to the estimated foreground object via Otsu thresholding.

    The background is inferred from the image border, so the method works whether the
    object is brighter or darker than its surroundings. Falls back to the original
    mask when the intersection with the foreground is empty.
    """

    def constrain(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Restrict the mask to the estimated foreground object."""
        foreground = self._foreground(image).to(mask)
        constrained = mask * foreground
        return constrained if constrained.any() else mask

    @staticmethod
    def _foreground(image: torch.Tensor) -> torch.Tensor:
        gray = (image.mean(dim=0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = binary > 0
        border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
        # The majority border class is background; foreground is the complement.
        if border.mean() > 0.5:
            binary = ~binary
        return torch.from_numpy(binary).float().unsqueeze(0)
