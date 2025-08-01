# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Feature extractors for deep learning models.

This module provides feature extraction utilities and classes for extracting
features from images using various backbone architectures.

Classes:
    TimmFeatureExtractor: Feature extractor using timm models.
    BackboneParams: Configuration parameters for backbone models.

Functions:
    dryrun_find_featuremap_dims: Utility to find feature map dimensions.

Example:
    >>> from anomalib.models.components.feature_extractors import (
    ...     TimmFeatureExtractor
    ... )
    >>> # Create feature extractor
    >>> feature_extractor = TimmFeatureExtractor(
    ...     backbone="resnet18",
    ...     layers=['layer1', 'layer2']
    ... )
    >>> # Extract features
    >>> features = feature_extractor(images)
"""

from .timm import TimmFeatureExtractor
from .utils import dryrun_find_featuremap_dims

__all__ = [
    "dryrun_find_featuremap_dims",
    "TimmFeatureExtractor",
]
