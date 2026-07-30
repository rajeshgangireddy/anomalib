# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities to generate synthetic data.

This module provides utilities for generating synthetic data for anomaly detection.
The utilities include:

- Perlin noise generation: Functions for creating Perlin noise patterns
- Anomaly generation: Classes for generating synthetic anomalies
- Composable synthetic anomalies: Mix-and-match mask, region, source and blend axes

Example:
    >>> from anomalib.data.utils.generators import generate_perlin_noise
    >>> # Generate 256x256 Perlin noise
    >>> noise = generate_perlin_noise(256, 256)
    >>> print(noise.shape)
    torch.Size([256, 256])

    >>> from anomalib.data.utils.generators import SyntheticAnomalyGenerator
    >>> # Build a composable generator from a named preset
    >>> generator = SyntheticAnomalyGenerator.from_preset("self_poisson")
"""

from .blend import AlphaBlend, BlendStrategy, HybridBlend, PoissonBlend
from .mask import MaskGenerator, PerlinMaskGenerator
from .perlin import PerlinAnomalyGenerator, generate_perlin_noise
from .region import ForegroundRegion, RegionSelector, WholeImageRegion
from .source import AnomalySource, SelfSource, TextureSource
from .synthetic_anomaly import PIPELINE_PRESETS, PipelineConfig, SyntheticAnomalyGenerator

__all__ = [
    "PIPELINE_PRESETS",
    "AlphaBlend",
    "AnomalySource",
    "BlendStrategy",
    "ForegroundRegion",
    "HybridBlend",
    "MaskGenerator",
    "PerlinAnomalyGenerator",
    "PerlinMaskGenerator",
    "PipelineConfig",
    "PoissonBlend",
    "RegionSelector",
    "SelfSource",
    "SyntheticAnomalyGenerator",
    "TextureSource",
    "WholeImageRegion",
    "generate_perlin_noise",
]
