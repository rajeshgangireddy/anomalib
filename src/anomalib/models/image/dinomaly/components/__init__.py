# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Components module for Dinomaly model.

This module provides all the necessary components for the Dinomaly Vision Transformer
architecture including layers, model loader, utilities, and vision transformer implementations.
"""

# Layer components
from .layers import (
    Attention,
    Block,
    DinomalyMLP,
    LinearAttention,
    MemEffAttention,
)

# Model loader
from .model_loader import DinoV2Loader, load

# Utility functions and classes
from .training_utils import (
    CosineHardMiningLoss,
    StableAdamW,
    WarmCosineScheduler,
)

# Vision transformer components
from .vision_transformer import (
    DinoVisionTransformer,
)

__all__ = [
    # Layers
    "Attention",
    "Block",
    "DinomalyMLP",
    "LinearAttention",
    "MemEffAttention",
    # Model loader
    "DinoV2Loader",
    "load",
    # Utils
    "StableAdamW",
    "WarmCosineScheduler",
    "CosineHardMiningLoss",
    # Vision transformer
    "DinoVisionTransformer",
]
