# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Manifold Projector for FoundAD model.

Implements the lightweight ViT-based predictor that projects encoder features
back onto the natural image manifold. This is the core trainable component
of the FoundAD architecture.

The key architectural difference from a standard ViT decoder is the **global
residual skip connection**: each transformer block's output is added to the
*initial* projected embedding, rather than using standard per-block residual
connections. This encourages the projector to learn small corrections to the
input features rather than completely overwriting them.

Reference:
    Zhai et al., "Foundation Visual Encoders Are Secretly Few-Shot Anomaly
    Detectors", ICLR 2026. https://arxiv.org/abs/2510.01934
    Original implementation: https://github.com/ymxlzgy/FoundAD
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn.init import trunc_normal_

from anomalib.models.image.dinomaly.components.layers import DinomalyMLP, LinearAttention

# Default initialization standard deviation
_INIT_STD = 0.02


class ProjectorBlock(nn.Module):
    """Transformer block for the manifold projector.

    Uses LinearAttention for OpenVINO export compatibility. Unlike the standard
    transformer block, the residual connection is handled externally (global skip).

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to input dim.
        qkv_bias: Whether to use bias in QKV projection.
        drop: Dropout rate for MLP and projection.
        attn_drop: Dropout rate for attention weights.
        drop_path: Drop path rate for stochastic depth.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = LinearAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = DinomalyMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop,
            bias=True,
            apply_input_dropout=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Standard pre-norm transformer block (no internal residual).

        The global residual skip is applied externally in ManifoldProjector.forward().

        Args:
            x: Input tensor of shape (B, N, D).

        Returns:
            Output tensor of shape (B, N, D).
        """
        y, _ = self.attn(self.norm1(x))
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension (must be even).
        grid_size: Grid height/width.

    Returns:
        Positional embeddings of shape (grid_size*grid_size, embed_dim).
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)

    pos_embed = np.zeros((grid_size * grid_size, embed_dim))
    half_dim = embed_dim // 2
    omega = np.arange(half_dim, dtype=float) / half_dim
    omega = 1.0 / (10000.0**omega)

    for i, g in enumerate(grid):
        flat = g.reshape(-1)
        out = np.einsum("m,d->md", flat, omega)
        start = i * half_dim
        pos_embed[:, start : start + half_dim // 2] = np.sin(out[:, : half_dim // 2])
        pos_embed[:, start + half_dim // 2 : start + half_dim] = np.cos(out[:, : half_dim // 2])

    return pos_embed


class ManifoldProjector(nn.Module):
    """Lightweight ViT predictor that projects features back onto the natural image manifold.

    Architecture:
        1. Linear embedding: embed_dim -> predictor_embed_dim
        2. Optional sinusoidal positional encoding
        3. N transformer blocks with global residual skip
        4. LayerNorm
        5. Linear projection: predictor_embed_dim -> embed_dim

    The global residual skip means each block's output is added to the *initial*
    projected embedding, encouraging the projector to learn manifold corrections
    rather than completely rewriting the features.

    Args:
        num_patches: Number of input patches (determines positional encoding size).
        embed_dim: Encoder feature dimension.
        predictor_embed_dim: Internal predictor dimension. Defaults to 384.
        depth: Number of transformer blocks. Defaults to 6.
        num_heads: Number of attention heads. Defaults to 12.
        mlp_ratio: MLP hidden dim ratio. Defaults to 4.0.
        use_pos_embed: Whether to use positional encoding. Defaults to False.
        feat_normed: Whether to L2-normalize output features. Defaults to False.

    Example:
        >>> projector = ManifoldProjector(num_patches=196, embed_dim=768)
        >>> features = torch.randn(2, 196, 768)
        >>> output = projector(features)
        >>> output.shape
        torch.Size([2, 196, 768])
    """

    def __init__(
        self,
        num_patches: int,
        embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_pos_embed: bool = False,
        feat_normed: bool = False,
    ) -> None:
        super().__init__()
        self.feat_normed = feat_normed

        # Dimension reduction: encoder dim -> predictor dim
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Optional positional encoding
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, predictor_embed_dim),
                requires_grad=False,
            )
            trunc_normal_(self.predictor_pos_embed, std=_INIT_STD)
        else:
            self.predictor_pos_embed = None

        # Transformer blocks
        self.predictor_blocks = nn.ModuleList([
            ProjectorBlock(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
            )
            for _ in range(depth)
        ])

        # Output normalization and projection back to encoder dim
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self._init_weights()
        self._fix_init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features back onto the natural image manifold.

        Args:
            x: Encoder features of shape (B, N, embed_dim).

        Returns:
            Projected features of shape (B, N, embed_dim).
        """
        x = self.predictor_embed(x)

        if self.use_pos_embed and self.predictor_pos_embed is not None:
            x = x + self.predictor_pos_embed

        # Global residual: each block adds to the initial embedding
        residual = x.clone()
        for blk in self.predictor_blocks:
            x = blk(x) + residual

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        if self.feat_normed:
            x = torch.nn.functional.normalize(x, dim=-1)

        return x

    def _init_weights(self) -> None:
        """Initialize weights with truncated normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=_INIT_STD)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _fix_init_weight(self) -> None:
        """Rescale attention and MLP weights by layer depth.

        This prevents the residual stream from growing too large in deep networks,
        following the approach from the original FoundAD implementation.
        """
        for layer_id, layer in enumerate(self.predictor_blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
