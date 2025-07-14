# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Consolidated layer implementations for Dinomaly model.

This module contains all layer-level components used in the Dinomaly Vision Transformer
architecture, including attention mechanisms, transformer blocks, and MLP layers.

References:
    https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
    https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py
"""

import logging
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from timm.layers import DropPath
from timm.models.vision_transformer import LayerScale

logger = logging.getLogger("dinov2")


class Attention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class MemEffAttention(Attention):
    """Memory-efficient attention using PyTorch's native scaled dot product attention."""

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = qkv.unbind(2)

        # Use PyTorch's native scaled dot product attention for memory efficiency.
        # Replaced xformers's method with pytorch's scaled dot product so openvino exporting be possible.
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_bias,
        )
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinearAttention(nn.Module):
    """Linear attention mechanism for efficient computation."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through linear attention."""
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = nn.functional.elu(q) + 1.0
        k = nn.functional.elu(k) + 1.0

        kv = torch.einsum("...sd,...se->...de", k, v)
        z = 1.0 / torch.einsum("...sd,...d->...s", q, k.sum(dim=-2))
        x = torch.einsum("...de,...sd,...s->...se", kv, q, z)
        x = x.transpose(1, 2).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv


class DinomalyMLP(nn.Module):
    """Unified MLP supporting bottleneck-style behavior, optional input dropout, and bias control.

    This can be used a simple MLP layer or as the BottleNeck layer in Dinomaly models.
    
    Example usage for BottleNeck:
        >>> embedding_dim = 768
        >>> mlp = DinomalyMLP(
        ...     in_features=embedding_dim,
        ...     hidden_features=embedding_dim * 4,
        ...     out_features=embedding_dim,
        ...     drop=0.2,
        ...     bias=False,
        ...     apply_input_dropout=True)
    
    Example usage for a Decoder's MLP:
        >>> embedding_dim = 768
        >>> mlp = DinomalyMLP(
        ...     in_features=embedding_dim,
        ...     hidden_features=embedding_dim * 4,
        ...     drop=0.2,
        ...     bias=False,
        ...     apply_input_dropout=False)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = False,
        apply_input_dropout: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)
        self.apply_input_dropout = apply_input_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, D).
        """
        if self.apply_input_dropout:
            x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = DinomalyMLP,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            apply_input_dropout=False,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x, return_attention=False):
        # Always use the MemEffAttention path for consistency
        if isinstance(self.attn, MemEffAttention):
            y = self.attn(self.norm1(x))
            attn = None
        else:
            y, attn = self.attn(self.norm1(x))

        x = x + self.ls1(y)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x
