# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from collections.abc import Callable

from torch import Tensor, nn
import torch


class Mlp(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int | None = None,
            out_features: int | None = None,
            act_layer: Callable[..., nn.Module] = nn.GELU,
            drop: float = 0.0,
            bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DinomalyMLP(nn.Module):
    """Unified MLP supporting bottleneck-style behavior,
    optional input dropout, and bias control.

    This can be used a simple MLP layer or as the BottleNeck layer in Dinomaly models.
    To use this as a BottleNeck layer,

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
        """
        Forward pass through the MLP.

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
