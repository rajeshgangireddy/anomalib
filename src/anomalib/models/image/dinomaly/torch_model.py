# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model for the Dinomaly model implementation.

Original Code : PyTorch Implementation of "Dinomaly" by guojiajeremy
https://github.com/guojiajeremy/Dinomaly

See Also:
    :class:`anomalib.models.image.dinomaly.lightning_model.Dinomaly`:
        Dinomaly Lightning model.
"""

import math
from functools import partial

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from anomalib.data import InferenceBatch

from .model_loader import load as load_dinov2_model


class ViTill(nn.Module):
    """ViTill: Vision Transformer-based anomaly detection model from Dinomaly.

    ViTill is a Vision Transformer-based anomaly detection model that uses an encoder-bottleneck-decoder
    architecture for feature reconstruction.

    The architecture comprises three main components:
    + An Encoder: A pre-trained Vision Transformer (ViT), by default a ViT-Base/14 based dinov2-reg model which
    extracts universal and discriminative features from input images.
    + Bottleneck: A simple MLP that collects feature representations from the encoder's middle-level layers.
    + Decoder: Composed of Transformer layers (by default 8 layers), it learns to reconstruct the middle-level features.

    Args:
        encoder_name (str): Name of the Vision Transformer encoder to use.
            Supports DINOv2 variants like "dinov2reg_vit_base_14".
            Defaults to "dinov2reg_vit_base_14".
        bottleneck_dropout (float): Dropout rate for the bottleneck MLP layer.
            Defaults to 0.2.
        decoder_depth (int): Number of Vision Transformer decoder layers.
            Defaults to 8.
        target_layers (list[int] | None): List of encoder layer indices to extract features from.
            If None, uses [2, 3, 4, 5, 6, 7, 8, 9] for base models.
            For large models, uses [4, 6, 8, 10, 12, 14, 16, 18].
        fuse_layer_encoder (list[list[int]] | None): Layer groupings for encoder feature fusion.
            If None, uses [[0, 1, 2, 3], [4, 5, 6, 7]].
        fuse_layer_decoder (list[list[int]] | None): Layer groupings for decoder feature fusion.
            If None, uses [[0, 1, 2, 3], [4, 5, 6, 7]].
        mask_neighbor_size (int): Size of neighborhood masking for attention.
            Set to 0 to disable masking. Defaults to 0.
        remove_class_token (bool): Whether to remove class token from features
            before processing. Defaults to False.
        encoder_require_grad_layer (list[int]): List of encoder layer indices
            that require gradients during training. Defaults to empty list.

    Example:
        >>> model = ViTill(
        ...     encoder_name="dinov2reg_vit_base_14",
        ...     decoder_depth=8,
        ...     bottleneck_dropout=0.2
        ... )
        >>> features = model(images)
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        bottleneck_dropout: float = 0.2,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        mask_neighbor_size: int = 0,
        remove_class_token: bool = False,
        encoder_require_grad_layer: list[int] | None = None,
    ) -> None:
        super().__init__()

        if target_layers is None:
            # 8 middle layers of the encoder are used for feature extraction.
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]

        # Instead of comparing layer to layer between encoder and decoder, dinomaly uses
        # layer groups to fuse features from multiple layers.
        if fuse_layer_encoder is None:
            fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        if fuse_layer_decoder is None:
            fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

        if encoder_require_grad_layer is None:
            encoder_require_grad_layer = []

        encoder = load_dinov2_model(encoder_name)
        if "small" in encoder_name:
            embed_dim, num_heads = 384, 6
        elif "base" in encoder_name:
            embed_dim, num_heads = 768, 12
        elif "large" in encoder_name:
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            msg = "Architecture not in small, base, large."
            raise ValueError(msg)

        # Add validation
        if decoder_depth <= 0:
            msg = f"decoder_depth must be greater than 1, got {decoder_depth}"
            raise ValueError(msg)

        bottleneck = []
        bottleneck.append(BottleNeckMLP(embed_dim, embed_dim * 4, embed_dim, drop=bottleneck_dropout))
        bottleneck = nn.ModuleList(bottleneck)

        decoder = []
        for _ in range(decoder_depth):
            blk = DecoderViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),  # type: ignore[arg-type]
                attn_drop=0.0,
                attn=LinearAttention2,
            )
            decoder.append(blk)
        decoder = nn.ModuleList(decoder)

        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def get_encoder_decoder_outputs(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Extract and process features through encoder and decoder.

        This method processes input images through the DINOv2 encoder to extract
        features from target layers, fuses them through a bottleneck MLP, and
        reconstructs them using the decoder. Features are reshaped for spatial
        anomaly map computation.

        Args:
            x (torch.Tensor): Input images with shape (B, C, H, W).

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: Tuple containing:
                - en: List of fused encoder features reshaped to spatial dimensions
                - de: List of fused decoder features reshaped to spatial dimensions
        """
        x = self.encoder.prepare_tokens(x)

        encoder_features = []
        decoder_features = []

        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                encoder_features.append(x)
        side = int(math.sqrt(encoder_features[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            encoder_features = [e[:, 1 + self.encoder.num_register_tokens :, :] for e in encoder_features]

        x = self._fuse_feature(encoder_features)
        for _i, blk in enumerate(self.bottleneck):
            x = blk(x)

        attn_mask = self.generate_mask(side, x.device) if self.mask_neighbor_size > 0 else None

        for _i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            decoder_features.append(x)
        decoder_features = decoder_features[::-1]

        en = [self._fuse_feature([encoder_features[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self._fuse_feature([decoder_features[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens :, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens :, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def forward(self, batch: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Forward pass of the Dinomaly model.

        During training, the model extracts features from the encoder and decoder
        and returns them for loss computation. During inference, it computes
        anomaly maps by comparing encoder and decoder features using cosine similarity,
        applies Gaussian smoothing, and returns anomaly scores and maps.

        Args:
            batch (torch.Tensor): Input batch of images with shape (B, C, H, W).

        Returns:
            torch.Tensor | InferenceBatch:
                - During training: Dictionary containing encoder and decoder features
                  for loss computation.
                - During inference: InferenceBatch with pred_score (anomaly scores)
                  and anomaly_map (pixel-level anomaly maps).

        Example:
            >>> model = ViTill()
            >>> images = torch.randn(4, 3, 224, 224)
            >>>
            >>> # Training mode
            >>> model.train()
            >>> features = model(images)  # Returns {"encoder_features": [...], "decoder_features": [...]}
            >>>
            >>> # Inference mode
            >>> model.eval()
            >>> result = model(images)  # Returns InferenceBatch
            >>> anomaly_scores = result.pred_score
            >>> anomaly_maps = result.anomaly_map
        """
        en, de = self.get_encoder_decoder_outputs(batch)

        if self.training:
            return {"encoder_features": en, "decoder_features": de}
        anomaly_map, _ = self.cal_anomaly_maps(en, de)
        anomaly_map_resized = anomaly_map.clone()
        resize_mask = 256
        if resize_mask is not None:
            anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode="bilinear", align_corners=False)

        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4, device=anomaly_map.device)
        anomaly_map = gaussian_kernel(anomaly_map)
        max_ratio = 0.01
        if max_ratio == 0:
            sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
        else:
            anomaly_map = anomaly_map.flatten(1)
            sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, : int(anomaly_map.shape[1] * max_ratio)]
            sp_score = sp_score.mean(dim=1)
        pred_score = sp_score

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map_resized)

    @staticmethod
    def cal_anomaly_maps(
        source_feature_maps: list[torch.Tensor],
        target_feature_maps: list[torch.Tensor],
        out_size: int | tuple[int, int] = 392,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Calculate anomaly maps by comparing encoder and decoder features.

        Computes pixel-level anomaly maps by calculating cosine similarity between
        corresponding encoder (source) and decoder (target) feature maps. Lower
        cosine similarity indicates higher anomaly likelihood.

        Args:
            source_feature_maps (list[torch.Tensor]): List of encoder feature maps
                from different layer groups.
            target_feature_maps (list[torch.Tensor]): List of decoder feature maps
                from different layer groups.
            out_size (int | tuple[int, int]): Output size for anomaly maps.
                Defaults to 392.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: Tuple containing:
                - anomaly_map: Combined anomaly map averaged across all feature scales
                - a_map_list: List of individual anomaly maps for each feature scale

        Example:
            >>> encoder_features = [torch.randn(2, 768, 28, 28), torch.randn(2, 768, 28, 28)]
            >>> decoder_features = [torch.randn(2, 768, 28, 28), torch.randn(2, 768, 28, 28)]
            >>> anomaly_map, map_list = ViTill.cal_anomaly_maps(encoder_features, decoder_features)
        """
        if not isinstance(out_size, tuple):
            out_size = (out_size, out_size)

        a_map_list = []
        for i in range(len(target_feature_maps)):
            fs = source_feature_maps[i]
            ft = target_feature_maps[i]
            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
            a_map_list.append(a_map)
        anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
        return anomaly_map, a_map_list

    @staticmethod
    def _fuse_feature(feat_list: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple feature tensors by averaging.

        Takes a list of feature tensors and computes their element-wise average
        to create a fused representation.

        Args:
            feat_list (list[torch.Tensor]): List of feature tensors to fuse.

        Returns:
            torch.Tensor: Averaged feature tensor.

        Example:
            >>> features = [torch.randn(2, 768, 196), torch.randn(2, 768, 196)]
            >>> fused = ViTill._fuse_feature(features)
            >>> fused.shape  # torch.Size([2, 768, 196])
        """
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def generate_mask(self, feature_size: int, device: str = "cuda") -> torch.Tensor:
        """Generate attention mask for neighborhood masking in decoder.

        Creates a square attention mask that restricts attention to local neighborhoods
        for each spatial position. This helps the decoder focus on local patterns
        during reconstruction.

        Args:
            feature_size (int): Size of the feature map (assumes square features).
            device (str): Device to create the mask on. Defaults to 'cuda'.

        Returns:
            torch.Tensor: Attention mask with shape (H, W, H, W) where masked
                positions are filled with 0 and unmasked positions with 1.

        Note:
            The mask size is controlled by self.mask_neighbor_size. Set to 0
            to disable masking.
        """
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                    idx_h1,
                    idx_w1,
                    idx_h2_start:idx_h2_end,
                    idx_w2_start:idx_w2_end,
                ] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        mask_all = torch.ones(
            h * w + 1 + self.encoder.num_register_tokens,
            h * w + 1 + self.encoder.num_register_tokens,
            device=device,
        )
        mask_all[1 + self.encoder.num_register_tokens :, 1 + self.encoder.num_register_tokens :] = mask
        return mask_all


def get_gaussian_kernel(
    kernel_size: int = 3,
    sigma: int = 2,
    channels: int = 1,
    device: torch.device | None = None,
) -> torch.nn.Conv2d:
    """Create a Gaussian kernel for smoothing operations.

    Args:
        kernel_size: Size of the Gaussian kernel. Defaults to 3.
        sigma: Standard deviation of the Gaussian distribution. Defaults to 2.
        channels: Number of channels. Defaults to 1.
        device: Device to place the kernel on. Defaults to None.

    Returns:
        Gaussian convolution filter as a Conv2d layer.
    """
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.0
    variance = sigma**2.0

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance),
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        groups=channels,
        bias=False,
        padding=kernel_size // 2,
    )

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    if device is not None:
        gaussian_filter = gaussian_filter.to(device)

    return gaussian_filter


class BottleNeckMLP(nn.Module):
    """Multi-layer perceptron with bottleneck architecture for feature processing."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        grad: float = 1.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.grad = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the bottleneck MLP."""
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class LinearAttention2(nn.Module):
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
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ARG002
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


class DropKey(nn.Module):
    """DropKey regularization for attention mechanisms."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        """Apply DropKey regularization to attention weights."""
        if self.training:
            m_r = torch.ones_like(attn) * self.p
            attn = attn + torch.bernoulli(m_r) * -1e12
        return attn


class Attention(nn.Module):
    """Standard multi-head attention mechanism."""

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
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = DropKey(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through multi-head attention."""
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        attn = self.attn_drop(attn)
        attn = attn.softmax(dim=-1)

        if attn_mask is not None:
            attn = attn.clone()
            attn[:, :, attn_mask == 0.0] = 0.0

        x = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Apply stochastic depth (drop path) regularization.

    Args:
        x: Input tensor.
        drop_prob: Probability of dropping a path.
        training: Whether in training mode.

    Returns:
        Output tensor with potential path dropping applied.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply drop path regularization."""
        return drop_path(x, self.drop_prob or 0.0, self.training)


class Mlp(nn.Module):
    """Multi-layer perceptron for transformer blocks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class DecoderViTBlock(nn.Module):
    """Vision Transformer decoder block with attention and MLP layers."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        attn: type[nn.Module] = Attention,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through decoder block."""
        if attn_mask is not None:
            y, attn = self.attn(self.norm1(x), attn_mask=attn_mask)
        else:
            y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x
