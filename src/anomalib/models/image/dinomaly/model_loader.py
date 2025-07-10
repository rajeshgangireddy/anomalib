"""Vision Transformer Encoder for DINOmaly model.

This module provides a simple interface for loading pre-trained DINOv2 Vision Transformer models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.hub import download_url_to_file

from .dinov2.models import vision_transformer as dinov2_models

logger = logging.getLogger(__name__)


class DinoV2Loader:
    """Simple loader for DINOv2 Vision Transformer models.

    Supports loading dinov2 and dinov2_reg models with small, base, and large architectures.
    """

    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"

    # Model configurations
    MODEL_CONFIGS = {
        "small": {"embed_dim": 384, "num_heads": 6},
        "base": {"embed_dim": 768, "num_heads": 12},
        "large": {"embed_dim": 1024, "num_heads": 16},
    }

    def __init__(self, cache_dir: str | Path = "backbones/weights") -> None:
        """Initialize model loader.

        Args:
            cache_dir: Directory to store downloaded model weights.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, model_name: str) -> torch.nn.Module:
        """Load a DINOv2 model by name.

        Args:
            model_name: Name like 'dinov2_vit_base_14' or 'dinov2reg_vit_small_14'.

        Returns:
            Loaded PyTorch model ready for inference.

        Raises:
            ValueError: If model name is invalid or unsupported.
        """
        # Parse model name
        model_type, architecture, patch_size = self._parse_name(model_name)

        # Create model
        model = self._create_model(model_type, architecture, patch_size)

        # Load weights
        self._load_weights(model, model_type, architecture, patch_size)

        logger.info(f"Loaded model: {model_name}")
        return model

    def _parse_name(self, name: str) -> tuple[str, str, int]:
        """Parse model name into components."""
        parts = name.split("_")

        if len(parts) < 3:
            raise ValueError(f"Invalid model name: {name}")

        # Determine model type and extract architecture/patch_size
        if "dinov2reg" in name or "reg" in name:
            model_type = "dinov2_reg"
            architecture = parts[-2]
            patch_size = int(parts[-1])
        else:
            model_type = "dinov2"
            architecture = parts[-2]
            patch_size = int(parts[-1])

        if architecture not in self.MODEL_CONFIGS:
            valid_archs = list(self.MODEL_CONFIGS.keys())
            raise ValueError(f"Unsupported architecture: {architecture}. Valid: {valid_archs}")

        return model_type, architecture, patch_size

    def _create_model(self, model_type: str, architecture: str, patch_size: int) -> torch.nn.Module:
        """Create model with appropriate configuration."""
        model_kwargs = {
            "patch_size": patch_size,
            "img_size": 518,
            "block_chunks": 0,
            "init_values": 1e-8,
            "interpolate_antialias": False,
            "interpolate_offset": 0.1,
        }

        # Add register tokens for reg models
        if model_type == "dinov2_reg":
            model_kwargs["num_register_tokens"] = 4

        # Get model constructor function
        model_fn = getattr(dinov2_models, f"vit_{architecture}", None)
        if model_fn is None:
            raise ValueError(f"Model function vit_{architecture} not found")

        return model_fn(**model_kwargs)

    def _load_weights(self, model: torch.nn.Module, model_type: str, architecture: str, patch_size: int) -> None:
        """Download and load model weights."""
        weight_path = self._get_weight_path(model_type, architecture, patch_size)

        if not weight_path.exists():
            self._download_weights(model_type, architecture, patch_size)

        state_dict = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    def _get_weight_path(self, model_type: str, architecture: str, patch_size: int) -> Path:
        """Get local path for model weights."""
        arch_code = architecture[0]  # s, b, or l

        if model_type == "dinov2_reg":
            filename = f"dinov2_vit{arch_code}{patch_size}_reg4_pretrain.pth"
        else:
            filename = f"dinov2_vit{arch_code}{patch_size}_pretrain.pth"

        return self.cache_dir / filename

    def _download_weights(self, model_type: str, architecture: str, patch_size: int) -> None:
        """Download model weights from official repository."""
        arch_code = architecture[0]
        weight_path = self._get_weight_path(model_type, architecture, patch_size)

        # Build download URL
        model_dir = f"dinov2_vit{arch_code}{patch_size}"
        url = f"{self.DINOV2_BASE_URL}/{model_dir}/{weight_path.name}"

        logger.info(f"Downloading weights from {url}")
        download_url_to_file(url, str(weight_path), progress=True)


def load(model_name: str) -> torch.nn.Module:
    """Convenience function to load a model.
    This can be later extended to be a factory method to load other models.

    Args:
        model_name: Name like 'dinov2_vit_base_14' or 'dinov2reg_vit_small_14'.

    Returns:
        Loaded PyTorch model.
    """
    loader = DinoV2Loader()
    return loader.load(model_name)
