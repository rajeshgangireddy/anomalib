"""Vision Transformer Encoder for DINOmaly model.

This module provides a clean interface for loading pre-trained DINOv2 Vision Transformer models
with proper weight caching and error handling.
"""

import logging
import os
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

import torch
from torch.hub import HASH_REGEX, download_url_to_file

from .dinov2.models import vision_transformer as vision_transformer_dinov2

logger = logging.getLogger(__name__)


class VisionTransformerEncoder:
    """Main class for loading Vision Transformer encoders.
    
    This class provides a unified interface for loading DINOv2 Vision Transformer models
    with proper weight management and caching.
    """
    
    # Model configuration constants
    DEFAULT_IMG_SIZE = 518
    DEFAULT_BLOCK_CHUNKS = 0
    DEFAULT_INIT_VALUES = 1e-8
    DEFAULT_NUM_REGISTER_TOKENS = 4
    DEFAULT_INTERPOLATE_ANTIALIAS = False
    DEFAULT_INTERPOLATE_OFFSET = 0.1
    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    
    # Supported architectures for each model type
    SUPPORTED_MODELS = {
        'dinov2': {'small', 'base'},
        'dinov2_reg': {'small', 'base', 'large'}
    }
    
    def __init__(self, weights_dir: str = "backbones/weights"):
        """Initialize the Vision Transformer encoder.
        
        Args:
            weights_dir: Directory to store downloaded weights.
        """
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
    
    def _parse_model_name(self, name: str) -> Tuple[str, str, int]:
        """Parse model name to extract components.
        
        Args:
            name: Model name in format like 'dinov2_vit_base_14' or 'dinov2_reg_vit_small_14'.
            
        Returns:
            Tuple of (model_type, architecture, patch_size).
            
        Raises:
            ValueError: If model name format is invalid.
        """
        parts = name.split("_")
        
        if len(parts) < 4 or "dino" not in name or "v2" not in name:
            raise ValueError(f"Invalid model name format: {name}. Expected format: 'dinov2_vit_<arch>_<patch>' or 'dinov2_reg_vit_<arch>_<patch>'")
        
        # Extract patch size and architecture
        try:
            patch_size = int(parts[-1])
            architecture = parts[-2]
        except (ValueError, IndexError):
            raise ValueError(f"Invalid patch size or architecture in model name: {name}")
        
        # Determine model type
        model_type = 'dinov2_reg' if 'reg' in name else 'dinov2'
        
        # Validate architecture support
        if architecture not in self.SUPPORTED_MODELS[model_type]:
            raise ValueError(f"Architecture '{architecture}' not supported for {model_type}. "
                           f"Supported: {list(self.SUPPORTED_MODELS[model_type])}")
        
        return model_type, architecture, patch_size
    
    def _download_cached_file(self, url: str, check_hash: bool = True, progress: bool = True) -> str:
        """Download a file and cache it locally.
        
        Args:
            url: URL to download from.
            check_hash: Whether to verify file hash.
            progress: Whether to show download progress.
            
        Returns:
            Path to the cached file.
        """
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = self.weights_dir / filename
        
        if not cached_file.exists():
            logger.info(f'Downloading: "{url}" to {cached_file}')
            hash_prefix = None
            if check_hash:
                r = HASH_REGEX.search(filename)
                hash_prefix = r.group(1) if r else None
            download_url_to_file(url, str(cached_file), hash_prefix, progress=progress)
        
        return str(cached_file)
    
    def _get_checkpoint_url(self, model_type: str, architecture: str, patch_size: int) -> str:
        """Get checkpoint URL for DINOv2 model.
        
        Args:
            model_type: Type of model ('dinov2' or 'dinov2_reg').
            architecture: Model architecture ('small', 'base', 'large').
            patch_size: Patch size for the model.
            
        Returns:
            URL to the checkpoint file.
        """
        arch_code = architecture[0]  # 's' for small, 'b' for base, 'l' for large
        
        if model_type == 'dinov2_reg':
            filename = f"dinov2_vit{arch_code}{patch_size}_reg4_pretrain.pth"
        else:
            filename = f"dinov2_vit{arch_code}{patch_size}_pretrain.pth"
        
        return f"{self.DINOV2_BASE_URL}/dinov2_vit{arch_code}{patch_size}/{filename}"
    
    def _create_model(self, model_type: str, architecture: str, patch_size: int) -> torch.nn.Module:
        """Create a DINOv2 model with the specified configuration.
        
        Args:
            model_type: Type of model ('dinov2' or 'dinov2_reg').
            architecture: Model architecture ('small', 'base', 'large').
            patch_size: Patch size for the model.
            
        Returns:
            Initialized DINOv2 model.
        """
        # Base model configuration
        model_kwargs = {
            'patch_size': patch_size,
            'img_size': self.DEFAULT_IMG_SIZE,
            'block_chunks': self.DEFAULT_BLOCK_CHUNKS,
            'init_values': self.DEFAULT_INIT_VALUES,
            'interpolate_antialias': self.DEFAULT_INTERPOLATE_ANTIALIAS,
            'interpolate_offset': self.DEFAULT_INTERPOLATE_OFFSET
        }
        
        # Add register tokens if needed
        if model_type == 'dinov2_reg':
            model_kwargs['num_register_tokens'] = self.DEFAULT_NUM_REGISTER_TOKENS
        
        model = vision_transformer_dinov2.__dict__[f'vit_{architecture}'](**model_kwargs)
        return model
    
    def load(self, name: str) -> torch.nn.Module:
        """Load a Vision Transformer model by name.
        
        Args:
            name: Model name (e.g., 'dinov2_vit_base_14', 'dinov2_reg_vit_small_14').
            
        Returns:
            Loaded and initialized PyTorch model.
            
        Raises:
            ValueError: If model name is invalid or unsupported.
            
        Example:
            >>> encoder = VisionTransformerEncoder()
            >>> model = encoder.load('dinov2_vit_base_14')
        """
        try:
            # Parse model name
            model_type, architecture, patch_size = self._parse_model_name(name)
            
            # Create model
            model = self._create_model(model_type, architecture, patch_size)
            
            # Download and load weights
            checkpoint_url = self._get_checkpoint_url(model_type, architecture, patch_size)
            checkpoint_path = self._download_cached_file(checkpoint_url)
            
            # Load state dict
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            
            logger.info(f"Successfully loaded model: {name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            raise


# Factory function for backward compatibility
def load(name: str) -> torch.nn.Module:
    """Load a Vision Transformer model by name.
    
    This is a convenience function that maintains backward compatibility with the original API.
    
    Args:
        name: Model name (e.g., 'dinov2_vit_base_14', 'dinov2_reg_vit_small_14').
        
    Returns:
        Loaded and initialized PyTorch model.
    """
    encoder = VisionTransformerEncoder()
    return encoder.load(name)




