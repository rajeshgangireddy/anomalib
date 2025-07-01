"""
vit_encoder_clean.py
A clean, modular, and extensible Vision Transformer (ViT) encoder loader for anomaly detection and related tasks.
"""
import os
import torch
import logging
from typing import Optional
from torch.nn import Module

# Import backbone sources
from .dinov2.models import vision_transformer as dino_v2

_logger = logging.getLogger(__name__)

WEIGHTS_DIR = "backbones/weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

class ViTEncoderLoader:
    """
    Loader for various Vision Transformer (ViT) backbones with pre-trained weights.
    Supports DINOv1, DINOv2, BEiT, and can be extended easily.
    """
    SUPPORTED_BACKBONES = {
        'dinov2': dino_v2,
    }

    @staticmethod
    def download_file(url: str, filename: Optional[str] = None) -> str:
        """Download file if not present in cache."""
        from torch.hub import download_url_to_file, urlparse
        if not filename:
            filename = os.path.basename(urlparse(url).path)
        cached_file = os.path.join(WEIGHTS_DIR, filename)
        if not os.path.exists(cached_file):
            _logger.info(f"Downloading: {url} to {cached_file}")
            download_url_to_file(url, cached_file, progress=True)
        return cached_file

    @classmethod
    def load_encoder(cls, name: str) -> Module:
        """
        Load a ViT encoder by name. Example names:
        - 'dinov2reg_vit_base_14'
        - 'dinov1_vit_base_16'
        - 'beitv2_vit_base_16'
        """
        name = name.lower()
        if name.startswith('dinov2'):
            return cls._load_dinov2(name)
        else:
            raise ValueError(f"Unknown backbone: {name}")

    @classmethod
    def _load_dinov2(cls, name: str) -> Module:
        # Example: dinov2reg_vit_base_14
        arch, patch = cls._parse_arch_patch(name)
        reg = 'reg' in name
        model = dino_v2.__dict__[f'vit_{arch}'](
            patch_size=int(patch),
            img_size=518,
            block_chunks=0,
            init_values=1e-8,
            num_register_tokens=4 if reg else 0,
            interpolate_antialias=False,
            interpolate_offset=0.1
        )
        # Download weights
        url = cls._dinov2_url(arch, patch, reg)
        ckpt = torch.load(cls.download_file(url), map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        return model


    @staticmethod
    def _parse_arch_patch(name: str):
        # Extracts arch and patch size from name
        parts = name.split('_')
        arch = parts[-2]
        patch = parts[-1]
        return arch, patch

    @staticmethod
    def _dinov2_url(arch: str, patch: str, reg: bool) -> str:
        base = f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vit{arch[0]}{patch}/dinov2_vit{arch[0]}{patch}"
        if reg:
            return f"{base}_reg4_pretrain.pth"
        else:
            return f"{base}_pretrain.pth"
