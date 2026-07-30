# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Composable synthetic anomaly generator.

``SyntheticAnomalyGenerator`` composes four independent pipeline axes -- mask shape,
region, content source and blend -- into a single ``v2.Transform`` that maps a normal
image to a synthetic anomalous image and its ground-truth mask. It is a drop-in
replacement for :class:`PerlinAnomalyGenerator` and plugs into
:func:`anomalib.data.utils.synthetic.make_synthetic_dataset`.

Example:
    >>> from anomalib.data.utils.generators import SyntheticAnomalyGenerator
    >>> generator = SyntheticAnomalyGenerator.from_preset("self_poisson")
    >>> image = torch.rand(3, 256, 256)
    >>> anomalous, mask = generator(image)
"""

from dataclasses import dataclass, field, replace

import torch
from torchvision.transforms import v2

from .blend import AlphaBlend, BlendStrategy, HybridBlend, PoissonBlend
from .mask import MaskGenerator, PerlinMaskGenerator
from .region import ForegroundRegion, RegionSelector, WholeImageRegion
from .source import AnomalySource, SelfSource, TextureSource

MASK_GENERATORS: dict[str, type[MaskGenerator]] = {"perlin": PerlinMaskGenerator}
REGIONS: dict[str, type[RegionSelector]] = {"whole": WholeImageRegion, "foreground": ForegroundRegion}
SOURCES: dict[str, type[AnomalySource]] = {"self": SelfSource, "texture": TextureSource}
BLENDS: dict[str, type[BlendStrategy]] = {"alpha": AlphaBlend, "poisson": PoissonBlend, "hybrid": HybridBlend}


@dataclass
class PipelineConfig:
    """Declarative configuration of a synthetic-anomaly pipeline.

    Args:
        mask (str): Mask generator key (see ``MASK_GENERATORS``).
        region (str): Region selector key (see ``REGIONS``).
        source (str): Content source key (see ``SOURCES``).
        blend (str): Blend strategy key (see ``BLENDS``).
        probability (float): Probability of applying an anomaly to an image.
        source_kwargs (dict): Extra arguments for the source (e.g. ``texture_path``).
        blend_kwargs (dict): Extra arguments for the blend (e.g. ``blend_factor``).
    """

    mask: str = "perlin"
    region: str = "whole"
    source: str = "self"
    blend: str = "alpha"
    probability: float = 0.5
    source_kwargs: dict = field(default_factory=dict)
    blend_kwargs: dict = field(default_factory=dict)


PIPELINE_PRESETS: dict[str, PipelineConfig] = {
    # Object-aware, same-material self-blend -- subtle, realistic defects.
    "self_alpha": PipelineConfig(region="foreground", source="self", blend="alpha"),
    "self_poisson": PipelineConfig(region="foreground", source="self", blend="poisson"),
    # Area-routed blend: alpha on small components, Poisson on large ones.
    "self_hybrid": PipelineConfig(region="foreground", source="self", blend="hybrid"),
    # DRAEM-style external texture paste (requires ``source_kwargs={"texture_path": ...}``).
    "texture_alpha": PipelineConfig(region="whole", source="texture", blend="alpha"),
}


class SyntheticAnomalyGenerator(v2.Transform):
    """Compose mask, region, source and blend strategies into an anomaly generator.

    Args:
        mask_generator (MaskGenerator): Produces the anomaly shape.
        region (RegionSelector): Constrains where the anomaly may appear.
        source (AnomalySource): Produces the anomaly content.
        blend (BlendStrategy): Composites the content into the image.
        probability (float): Probability of applying an anomaly to an image.
    """

    def __init__(
        self,
        mask_generator: MaskGenerator,
        region: RegionSelector,
        source: AnomalySource,
        blend: BlendStrategy,
        probability: float = 0.5,
    ) -> None:
        super().__init__()
        self.mask_generator = mask_generator
        self.region = region
        self.source = source
        self.blend = blend
        self.probability = probability

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "SyntheticAnomalyGenerator":
        """Build a generator from a :class:`PipelineConfig`."""
        return cls(
            mask_generator=MASK_GENERATORS[config.mask](),
            region=REGIONS[config.region](),
            source=SOURCES[config.source](**config.source_kwargs),
            blend=BLENDS[config.blend](**config.blend_kwargs),
            probability=config.probability,
        )

    @classmethod
    def from_preset(cls, name: str, **overrides: object) -> "SyntheticAnomalyGenerator":
        """Build a generator from a named preset in ``PIPELINE_PRESETS``.

        Args:
            name (str): Preset key.
            **overrides: Fields to override on the preset config.

        Raises:
            KeyError: If ``name`` is not a known preset.
        """
        if name not in PIPELINE_PRESETS:
            msg = f"Unknown preset '{name}'. Available: {sorted(PIPELINE_PRESETS)}."
            raise KeyError(msg)
        config = replace(PIPELINE_PRESETS[name], **overrides) if overrides else PIPELINE_PRESETS[name]
        return cls.from_config(config)

    def _apply(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, height, width = image.shape
        if torch.rand(1, device=image.device) > self.probability:
            return image, torch.zeros((1, height, width), device=image.device)

        mask = torch.zeros((1, height, width), device=image.device)
        for _ in range(10):
            candidate = self.region.constrain(image, self.mask_generator.generate(height, width, image.device))
            if candidate.max() > 0:
                mask = candidate
                break
        if mask.max() == 0:
            return image, mask

        source = self.source.get(image, mask)
        return self.blend.blend(image, source, mask), mask

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply synthetic anomalies to a single image or a batch.

        Args:
            image (torch.Tensor): Image of shape ``[C, H, W]`` or batch ``[B, C, H, W]``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The anomalous image(s) and mask(s), with a
            leading batch dimension only when the input has one.
        """
        if image.ndim == 4:
            results = [self._apply(sample) for sample in image]
            images = torch.stack([result[0] for result in results])
            masks = torch.stack([result[1] for result in results])
            return images, masks
        return self._apply(image)
