# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for HybridBlend, which routes mask components by area."""

import torch

from anomalib.data.utils.generators import HybridBlend, SyntheticAnomalyGenerator
from anomalib.data.utils.generators.blend import AlphaBlend, PoissonBlend

SIZE = 128


def _disc(radius: int, center_y: int, center_x: int) -> torch.Tensor:
    """Return a ``[1, SIZE, SIZE]`` binary disc mask."""
    ys, xs = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
    return (((ys - center_y) ** 2 + (xs - center_x) ** 2) < radius**2).float().unsqueeze(0)


def _fixtures() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    image = torch.full((3, SIZE, SIZE), 0.4)
    source = torch.full((3, SIZE, SIZE), 0.9)
    return image, source


def test_hybrid_blend_preserves_shape() -> None:
    """Verify the blended output keeps the input shape."""
    image, source = _fixtures()
    blended = HybridBlend().blend(image, source, _disc(20, 64, 64))
    assert blended.shape == image.shape


def test_small_component_matches_alpha_not_poisson() -> None:
    """A component below the threshold should stay visible like alpha blending."""
    image, source = _fixtures()
    mask = _disc(6, 64, 64)
    hybrid = HybridBlend(area_threshold=10_000, blend_factor=1.0).blend(image, source, mask)
    alpha = AlphaBlend(blend_factor=1.0).blend(image, source, mask)
    poisson = PoissonBlend().blend(image, source, mask)

    selected = mask[0] > 0
    hybrid_visibility = (hybrid - image).abs().mean(0)[selected].mean()
    poisson_visibility = (poisson - image).abs().mean(0)[selected].mean()

    assert torch.allclose(hybrid, alpha, atol=1e-5)
    assert hybrid_visibility > poisson_visibility


def test_large_component_uses_poisson() -> None:
    """A component at or above the threshold should be Poisson blended."""
    image, source = _fixtures()
    mask = _disc(30, 64, 64)
    hybrid = HybridBlend(area_threshold=100, blend_factor=1.0).blend(image, source, mask)
    poisson = PoissonBlend().blend(image, source, mask)
    assert torch.allclose(hybrid, poisson, atol=1e-5)


def test_mixed_mask_routes_each_component() -> None:
    """With both sizes present, the small blob stays more visible than the large one."""
    image, source = _fixtures()
    small, large = _disc(5, 30, 30), _disc(28, 90, 90)
    blended = HybridBlend(area_threshold=1000, blend_factor=1.0).blend(image, source, (small + large).clamp(0, 1))
    delta = (blended - image).abs().mean(0)
    assert delta[small[0] > 0].mean() > delta[large[0] > 0].mean()


def test_empty_mask_returns_image_unchanged() -> None:
    """An all-zero mask should leave the image untouched."""
    image, source = _fixtures()
    blended = HybridBlend().blend(image, source, torch.zeros(1, SIZE, SIZE))
    assert torch.allclose(blended, image)


def test_self_hybrid_preset_builds_hybrid_blend() -> None:
    """The ``self_hybrid`` preset should wire up HybridBlend."""
    generator = SyntheticAnomalyGenerator.from_preset("self_hybrid", probability=1.0)
    assert isinstance(generator.blend, HybridBlend)

    image, mask = generator(torch.rand(3, SIZE, SIZE))
    assert image.shape == (3, SIZE, SIZE)
    assert mask.shape == (1, SIZE, SIZE)
