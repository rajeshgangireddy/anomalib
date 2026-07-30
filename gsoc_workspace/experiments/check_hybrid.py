# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sanity check that HybridBlend routes components by area as intended."""

from __future__ import annotations

import torch

from anomalib.data.utils.generators import HybridBlend, SyntheticAnomalyGenerator
from anomalib.data.utils.generators.blend import AlphaBlend, PoissonBlend

torch.manual_seed(0)
size = 256
image = torch.rand(3, size, size) * 0.2 + 0.4
source = torch.rand(3, size, size) * 0.2 + 0.7


def disc(radius: int, cy: int, cx: int) -> torch.Tensor:
    """Return a binary disc mask of the given radius centred at ``(cy, cx)``."""
    ys, xs = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    return (((ys - cy) ** 2 + (xs - cx) ** 2) < radius**2).float().unsqueeze(0)


small, large = disc(8, 60, 60), disc(45, 170, 170)
mask = (small + large).clamp(0, 1)
print(f"small blob = {int(small.sum())} px, large blob = {int(large.sum())} px, threshold = 2500 px")

hybrid = HybridBlend(area_threshold=2500, blend_factor=1.0)
alpha = AlphaBlend(blend_factor=1.0)
poisson = PoissonBlend()

for name, blend in [("alpha", alpha), ("poisson", poisson), ("hybrid", hybrid)]:
    out = blend.blend(image, source, mask)
    delta = (out - image).abs().mean(0)
    vis_small = float(delta[small[0] > 0].mean())
    vis_large = float(delta[large[0] > 0].mean())
    print(f"{name:8s} visibility: small blob {vis_small:.4f} | large blob {vis_large:.4f}")

print("\n-- preset wiring --")
gen = SyntheticAnomalyGenerator.from_preset("self_hybrid", probability=1.0)
print("blend class:", type(gen.blend).__name__, "| threshold:", gen.blend.area_threshold)
img, msk = gen(torch.rand(3, size, size))
print("output image:", tuple(img.shape), "| mask:", tuple(msk.shape), "| mask px:", int(msk.sum()))
