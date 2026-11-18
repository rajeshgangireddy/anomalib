# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Registered single-axis Phase 1 configurations and curated Phase 2 combinations."""

from __future__ import annotations

from dataclasses import dataclass

CATEGORY_SYNTH_TARGETS: dict[str, int] = {
    "can": 90,
    "fabric": 90,
    "fruit_jelly": 60,
    "rice": 90,
    "sheet_metal": 90,
    "vial": 105,
    "wallplugs": 90,
    "walnuts": 90,
}
PROTOCOL_ID = "paper_heldout_v1"
GENERATION_ID = "paper_components_v1"


@dataclass(frozen=True)
class AblationSpec:
    """One synthesis configuration, with the verified original paper settings as defaults."""

    name: str
    axis: str
    description: str
    obs: str = "full"
    mrsp: str = "object"
    adaptive_fraction: float = 0.5
    harmonise: bool = True
    blend: str = "paper_hybrid"
    retention_rescue: bool = False
    donors_per_category: int = 3
    n_synth: int | None = None

    def n_synth_for(self, category: str) -> int:
        """Return the category budget, or an explicit budget-sensitivity override.

        Args:
            category (str): MVTec AD 2 category.

        Returns:
            int: Number of unique synthetic positives to generate and calibrate on.

        Raises:
            KeyError: If the category is not in the paper evaluation.
        """
        target = CATEGORY_SYNTH_TARGETS[category]
        return target if self.n_synth is None else self.n_synth

    @property
    def source_dir(self) -> str:
        """Return the directory suffix used by the evaluation harness."""
        return f"{GENERATION_ID}_{self.name}"


ARM_SPECS: tuple[AblationSpec, ...] = (
    AblationSpec("baseline", "baseline", "Restored original paper-generation reference."),
    AblationSpec(
        "obs_color_only",
        "obs",
        "OBS with the colour cue only.",
        obs="color",
    ),
    AblationSpec(
        "obs_texture_only",
        "obs",
        "OBS with the texture cue only.",
        obs="texture",
    ),
    AblationSpec(
        "obs_off",
        "obs",
        "Use the full frame instead of an OBS region.",
        obs="off",
    ),
    AblationSpec(
        "mrsp_frame_relative",
        "mrsp",
        "Threshold the MRSP field relative to the full image.",
        mrsp="frame",
    ),
    AblationSpec(
        "mrsp_uniform_blob",
        "mrsp",
        "Place a compact, uniformly located blob with matched object-relative area.",
        mrsp="uniform",
    ),
    AblationSpec(
        "placement_original",
        "placement",
        "Use original placement for every sample.",
        adaptive_fraction=0.0,
    ),
    AblationSpec(
        "placement_adaptive",
        "placement",
        "Use adaptive placement for every sample.",
        adaptive_fraction=1.0,
    ),
    AblationSpec(
        "harmonisation_off",
        "harmonisation",
        "Disable CIELAB background harmonisation.",
        harmonise=False,
    ),
    AblationSpec(
        "blend_alpha_only",
        "blending",
        "Use alpha compositing for every sample.",
        blend="alpha",
    ),
    AblationSpec(
        "blend_poisson_collar",
        "blending",
        "Force Poisson compositing with a substrate collar.",
        blend="poisson_collar",
    ),
    AblationSpec(
        "blend_poisson_no_collar",
        "blending",
        "Force Poisson compositing without a substrate collar.",
        blend="poisson_no_collar",
    ),
    AblationSpec(
        "bank_1_donor",
        "bank_size",
        "Subsample to one donor per category; rotate donor identity across seeds.",
        donors_per_category=1,
    ),
    AblationSpec(
        "bank_2_donors",
        "bank_size",
        "Subsample to two donors per category; rotate omitted donor across seeds.",
        donors_per_category=2,
    ),
    AblationSpec(
        "reuse_8",
        "bank_reuse",
        "Calibrate on eight unique anomalies without resampling (budget sensitivity).",
        n_synth=8,
    ),
    AblationSpec(
        "reuse_16",
        "bank_reuse",
        "Calibrate on 16 unique anomalies without resampling (budget sensitivity).",
        n_synth=16,
    ),
)

EXTENSION_SPECS: tuple[AblationSpec, ...] = (
    AblationSpec(
        "blend_retention_rescue",
        "blending_extension",
        "Add bounded retention rescue to the original paper routing; not a component removal.",
        retention_rescue=True,
    ),
)

# Phase 2: cross-stage combinations selected after the Phase 1 single-axis results. They are
# evaluated post hoc on the same categories and are reported as such, never as the method default.
COMBINATION_SPECS: tuple[AblationSpec, ...] = (
    AblationSpec(
        "combo_color_obs_harmonisation_off",
        "combination",
        "Colour-only OBS with CIELAB harmonisation disabled.",
        obs="color",
        harmonise=False,
    ),
    AblationSpec(
        "combo_color_obs_2_donors",
        "combination",
        "Colour-only OBS with two donors per category.",
        obs="color",
        donors_per_category=2,
    ),
    AblationSpec(
        "combo_harmonisation_off_alpha_only",
        "combination",
        "Harmonisation disabled with alpha compositing for every sample.",
        harmonise=False,
        blend="alpha",
    ),
    AblationSpec(
        "combo_pixel_positive_stack",
        "combination",
        "Colour-only OBS, harmonisation off, adaptive placement only and two donors per category.",
        obs="color",
        adaptive_fraction=1.0,
        harmonise=False,
        donors_per_category=2,
    ),
    AblationSpec(
        "combo_obs_off_uniform_blob",
        "combination",
        "Full frame instead of OBS with a uniformly located blob (both localisation priors removed).",
        obs="off",
        mrsp="uniform",
    ),
    AblationSpec(
        "combo_naive_cutpaste",
        "combination",
        "Naive cut-paste control: full frame, uniform blob, original placement, no harmonisation, alpha.",
        obs="off",
        mrsp="uniform",
        adaptive_fraction=0.0,
        harmonise=False,
        blend="alpha",
    ),
)

ARM_BY_NAME = {spec.name: spec for spec in (*ARM_SPECS, *EXTENSION_SPECS, *COMBINATION_SPECS)}
