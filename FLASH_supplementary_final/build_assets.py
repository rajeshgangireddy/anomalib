# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build publication assets from immutable measurements, without running detectors."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import math
import shutil
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
FLASH = Path(
    "/home/devuser/.copilot/session-state/8a2f0e48-4fe2-4d47-ba4d-4bb694881422/files/flash",
)
LAUNCHES = FLASH / "evaluation/ablations/results/launches"
CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
BANK_CATEGORIES = ("rice", "fruit_jelly", "walnuts")
BANK_SIZES = (3, 5, 7, 9, 11, 14)
SHORTLIST = ("baseline", "obs_color_only", "combo_color_obs_2_donors", "combo_naive_cutpaste")
PIXEL_COLORS = {
    "rice": ("#0046C8", "#6CC4F5", "o"),
    "fruit_jelly": ("#E96115", "#FFC599", "s"),
    "walnuts": ("#708541", "#D7F3A2", "^"),
}
PIXEL_AXIS_RANGES: dict[str, tuple[int, int]] = {
    "SuperADD": (48, 72),
    "Dinomaly": (14, 52),
    "PatchCore": (0, 56),
    "PaDiM": (0, 20),
}
PIXEL_V2_AXIS_RANGES = {
    "SuperADD": (48, 72),
    "Dinomaly": (16, 49),
    "PatchCore": (0, 53),
    "PaDiM": (0, 20),
}
PIXEL_V3_AXIS_RANGES = {
    "SuperADD": (48, 72),
    "Dinomaly": (16, 49),
    "PatchCore": (0, 53),
    "PaDiM": (0, 20),
}
PIXEL_V2_AXIS_BREAKS = {
    "SuperADD": (60.91, 61.96),
    "Dinomaly": None,
    "PatchCore": (34.0, 46.0),
    "PaDiM": (6.0, 9.5),
}
PIXEL_V3_AXIS_BREAKS = {
    "SuperADD": (60.15, 62.7),
    "Dinomaly": None,
    "PatchCore": (32.5, 47.5),
    "PaDiM": (4.7, 10.9),
}
PIXEL_V3_MIN_BREAK_CLEARANCE = 0.2
PIXEL_V4_AXIS_RANGES = {
    "SuperADD": (48, 72),
    "Dinomaly": (16, 49),
    "PatchCore": (0, 53),
    "PaDiM": (0, 20),
}
PIXEL_V4_AXIS_BREAKS = {
    "SuperADD": (58.85, 65.6),
    "Dinomaly": None,
    "PatchCore": (25.05, 48.3),
    "PaDiM": (3.25, 13.8),
}
PIXEL_V4_MIN_BREAK_CLEARANCE = 0.5
PIXEL_V5_AXIS_RANGES = {**PIXEL_V4_AXIS_RANGES, "PaDiM": (0, 16)}
SOURCES = {
    "component_p1.csv": (
        LAUNCHES / "20261022_phase1_paper_components_v1/phase1_results.csv",
        "c2468fc11cb86d345fbc013b6acf3b6f36fed768a1755e6946e1c25f8bd1a52b",
    ),
    "component_p2.csv": (
        LAUNCHES / "20261027_phase2_paper_combinations_v2/phase1_results.csv",
        "ad9e864949c1c95ea72f3633f3a864289f331a05e7f382ad49ee0fec9d8de304",
    ),
    "component_extra_seeds.csv": (
        LAUNCHES / "20261027_threshold_audit_stability_patchcore_v2/stability_results.csv",
        "77fb7a9d6995cbddfeae0fbb12f89813857a35df7d83613018a7eb34f0ef5dc2",
    ),
    "component_audit.csv": (
        LAUNCHES / "20261027_threshold_audit_stability_patchcore_v2/audit_results.csv",
        "f9295fe4a0a1cc9d09b4545572895bec0e433242b2bd98f0cb76db18081a1b64",
    ),
    "legacy_patchcore_components.csv": (
        LAUNCHES / "20261027_threshold_audit_stability_patchcore_v2/patchcore_results.csv",
        "0b0988159707753286e51c2f992665dcbb865a06593e1873139462ca355a1e5c",
    ),
    "SuperADD": (
        ROOT / "bank_size_results.csv",
        "8c86beda33297008d47ada2e820f5b432d038abc8476c36df9381867a43aa803",
    ),
    "Dinomaly": (
        ROOT / "dinomaly_bank_size_results.csv",
        "51e9a3b8d52487a04c989e4471748c0b60ca4a21e9866df5a994343c2afdba68",
    ),
    "PatchCore": (
        ROOT / "patchcore_bank_size_results.csv",
        "49491d742af25793717fe9ddca061c5fe71f18a81d8952258d5bcf3dfc9c3384",
    ),
    "PaDiM": (
        ROOT / "padim_bank_size_results.csv",
        "933e643b4d8717bdc9fe1326eadd20077085a030ad08fbc1a43a45a1d9e88dc5",
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict], seed_key: str) -> dict:
    seeds = sorted({int(row[seed_key]) for row in rows})
    categories = sorted({row["category"] for row in rows})
    if len(rows) != len(seeds) * len(categories):
        raise ValueError("Unbalanced or duplicate category/seed measurements")
    result = {"n_categories": len(categories), "n_seeds": len(seeds)}
    for level in ("pixel", "image"):
        key = f"{level}_F1Score"
        values = [
            100 * statistics.mean(float(row[key]) for row in rows if int(row[seed_key]) == seed)
            for seed in seeds
        ]
        result[f"{level}_f1_mean_percent"] = statistics.mean(values)
        result[f"{level}_f1_seed_sd_pp"] = statistics.stdev(values)
    return result


def _component_tables(data: dict[str, list[dict]], specs: dict) -> tuple[list[dict], list[dict]]:
    p1, p2 = data["component_p1.csv"], data["component_p2.csv"]
    for rows, count in ((p1, 16), (p2, 7)):
        arms = {row["arm"] for row in rows}
        keys = {(row["arm"], row["category"], int(row["seed"])) for row in rows}
        if (
            len(arms) != count
            or len(rows) != count * 24
            or keys
            != set(
                itertools.product(arms, CATEGORIES, range(3)),
            )
        ):
            raise ValueError("Incomplete original component matrix")
    baseline = {(row["category"], row["seed"]): row for row in p1 if row["arm"] == "baseline"}
    for row in p2:
        if row["arm"] == "baseline":
            reference = baseline[(row["category"], row["seed"])]
            if any(row[f"{level}_F1Score"] != reference[f"{level}_F1Score"] for level in ("pixel", "image")):
                raise ValueError("P2 baseline does not reproduce original paired F1")
    combined = p1 + [row for row in p2 if row["arm"] != "baseline"]
    macro, category = [], []
    for arm in dict.fromkeys(row["arm"] for row in combined):
        rows = [row for row in combined if row["arm"] == arm]
        source = "P2" if arm.startswith("combo_") else "P1"
        config = asdict(specs[arm])
        for cat in (None, *CATEGORIES):
            selected = rows if cat is None else [row for row in rows if row["category"] == cat]
            control = (
                list(baseline.values())
                if cat is None
                else [row for row in baseline.values() if row["category"] == cat]
            )
            stats = _summary(selected, "seed")
            reference = _summary(control, "seed")
            record = {
                "arm": arm,
                "category": cat or "macro",
                "model": "SuperADD",
                "cohort": "original_seeds_0_1_2",
                "source": source,
                "exploratory_combination": source == "P2",
                "budget_sensitivity": arm in {"reuse_8", "reuse_16"},
                "obs": config["obs"],
                "mrsp": config["mrsp"],
                "adaptive_fraction": config["adaptive_fraction"],
                "harmonisation": config["harmonise"],
                "blend": config["blend"],
                "source_images": 3
                if config["donors_per_category"] is None
                else config["donors_per_category"],
                **stats,
                "delta_pixel_vs_baseline_pp": stats["pixel_f1_mean_percent"]
                - reference["pixel_f1_mean_percent"],
                "delta_image_vs_baseline_pp": stats["image_f1_mean_percent"]
                - reference["image_f1_mean_percent"],
                "invalid_source_pixel_auroc": any(
                    not 0 <= float(row["pixel_AUROC"]) <= 1 for row in selected
                ),
                "source_csv_sha256": SOURCES[f"component_{source.lower()}.csv"][1],
            }
            (macro if cat is None else category).append(record)
    _write("component_settings_macro.csv", macro)
    _write("component_settings_by_category.csv", category)
    extra = data["component_extra_seeds.csv"]
    keys = {(row["arm"], row["category"], int(row["generation_seed"])) for row in extra}
    if len(extra) != 64 or keys != set(itertools.product(SHORTLIST, CATEGORIES, (3, 4))):
        raise ValueError("Extra seeds are incomplete or not physical seeds3/4")
    five = [{**row, "physical_seed": int(row["seed"])} for row in combined if row["arm"] in SHORTLIST] + [
        {**row, "physical_seed": int(row["generation_seed"])} for row in extra
    ]
    stability_macro, stability_category = [], []
    for arm in SHORTLIST:
        for cat in (None, *CATEGORIES):
            selected = [row for row in five if row["arm"] == arm and (cat is None or row["category"] == cat)]
            control = [
                row for row in five if row["arm"] == "baseline" and (cat is None or row["category"] == cat)
            ]
            stats, reference = _summary(selected, "physical_seed"), _summary(control, "physical_seed")
            record = {
                "arm": arm,
                "category": cat or "macro",
                "model": "SuperADD",
                "cohort": "original_0_1_2_plus_physical_3_4",
                "exploratory_shortlist_not_independent_confirmation": True,
                **stats,
                "delta_pixel_vs_baseline_pp": stats["pixel_f1_mean_percent"]
                - reference["pixel_f1_mean_percent"],
                "delta_image_vs_baseline_pp": stats["image_f1_mean_percent"]
                - reference["image_f1_mean_percent"],
                "audit_replays_counted_as_seeds": False,
            }
            (stability_macro if cat is None else stability_category).append(record)
    _write("five_seed_shortlist_macro.csv", stability_macro)
    _write("five_seed_shortlist_by_category.csv", stability_category)
    return macro, category


def _bank_tables(data: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    macro, category = [], []
    for model in ("SuperADD", "Dinomaly", "PatchCore", "PaDiM"):
        rows = data[model]
        keys = {
            (row["category"], int(row["generation_seed"]), int(row["bank_size"]), row["eval_set"])
            for row in rows
        }
        if len(rows) != 108 or keys != set(
            itertools.product(
                BANK_CATEGORIES,
                range(3),
                BANK_SIZES,
                ("real_test", "heldout_synthetic"),
            )
        ):
            raise ValueError(f"Incomplete bank-size matrix: {model}")
        for evaluation in ("real_test", "heldout_synthetic"):
            for size in BANK_SIZES:
                for cat in (None, *BANK_CATEGORIES):
                    pool = [
                        row
                        for row in rows
                        if row["eval_set"] == evaluation and (cat is None or row["category"] == cat)
                    ]
                    selected = [row for row in pool if int(row["bank_size"]) == size]
                    control = [row for row in pool if int(row["bank_size"]) == 3]
                    stats = _summary(selected, "generation_seed")
                    reference = _summary(control, "generation_seed")
                    record = {
                        "model": model,
                        "category": cat or "macro",
                        "bank_size": size,
                        "eval_set": evaluation,
                        **stats,
                        "delta_pixel_vs_b3_pp": stats["pixel_f1_mean_percent"]
                        - reference["pixel_f1_mean_percent"],
                        "delta_image_vs_b3_pp": stats["image_f1_mean_percent"]
                        - reference["image_f1_mean_percent"],
                        "same_fit_including_b3": model != "SuperADD",
                        "auxiliary_synthetic_evaluation": evaluation != "real_test",
                        "source_csv_sha256": SOURCES[model][1],
                    }
                    (macro if cat is None else category).append(record)
    _write("bank_size_all_models_macro.csv", macro)
    _write("bank_size_all_models_by_category.csv", category)
    return macro, category


def _save(figure: plt.Figure, stem: str) -> None:
    figure.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _pixel_axis_range(rows: list[dict]) -> tuple[int, int]:
    lower = min(row["pixel_f1_mean_percent"] - row["pixel_f1_seed_sd_pp"] for row in rows)
    upper = max(row["pixel_f1_mean_percent"] + row["pixel_f1_seed_sd_pp"] for row in rows)
    padding = max((upper - lower) * 0.08, 0.5)
    tick_step = 2
    axis_min = max(0, math.floor((lower - padding) / tick_step) * tick_step)
    axis_max = min(100, math.ceil((upper + padding) / tick_step) * tick_step)
    return axis_min, axis_max


def _pixel_bank_figure(bank: list[dict], layout: str) -> None:
    if layout == "full_width":
        figsize, font_size, tick_size, line_width, marker_size = (7.2, 4.8), 8, 7, 1.55, 4
        margins = {"left": 0.10, "right": 0.99, "bottom": 0.14, "top": 0.84, "hspace": 0.42, "wspace": 0.28}
    else:
        figsize, font_size, tick_size, line_width, marker_size = (3.45, 4.2), 6.5, 5.5, 1.25, 3
        margins = {"left": 0.15, "right": 0.99, "bottom": 0.12, "top": 0.86, "hspace": 0.46, "wspace": 0.28}

    figure, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    figure.subplots_adjust(**margins)
    models = ("SuperADD", "Dinomaly", "PatchCore", "PaDiM")
    for axis, model in zip(axes.flat, models, strict=True):
        model_rows = [row for row in bank if row["model"] == model and row["eval_set"] == "real_test"]
        ymin, ymax = _pixel_axis_range(model_rows)
        PIXEL_AXIS_RANGES[model] = (ymin, ymax)
        for category, (line_color, band_color, marker) in PIXEL_COLORS.items():
            rows = sorted(
                (row for row in model_rows if row["category"] == category),
                key=lambda row: row["bank_size"],
            )
            means = np.asarray([row["pixel_f1_mean_percent"] for row in rows], dtype=float)
            deviations = np.asarray([row["pixel_f1_seed_sd_pp"] for row in rows], dtype=float)
            axis.fill_between(
                BANK_SIZES,
                means - deviations,
                means + deviations,
                color=band_color,
                alpha=0.52,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                BANK_SIZES,
                means,
                color=line_color,
                marker=marker,
                markersize=marker_size,
                linewidth=line_width,
                label=category.replace("_", " ").capitalize(),
                zorder=2,
            )
        title = model + (" (B=3 separate fit)" if model == "SuperADD" else "")
        axis.set_title(title, fontsize=font_size + 1, pad=3)
        axis.set_xticks(BANK_SIZES)
        axis.set_xlim(2.5, 14.5)
        axis.set_ylim(ymin, ymax)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        axis.tick_params(axis="both", labelsize=tick_size, length=2.5, pad=2)
        axis.grid(axis="y", color="#E9E9E9", linewidth=0.55, zorder=0)
        axis.spines["left"].set_color("#777777")
        axis.spines["bottom"].set_color("#777777")
        axis.spines["left"].set_linewidth(0.65)
        axis.spines["bottom"].set_linewidth(0.65)
        axis.text(
            0.98,
            0.96,
            f"{ymin}\u2013{ymax}%",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=tick_size - 0.5,
            color="#555555",
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        fontsize=tick_size,
        handlelength=1.7,
        columnspacing=1.3,
    )
    axes[0, 0].set_ylabel("Pixel F1 (%)", fontsize=font_size)
    axes[1, 0].set_ylabel("Pixel F1 (%)", fontsize=font_size)
    figure.supxlabel("Source-image bank size B", y=0.025, fontsize=font_size)
    stem = "bank_size_pixel_f1" if layout == "full_width" else "bank_size_pixel_f1_single_column"
    _save(figure, stem)


def _draw_pixel_series(
    axis: plt.Axes,
    rows: list[dict],
    marker_size: float,
    line_width: float,
    *,
    show_sd: bool = True,
) -> None:
    for category, (line_color, band_color, marker) in PIXEL_COLORS.items():
        series = sorted(
            (row for row in rows if row["category"] == category),
            key=lambda row: row["bank_size"],
        )
        means = np.asarray([row["pixel_f1_mean_percent"] for row in series], dtype=float)
        deviations = np.asarray([row["pixel_f1_seed_sd_pp"] for row in series], dtype=float)
        if show_sd:
            axis.fill_between(
                BANK_SIZES,
                means - deviations,
                means + deviations,
                color=band_color,
                alpha=0.32,
                linewidth=0,
                zorder=1,
            )
        axis.plot(
            BANK_SIZES,
            means,
            color=line_color,
            marker=marker,
            markersize=marker_size,
            linewidth=line_width,
            label=category.replace("_", " ").capitalize(),
            zorder=2,
        )


def _pixel_v2_figure(bank: list[dict], layout: str) -> None:
    if layout == "full_width":
        figsize, fontsize, ticksize = (7.2, 7.2), 8, 6.7
        margins = {"left": 0.14, "right": 0.99, "bottom": 0.18, "top": 0.96, "hspace": 0.30, "wspace": 0.34}
        marker_size, line_width, legend_y = 3.8, 1.35, 0.055
    else:
        figsize, fontsize, ticksize = (3.45, 4.3), 6.2, 5.0
        margins = {"left": 0.19, "right": 0.99, "bottom": 0.23, "top": 0.96, "hspace": 0.32, "wspace": 0.34}
        marker_size, line_width, legend_y = 2.8, 1.1, 0.07

    figure = plt.figure(figsize=figsize)
    grid = figure.add_gridspec(2, 2, **margins)
    models = ("SuperADD", "Dinomaly", "PatchCore", "PaDiM")
    panel_handles: list[plt.Line2D] = []
    panel_labels: list[str] = []
    for index, model in enumerate(models):
        model_rows = [row for row in bank if row["model"] == model and row["eval_set"] == "real_test"]
        ymin, ymax = PIXEL_V2_AXIS_RANGES[model]
        axis_break = PIXEL_V2_AXIS_BREAKS[model]
        intervals = [
            (
                row["pixel_f1_mean_percent"] - row["pixel_f1_seed_sd_pp"],
                row["pixel_f1_mean_percent"] + row["pixel_f1_seed_sd_pp"],
            )
            for row in model_rows
        ]
        if axis_break is not None:
            break_low, break_high = axis_break
            lower_cluster = [high for low, high in intervals if high <= break_low]
            upper_cluster = [low for low, high in intervals if low >= break_high]
            if (
                not lower_cluster
                or not upper_cluster
                or break_low - max(lower_cluster) < 1
                or min(upper_cluster) - break_high < 1
            ):
                raise ValueError(f"Pixel-F1 axis break lacks a clear band-free interval for {model}")
        row_index, column_index = divmod(index, 2)

        if axis_break is None:
            axis = figure.add_subplot(grid[row_index, column_index])
            axes = (axis,)
        else:
            break_low, break_high = axis_break
            lower_span = break_low - ymin
            upper_span = ymax - break_high
            if lower_span <= 0 or upper_span <= 0:
                raise ValueError(f"Invalid pixel-F1 axis break for {model}")
            cell = grid[row_index, column_index].subgridspec(
                2,
                1,
                height_ratios=(upper_span, lower_span),
                hspace=0.035,
            )
            upper_axis = figure.add_subplot(cell[0])
            lower_axis = figure.add_subplot(cell[1], sharex=upper_axis)
            upper_axis.set_ylim(break_high, ymax)
            lower_axis.set_ylim(ymin, break_low)
            upper_axis.spines["bottom"].set_visible(False)
            lower_axis.spines["top"].set_visible(False)
            upper_axis.tick_params(labelbottom=False, bottom=False)
            axes = (upper_axis, lower_axis)

        for axis in axes:
            _draw_pixel_series(axis, model_rows, marker_size, line_width)
            axis.set_xlim(2.75, 14.25)
            axis.set_xticks(BANK_SIZES)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            axis.tick_params(axis="both", labelsize=ticksize, length=2.2, pad=1.6)
            axis.grid(axis="y", color="#E9E9E9", linewidth=0.5, zorder=0)
            axis.spines["left"].set_color("#666666")
            axis.spines["bottom"].set_color("#666666")
            axis.spines["left"].set_linewidth(0.65)
            axis.spines["bottom"].set_linewidth(0.65)
            if axis_break is None:
                axis.set_box_aspect(1)

        title_axis = axes[0]
        title_axis.set_title(model, fontsize=fontsize + 1, pad=3)
        if len(axes) == 1:
            axes[0].set_xlabel("Bank size B", fontsize=fontsize)
            axes[0].set_ylabel("Pixel F1 (%)", fontsize=fontsize)
        else:
            axes[1].set_xlabel("Bank size B", fontsize=fontsize)
            axes[1].set_ylabel("Pixel F1 (%)", fontsize=fontsize)
            diagonal = 0.012
            mark_style = {"color": "#444444", "linewidth": 0.8, "clip_on": False}
            axes[0].plot(
                (-diagonal, diagonal),
                (-diagonal, diagonal),
                transform=axes[0].transAxes,
                **mark_style,
            )
            axes[1].plot(
                (-diagonal, diagonal),
                (1 - diagonal, 1 + diagonal),
                transform=axes[1].transAxes,
                **mark_style,
            )
        if axis_break is not None and any(
            low < axis_break[1] and high > axis_break[0] for low, high in intervals
        ):
            raise ValueError(f"Pixel-F1 axis break intersects a seed-spread band for {model}")
        handles, labels = axes[0].get_legend_handles_labels()
        if not panel_handles:
            panel_handles, panel_labels = handles, labels

    figure.legend(
        panel_handles,
        panel_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=3,
        frameon=False,
        fontsize=ticksize,
        handlelength=1.6,
        columnspacing=1.15,
    )
    stem = "bank_size_pixel_f1_v2" if layout == "full_width" else "bank_size_pixel_f1_single_column_v2"
    _save(figure, stem)


def _pixel_broken_figure(
    bank: list[dict],
    layout: str,
    *,
    version: str,
    axis_ranges: dict[str, tuple[float, float]],
    axis_breaks: dict[str, tuple[float, float] | None],
    min_break_clearance: float,
    show_sd: bool,
    compact_spacing: bool = False,
    xlabel: str = "Bank size B",
) -> None:
    if layout == "full_width":
        figsize, fontsize, ticksize = (7.2, 7.2), 8, 6.7
        margins = {"left": 0.14, "right": 0.99, "bottom": 0.18, "top": 0.96, "hspace": 0.30, "wspace": 0.34}
        marker_size, line_width, legend_y, label_offset = 3.8, 1.35, 0.055, 0.055
    else:
        figsize, fontsize, ticksize = (3.45, 4.3), 6.2, 5.0
        margins = {"left": 0.19, "right": 0.99, "bottom": 0.23, "top": 0.96, "hspace": 0.32, "wspace": 0.34}
        marker_size, line_width, legend_y, label_offset = 2.8, 1.1, 0.07, 0.07

    if compact_spacing:
        figsize = (7.2, 6.9) if layout == "full_width" else (3.45, 4.1)
        margins["bottom"] -= 0.02
        margins["hspace"] -= 0.05

    figure = plt.figure(figsize=figsize)
    grid = figure.add_gridspec(2, 2, **margins)
    models = ("SuperADD", "Dinomaly", "PatchCore", "PaDiM")
    panel_handles: list[plt.Line2D] = []
    panel_labels: list[str] = []
    for index, model in enumerate(models):
        model_rows = [row for row in bank if row["model"] == model and row["eval_set"] == "real_test"]
        ymin, ymax = axis_ranges[model]
        axis_break = axis_breaks[model]
        intervals = [
            (
                row["pixel_f1_mean_percent"] - (row["pixel_f1_seed_sd_pp"] if show_sd else 0.0),
                row["pixel_f1_mean_percent"] + (row["pixel_f1_seed_sd_pp"] if show_sd else 0.0),
            )
            for row in model_rows
        ]
        if axis_break is not None:
            break_low, break_high = axis_break
            lower_cluster = [high for low, high in intervals if high <= break_low]
            upper_cluster = [low for low, high in intervals if low >= break_high]
            if (
                not lower_cluster
                or not upper_cluster
                or break_low - max(lower_cluster) < min_break_clearance
                or min(upper_cluster) - break_high < min_break_clearance
            ):
                raise ValueError("Pixel-F1 axis break lacks a clear band-free interval")
            if not show_sd:
                category_means = {
                    category: [
                        row["pixel_f1_mean_percent"] for row in model_rows if row["category"] == category
                    ]
                    for category in BANK_CATEGORIES
                }
                if any(
                    min(values) < break_high and max(values) > break_low
                    for values in category_means.values()
                ):
                    raise ValueError("Pixel-F1 axis break crosses a connected category mean line")

        row_index, column_index = divmod(index, 2)
        panel = grid[row_index, column_index].subgridspec(
            2,
            1,
            height_ratios=(0.13, 0.87),
            hspace=0.0,
        )
        title_axis = figure.add_subplot(panel[0])
        title_axis.set_axis_off()
        title_axis.text(0.5, 0.45, model, ha="center", va="center", fontsize=fontsize + 1)
        plot_spec = panel[1]

        if axis_break is None:
            axis = figure.add_subplot(plot_spec)
            axis.set_ylim(ymin, ymax)
            axes = (axis,)
        else:
            break_low, break_high = axis_break
            lower_span = break_low - ymin
            upper_span = ymax - break_high
            if lower_span <= 0 or upper_span <= 0:
                raise ValueError("Invalid pixel-F1 axis break")
            cell = plot_spec.subgridspec(
                2,
                1,
                height_ratios=(upper_span, lower_span),
                hspace=0.035,
            )
            upper_axis = figure.add_subplot(cell[0])
            lower_axis = figure.add_subplot(cell[1], sharex=upper_axis)
            upper_axis.set_ylim(break_high, ymax)
            lower_axis.set_ylim(ymin, break_low)
            upper_axis.spines["bottom"].set_visible(False)
            lower_axis.spines["top"].set_visible(False)
            upper_axis.tick_params(labelbottom=False, bottom=False)
            axes = (upper_axis, lower_axis)

        for axis in axes:
            _draw_pixel_series(axis, model_rows, marker_size, line_width, show_sd=show_sd)
            axis.set_xlim(2.75, 14.25)
            axis.set_xticks(BANK_SIZES)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            if version == "v5" and model == "PaDiM" and axis is axes[0]:
                axis.set_yticks((14, 15, 16))
            axis.tick_params(axis="both", labelsize=ticksize, length=2.2, pad=1.6)
            axis.grid(axis="y", color="#E9E9E9", linewidth=0.5, zorder=0)
            axis.spines["left"].set_color("#666666")
            axis.spines["bottom"].set_color("#666666")
            axis.spines["left"].set_linewidth(0.65)
            axis.spines["bottom"].set_linewidth(0.65)

        axes[-1].set_xlabel(xlabel, fontsize=fontsize)
        plot_position = plot_spec.get_position(figure)
        figure.text(
            plot_position.x0 - label_offset,
            (plot_position.y0 + plot_position.y1) / 2,
            "Pixel F1 (%)",
            rotation=90,
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        if axis_break is not None:
            diagonal = 0.012
            mark_style = {"color": "#444444", "linewidth": 0.8, "clip_on": False}
            axes[0].plot(
                (-diagonal, diagonal),
                (-diagonal, diagonal),
                transform=axes[0].transAxes,
                **mark_style,
            )
            axes[1].plot(
                (-diagonal, diagonal),
                (1 - diagonal, 1 + diagonal),
                transform=axes[1].transAxes,
                **mark_style,
            )
            if any(low < axis_break[1] and high > axis_break[0] for low, high in intervals):
                raise ValueError("Pixel-F1 axis break intersects a seed-spread band")
        handles, labels = axes[0].get_legend_handles_labels()
        if not panel_handles:
            panel_handles, panel_labels = handles, labels

    figure.legend(
        panel_handles,
        panel_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=3,
        frameon=False,
        fontsize=ticksize,
        handlelength=1.6,
        columnspacing=1.15,
    )
    stem = f"bank_size_pixel_f1_{version}"
    if layout == "single_column":
        stem = f"bank_size_pixel_f1_single_column_{version}"
    _save(figure, stem)


def _pixel_v3_figure(bank: list[dict], layout: str) -> None:
    _pixel_broken_figure(
        bank,
        layout,
        version="v3",
        axis_ranges=PIXEL_V3_AXIS_RANGES,
        axis_breaks=PIXEL_V3_AXIS_BREAKS,
        min_break_clearance=PIXEL_V3_MIN_BREAK_CLEARANCE,
        show_sd=True,
    )


def _pixel_v4_figure(bank: list[dict], layout: str) -> None:
    _pixel_broken_figure(
        bank,
        layout,
        version="v4",
        axis_ranges=PIXEL_V4_AXIS_RANGES,
        axis_breaks=PIXEL_V4_AXIS_BREAKS,
        min_break_clearance=PIXEL_V4_MIN_BREAK_CLEARANCE,
        show_sd=False,
    )


def _pixel_v5_figure(bank: list[dict], layout: str) -> None:
    _pixel_broken_figure(
        bank,
        layout,
        version="v5",
        axis_ranges=PIXEL_V5_AXIS_RANGES,
        axis_breaks=PIXEL_V4_AXIS_BREAKS,
        min_break_clearance=PIXEL_V4_MIN_BREAK_CLEARANCE,
        show_sd=False,
        compact_spacing=True,
        xlabel="Bank size",
    )


def _figures(bank: list[dict], component: list[dict]) -> None:
    colors, markers = ("#0072B2", "#D55E00", "#009E73"), ("o", "s", "^")
    plt.rcParams.update(
        {"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    )
    _pixel_v5_figure(bank, "full_width")
    _pixel_v5_figure(bank, "single_column")
    for level in ("image",):
        figure, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
        figure.subplots_adjust(left=0.085, right=0.99, bottom=0.12, top=0.88, hspace=0.22, wspace=0.07)
        for axis, model in zip(axes.flat, ("SuperADD", "Dinomaly", "PatchCore", "PaDiM"), strict=True):
            for cat, color, marker in zip(BANK_CATEGORIES, colors, markers, strict=True):
                rows = sorted(
                    (
                        row
                        for row in bank
                        if row["model"] == model and row["category"] == cat and row["eval_set"] == "real_test"
                    ),
                    key=lambda row: row["bank_size"],
                )
                means = np.asarray([row[f"{level}_f1_mean_percent"] for row in rows], dtype=float)
                deviations = np.asarray([row[f"{level}_f1_seed_sd_pp"] for row in rows], dtype=float)
                axis.plot(
                    BANK_SIZES, means, marker=marker, color=color, label=cat.replace("_", " ").capitalize()
                )
                axis.fill_between(BANK_SIZES, means - deviations, means + deviations, color=color, alpha=0.12)
            axis.set_title(model + (" (B=3 separate fit)" if model == "SuperADD" else " (same fit)"))
            axis.set_xticks(BANK_SIZES)
            axis.set_ylim(0, 80 if level == "pixel" else 100)
            axis.grid(alpha=0.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1), ncol=3, frameon=False)
        figure.supxlabel("Source-image bank size B", y=0.02)
        figure.supylabel(f"Real-test {level} F1 (%)", x=0.01)
        _save(figure, f"bank_size_{level}_f1")
    arms = ("obs_off", "mrsp_uniform_blob", "obs_color_only", "harmonisation_off", "blend_alpha_only")
    values = np.asarray(
        [
            [
                next(
                    row["delta_pixel_vs_baseline_pp"]
                    for row in component
                    if row["arm"] == arm and row["category"] == cat
                )
                for cat in CATEGORIES
            ]
            for arm in arms
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(11, 4), layout="constrained")
    image = axis.imshow(values, cmap="RdBu_r", vmin=-28, vmax=28, aspect="auto")
    axis.set_xticks(range(8), [cat.replace("_", " ") for cat in CATEGORIES], rotation=25, ha="right")
    axis.set_yticks(
        range(5), ["OBS off", "Uniform blob", "Colour-only OBS", "Harmonisation off", "Alpha-only blend"]
    )
    for i, j in itertools.product(range(5), range(8)):
        axis.text(
            j,
            i,
            f"{values[i, j]:+.2f}",
            ha="center",
            va="center",
            color="white" if abs(values[i, j]) > 18 else "black",
        )
    figure.colorbar(image, ax=axis, label="Pixel F1 change from full FLASH (pp)")
    _save(figure, "component_category_pixel_deltas")


def main() -> None:
    """Rebuild complete numerical views and figures from checksummed source copies."""
    directory = OUT / "sources"
    directory.mkdir(exist_ok=True)
    data, evidence = {}, {}
    for name, (original, expected) in SOURCES.items():
        filename = (
            f"bank_{name.lower()}.csv" if name in {"SuperADD", "Dinomaly", "PatchCore", "PaDiM"} else name
        )
        saved = directory / filename
        if not saved.exists():
            if _sha(original) != expected:
                raise ValueError(f"Original numerical source changed: {original}")
            shutil.copy2(original, saved)
        if _sha(saved) != expected:
            raise ValueError(f"Archived numerical source changed: {saved}")
        data[name] = _read(saved)
        for row in data[name]:
            if any(
                not math.isfinite(float(row[f"{level}_F1Score"]))
                or not 0 <= float(row[f"{level}_F1Score"]) <= 1
                for level in ("pixel", "image")
            ):
                raise ValueError("Invalid source F1")
        evidence[filename] = {"sha256": expected, "rows": len(data[name]), "original": str(original)}
    specs_path = directory / "component_specs.py"
    if not specs_path.exists():
        shutil.copy2(FLASH / "evaluation/ablations/specs.py", specs_path)
    if _sha(specs_path) != "24275db88bfefeb23d4cba5d2aa5f5f799fa45849e2f92d20f199f0c0cdea108":
        raise ValueError("Archived component settings changed")
    spec = importlib.util.spec_from_file_location("_supplementary_frozen_specs", specs_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load archived component settings")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    macro, component = _component_tables(data, module.ARM_BY_NAME)
    _, bank = _bank_tables(data)
    _figures(bank, component)
    check_sources = {
        "padim_export_verification.json": "20261029_feature_bank_size_v1/padim_export_verification.json",
        "patchcore_export_verification.json": (
            "20261029_feature_bank_size_v1/patchcore_export_verification.json"
        ),
        "padim_approved_protocol.json": "20261029_feature_bank_size_v1/padim_boundary_resume_launch.json",
        "padim_checkpoint_evaluation_provenance.json": (
            "20261029_feature_bank_size_v1/padim_boundary_resume_v1/checkpoint_evaluation_provenance.json"
        ),
        "original_feature_launch_failed_status.json": "20261029_feature_bank_size_v1/distributed_status.json",
        "dinomaly_export_verification.json": "20261029_dinomaly_bank_size_v1/export_verification.json",
        "legacy_followup_completion.json": (
            "20261027_threshold_audit_stability_patchcore_v2/completion_verified.json"
        ),
    }
    checks = directory / "protocol_checks"
    checks.mkdir(exist_ok=True)
    check_hashes = {}
    for name, relative in check_sources.items():
        destination = checks / name
        if not destination.exists():
            shutil.copy2(LAUNCHES / relative, destination)
        check_hashes[name] = _sha(destination)
    audit = [row for row in data["component_audit.csv"] if row["audit_pass"] == "primary"]
    diagnostic = {}
    for arm in SHORTLIST:
        rows = [row for row in audit if row["arm"] == arm]
        if len(rows) != 24:
            raise ValueError("Incomplete audit cohort")
        diagnostic[arm] = {
            "image_fpr_percent": 100
            * statistics.mean(float(row["processed_image_false_positive_rate"]) for row in rows),
            "all_positive_runs": sum(
                int(row["processed_image_fn"]) == 0 and int(row["processed_image_tn"]) == 0 for row in rows
            ),
        }
    manifest = {
        "sources": evidence,
        "protocol_check_sha256": check_hashes,
        "component_specs_sha256": _sha(specs_path),
        "component_macro_rows": len(macro),
        "component_category_rows": len(component),
        "bank_macro_rows": 48,
        "bank_category_rows": len(bank),
        "five_seed_macro_rows": 4,
        "five_seed_category_rows": 32,
        "original_p2_baselines_checked_and_not_double_counted": True,
        "legacy_component_audit": diagnostic,
        "legacy_patchcore_components_are_not_new_bank_results_or_a_passed_reproduction_gate": True,
        "invalid_source_pixel_auroc_rows": [
            {"arm": row["arm"], "category": row["category"], "seed": row["seed"], "value": row["pixel_AUROC"]}
            for row in data["component_p2.csv"]
            if not 0 <= float(row["pixel_AUROC"]) <= 1
        ],
        "no_new_experiments": True,
        "checkpoints_and_corpora_not_included": True,
        "builder_sha256": _sha(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "pixel_figure": {
            "version": 1,
            "preserved_unchanged": True,
            "primary": "bank_size_pixel_f1.pdf",
            "primary_layout": "full_width_two_column_2x2",
            "single_column_variant": "bank_size_pixel_f1_single_column.pdf",
            "intel_palette": {
                category: {"line": line, "sd_band_tint": tint, "marker": marker}
                for category, (line, tint, marker) in PIXEL_COLORS.items()
            },
            "axis_range_percent_by_model": PIXEL_AXIS_RANGES,
            "shading": "category-wise mean plus/minus one sample SD across three generation seeds",
            "independent_detector_y_ranges": True,
        },
        "pixel_figure_v2": {
            "primary": "bank_size_pixel_f1_v2.pdf",
            "primary_layout": "full_width_two_column_2x2",
            "single_column_variant": "bank_size_pixel_f1_single_column_v2.pdf",
            "titles": "model names only",
            "axis_range_percent_by_model": PIXEL_V2_AXIS_RANGES,
            "axis_break_percent_by_model": PIXEL_V2_AXIS_BREAKS,
            "break_policy": (
                "Only omit an empty score interval outside all mean±SD bands; Dinomaly remains continuous "
                "because its category envelopes overlap."
            ),
            "minimum_clearance_points_each_side": 1.0,
            "shading": "category-wise mean plus/minus one sample SD across three generation seeds",
            "legend": "single shared legend below panels",
            "axis_labels": "each model panel labels both axes",
            "independent_detector_y_ranges": True,
            "preserves_v1_files": True,
        },
        "pixel_figure_v3": {
            "primary": "bank_size_pixel_f1_v3.pdf",
            "primary_layout": "full_width_two_column_2x2",
            "single_column_variant": "bank_size_pixel_f1_single_column_v3.pdf",
            "titles": "model names aligned in equal-height title bands",
            "equal_subplot_panels": True,
            "axis_range_percent_by_model": PIXEL_V3_AXIS_RANGES,
            "axis_break_percent_by_model": PIXEL_V3_AXIS_BREAKS,
            "break_policy": (
                "Widened breaks remain within empty intervals outside all mean±SD bands; "
                "Dinomaly remains continuous because its category envelopes overlap."
            ),
            "minimum_clearance_points_each_side": PIXEL_V3_MIN_BREAK_CLEARANCE,
            "shading": "category-wise mean plus/minus one sample SD across three generation seeds",
            "legend": "single shared legend below panels",
            "axis_labels": "each model panel labels both axes",
            "independent_detector_y_ranges": True,
            "preserves_v1_and_v2_files": True,
        },
        "pixel_figure_v4": {
            "primary": "bank_size_pixel_f1_v4.pdf",
            "primary_layout": "full_width_two_column_2x2",
            "single_column_variant": "bank_size_pixel_f1_single_column_v4.pdf",
            "titles": "model names aligned in equal-height title bands",
            "equal_subplot_panels": True,
            "axis_range_percent_by_model": PIXEL_V4_AXIS_RANGES,
            "axis_break_percent_by_model": PIXEL_V4_AXIS_BREAKS,
            "break_policy": (
                "Break only where no category mean envelope or connected mean line crosses; "
                "Dinomaly remains continuous because its envelopes overlap."
            ),
            "minimum_clearance_points_each_side": PIXEL_V4_MIN_BREAK_CLEARANCE,
            "shading": "none; plotted lines and markers show means across three seeds",
            "variance_bands": False,
            "legend": "single shared legend below panels",
            "axis_labels": "each model panel labels both axes",
            "independent_detector_y_ranges": True,
            "preserves_v1_v3_files": True,
        },
        "pixel_figure_v5": {
            "primary": "bank_size_pixel_f1_v5.pdf",
            "primary_layout": "full_width_two_column_2x2",
            "single_column_variant": "bank_size_pixel_f1_single_column_v5.pdf",
            "titles": "model names aligned in equal-height title bands",
            "equal_subplot_panels": True,
            "axis_range_percent_by_model": PIXEL_V5_AXIS_RANGES,
            "axis_break_percent_by_model": PIXEL_V4_AXIS_BREAKS,
            "break_policy": "Unchanged V4 category-mean gaps; no connected mean line is omitted.",
            "minimum_clearance_points_each_side": PIXEL_V4_MIN_BREAK_CLEARANCE,
            "shading": "none; plotted lines and markers show means across three seeds",
            "variance_bands": False,
            "legend": "single shared legend below panels, with reduced vertical whitespace",
            "axis_labels": "each model panel labels both axes; x-axis reads Bank size",
            "compact_vertical_spacing": True,
            "independent_detector_y_ranges": True,
            "preserves_v1_v4_files": True,
        },
        "generated_sha256": {
            path.name: _sha(path) for path in OUT.iterdir() if path.suffix in {".csv", ".pdf", ".png"}
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
