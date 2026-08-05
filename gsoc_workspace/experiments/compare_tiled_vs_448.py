#!/usr/bin/env python3
# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compare phase 6 (native-resolution tiled) results against phase 5 (448px) baseline.

This is the analysis that answers the question the whole tiling effort was for: does
native-resolution tiled inference actually close the gap to published SOTA (SuperADD
57.42% SegF1, RoBiS 51.00%), or was the 448px downsize not the dominant bottleneck after
all? Runs on whatever jobs have completed, so it is safe to use while phase 6 is still
in flight -- comparisons are per (category, model) and simply skip pairs where one side
is missing.

Usage:
    uv run python -m gsoc_workspace.experiments.compare_tiled_vs_448
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS_CSV = Path(__file__).parent / "results" / "results.csv"
RAW_DIR = Path(__file__).parent / "results" / "raw"

# From the MVTec AD 2 challenge technical reports (VAND 3.0 / 4.0), test_private SegF1.
# Used only as an external reference point -- not something our single-seed, tiled-only
# pilot is expected to match, since those methods use dedicated per-category tuning.
PUBLISHED_SEGF1 = {
    "can": 1.86, "fabric": 87.46, "fruit_jelly": 53.63, "rice": 63.86,
    "sheet_metal": 70.98, "vial": 48.73, "wallplugs": 14.38, "walnuts": 67.13,
}
PUBLISHED_MEAN_SEGF1 = 51.00  # RoBiS, VAND 3.0 winner, test_private


def load_arm_a(phase: str) -> pd.DataFrame:
    """Load oracle-threshold (arm A) rows for one phase, averaged to one row per (category, model).

    Phase 5 has 3 seeds per (category, model); phase 6 has 1. Averaging over seed here
    keeps the merge in ``main`` from producing spurious cross-product duplicates.
    """
    df = pd.read_csv(RESULTS_CSV)
    sub = df[(df.phase == phase) & (df.arm == "A") & (df.dataset == "mvtec2")]
    cols = ["category", "model", "image_AUROC", "pixel_F1Score", "pixel_AUPRO", "pixel005_AUPRO"]
    return sub[cols].groupby(["category", "model"], as_index=False).mean(numeric_only=True)


def load_arm_a_from_raw(phase: str) -> pd.DataFrame:
    """Load arm-A rows for one phase directly from raw JSONs, bypassing the aggregated CSV.

    ``results.csv`` is only rewritten when ``sweep.py --aggregate-only`` (or the end of a
    chain) runs, so it can lag behind or reference raw files that were since deleted (e.g.
    after purging a corrupted pre-fix run). Reading raw JSONs directly means comparisons
    always reflect what has actually landed on disk right now, even mid-sweep.
    """
    rows: list[dict] = []
    for path in sorted(RAW_DIR.glob(f"{phase}_*.json")):
        data = json.loads(path.read_text())
        rows.extend(data if isinstance(data, list) else [data])
    cols = ["category", "model", "image_AUROC", "pixel_F1Score", "pixel_AUPRO", "pixel005_AUPRO"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    sub = df[(df.arm == "A") & (df.dataset == "mvtec2")]
    return sub[cols].groupby(["category", "model"], as_index=False).mean(numeric_only=True)


def main() -> None:
    """Print a side-by-side comparison table and summary statistics."""
    if not RESULTS_CSV.exists():
        print(f"No results file at {RESULTS_CSV} yet.")
        return

    p5 = load_arm_a("phase5").rename(columns={
        "image_AUROC": "auroc_448", "pixel_F1Score": "segf1_448",
        "pixel_AUPRO": "aupro_448", "pixel005_AUPRO": "aupro05_448",
    })
    p6 = load_arm_a_from_raw("phase6").rename(columns={
        "image_AUROC": "auroc_tiled", "pixel_F1Score": "segf1_tiled",
        "pixel_AUPRO": "aupro_tiled", "pixel005_AUPRO": "aupro05_tiled",
    })

    if p6.empty:
        print("No phase6 (tiled) jobs have completed yet -- nothing to compare.")
        print(f"phase5 (448px) baseline available for {len(p5)} (category, model) pairs.")
        return

    n_raw_files = len(list(RAW_DIR.glob("phase6_*.json")))
    print(f"(reading phase6 directly from {n_raw_files} raw JSON files on disk, "
          "ignoring possibly-stale results.csv)\n")

    merged = p5.merge(p6, on=["category", "model"], how="outer")
    merged["segf1_delta"] = merged["segf1_tiled"] - merged["segf1_448"]
    merged["published_segf1_pct"] = merged["category"].map(PUBLISHED_SEGF1)

    print(f"=== Phase 6 (tiled) vs Phase 5 (448px) -- {len(p6)}/{len(p5) if not p5.empty else '?'} "
          "tiled rows so far ===\n")

    cols = [
        "category", "model", "auroc_448", "auroc_tiled",
        "segf1_448", "segf1_tiled", "segf1_delta", "published_segf1_pct",
    ]
    print(merged[cols].sort_values(["model", "category"]).round(4).to_string(index=False))

    print("\n=== Per-model mean SegF1: 448px vs tiled (only rows where both exist) ===")
    both = merged.dropna(subset=["segf1_448", "segf1_tiled"])
    if not both.empty:
        summary = both.groupby("model").agg(
            n=("segf1_448", "size"),
            segf1_448_mean=("segf1_448", "mean"),
            segf1_tiled_mean=("segf1_tiled", "mean"),
            segf1_delta_mean=("segf1_delta", "mean"),
        ).round(4)
        print(summary.to_string())
        print(f"\nOverall: 448px mean SegF1 = {both.segf1_448.mean():.4f}, "
              f"tiled mean SegF1 = {both.segf1_tiled.mean():.4f} "
              f"(delta {both.segf1_delta.mean():+.4f})")
        print(f"Published SOTA reference (RoBiS, test_private): {PUBLISHED_MEAN_SEGF1 / 100:.4f}")
    else:
        print("No (category, model) pairs with both phase5 and phase6 results yet.")

    print("\n=== Tiled-only: how far from published per-category SegF1? ===")
    tiled_only = merged.dropna(subset=["segf1_tiled", "published_segf1_pct"]).copy()
    if not tiled_only.empty:
        tiled_only["gap_to_published"] = tiled_only["published_segf1_pct"] / 100 - tiled_only["segf1_tiled"]
        print(tiled_only[["category", "model", "segf1_tiled", "published_segf1_pct", "gap_to_published"]]
              .sort_values("gap_to_published").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
