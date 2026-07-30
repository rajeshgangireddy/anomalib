# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Analyze the phase4 hybrid-blend area-threshold ablation."""

from __future__ import annotations

import pandas as pd

pd.set_option("display.width", 220)

THRESHOLDS = {"P3": 0, "P4a": 500, "P4b": 1000, "P4": 2500, "P4c": 6000, "P4d": 12000}
# Fraction of total mask area held by components below 2500 px (blob census).
AREA_BELOW = {
    ("mvtec", "carpet"): 0.687, ("mvtec", "screw"): 0.323, ("visa", "pcb1"): 0.323,
    ("mvtec", "bottle"): 0.273, ("visa", "macaroni1"): 0.178, ("visa", "cashew"): 0.159,
    ("mvtec", "hazelnut"): 0.124, ("visa", "candle"): 0.075,
}

df = pd.read_csv("gsoc_workspace/experiments/results/results.csv")
p4 = df[df.phase == "phase4"]
print(f"phase4 rows: {len(p4)} | NaN F1: {p4.image_F1Score.isna().sum()}")

key = ["dataset", "category", "model"]
a = p4[p4.arm == "A"]
print(f"[confound] max arm-A AUROC spread across pipelines: "
      f"{a.groupby(key)['image_AUROC'].agg(lambda s: s.max() - s.min()).max():.4f}")

oracle = a.groupby(key)["image_F1Score"].max().rename("F1_A")
proxy = p4[p4.arm == "B"].set_index([*key, "pipeline"])["image_F1Score"].rename("F1_B")
d = proxy.reset_index().merge(oracle.reset_index(), on=key)
d["gap"] = d.F1_A - d.F1_B
d["threshold"] = d.pipeline.map(THRESHOLDS)
d = d.sort_values("threshold")

order = ["P3", "P4a", "P4b", "P4", "P4c", "P4d"]
print("\n== mean F1 gap by area_threshold (lower is better) ==")
overall = d.groupby(["threshold", "pipeline"]).gap.mean().round(4)
print(overall.to_string())

print("\n== by dataset ==")
print(d.pivot_table(index="dataset", columns="pipeline", values="gap", aggfunc="mean")[order].round(4).to_string())

print("\n== by model ==")
print(
    d.pivot_table(index=["dataset", "model"], columns="pipeline", values="gap", aggfunc="mean")[order]
    .round(4)
    .to_string(),
)

print("\n== by category (sorted by fraction of mask area below 2500 px) ==")
cat = d.pivot_table(index=["dataset", "category"], columns="pipeline", values="gap", aggfunc="mean")[order]
cat["area_below"] = [AREA_BELOW[i] for i in cat.index]
print(cat.sort_values("area_below", ascending=False).round(4).to_string())

# Best threshold per cell, and paired tests vs the P3 anchor.
wide = d.pivot_table(index=key, columns="pipeline", values="gap")[order].dropna()
print(f"\n== best pipeline per cell ({len(wide)} cells) ==")
print(wide.idxmin(axis=1).value_counts().to_string())

try:
    from scipy.stats import wilcoxon

    print("\n== Wilcoxon vs P3 anchor ==")
    for p in order[1:]:
        stat, pval = wilcoxon(wide[p], wide.P3)
        delta = wide[p].mean() - wide.P3.mean()
        print(f"{p:4s} (thr={THRESHOLDS[p]:5d}): mean delta={delta:+.4f}  p={pval:.4f}")
except ImportError:
    print("scipy unavailable")
