# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Paired P2 vs P3 vs P4 comparison on the phase3 sweep."""

from __future__ import annotations

import pandas as pd

pd.set_option("display.width", 200)

df = pd.read_csv("gsoc_workspace/experiments/results/results.csv")
p3 = df[df.phase == "phase3"]

print(f"phase3 rows: {len(p3)} | NaN in image_F1Score: {p3.image_F1Score.isna().sum()}")
print("arms:", sorted(p3.arm.unique()), "| pipelines:", sorted(p3.pipeline.dropna().unique()))

key = ["dataset", "category", "model"]

# --- confound check: arm A must be identical across pipelines (same trained model)
a = p3[p3.arm == "A"]
spread = a.groupby(key)["image_AUROC"].agg(lambda s: s.max() - s.min())
print(f"\n[confound] max arm-A AUROC spread across pipelines: {spread.max():.4f}")

# --- paired F1 gap (oracle - proxy), lower is better
oracle = a.groupby(key)["image_F1Score"].max().rename("F1_A")
proxy = p3[p3.arm == "B"].set_index([*key, "pipeline"])["image_F1Score"].rename("F1_B")
paired = proxy.reset_index().merge(oracle.reset_index(), on=key)
paired["gap"] = paired.F1_A - paired.F1_B
paired["recovery"] = paired.F1_B / paired.F1_A

print("\n== F1 gap (oracle - proxy), mean by dataset x pipeline ==")
print(paired.pivot_table(index="dataset", columns="pipeline", values="gap", aggfunc="mean").round(4).to_string())
print("\n== F1 recovery (proxy / oracle) ==")
print(paired.pivot_table(index="dataset", columns="pipeline", values="recovery", aggfunc="mean").round(4).to_string())
print("\n== F1 gap by model x pipeline ==")
print(
    paired.pivot_table(index=["dataset", "model"], columns="pipeline", values="gap", aggfunc="mean")
    .round(4)
    .to_string(),
)

# --- head-to-head: does P4 beat both parents on the same (dataset, category, model)?
wide = paired.pivot_table(index=key, columns="pipeline", values="gap")
wide = wide.dropna()
print(f"\n== head-to-head on {len(wide)} paired cells ==")
print(f"P4 better than P2: {(wide.P4 < wide.P2).sum():3d} / {len(wide)}")
print(f"P4 better than P3: {(wide.P4 < wide.P3).sum():3d} / {len(wide)}")
print(f"P4 best of three : {((wide.P4 <= wide.P2) & (wide.P4 <= wide.P3)).sum():3d} / {len(wide)}")
print(f"mean gap  P2={wide.P2.mean():.4f}  P3={wide.P3.mean():.4f}  P4={wide.P4.mean():.4f}")

# Wilcoxon signed-rank, P4 vs each parent
try:
    from scipy.stats import wilcoxon

    for parent in ("P2", "P3"):
        stat, pval = wilcoxon(wide.P4, wide[parent])
        better = "P4 lower" if wide.P4.mean() < wide[parent].mean() else "P4 higher"
        print(f"wilcoxon P4 vs {parent}: p={pval:.4g}  ({better})")
except ImportError:
    print("scipy unavailable; skipping Wilcoxon")

# --- arm C difficulty: is the synthetic set closer to real-test difficulty?
real = a.groupby(key)["image_AUROC"].max().rename("AUROC_real")
synth = p3[p3.arm == "C"].set_index([*key, "pipeline"])["image_AUROC"].rename("AUROC_synth")
diff = synth.reset_index().merge(real.reset_index(), on=key)
diff["abs_delta"] = (diff.AUROC_synth - diff.AUROC_real).abs()
print("\n== arm C: synthetic-set AUROC vs real-test AUROC ==")
print(
    diff.pivot_table(index="dataset", columns="pipeline", values=["AUROC_synth", "abs_delta"], aggfunc="mean")
    .round(4)
    .to_string(),
)
print(f"\nreal-test AUROC (arm A) mean: {diff.groupby('dataset').AUROC_real.mean().round(4).to_dict()}")
