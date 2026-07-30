# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ad-hoc analysis producing the numbers used in the synthetic-anomaly report."""

from __future__ import annotations

import numpy as np
import pandas as pd

df = pd.read_csv("gsoc_workspace/experiments/results/results.csv")
# phase2 has all 5 models on both datasets with unique keys; phase1 is a 3-model
# mvtec subset that duplicates phase2 mvtec keys, so use phase2 as the main table.
main = df[df.phase == "phase2"]
key = ["dataset", "category", "model", "seed"]
A = main[main.arm == "A"].set_index(key)

print("== 1) Pearson r: synthetic tau_B vs oracle tau_A across cats/seeds (P3) ==")
for ds in ["mvtec", "visa"]:
    out = []
    sub = main[(main.arm == "B") & (main.pipeline == "P3") & (main.dataset == ds)]
    for model, g in sub.groupby("model"):
        m = pd.concat(
            [A["image_threshold"].rename("A"), g.set_index(key)["image_threshold"].rename("B")],
            axis=1,
        ).dropna()
        r = np.corrcoef(m.A, m.B)[0, 1] if len(m) > 2 else float("nan")
        out.append(f"{model}:{r:.2f}")
    print(f"  {ds}: " + "  ".join(out))

print("\n== 2) F1_gap seed-stability: mean +/- std across seeds (P3, over cats) ==")
b = main[(main.arm == "B") & (main.pipeline == "P3")].set_index(key).join(A["image_F1Score"].rename("A"))
b["gap"] = b["A"] - b["image_F1Score"]
gg = b.reset_index().groupby(["dataset", "model", "category", "seed"])["gap"].mean().reset_index()
st = gg.groupby(["dataset", "model"])["gap"].agg(["mean", "std"]).round(3)
print(st.to_string())

print("\n== 3) model-ranking Spearman: oracle-F1 vs synthetic-F1 ordering (P3) ==")
for ds in ["mvtec", "visa"]:
    ra = main[(main.arm == "A") & (main.dataset == ds)].groupby("model")["image_F1Score"].mean().rank()
    rb = (
        main[(main.arm == "B") & (main.pipeline == "P3") & (main.dataset == ds)]
        .groupby("model")["image_F1Score"]
        .mean()
        .rank()
    )
    print(f"  {ds}: rho={ra.corr(rb, method='spearman'):.3f}")

print("\n== 4) pixel AUPRO (localization, arm A mean over cats/seeds) ==")
print(main[main.arm == "A"].groupby(["dataset", "model"])["pixel_AUPRO"].mean().round(3).to_string())

print("\n== 5) Arm C (phase0): real vs synthetic-set difficulty ==")
p0 = df[df.phase == "phase0"]
print("Arm A real AUROC:", round(p0[p0.arm == "A"].image_AUROC.mean(), 3))
print(p0[p0.arm == "C"].groupby(["model", "pipeline"])["image_AUROC"].mean().round(3).to_string())

print("\n== 6) overall F1 recovery (B/A ratio) per pipeline, both datasets ==")
for pipe in ["P1", "P2", "P3"]:
    g = main[(main.arm == "B") & (main.pipeline == pipe)].set_index(key).join(A["image_F1Score"].rename("A"))
    ratio = (g["image_F1Score"] / g["A"]).mean()
    print(f"  {pipe}: mean F1_B/F1_A = {ratio:.3f}")
