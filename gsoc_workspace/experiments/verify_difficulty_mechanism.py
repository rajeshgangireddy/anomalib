#!/usr/bin/env python3
# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Full-population verification of the difficulty-mismatch mechanism.

Resolves the outstanding 'task A0': exact means, medians and Spearman
correlations over all rows of results.csv (previously only sampled by hand).
"""

import pandas as pd
from scipy import stats

CSV = "gsoc_workspace/experiments/results/results.csv"
KEY = ["phase", "dataset", "category", "model", "seed"]
COLS = [
    "image_AUROC", "image_F1Score", "image_BinaryPrecision",
    "image_BinaryRecall", "image_threshold",
]

df = pd.read_csv(CSV)
arm_a = df[df["arm"] == "A"]
merged = df[df["arm"] != "A"].merge(arm_a[KEY + COLS], on=KEY, suffixes=("_B", "_A"))

print(f"rows={len(df)}  merged={len(merged)}")

# --- Q6 sanity: arm-A must not vary with pipeline -------------------------
spread = arm_a.groupby(KEY)["image_AUROC"].agg(lambda x: x.max() - x.min())
print(f"\n[Q6] max arm-A AUROC spread within group: {spread.max():.10f}  "
      f"(groups>1e-6: {(spread > 1e-6).sum()})")

p2 = merged[(merged["phase"] == "phase2") & (merged["arm"] == "B")].copy()
p2["f1_gap"] = p2["image_F1Score_A"] - p2["image_F1Score_B"]
p2["d_prec"] = p2["image_BinaryPrecision_B"] - p2["image_BinaryPrecision_A"]
p2["d_rec"] = p2["image_BinaryRecall_B"] - p2["image_BinaryRecall_A"]

ok = (p2["image_threshold_A"] > 0) & p2["image_threshold_A"].notna() & p2["image_threshold_B"].notna()
print(f"[Q1] dropped {(~ok).sum()} rows with tau_A<=0 or NaN")
p2v = p2[ok].copy()
p2v["rel_err"] = (p2v["image_threshold_B"] - p2v["image_threshold_A"]) / p2v["image_threshold_A"]

# --- Q1 threshold direction ----------------------------------------------
print("\n[Q1] THRESHOLD DIRECTION (phase2, arm B)")
print(f"{'ds':6} {'pipe':4} {'n':>4} {'frac tau_B>tau_A':>17} {'mean rel':>10} {'median rel':>11}")
for ds in ["all", "mvtec", "visa"]:
    for pipe in ["P1", "P2", "P3"]:
        s = p2v[p2v["pipeline"] == pipe]
        if ds != "all":
            s = s[s["dataset"] == ds]
        print(f"{ds:6} {pipe:4} {len(s):>4} {(s['image_threshold_B'] > s['image_threshold_A']).mean():>17.3f}"
              f" {s['rel_err'].mean():>+10.3f} {s['rel_err'].median():>+11.3f}")

# --- Q2 precision / recall direction --------------------------------------
print("\n[Q2] PRECISION / RECALL DELTA (arm B minus arm A)")
print(f"{'ds':6} {'pipe':4} {'n':>4} {'d_precision':>12} {'d_recall':>10}")
for ds in ["all", "mvtec", "visa"]:
    for pipe in ["P1", "P2", "P3"]:
        s = p2[p2["pipeline"] == pipe]
        if ds != "all":
            s = s[s["dataset"] == ds]
        print(f"{ds:6} {pipe:4} {len(s):>4} {s['d_prec'].mean():>+12.4f} {s['d_rec'].mean():>+10.4f}")

# --- Q3 degenerate cells ---------------------------------------------------
print("\n[Q3] DEGENERATE CELLS (recall>=0.999 and precision<0.999 => flags ~everything)")
print(f"{'ds':6} {'pipe':4} {'n':>4} {'prec=1':>8} {'rec=1':>8} {'degenerate':>11} {'%':>6}")
for ds in ["all", "mvtec", "visa"]:
    for pipe in ["P1", "P2", "P3"]:
        s = p2[p2["pipeline"] == pipe]
        if ds != "all":
            s = s[s["dataset"] == ds]
        pr1 = (s["image_BinaryPrecision_B"] >= 0.999).sum()
        rc1 = (s["image_BinaryRecall_B"] >= 0.999).sum()
        deg = ((s["image_BinaryRecall_B"] >= 0.999) & (s["image_BinaryPrecision_B"] < 0.999)).sum()
        print(f"{ds:6} {pipe:4} {len(s):>4} {pr1:>8} {rc1:>8} {deg:>11} {deg / max(len(s), 1) * 100:>5.1f}%")

# --- Q3b trivial all-positive baseline ------------------------------------
print("\n[Q3b] TRIVIAL ALL-POSITIVE BASELINE  F1 = 2*pi/(1+pi)")
deg_rows = p2[(p2["image_BinaryRecall_B"] >= 0.999) & (p2["image_BinaryPrecision_B"] < 0.999)]
if len(deg_rows):
    pi = deg_rows["image_BinaryPrecision_B"]  # precision == prevalence when all flagged
    triv = 2 * pi / (1 + pi)
    print(f"  degenerate cells: n={len(deg_rows)}  prevalence {pi.min():.3f}-{pi.max():.3f} (mean {pi.mean():.3f})")
    print(f"  trivial F1 on those cells: {triv.min():.3f}-{triv.max():.3f} (mean {triv.mean():.3f})")
    print(f"  their actual reported F1_B mean: {deg_rows['image_F1Score_B'].mean():.3f}")
    print(f"  oracle F1_A mean on same cells: {deg_rows['image_F1Score_A'].mean():.3f}")

# --- Q4 arm-C deficit predicts the F1 gap ---------------------------------
print("\n[Q4] ARM-C DEFICIT vs F1 GAP (phase3)")
p3 = merged[merged["phase"] == "phase3"]
b = p3[p3["arm"] == "B"][[*KEY, "pipeline", "image_F1Score_B", "image_F1Score_A"]]
c = p3[p3["arm"] == "C"][[*KEY, "pipeline", "image_AUROC_B", "image_AUROC_A"]]
j = b.merge(c, on=[*KEY, "pipeline"])
j["deficit"] = j["image_AUROC_B"] - j["image_AUROC_A"]   # armC AUROC - armA AUROC
j["f1_gap"] = j["image_F1Score_A"] - j["image_F1Score_B"]

print(f"{'ds':6} {'pipe':4} {'n':>4} {'mean deficit':>13} {'mean F1 gap':>12}")
for ds in ["mvtec", "visa"]:
    for pipe in ["P2", "P3", "P4"]:
        s = j[(j["dataset"] == ds) & (j["pipeline"] == pipe)]
        if len(s):
            print(f"{ds:6} {pipe:4} {len(s):>4} {s['deficit'].mean():>+13.4f} {s['f1_gap'].mean():>12.4f}")

v = j[["deficit", "f1_gap"]].dropna()
r1, p1 = stats.spearmanr(v["deficit"], v["f1_gap"])
r2, p2_ = stats.spearmanr(v["deficit"].abs(), v["f1_gap"])
print(f"\n  Spearman  deficit  vs F1gap: rho={r1:+.4f} p={p1:.3e} n={len(v)}")
print(f"  Spearman |deficit| vs F1gap: rho={r2:+.4f} p={p2_:.3e} n={len(v)}")
for pipe in ["P2", "P3", "P4"]:
    s = j[j["pipeline"] == pipe][["deficit", "f1_gap"]].dropna()
    rr, pp = stats.spearmanr(s["deficit"].abs(), s["f1_gap"])
    print(f"    {pipe}: rho(|deficit|,F1gap)={rr:+.4f} p={pp:.3e} n={len(s)}")

print("\n  worst MVTec categories by P3 arm-C deficit:")
w = j[(j["dataset"] == "mvtec") & (j["pipeline"] == "P3")].groupby("category")["deficit"].mean().sort_values()
for cat, d in w.head(6).items():
    print(f"    {cat:14} {d:+.4f}")

# --- Q5 threshold error vs F1 gap -----------------------------------------
print("\n[Q5] |relative threshold error| vs F1 GAP (phase2)")
q = p2v[["rel_err", "f1_gap", "model", "pipeline"]].dropna()
rr, pp = stats.spearmanr(q["rel_err"].abs(), q["f1_gap"])
print(f"  OVERALL rho={rr:+.4f} p={pp:.3e} n={len(q)}")
for m in sorted(q["model"].unique()):
    s = q[q["model"] == m]
    rr, pp = stats.spearmanr(s["rel_err"].abs(), s["f1_gap"])
    print(f"    {m:14} rho={rr:+.4f} p={pp:.3e} n={len(s)}")

print("\nDone.")
