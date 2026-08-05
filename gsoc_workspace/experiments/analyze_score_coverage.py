#!/usr/bin/env python3
# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test whether score-space coverage predicts threshold-transfer quality.

This is the load-bearing empirical bet behind the "score-space coverage" idea: the
claim is that a synthetic anomaly set is good precisely when the anomaly *score
distribution* it induces matches the real one, and that this predicts the downstream
gap better than a scalar difficulty summary (the arm-C AUROC deficit) does.

If the distribution distances do not correlate with the gap, the coverage objective is
not supported and should be dropped before any method is built on it.

Runs on whatever jobs have completed, so it can be used while a sweep is in flight.

Usage:
    uv run python -m gsoc_workspace.experiments.analyze_score_coverage --phase phase5
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).parent / "results"
RAW_DIR = RESULTS / "raw"
SCORES_DIR = RESULTS / "scores"


def _coverage(synthetic: np.ndarray, real: np.ndarray) -> float:
    """Fraction of the real anomaly score range that synthetic scores span.

    1.0 means the synthetic scores cover the full spread of real anomaly scores;
    values near 0 mean they collapse into a narrow band and leave the rest of the
    operating range unrepresented.
    """
    lo, hi = float(real.min()), float(real.max())
    if hi <= lo:
        return float("nan")
    clipped = synthetic[(synthetic >= lo) & (synthetic <= hi)]
    if clipped.size < 2:
        return 0.0
    return float((clipped.max() - clipped.min()) / (hi - lo))


def _quantile_gap(synthetic: np.ndarray, real: np.ndarray, n_bins: int = 10) -> float:
    """Fraction of real-anomaly score deciles containing no synthetic sample."""
    edges = np.quantile(real, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    counts, _ = np.histogram(synthetic, bins=edges)
    return float((counts == 0).mean())


def load_records(phase: str) -> pd.DataFrame:
    """Join per-image score distributions to the aggregate arm metrics."""
    records: list[dict] = []
    for raw_path in sorted(RAW_DIR.glob(f"{phase}_*.json")):
        score_path = SCORES_DIR / raw_path.name
        if not score_path.exists():
            continue
        rows = json.loads(raw_path.read_text())
        scores = json.loads(score_path.read_text())

        arm_a = next((r for r in rows if r["arm"] == "A"), None)
        real = scores.get("real_test")
        if arm_a is None or real is None:
            continue
        real_arr = np.asarray(real["scores"], dtype=float)
        real_lbl = np.asarray(real["labels"], dtype=int)
        real_anom = real_arr[real_lbl == 1]
        if real_anom.size < 2:
            continue

        by_arm = {(r["arm"], r["pipeline"]): r for r in rows}
        for pipeline, payload in scores.items():
            if pipeline == "real_test":
                continue
            arm_b = by_arm.get(("B", pipeline))
            if arm_b is None:
                continue
            syn = np.asarray(payload["scores"], dtype=float)
            lbl = np.asarray(payload["labels"], dtype=int)
            syn_anom = syn[lbl == 1]
            if syn_anom.size < 2:
                continue
            arm_c = by_arm.get(("C", pipeline))

            records.append({
                "dataset": arm_a["dataset"],
                "category": arm_a["category"],
                "model": arm_a["model"],
                "seed": arm_a["seed"],
                "pipeline": pipeline,
                # --- distribution distances (candidate predictors) ---
                "w1": stats.wasserstein_distance(syn_anom, real_anom),
                "ks": stats.ks_2samp(syn_anom, real_anom).statistic,
                "energy": stats.energy_distance(syn_anom, real_anom),
                "mean_shift": float(syn_anom.mean() - real_anom.mean()),
                "coverage": _coverage(syn_anom, real_anom),
                "quantile_gap": _quantile_gap(syn_anom, real_anom),
                # --- scalar difficulty summary (the incumbent predictor) ---
                "armc_deficit": (
                    arm_c["image_AUROC"] - arm_a["image_AUROC"] if arm_c is not None else np.nan
                ),
                # --- outcomes ---
                "f1_gap": arm_a["image_F1Score"] - arm_b["image_F1Score"],
                "segf1_gap": arm_a["pixel_F1Score"] - arm_b["pixel_F1Score"],
                "auroc_real": arm_a["image_AUROC"],
            })
    return pd.DataFrame(records)


def _report_correlations(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Spearman correlation of every candidate predictor against one outcome."""
    predictors = ["w1", "ks", "energy", "coverage", "quantile_gap", "armc_deficit"]
    out = []
    for name in predictors:
        sub = df[[name, outcome]].dropna()
        # coverage is protective, so flip it to keep "higher = worse" comparable
        values = -sub[name] if name == "coverage" else sub[name]
        if len(sub) < 8 or values.nunique() < 3:
            out.append({"predictor": name, "rho": np.nan, "p": np.nan, "n": len(sub)})
            continue
        rho, p = stats.spearmanr(values.abs() if name in {"mean_shift"} else values, sub[outcome])
        out.append({"predictor": name, "rho": rho, "p": p, "n": len(sub)})
    return pd.DataFrame(out).sort_values("rho", key=lambda s: s.abs(), ascending=False)


def main() -> None:
    """Run the coverage-vs-gap analysis over completed jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="phase5")
    args = parser.parse_args()

    df = load_records(args.phase)
    if df.empty:
        print(f"No completed jobs with scores for phase {args.phase!r} yet.")
        return

    print(f"=== {args.phase}: {len(df)} (job x pipeline) records "
          f"from {df.groupby(['category', 'model', 'seed']).ngroups} jobs ===\n")

    print("--- per-pipeline distribution distances vs real anomalies ---")
    summary = df.groupby("pipeline").agg(
        n=("w1", "size"),
        w1=("w1", "mean"),
        ks=("ks", "mean"),
        coverage=("coverage", "mean"),
        quantile_gap=("quantile_gap", "mean"),
        mean_shift=("mean_shift", "mean"),
        armc_deficit=("armc_deficit", "mean"),
        f1_gap=("f1_gap", "mean"),
        segf1_gap=("segf1_gap", "mean"),
    )
    print(summary.round(4).to_string(), "\n")

    for outcome in ("f1_gap", "segf1_gap"):
        print(f"--- Spearman: predictor vs {outcome} (higher rho = better predictor) ---")
        print(_report_correlations(df, outcome).round(4).to_string(index=False), "\n")

    # Within-pipeline correlation: does coverage still explain variation once the
    # generator is held fixed? A predictor that only separates pipelines is much
    # weaker evidence than one that also tracks the gap inside a pipeline.
    print("--- within-pipeline Spearman (w1 vs segf1_gap) ---")
    for pipeline, sub in df.groupby("pipeline"):
        s = sub[["w1", "segf1_gap"]].dropna()
        if len(s) >= 8:
            rho, p = stats.spearmanr(s["w1"], s["segf1_gap"])
            print(f"  {pipeline}: rho={rho:+.4f} p={p:.3e} n={len(s)}")

    print("\n--- paired pipeline comparison (Wilcoxon on segf1_gap) ---")
    wide = df.pivot_table(index=["category", "model", "seed"], columns="pipeline", values="segf1_gap")
    for a, b in combinations(sorted(wide.columns), 2):
        pair = wide[[a, b]].dropna()
        if len(pair) >= 8 and not np.allclose(pair[a], pair[b]):
            _stat, p = stats.wilcoxon(pair[a], pair[b])
            print(f"  {a} vs {b}: median diff={float((pair[a] - pair[b]).median()):+.4f} p={p:.3e} n={len(pair)}")

    out_path = RESULTS / f"score_coverage_{args.phase}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
