# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Estimate phase3 wall-clock time from measured phase2 job timings."""

from __future__ import annotations

import pandas as pd

df = pd.read_csv("gsoc_workspace/experiments/results/results.csv")
p2 = df[df.phase == "phase2"]
key = ["dataset", "category", "model", "seed"]

# Per job: fit happens once; each row carries its own test time.
fit = p2.groupby(key)["fit_seconds"].max()
test = p2.groupby(key)["test_seconds"].sum()
job = (fit + test).rename("phase2_job_s").reset_index()

# phase2 job = fit + 4 evaluation passes (A + 3xB). phase3 = fit + 7 passes
# (A + 3xB + 3xC), so add ~3 more test passes worth of time.
per_pass = (test / 4).rename("per_pass_s").reset_index()
job = job.merge(per_pass, on=key)
job["phase3_job_s"] = job["phase2_job_s"] + 3 * job["per_pass_s"]

print("== measured phase2 vs projected phase3 job time (seconds, mean) ==")
summary = job.groupby(["dataset", "model"])[["phase2_job_s", "phase3_job_s"]].mean().round(0)
print(summary.to_string())

# phase3 = all 27 categories x 5 models x seed 1
counts = {"mvtec": 15, "visa": 12}
per_model = job.groupby(["dataset", "model"])["phase3_job_s"].mean()

heavy = ["patchcore"]
total_heavy = total_light = 0.0
for (dataset, model), secs in per_model.items():
    total = secs * counts[dataset]
    if model in heavy:
        total_heavy += total
    else:
        total_light += total

print(f"\nphase3 jobs: {sum(counts.values()) * 5} (135)")
print(f"serial GPU-time  light models (4): {total_light / 3600:6.1f} h")
print(f"serial GPU-time  patchcore    (1): {total_heavy / 3600:6.1f} h")

light_wall = total_light / 4  # 2 GPUs x 2 procs
heavy_wall = total_heavy / 2  # 2 GPUs x 1 proc
print(f"\npass 1 (4 models @ 2 proc/GPU, 4 workers): {light_wall / 3600:5.1f} h")
print(f"pass 2 (patchcore @ 1 proc/GPU, 2 workers): {heavy_wall / 3600:5.1f} h")
print(f"TOTAL estimated wall clock:                {(light_wall + heavy_wall) / 3600:5.1f} h")
print("\n(+~20% overhead for data loading, coreset, contention)")
print(f"realistic range: {(light_wall + heavy_wall) / 3600 * 1.0:.1f}-{(light_wall + heavy_wall) / 3600 * 1.4:.1f} h")
