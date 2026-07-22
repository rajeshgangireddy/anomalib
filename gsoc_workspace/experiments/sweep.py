# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Enumerate and schedule experiment runs across multiple GPUs.

Each run is launched as an isolated subprocess pinned to a single GPU via
``CUDA_VISIBLE_DEVICES``. Completed runs (existing result JSON) are skipped, so
the sweep is resumable. Results are aggregated into a single CSV.

Usage:
    python -m gsoc_workspace.experiments.sweep --phase phase0
    python -m gsoc_workspace.experiments.sweep --phase phase1 --gpus 0 1 --procs-per-gpu 2
    python -m gsoc_workspace.experiments.sweep --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from queue import Queue

import pandas as pd

from gsoc_workspace.experiments.harness import MVTEC_CATEGORIES, RunConfig

RESULTS_DIR = Path(__file__).parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
RESULTS_CSV = RESULTS_DIR / "results.csv"

PHASES: dict[str, dict] = {
    "phase0": {
        "datasets": ["mvtec"],
        "categories": ["bottle", "cable", "hazelnut"],
        "models": ["patchcore", "draem"],
        "pipelines": ["P1", "P3"],
        "arms": ["A", "B", "C"],
        "seeds": [1],
    },
    "phase1": {
        "datasets": ["mvtec"],
        "categories": MVTEC_CATEGORIES,
        "models": ["patchcore", "padim", "efficient_ad"],
        "pipelines": ["P1", "P2", "P3"],
        "arms": ["A", "B"],
        "seeds": [1, 2, 3],
    },
}

# Column order for the aggregated CSV.
METRIC_COLUMNS = [
    "image_AUROC", "image_F1Score", "image_AUPR", "image_BinaryPrecision", "image_BinaryRecall",
    "pixel_AUROC", "pixel_F1Score", "pixel_AUPRO",
]
COLUMN_ORDER = [
    "phase", "dataset", "category", "model", "pipeline", "arm", "seed",
    *METRIC_COLUMNS,
    "n_train", "n_val", "n_test", "image_threshold", "normalized_image_threshold",
    "fit_seconds", "test_seconds", "anomalib_version", "timestamp",
]


def enumerate_jobs(phase: str) -> list[RunConfig]:
    """Expand a phase specification into individual run configs.

    Arm A is pipeline-independent, so it is emitted once per
    (dataset, category, model, seed) rather than per pipeline.
    """
    spec = PHASES[phase]
    jobs: list[RunConfig] = []
    for dataset, category, model, seed in product(
        spec["datasets"], spec["categories"], spec["models"], spec["seeds"],
    ):
        if "A" in spec["arms"]:
            jobs.append(RunConfig(phase, dataset, category, model, "A", "-", seed))
        for pipeline, arm in product(spec["pipelines"], spec["arms"]):
            if arm != "A":
                jobs.append(RunConfig(phase, dataset, category, model, arm, pipeline, seed))
    return jobs


def run_job(config: RunConfig, gpu: int) -> None:
    """Launch a single run as a subprocess pinned to ``gpu``."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    cmd = [
        sys.executable, "-m", "gsoc_workspace.experiments.run_one",
        "--phase", config.phase, "--dataset", config.dataset, "--category", config.category,
        "--model", config.model, "--arm", config.arm, "--pipeline", config.pipeline,
        "--seed", str(config.seed),
    ]
    subprocess.run(cmd, env=env, check=False)  # noqa: S603  # fixed internal command, no shell


def schedule(jobs: list[RunConfig], gpus: list[int], procs_per_gpu: int) -> None:
    """Run jobs concurrently, pinning each to a free GPU slot."""
    slots: Queue[int] = Queue()
    for gpu in gpus:
        for _ in range(procs_per_gpu):
            slots.put(gpu)

    def worker(config: RunConfig) -> None:
        gpu = slots.get()
        try:
            run_job(config, gpu)
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus) * procs_per_gpu) as pool:
        list(pool.map(worker, jobs))


def aggregate() -> int:
    """Aggregate all raw result JSONs into a single CSV. Returns row count."""
    rows = [json.loads(path.read_text()) for path in sorted(RAW_DIR.glob("*.json"))]
    if not rows:
        print("No results to aggregate.")
        return 0
    frame = pd.DataFrame(rows)
    ordered = [c for c in COLUMN_ORDER if c in frame.columns]
    remainder = [c for c in frame.columns if c not in ordered]
    frame = frame[ordered + remainder]
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_CSV, index=False)
    print(f"Aggregated {len(frame)} runs -> {RESULTS_CSV}")
    return len(frame)


def pending(jobs: list[RunConfig]) -> list[RunConfig]:
    """Filter out jobs whose result JSON already exists."""
    return [job for job in jobs if not (RAW_DIR / f"{job.key}.json").exists()]


def parse_args() -> argparse.Namespace:
    """Parse sweep command-line arguments."""
    parser = argparse.ArgumentParser(description="Schedule synthetic-anomaly experiments.")
    parser.add_argument("--phase", choices=list(PHASES))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--procs-per-gpu", type=int, default=2)
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Entry point: schedule a phase and aggregate results."""
    args = parse_args()
    if args.aggregate_only:
        aggregate()
        return
    if not args.phase:
        print("Specify --phase or --aggregate-only.")
        return

    jobs = enumerate_jobs(args.phase)
    todo = pending(jobs)
    print(f"{args.phase}: {len(jobs)} jobs, {len(todo)} pending, {len(jobs) - len(todo)} done.")
    schedule(todo, args.gpus, args.procs_per_gpu)
    aggregate()


if __name__ == "__main__":
    main()
