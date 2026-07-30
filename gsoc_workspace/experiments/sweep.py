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

from gsoc_workspace.experiments.harness import DATASETS, MVTEC_CATEGORIES, JobConfig

RESULTS_DIR = Path(__file__).parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
RESULTS_CSV = RESULTS_DIR / "results.csv"

# Each job trains one model once and emits arm A + arm B per pipeline (+ arm C per
# pipeline when ``include_c`` is set), so arms share the identical trained model.
PHASES: dict[str, dict] = {
    "phase0": {
        "datasets": ["mvtec"],
        "categories": ["bottle", "cable", "hazelnut"],
        "models": ["patchcore", "draem"],
        "pipelines": ["P1", "P3"],
        "seeds": [1],
        "include_c": True,
    },
    "phase1": {
        "datasets": ["mvtec"],
        "categories": MVTEC_CATEGORIES,
        "models": ["patchcore", "padim", "efficient_ad"],
        "pipelines": ["P1", "P2", "P3"],
        "seeds": [1, 2, 3],
        "include_c": False,
    },
    # Phase 2 - grand comparison across all val-only models and both datasets.
    # The three implemented pipelines (P1/P2/P3) serve as the shortlist.
    "phase2": {
        "datasets": ["mvtec", "visa"],
        "categories": None,  # use each dataset's full category list
        "models": ["patchcore", "padim", "efficient_ad", "anomaly_dino", "dinomaly"],
        "pipelines": ["P1", "P2", "P3"],
        "seeds": [1, 2, 3],
        "include_c": False,
    },
    # Phase 3 - does the area-routed hybrid blend (P4) beat its two parents?
    # P2/P3/P4 are evaluated on the identical trained model, so the comparison is
    # paired. Arm C is included to test whether P4 lands closer to the real-test
    # difficulty than the too-hard P3.
    "phase3": {
        "datasets": ["mvtec", "visa"],
        "categories": None,
        "models": ["patchcore", "padim", "efficient_ad", "anomaly_dino", "dinomaly"],
        "pipelines": ["P2", "P3", "P4"],
        "seeds": [1],
        "include_c": True,
    },
    # Phase 4 - is the P3/P4 tie fundamental, or is 2500 px simply a mistuned knob?
    # Sweeps the hybrid routing threshold on a category subset chosen from the blob-area
    # census to span the full range of "fraction of mask area below threshold":
    # carpet 68.7% / screw 32.3% / pcb1 32.3% / bottle 27.3% / macaroni1 17.8% /
    # cashew 15.9% / hazelnut 12.4% / candle 7.5%. P3 is re-run inside the same jobs
    # as a paired anchor.
    "phase4": {
        "datasets": ["mvtec", "visa"],
        "categories": {
            "mvtec": ["bottle", "hazelnut", "screw", "carpet"],
            "visa": ["candle", "macaroni1", "pcb1", "cashew"],
        },
        "models": ["patchcore", "padim", "efficient_ad", "anomaly_dino", "dinomaly"],
        "pipelines": ["P3", "P4a", "P4b", "P4", "P4c", "P4d"],
        "seeds": [1],
        "include_c": False,
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


def enumerate_jobs(phase: str) -> list[JobConfig]:
    """Expand a phase specification into individual training jobs.

    One job is emitted per (dataset, category, model, seed); each job trains once
    and internally evaluates all arms/pipelines from the identical model.
    """
    spec = PHASES[phase]
    pipelines = tuple(spec["pipelines"])
    include_c = spec["include_c"]
    jobs: list[JobConfig] = []
    for dataset in spec["datasets"]:
        subset = spec["categories"]
        if isinstance(subset, dict):
            subset = subset.get(dataset)
        categories = subset or DATASETS[dataset][2]
        for category, model, seed in product(categories, spec["models"], spec["seeds"]):
            jobs.append(JobConfig(phase, dataset, category, model, seed, pipelines, include_c))
    return jobs


def run_job(config: JobConfig, gpu: int) -> None:
    """Launch a single job as a subprocess pinned to ``gpu``."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    cmd = [
        sys.executable, "-m", "gsoc_workspace.experiments.run_one",
        "--phase", config.phase, "--dataset", config.dataset, "--category", config.category,
        "--model", config.model, "--seed", str(config.seed),
        "--pipelines", *config.pipelines,
    ]
    if config.include_c:
        cmd.append("--include-c")
    subprocess.run(cmd, env=env, check=False)  # noqa: S603  # fixed internal command, no shell


def schedule(jobs: list[JobConfig], gpus: list[int], procs_per_gpu: int) -> None:
    """Run jobs concurrently, pinning each to a free GPU slot."""
    slots: Queue[int] = Queue()
    for gpu in gpus:
        for _ in range(procs_per_gpu):
            slots.put(gpu)

    def worker(config: JobConfig) -> None:
        gpu = slots.get()
        try:
            run_job(config, gpu)
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus) * procs_per_gpu) as pool:
        list(pool.map(worker, jobs))


def aggregate() -> int:
    """Aggregate all raw result JSONs into a single CSV. Returns row count.

    Each JSON holds a list of rows (one job emits several arm/pipeline rows).
    """
    rows: list[dict] = []
    for path in sorted(RAW_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        rows.extend(data if isinstance(data, list) else [data])
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


def pending(jobs: list[JobConfig]) -> list[JobConfig]:
    """Filter out jobs whose result JSON already exists."""
    return [job for job in jobs if not (RAW_DIR / f"{job.key}.json").exists()]


def parse_args() -> argparse.Namespace:
    """Parse sweep command-line arguments."""
    parser = argparse.ArgumentParser(description="Schedule synthetic-anomaly experiments.")
    parser.add_argument("--phase", choices=list(PHASES))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--procs-per-gpu", type=int, default=2)
    parser.add_argument("--models", nargs="+", help="Restrict the phase to these models.")
    parser.add_argument("--exclude-models", nargs="+", help="Skip these models in the phase.")
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
    if args.models:
        jobs = [job for job in jobs if job.model in set(args.models)]
    if args.exclude_models:
        jobs = [job for job in jobs if job.model not in set(args.exclude_models)]
    todo = pending(jobs)
    print(f"{args.phase}: {len(jobs)} jobs, {len(todo)} pending, {len(jobs) - len(todo)} done.")
    schedule(todo, args.gpus, args.procs_per_gpu)
    aggregate()


if __name__ == "__main__":
    main()
