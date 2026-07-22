# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run a single experiment and persist its result as JSON.

Each run writes ``results/raw/<key>.json`` so runs are independent and the sweep
is resumable (an existing file means the run is skipped). Aggregation into a CSV
is handled by ``sweep.py``.

Usage:
    python -m gsoc_workspace.experiments.run_one \
        --phase phase0 --dataset mvtec --category bottle \
        --model patchcore --arm B --pipeline P3 --seed 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gsoc_workspace.experiments.harness import RunConfig, run_single

RAW_DIR = Path(__file__).parent / "results" / "raw"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a single run."""
    parser = argparse.ArgumentParser(description="Run a single synthetic-anomaly experiment.")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True, choices=["A", "B", "C"])
    parser.add_argument("--pipeline", default="-")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Execute one run and write its result JSON (unless already present)."""
    args = parse_args()
    config = RunConfig(
        phase=args.phase,
        dataset=args.dataset,
        category=args.category,
        model=args.model,
        arm=args.arm,
        pipeline=args.pipeline,
        seed=args.seed,
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{config.key}.json"
    if out_path.exists():
        print(f"[skip] {config.key}")
        return

    result = run_single(config)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[done] {config.key}")


if __name__ == "__main__":
    main()
