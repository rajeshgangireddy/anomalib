# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dry run for the synthetic-anomaly experiment harness.

Exercises the full path -- datamodule, custom synthetic validation split, model,
engine fit/test -- with ``fast_dev_run`` so integration errors surface quickly
before launching the real experiment grid.

Usage:
    python gsoc_workspace/dry_run.py
"""

from anomalib.data import MVTecAD
from anomalib.data.utils import ValSplitMode
from anomalib.data.utils.generators import SyntheticAnomalyGenerator
from anomalib.data.utils.synthetic import SyntheticAnomalyDataset
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore

CATEGORY = "bottle"


def build_datamodule(val_split_mode: ValSplitMode) -> MVTecAD:
    """Return an MVTecAD datamodule for a single category."""
    datamodule = MVTecAD(
        root="./datasets/MVTecAD",
        category=CATEGORY,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=4,
        val_split_mode=val_split_mode,
        val_split_ratio=0.5,
        seed=0,
    )
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def inject_synthetic_val(datamodule: MVTecAD, preset: str) -> None:
    """Replace the validation set with synthetic anomalies from held-out train normals."""
    generator = SyntheticAnomalyGenerator.from_preset(preset, probability=1.0)
    datamodule.val_data = SyntheticAnomalyDataset.from_dataset(datamodule.val_data, augmenter=generator)


def run(name: str, model: object, datamodule: MVTecAD) -> None:
    """Fit and test a single (model, arm) combination in fast_dev_run mode."""
    engine = Engine(fast_dev_run=True, accelerator="gpu", devices=1)
    engine.fit(model=model, datamodule=datamodule)
    engine.test(model=model, datamodule=datamodule)
    print(f"[OK] {name}")


def main() -> None:
    """Run the dry-run combinations."""
    # Arm A: real anomalies carved from the test set for validation.
    run("PatchCore | Arm A (real-val)", Patchcore(), build_datamodule(ValSplitMode.FROM_TEST))

    # Arm B: synthetic validation from held-out train normals (our generator).
    datamodule = build_datamodule(ValSplitMode.FROM_TRAIN)
    inject_synthetic_val(datamodule, preset="self_poisson")
    run("PaDiM | Arm B (synthetic-val, self_poisson)", Padim(), datamodule)

    print("Dry run complete: no errors.")


if __name__ == "__main__":
    main()
