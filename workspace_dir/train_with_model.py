"""Simple training script with model selection via command line arguments."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
from pathlib import Path

import torch
from lightning import seed_everything

from anomalib.data import Folder
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import get_model

DATA_ROOT = Path("/home/devuser/workspace/code/Anomalib/dev/anomalib/datasets/Custom/apples_on_belt")


def main() -> None:
    """Main function for training and testing anomaly detection models."""
    seed_everything(42, workers=True)
    parser = argparse.ArgumentParser(description="Train and test an Anomalib model.")
    parser.add_argument("--model", type=str, default="Padim", help="Model name to use (e.g., patchcore, padim, etc.)")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum number of epochs for training")
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    normal_dir = DATA_ROOT / "normal"
    abnormal_dir = DATA_ROOT / "abnormal"
    datamodule = Folder(name="Apples", normal_dir=normal_dir)
    datamodule_test = Folder(name="Apples", normal_dir=normal_dir, abnormal_dir=abnormal_dir)

    model = get_model(args.model)

    if not cuda_available:
        print("CUDA is not available. Using XPU strategy.")
        print(f"XPU Available: {torch.xpu.is_available()}")
        engine = Engine(
            strategy=SingleXPUStrategy(),
            accelerator=XPUAccelerator(),
        )
    else:
        print("CUDA is available. Using default strategy.")
        engine = Engine()

    tic = time.time()
    engine.fit(datamodule=datamodule, model=model)
    training_time = time.time() - tic
    tic = time.time()
    _ = engine.test(datamodule=datamodule_test, model=model)
    export_path = engine.export(model=model, export_type="torch", model_file_name=f"{model.__class__.__name__}")
    print(f"Model exported to: {export_path}")

    testing_time = time.time() - tic
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Testing time: {testing_time:.2f} seconds")


if __name__ == "__main__":
    main()
