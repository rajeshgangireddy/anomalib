"""Enhanced Training Script.

A script that takes category and max_steps as inputs, performs training and testing,
and saves results to an Excel sheet.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from lightning import seed_everything

from anomalib.data import MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.loggers import AnomalibWandbLogger
from anomalib.metrics import AUROC, F1Max, F1Score
from anomalib.metrics.evaluator import Evaluator
from anomalib.models import Glass as ModelClass

SEED = 42
BATCH_SIZE = 8  # As used by the paper
PROJECT_NAME = "Glass-BM-MVTEC-T2"
ENABLE_WANDB = True
COMPRESS_TYPES: list[str] = []  # ["fp32", "fp16", "int8"]
INPUT_SIZE = (288, 288)


def main() -> bool | None:
    """Main function to run training and testing with configurable parameters."""
    parser = argparse.ArgumentParser(description="Train and test anomaly detection model")
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        help="Dataset category (e.g., carpet, bottle, cable)",
    )
    parser.add_argument(
        "--train_iterations",
        type=str,
        default="max_epochs",
        help="Train until 'max_steps' or 'max_epochs'. Default is 'max_epochs'.",
    )
    parser.add_argument(
        "--max_train_iterations",
        type=int,
        default=-1,
        help="Maximum number of training epochs (default: -1, use default from model)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Dinomaly",
        help="Model name for results tracking",
    )
    seed_everything(seed=SEED, workers=True)  # Set a seed for reproducibility

    args = parser.parse_args()

    # Create an output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Starting training for category: {args.category}")
    print(f"Train iterations mode: {args.train_iterations}")
    print(
        f"Max train iterations: "
        f"{args.max_train_iterations if args.max_train_iterations != -1 else 'default from model'}",
    )
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {SEED}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Input size: {INPUT_SIZE}")
    print(f"Project name: {PROJECT_NAME}")
    print(f"Enable Weights & Biases logging: {ENABLE_WANDB}")
    print(f"Compress types: {COMPRESS_TYPES}")
    print(f"Model: {args.model_name}")

    image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)
    pixel_f1score = F1Score(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    pixel_f1max = F1Max(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)

    test_metrics = [image_auroc, image_f1score, pixel_auroc, pixel_f1score, pixel_f1max]
    evaluator = Evaluator(test_metrics=test_metrics)

    # Initialize components
    datamodule = MVTecAD(category=args.category, train_batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE)
    model = ModelClass(evaluator=evaluator)
    wandb_name = f"{args.model_name}_{args.category}"

    wandb_logger = (
        AnomalibWandbLogger(name=wandb_name, save_dir=args.output_dir, project=PROJECT_NAME) if ENABLE_WANDB else None
    )

    # Engine creation logic based on train_iterations and max_train_iterations
    engine_kwargs = {
        "accelerator": "gpu",
        "devices": 1,
        "logger": wandb_logger,
    }
    if args.max_train_iterations != -1:
        if args.train_iterations == "max_steps":
            engine_kwargs["max_steps"] = args.max_train_iterations
        else:
            engine_kwargs["max_epochs"] = args.max_train_iterations
    engine = Engine(**engine_kwargs)

    # Record start time
    start_time = time.time()
    training_start = time.time()

    try:
        # Train the model
        print("Starting training...")
        engine.fit(datamodule=datamodule, model=model)
        training_time = time.time() - training_start
        print(f"Training completed in {training_time:.2f} seconds")

        # Test the model
        print("Starting testing...")
        testing_start = time.time()
        test_results = engine.test(datamodule=datamodule, model=model)
        testing_time = time.time() - testing_start
        print(f"Testing completed in {testing_time:.2f} seconds")

        # Export model to OPENVINO
        for compress_type in COMPRESS_TYPES:
            print(f"Exporting model with compression type: {compress_type}")
            tic = time.time()
            compress_type_param = None if compress_type == "fp32" else compress_type
            export_path = Path(args.output_dir) / f"{args.category}/openvino/{compress_type}/"
            export_path.mkdir(parents=True, exist_ok=True)

            exported_path = engine.export(
                model=model,
                export_type=ExportType.OPENVINO,
                input_size=INPUT_SIZE,
                compression_type=compress_type_param,
                export_root=export_path,
            )

            export_time = time.time() - tic
            print(f"Model exported to: {exported_path} in {export_time:.2f} seconds")

        #
        # Calculate total time
        total_time = time.time() - start_time

        # Extract metrics from test results
        # test_results is typically a list of dictionaries containing metrics
        metrics = {}
        if test_results and len(test_results) > 0:
            # Get the first (and usually only) result dictionary
            result_dict = test_results[0] if isinstance(test_results, list) else test_results

            # Extract common anomaly detection metrics
            metrics.update({key: value for key, value in result_dict.items() if isinstance(value, (int, float))})

        # Prepare results data
        results_data = {
            "seed": [SEED],
            "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "model": [args.model_name],
            "category": [args.category],
            "train_iterations": [args.train_iterations],
            "max_train_iterations": [args.max_train_iterations],
            "training_time_seconds": [training_time],
            "testing_time_seconds": [testing_time],
            "total_time_seconds": [total_time],
        }

        # Add metrics to results data
        for metric_name, metric_value in metrics.items():
            results_data[metric_name] = [metric_value]

        # Create DataFrame
        df = pd.DataFrame(results_data)

        # Define an output file path
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file = Path(args.output_dir) / f"{args.category}_{timestamp_str}.xlsx"

        # Save to Excel
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Results", index=False)

            # Add a summary sheet with key information
            summary_data = {
                "Parameter": [
                    "Seed",
                    "Model",
                    "Category",
                    "Train Iterations Mode",
                    "Max Train Iterations",
                    "Training Time (s)",
                    "Testing Time (s)",
                    "Total Time (s)",
                ],
                "Value": [
                    SEED,
                    args.model_name,
                    args.category,
                    args.train_iterations,
                    args.max_train_iterations,
                    f"{training_time:.2f}",
                    f"{testing_time:.2f}",
                    f"{total_time:.2f}",
                ],
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        print(f"\nResults saved to: {excel_file}")
        print(f"Total elapsed time: {total_time:.2f} seconds")

        # Print summary
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Seed: {SEED}")
        print(f"Model: {args.model_name}")
        print(f"Category: {args.category}")
        print(f"Train Iterations Mode: {args.train_iterations}")
        print(
            f"Max Train Iterations: "
            f"{args.max_train_iterations if args.max_train_iterations != -1 else 'default from model'}",
        )
        print(f"Training Time: {training_time:.2f}s")
        print(f"Testing Time: {testing_time:.2f}s")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Results saved to: {excel_file}")
        print(f"Model exported to OpenVINO formats in {args.output_dir}")
        print("=" * 50)

        if metrics:
            print("\nTest Metrics:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value}")

    except (RuntimeError, ValueError, OSError) as e:
        print(f"Error during training/testing: {e!s}")

        # Save error information to Excel
        error_data = {
            "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "model": [args.model_name],
            "category": [args.category],
            "train_iterations": [args.train_iterations],
            "max_train_iterations": [args.max_train_iterations],
            "status": ["FAILED"],
            "error": [str(e)],
            "partial_time_seconds": [time.time() - start_time],
        }

        error_df = pd.DataFrame(error_data)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = Path(args.output_dir) / f"anomaly_detection_error_{timestamp_str}.xlsx"
        error_df.to_excel(error_file, index=False)

        print(f"Error details saved to: {error_file}")
        return False
    else:
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
