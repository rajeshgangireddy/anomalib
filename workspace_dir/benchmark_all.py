"""Comprehensive benchmarking script for all Anomalib image models.

This script runs all available image models for anomaly detection, measures their
performance, and saves results to an Excel file with detailed system information.

Usage:
    python benchmark_all_models.py --device xpu --category toothbrush --epochs 100
    python benchmark_all_models.py --device gpu --category bottle --epochs 50
    python benchmark_all_models.py --device cpu --category tile --epochs 25
"""

import argparse
import logging
import platform
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from lightning import seed_everything

from anomalib.data import MVTecAD
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import get_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# All available image models
IMAGE_MODELS = [
    'Cfa',
    'Cflow',
    'Csflow',
    'Dfkde',
    'Dfm',
    'Dinomaly',
    'Draem',
    'Dsr',
    'EfficientAd',
    'Fastflow',
    'Fre',
    'Ganomaly',
    'Padim',
    'Patchcore',
    'ReverseDistillation',
    'Stfpm',
    'Supersimplenet',
    'Uflow',
    'UniNet',
    'VlmAd',
    'WinClip',
]


def get_device_info(device_type: str) -> dict[str, Any]:
    """Get detailed device information.

    Args:
        device_type: Either 'gpu', 'xpu', or 'cpu'

    Returns:
        Dictionary with device information
    """
    device_info = {'device_type': device_type}

    if device_type == 'gpu' and torch.cuda.is_available():
        device_info.update({
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory': f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            'cuda_version': torch.version.cuda,
            'gpu_count': torch.cuda.device_count(),
        })
    elif device_type == 'xpu':
        try:
            import intel_extension_for_pytorch as ipex
            device_info['ipex_version'] = ipex.__version__

            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                props = torch.xpu.get_device_properties()
                device_info.update({
                    'xpu_device': props.name,
                    'xpu_units': props.gpu_eu_count,
                    'xpu_memory': f"{props.total_memory / 1e9:.1f} GB" if hasattr(props, 'total_memory') else 'N/A',
                    'xpu_count': torch.xpu.device_count(),
                })
            else:
                device_info['xpu_available'] = False
        except ImportError:
            device_info['ipex_installed'] = False
    elif device_type == 'cpu':
        # For CPU, we can add CPU-specific information
        device_info.update({
            'cpu_available': True,
            'cpu_cores': torch.get_num_threads(),
        })

    # Add system info
    device_info.update({
        'torch_version': torch.__version__,
        'python_version': f"{platform.python_version()}",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })

    # Add CPU info
    device_info['cpu_name'] = platform.processor() or platform.uname().processor or platform.uname().machine
    device_info['cpu_node'] = platform.uname().node

    return device_info


def create_engine(device_type: str, epochs: int) -> Engine:
    """Create appropriate engine based on device type.

    Args:
        device_type: Either 'gpu', 'xpu', or 'cpu'
        epochs: Number of training epochs

    Returns:
        Configured Engine instance
    """
    if device_type == 'xpu':
        return Engine(
            strategy=SingleXPUStrategy(),
            accelerator=XPUAccelerator(),
            max_epochs=epochs,
        )
    if device_type == 'cpu':
        return Engine(
            accelerator='cpu',
            max_epochs=epochs,
        )
    return Engine(max_epochs=epochs)


def benchmark_model(
    model_name: str,
    datamodule: MVTecAD,
    device_type: str,
    epochs: int,
) -> dict[str, Any]:
    """Benchmark a single model.

    Args:
        model_name: Name of the model
        datamodule: Data module for training/testing
        device_type: Device type ('gpu', 'xpu', or 'cpu')
        epochs: Number of training epochs

    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Starting benchmark for {model_name}")

    result = {
        'model_name': model_name,
        'status': 'failed',
        'error_message': None,
        'training_time': None,
        'testing_time': None,
        'epochs_completed': 0,
    }

    try:
        # Initialize model using get_model
        model = get_model(model_name)

        # Create engine
        engine = create_engine(device_type, epochs)

        # Training phase
        logger.info(f"Training {model_name} for {epochs} epochs...")
        train_start_time = time.time()

        engine.fit(datamodule=datamodule, model=model)

        train_end_time = time.time()
        training_time = train_end_time - train_start_time

        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Testing phase
        logger.info(f"Testing {model_name}...")
        test_start_time = time.time()

        test_results = engine.test(datamodule=datamodule, model=model)

        test_end_time = time.time()
        testing_time = test_end_time - test_start_time

        logger.info(f"Testing completed in {testing_time:.2f} seconds")

        # Extract test metrics
        if test_results and len(test_results) > 0:
            test_metrics = test_results[0]  # Get the first result
            for key, value in test_metrics.items():
                if isinstance(value, torch.Tensor):
                    result[key] = float(value.item())
                else:
                    result[key] = value

        # Update result
        result.update({
            'status': 'success',
            'training_time': training_time,
            'testing_time': testing_time,
            'epochs_completed': epochs,
            'total_time': training_time + testing_time,
        })

        logger.info(f"Successfully completed {model_name}")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to benchmark {model_name}: {error_msg}")
        logger.error(traceback.format_exc())

        result.update({
            'status': 'failed',
            'error_message': error_msg,
        })

    return result


def save_results_to_excel(
    results: list[dict[str, Any]],
    device_info: dict[str, Any],
    category: str,
    device_type: str,
    output_dir: Path,
    model_name: str = None,
    num_runs: int = None,
) -> Path:
    """Save benchmark results to Excel file.

    Args:
        results: List of benchmark results
        device_info: Device information
        category: MVTec category
        device_type: Device type
        output_dir: Output directory

    Returns:
        Path to saved Excel file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Build filename with model name and num_runs if provided
    filename = f"anomalib_benchmark_{timestamp}_{device_type}"
    if model_name:
        filename += f"_{model_name}"
    if num_runs:
        filename += f"_{num_runs}runs"
    filename += ".xlsx"
    filepath = output_dir / filename

    # Create Excel writer
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Create summary sheet with device info
        summary_data = {
            'Parameter': ['MVTec Category'] + list(device_info.keys()),
            'Value': [category] + list(device_info.values()),
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='System_Info', index=False)

        # Create the result sheet
        results_df = pd.DataFrame(results)

        # Reorder columns for better readability
        column_order = [
            'model_name', 'status', 'epochs_completed', 'training_time',
            'testing_time', 'total_time', 'error_message',
        ]

        # Add any metric columns that exist
        metric_columns = [col for col in results_df.columns if col not in column_order]
        column_order.extend(metric_columns)

        # Reorder DataFrame columns
        available_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[available_columns]

        results_df.to_excel(writer, sheet_name='Benchmark_Results', index=False)

        # --- Add averages and stddev columns in the same sheet for each model ---
        if 'run' in results_df.columns:
            # Define potential average columns, but only use ones that exist in the DataFrame
            potential_avg_columns = [
                'training_time', 'testing_time', 'total_time', 'epochs_completed',
                'image_AUROC', 'image_F1Score', 'pixel_AUROC', 'pixel_F1Score',
            ]
            # Filter to only columns that actually exist in the DataFrame and are numeric
            average_columns = [col for col in potential_avg_columns
                             if col in results_df.columns and results_df[col].dtype.kind in 'biufc']

            if average_columns:  # Only proceed if we have columns to average
                avg_df = results_df.groupby('model_name')[average_columns].mean(numeric_only=True)
                std_df = results_df.groupby('model_name')[average_columns].std(numeric_only=True)
            if average_columns:  # Only proceed if we have columns to average
                avg_df = results_df.groupby('model_name')[average_columns].mean(numeric_only=True)
                std_df = results_df.groupby('model_name')[average_columns].std(numeric_only=True)
                # Rename std columns
                std_df = std_df.rename(columns={col: f"{col}_std" for col in std_df.columns})
                # Concatenate avg and std columns
                merged_df = pd.concat([avg_df, std_df], axis=1).reset_index()
                # Ensure avg columns come first, then std columns
                ordered_cols = ['model_name'] + average_columns + [f"{col}_std" for col in average_columns if f"{col}_std" in merged_df.columns]
                merged_df = merged_df[ordered_cols]
                merged_df.to_excel(writer, sheet_name='Averages', index=False)
            else:
                # Create an empty averages sheet if no numeric columns to average
                empty_avg_df = pd.DataFrame({'Message': ['No numeric columns available for averaging']})
                empty_avg_df.to_excel(writer, sheet_name='Averages', index=False)

        # Create a success / failure summary
        status_summary = results_df['status'].value_counts().to_frame()
        status_summary.reset_index(inplace=True)
        status_summary.columns = ['Status', 'Count']
        status_summary.to_excel(writer, sheet_name='Summary', index=False)

    logger.info(f"Results saved to: {filepath}")
    return filepath


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description='Benchmark all Anomalib image models')
    parser.add_argument(
        '--device',
        type=str,
        choices=['gpu', 'xpu', 'cpu'],
        default='gpu',
        help='Device type to use (gpu, xpu, or cpu)',
    )
    parser.add_argument(
        '--category',
        type=str,
        default='toothbrush',
        help='MVTec category to use for benchmarking',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(),
        help='Output directory for results',
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='*',
        help='Specific models to benchmark (default: all models)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=3,
        help='Number of times to run each model (default: 3)',
    )

    args = parser.parse_args()

    # Set random seed
    seed_everything(args.seed)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get device information
    device_info = get_device_info(args.device)

    # Log system information
    logger.info("=== Benchmark Configuration ===")
    logger.info(f"Device: {args.device}")
    logger.info(f"Category: {args.category}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Models: {args.models if args.models else 'All models'}")
    logger.info("=== Device Information ===")
    for key, value in device_info.items():
        logger.info(f"{key}: {value}")

    # Initialize datamodule
    logger.info(f"Initializing MVTecAD datamodule for category: {args.category}")
    datamodule = MVTecAD(category=args.category)

    # If benchmarking EfficientAd, set train_batch_size=1
    if args.models and len(args.models) == 1 and args.models[0].lower() == "efficientad" or (
        not args.models and "EfficientAd" in IMAGE_MODELS
    ):
        if hasattr(datamodule, 'train_batch_size'):
            logger.info("Setting train_batch_size=1 for EfficientAd model.")
            datamodule.train_batch_size = 1

    # Determine which models to benchmark
    models_to_benchmark = IMAGE_MODELS
    if args.models:
        models_to_benchmark = [name for name in IMAGE_MODELS if name in args.models]
        logger.info(f"Benchmarking specific models: {models_to_benchmark}")
    else:
        logger.info(f"Benchmarking all {len(models_to_benchmark)} models")

    # Run benchmarks
    results = []
    total_models = len(models_to_benchmark)

    for i, model_name in enumerate(models_to_benchmark, 1):
        logger.info(f"Progress: {i}/{total_models} - Starting {model_name}")
        for run_idx in range(args.num_runs):
            logger.info(f"Run {run_idx+1}/{args.num_runs} for {model_name}")
            result = benchmark_model(
                model_name=model_name,
                datamodule=datamodule,
                device_type=args.device,
                epochs=args.epochs,
            )
            result['run'] = run_idx + 1
            results.append(result)

            # Log intermediate results
            if result['status'] == 'success':
                logger.info(f"✓ {model_name} run {run_idx+1} completed successfully")
            else:
                logger.warning(f"✗ {model_name} run {run_idx+1} failed: {result['error_message']}")

            # Wait 20 seconds between runs, except after the last run of the last model
            is_last_model = (i == total_models)
            is_last_run = (run_idx == args.num_runs - 1)
            if not (is_last_model and is_last_run):
                logger.info("Waiting 20 seconds before next run/model...")
                time.sleep(20)

    # Save results
    # If only one model, add model name and num_runs to filename
    model_name_for_file = models_to_benchmark[0] if len(models_to_benchmark) == 1 else None
    excel_path = save_results_to_excel(
        results=results,
        device_info=device_info,
        category=args.category,
        device_type=args.device,
        output_dir=args.output_dir,
        model_name=model_name_for_file,
        num_runs=args.num_runs if model_name_for_file else None,
    )

    # Print summary
    successful_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - successful_count

    logger.info("=== Benchmark Summary ===")
    logger.info(f"Total runs: {len(results)}")
    logger.info(f"Successful: {successful_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Results saved to: {excel_path}")

    if successful_count > 0:
        successful_results = [r for r in results if r['status'] == 'success']
        avg_training_time = sum(r['training_time'] for r in successful_results) / len(successful_results)
        avg_testing_time = sum(r['testing_time'] for r in successful_results) / len(successful_results)
        logger.info(f"Average training time: {avg_training_time:.2f} seconds")
        logger.info(f"Average testing time: {avg_testing_time:.2f} seconds")


if __name__ == "__main__":
    main()
