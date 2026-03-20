"""FoundAD few-shot experiment on MVTecAD Transistor.

Runs FoundAD with 1-shot, 2-shot, and 4-shot settings on the Transistor
category of MVTecAD. For each shot count, the model trains on exactly N
normal images and is evaluated on the full test set.

Usage:
    python experiments/foundad_fewshot.py
    python experiments/foundad_fewshot.py --epochs 50 --encoder dinov2_vit_base_14
"""

import argparse
import random
import time
from pathlib import Path

import torch
from tabulate import tabulate

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import FoundAD

SEED = 42
CATEGORY = "transistor"
DATASET_ROOT = "./datasets/MVTecAD"
SHOTS = [1, 2, 4]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FoundAD few-shot experiment")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument(
        "--encoder",
        type=str,
        default="dinov2_vit_small_14",
        help="DINOv2 encoder name (small/base/large)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=266,
        help="Input image size (266=19*14 for DINOv2, 518=37*14)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--category", type=str, default=CATEGORY)
    parser.add_argument("--shots", type=int, nargs="+", default=SHOTS)
    return parser.parse_args()


def run_experiment(
    n_shot: int,
    max_epochs: int,
    encoder_name: str,
    category: str,
    image_size: int = 266,
    seed: int = SEED,
) -> dict:
    """Run a single few-shot experiment.

    Args:
        n_shot: Number of normal training images.
        max_epochs: Maximum training epochs.
        encoder_name: DINOv2 encoder variant.
        category: MVTecAD category name.
        image_size: Input image resolution.
        seed: Random seed for reproducibility.

    Returns:
        Dict with metrics from the test run.
    """
    results_dir = Path(f"results/foundad_fewshot_{category}")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  FoundAD {n_shot}-shot on MVTecAD/{category}")
    print(f"  encoder={encoder_name}, epochs={max_epochs}, seed={seed}, img={image_size}")
    print(f"{'=' * 60}\n")

    # Seed everything
    random.seed(seed)
    torch.manual_seed(seed)

    # Create datamodule
    datamodule = MVTecAD(
        root=DATASET_ROOT,
        category=category,
        train_batch_size=min(n_shot, 8),
        eval_batch_size=32,
        num_workers=2,
        seed=seed,
    )

    # Setup to get access to train_data
    datamodule.setup(stage="fit")

    # Subsample training data to n_shot images
    total_train = len(datamodule.train_data)
    indices = list(range(total_train))
    random.shuffle(indices)
    selected = sorted(indices[:n_shot])
    datamodule.train_data = datamodule.train_data.subsample(selected)
    print(f"Training samples: {len(datamodule.train_data)} (from {total_train})")

    # Create model
    model = FoundAD(
        encoder_name=encoder_name,
        pred_depth=6,
        pred_emb_dim=384,
        n_layer=3,
        top_k=10,
        dropout=0.2,
        image_size=image_size,
        lr=1e-3,
        weight_decay=1e-4,
        color_jitter=0.5,
    )

    # Create engine — validate only at end to save CPU time
    engine = Engine(
        max_epochs=max_epochs,
        default_root_dir=str(results_dir / f"{n_shot}shot"),
        devices=1,
        accelerator="auto",
        check_val_every_n_epoch=max_epochs,
        enable_progress_bar=True,
    )

    # Train
    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s")

    # Test
    datamodule.setup(stage="test")
    test_results = engine.test(model=model, datamodule=datamodule)

    metrics = test_results[0] if test_results else {}
    metrics["n_shot"] = n_shot
    metrics["train_time_s"] = round(train_time, 1)
    return metrics


def main() -> None:
    """Run all few-shot experiments and print summary table."""
    args = parse_args()

    all_results = []
    for n_shot in args.shots:
        result = run_experiment(
            n_shot=n_shot,
            max_epochs=args.epochs,
            encoder_name=args.encoder,
            category=args.category,
            image_size=args.image_size,
            seed=args.seed,
        )
        all_results.append(result)
        print(f"\n{n_shot}-shot result: {result}\n")

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"  FoundAD Few-Shot Results on MVTecAD/{args.category}")
    print(f"  encoder={args.encoder}, epochs={args.epochs}, img={args.image_size}")
    print(f"{'=' * 70}\n")

    # Extract table rows
    headers = ["Shots", "Image AUROC", "Pixel AUROC", "Image F1", "Pixel F1", "Time (s)"]
    rows = []
    for r in all_results:
        rows.append([
            r.get("n_shot", "?"),
            f"{r.get('image_AUROC', 0):.4f}" if isinstance(r.get("image_AUROC"), float) else "N/A",
            f"{r.get('pixel_AUROC', 0):.4f}" if isinstance(r.get("pixel_AUROC"), float) else "N/A",
            f"{r.get('image_F1Score', 0):.4f}" if isinstance(r.get("image_F1Score"), float) else "N/A",
            f"{r.get('pixel_F1Score', 0):.4f}" if isinstance(r.get("pixel_F1Score"), float) else "N/A",
            r.get("train_time_s", "?"),
        ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
