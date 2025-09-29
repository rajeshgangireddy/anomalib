# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Creates a dataset based on one of the MVTEC dataset categories. Ideal for testing."""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    """Create dataset subsets with different numbers of training samples."""
    parser = argparse.ArgumentParser(description="Create a subset of the MVTEC dataset.")
    parser.add_argument(
        "--category",
        type=str,
        default="screw",
        required=False,
        help="Category of the dataset (e.g., screw, toothbrush, etc.)",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200],
        required=False,
        help="List of numbers of training samples to create subsets for.",
    )
    args = parser.parse_args()

    category = args.category
    mvtec_location = Path("datasets/MVTecAD")
    output_location = Path(f"datasets/Custom/MVTecCustom/{category}/")
    num_train_samples_list = args.num_train_samples

    category_path = mvtec_location / category
    if not category_path.exists():
        print(f"Error: Category '{category}' does not exist in {mvtec_location}.")
        return

    train_source = category_path / "train" / "good"
    if not train_source.exists():
        print(f"Error: 'train' folder does not exist for category '{category}'.")
        return

    train_images = [item.name for item in train_source.iterdir() if item.is_file()]
    if not train_images:
        print(f"Error: No training images found in {train_source}.")
        return

    dataset_num = 0
    for num_train_samples in num_train_samples_list:
        if num_train_samples > len(train_images):
            msg = (
                f"Warning: Requested {num_train_samples} samples, but only {len(train_images)} are available. Skipping."
            )
            print(msg)
            continue
        # make a dataset prefix like 01, 02, 03,
        dataset_num += 1
        num_str = str(dataset_num).zfill(2)
        dataset_dir = f"{num_str}_MVTeC_{category}_{num_train_samples}"
        dataset_path = output_location / dataset_dir
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Copy all of the mvtec folder except the "train" folder
        for item in category_path.iterdir():
            s = item
            d = dataset_path / item.name
            if s.is_dir() and item.name != "train":
                shutil.copytree(s, d)
            elif s.is_file():
                shutil.copy2(s, d)

        # Create a new train folder with only a subset of the training images
        train_dest = dataset_path / "train" / "good"
        train_dest.mkdir(parents=True, exist_ok=True)
        for img in train_images[:num_train_samples]:
            s = train_source / img
            d = train_dest / img
            shutil.copy2(s, d)

        # Move the 'good' folder outside the 'test' folder and rename it to 'test_good'
        test_good_source = dataset_path / "test" / "good"
        test_good_dest = dataset_path / "test_good"
        if test_good_source.exists():
            shutil.move(test_good_source, test_good_dest)

        # Remove the now-empty 'test' folder if it exists
        test_folder = dataset_path / "test"
        if test_folder.exists() and not any(test_folder.iterdir()):
            test_folder.rmdir()

        print(f"Created dataset at: {dataset_path}")


if __name__ == "__main__":
    main()
