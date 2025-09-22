# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Creates a dataset based on one of the MVTEC dataset categories. Ideal for testing."""

import argparse
import os
import shutil


def main():
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

    CATEGORY = args.category
    MVTEC_LOCATION = "datasets/MVTecAD"
    OUTPUT_LOCATION = f"datasets/Custom/MVTecCustom/{CATEGORY}/"
    NUM_TRAIN_SAMPLES = args.num_train_samples

    category_path = os.path.join(MVTEC_LOCATION, CATEGORY)
    if not os.path.exists(category_path):
        print(f"Error: Category '{CATEGORY}' does not exist in {MVTEC_LOCATION}.")
        return

    train_source = os.path.join(category_path, "train", "good")
    if not os.path.exists(train_source):
        print(f"Error: 'train' folder does not exist for category '{CATEGORY}'.")
        return

    train_images = os.listdir(train_source)
    if not train_images:
        print(f"Error: No training images found in {train_source}.")
        return

    dataset_num = 0
    for num_train_samples in NUM_TRAIN_SAMPLES:
        if num_train_samples > len(train_images):
            print(
                f"Warning: Requested {num_train_samples} samples, but only {len(train_images)} are available. Skipping.",
            )
            continue
        # make a dataset prefix like 01, 02, 03,
        dataset_num += 1
        num_str = str(dataset_num).zfill(2)
        dataset_dir = f"{num_str}_MVTeC_{CATEGORY}_{num_train_samples}"
        dataset_path = os.path.join(OUTPUT_LOCATION, dataset_dir)
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        os.makedirs(dataset_path)

        # Copy all of the mvtec folder except the "train" folder
        for item in os.listdir(category_path):
            s = os.path.join(category_path, item)
            d = os.path.join(dataset_path, item)
            if os.path.isdir(s) and item != "train":
                shutil.copytree(s, d)
            elif os.path.isfile(s):
                shutil.copy2(s, d)

        # Create a new train folder with only a subset of the training images
        train_dest = os.path.join(dataset_path, "train", "good")
        os.makedirs(train_dest)
        for img in train_images[:num_train_samples]:
            s = os.path.join(train_source, img)
            d = os.path.join(train_dest, img)
            shutil.copy2(s, d)

        # Move the 'good' folder outside the 'test' folder and rename it to 'test_good'
        test_good_source = os.path.join(dataset_path, "test", "good")
        test_good_dest = os.path.join(dataset_path, "test_good")
        if os.path.exists(test_good_source):
            shutil.move(test_good_source, test_good_dest)

        # Remove the now-empty 'test' folder if it exists
        test_folder = os.path.join(dataset_path, "test")
        if os.path.exists(test_folder) and not os.listdir(test_folder):
            os.rmdir(test_folder)

        print(f"Created dataset at: {dataset_path}")


if __name__ == "__main__":
    main()
