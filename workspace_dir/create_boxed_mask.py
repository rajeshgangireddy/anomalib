# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Script for creating synthetic anomalous images with boxed masks."""

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# Create a random number generator for reproducible results
rng = np.random.default_rng(42)

path_to_dataset = Path("./datasets/Custom/toothbrush")
output_dir = Path("./datasets/Custom/toothbrush-BoxedMasks")

normal_images_dir_train = path_to_dataset / "train/good"
normal_images_dir_test = path_to_dataset / "test/good"
abnormal_images_dir = path_to_dataset / "test/defective"
mask_dir = path_to_dataset / "ground_truth/defective"

anomalous_shape = (100, 160)  # Square anomaly size (width x height)


def create_synthetic_dataset() -> None:
    """Create a new dataset with synthetic anomalous images using Perlin noise."""
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train/good").mkdir(parents=True, exist_ok=True)
    (output_dir / "test/good").mkdir(parents=True, exist_ok=True)
    (output_dir / "test/defective").mkdir(parents=True, exist_ok=True)
    (output_dir / "ground_truth/defective").mkdir(parents=True, exist_ok=True)

    print(f"Creating synthetic dataset at: {output_dir}")

    # Copy all normal images (training and test)
    print("Copying normal training images...")
    copy_normal_images(normal_images_dir_train, output_dir / "train/good")

    print("Copying normal test images...")
    copy_normal_images(normal_images_dir_test, output_dir / "test/good")

    # Get the count of original defective images
    original_defective_files = list(abnormal_images_dir.glob("*.png"))
    num_anomalous_images = len(original_defective_files)

    print(f"Found {num_anomalous_images} original defective images")
    print("Creating synthetic anomalous images...")

    # Create synthetic anomalous images
    create_synthetic_anomalous_images(
        normal_images_dir_test,
        output_dir / "test/defective",
        output_dir / "ground_truth/defective",
        num_anomalous_images,
    )

    # Copy license and readme files if they exist
    copy_metadata_files()

    print(f"✅ Synthetic dataset created successfully at: {output_dir}")


def copy_normal_images(source_dir: Path, target_dir: Path) -> None:
    """Copy normal images from source to target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)

    for img_file in source_dir.glob("*.png"):
        shutil.copy2(img_file, target_dir / img_file.name)

    print(f"  Copied {len(list(source_dir.glob('*.png')))} images to {target_dir}")


def create_synthetic_anomalous_images(
    source_normal_dir: Path,
    target_defective_dir: Path,
    target_mask_dir: Path,
    num_images: int,
) -> None:
    """Create synthetic anomalous images using Perlin noise."""
    target_defective_dir.mkdir(parents=True, exist_ok=True)
    target_mask_dir.mkdir(parents=True, exist_ok=True)

    # Get list of normal images to use as base
    normal_images = list(source_normal_dir.glob("*.png"))

    for i in range(num_images):
        # Use modulo to cycle through normal images if we need more anomalous images than normal ones
        base_image_path = normal_images[i % len(normal_images)]

        # Load the base normal image
        base_image = Image.open(base_image_path)
        base_array = np.array(base_image)
        height, width = base_array.shape[:2]

        # Create binary mask with configurable anomalous shape
        # Create a square anomaly in the center region
        center_y, center_x = height // 2, width // 2
        anomaly_height, anomaly_width = anomalous_shape

        # Add some randomness to the position
        offset_y = rng.integers(-height // 6, height // 6)
        offset_x = rng.integers(-width // 6, width // 6)

        start_y = max(0, center_y - anomaly_height // 2 + offset_y)
        end_y = min(height, center_y + anomaly_height // 2 + offset_y)
        start_x = max(0, center_x - anomaly_width // 2 + offset_x)
        end_x = min(width, center_x + anomaly_width // 2 + offset_x)

        # Create anomaly mask (binary)
        anomaly_mask = np.zeros((height, width), dtype=np.uint8)

        # Create a simple square anomaly (solid square)
        anomaly_mask[start_y:end_y, start_x:end_x] = 255

        # Create anomalous image by blending with noise
        anomalous_image = base_array.copy()

        # Add some color distortion where the mask is active
        mask_bool = anomaly_mask > 0
        if len(base_array.shape) == 3:  # RGB image
            # Add reddish tint to anomalous regions
            anomalous_image[mask_bool, 0] = np.clip(anomalous_image[mask_bool, 0] * 1.3, 0, 255)  # Red channel
            anomalous_image[mask_bool, 1] = np.clip(anomalous_image[mask_bool, 1] * 0.7, 0, 255)  # Green channel
            anomalous_image[mask_bool, 2] = np.clip(anomalous_image[mask_bool, 2] * 0.7, 0, 255)  # Blue channel

        # Add some noise to make it more realistic
        noise_strength = 0.1
        noise = rng.normal(0, noise_strength * 255, anomalous_image.shape)
        anomalous_image = anomalous_image.astype(np.float32) + noise * mask_bool[..., np.newaxis]
        anomalous_image = np.clip(anomalous_image, 0, 255).astype(np.uint8)

        # Save anomalous image
        anomalous_img_filename = f"{i:03d}.png"
        anomalous_img_path = target_defective_dir / anomalous_img_filename
        Image.fromarray(anomalous_image).save(anomalous_img_path)

        # Save corresponding mask
        mask_filename = f"{i:03d}_mask.png"
        mask_path = target_mask_dir / mask_filename
        Image.fromarray(anomaly_mask).save(mask_path)

        if (i + 1) % 10 == 0:
            print(f"  Created {i + 1}/{num_images} synthetic anomalous images")

    print(f"  ✅ Created {num_images} synthetic anomalous images and masks")


def copy_metadata_files() -> None:
    """Copy license and readme files to the new dataset."""
    for filename in ["license.txt", "readme.txt"]:
        source_file = path_to_dataset / filename
        if source_file.exists():
            shutil.copy2(source_file, output_dir / filename)
            print(f"  Copied {filename}")


if __name__ == "__main__":
    create_synthetic_dataset()
