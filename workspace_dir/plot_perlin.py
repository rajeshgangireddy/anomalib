"""Perline noise visualization script."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from anomalib.data.utils.generators.perlin import generate_perlin_noise

if os.environ.get("DISPLAY") is None:
    # If no display is available (like in SSH or headless environments)
    matplotlib.use("Agg")  # Use non-interactive backend
    SAVE_PLOTS = True
else:
    matplotlib.use("TkAgg")  # Use interactive backend
    SAVE_PLOTS = False

plt.style.use("default")  # Ensure consistent styling


def plot_perlin_noise_scales() -> None:
    """Plot Perlin noise patterns at various scales to visualize the differences."""
    # Set up the plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Perlin Noise at Different Scales", fontsize=16)

    # Define different scales to test
    scales = [
        (1, 1),  # Very fine noise
        (2, 2),  # Fine noise
        (4, 4),  # Medium-fine noise
        (8, 8),  # Medium noise
        (16, 16),  # Medium-coarse noise
        (32, 32),  # Coarse noise
        (64, 64),  # Very coarse noise
        None,  # Random scale (default behavior)
    ]

    # Additional test cases
    scales.extend([
        (4, 16),  # Different X and Y scales
        (16, 4),  # Different X and Y scales (reversed)
        (2, 32),  # Very different X and Y scales
        (32, 2),  # Very different X and Y scales (reversed)
    ])

    height, width = 256, 256
    device = torch.device("cpu")  # Use CPU for visualization

    for idx, scale in enumerate(scales):
        row = idx // 4
        col = idx % 4

        # Generate Perlin noise
        if scale is None:
            noise = generate_perlin_noise(height, width, device=device)
            title = "Random Scale"
        else:
            noise = generate_perlin_noise(height, width, scale=scale, device=device)
            title = f"Scale: {scale}"

        # Convert to numpy for plotting
        noise_np = noise.cpu().numpy()

        # Plot the noise
        im = axes[row, col].imshow(noise_np, cmap="gray", vmin=-1, vmax=1)
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis("off")

        # Add colorbar to the first plot as reference
        if idx == 0:
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(
            "/home/devuser/workspace/code/Anomalib/dev/anomalib/workspace_dir/perlin_scales.png",
            dpi=150,
            bbox_inches="tight",
        )
        print("Saved plot: perlin_scales.png")
    plt.show()
    plt.close()


def plot_perlin_noise_comparison() -> None:
    """Compare Perlin noise with different properties side by side."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Perlin Noise Properties Comparison", fontsize=16)

    height, width = 256, 256
    device = torch.device("cpu")

    # Row 1: Different scales
    scales = [(2, 2), (8, 8), (32, 32)]
    for i, scale in enumerate(scales):
        noise = generate_perlin_noise(height, width, scale=scale, device=device)
        noise_np = noise.cpu().numpy()

        im = axes[0, i].imshow(noise_np, cmap="viridis", vmin=-1, vmax=1)
        axes[0, i].set_title(f"Scale: {scale}", fontsize=12)
        axes[0, i].axis("off")

        if i == 2:  # Add colorbar to last plot
            plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

    # Row 2: Different sizes with same scale
    sizes = [(128, 128), (256, 256), (512, 512)]
    scale = (8, 8)

    for i, size in enumerate(sizes):
        h, w = size
        noise = generate_perlin_noise(h, w, scale=scale, device=device)
        noise_np = noise.cpu().numpy()

        im = axes[1, i].imshow(noise_np, cmap="plasma", vmin=-1, vmax=1)
        axes[1, i].set_title(f"Size: {size}, Scale: {scale}", fontsize=12)
        axes[1, i].axis("off")

        if i == 2:  # Add colorbar to last plot
            plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(
            "/home/devuser/workspace/code/Anomalib/dev/anomalib/workspace_dir/perlin_comparison.png",
            dpi=150,
            bbox_inches="tight",
        )
        print("Saved plot: perlin_comparison.png")
    plt.show()
    plt.close()


def plot_perlin_statistics() -> None:
    """Plot statistics and histograms of Perlin noise at different scales."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Perlin Noise Statistics", fontsize=16)

    scales = [(2, 2), (8, 8), (16, 16), (32, 32)]
    colors = ["blue", "red", "green", "orange"]
    height, width = 256, 256
    device = torch.device("cpu")

    # Collect statistics
    stats = []
    all_noise = []

    for scale in scales:
        noise = generate_perlin_noise(height, width, scale=scale, device=device)
        noise_np = noise.cpu().numpy()
        all_noise.append(noise_np)

        stats.append({
            "scale": scale,
            "mean": float(noise.mean()),
            "std": float(noise.std()),
            "min": float(noise.min()),
            "max": float(noise.max()),
        })

    # Plot 1: Sample noise patterns (small versions)
    for i, (noise_np, _scale) in enumerate(zip(all_noise, scales, strict=False)):
        axes[0, 0].imshow(noise_np[60:100, 60:100], cmap="gray", extent=[i * 50, (i + 1) * 50 - 10, 0, 40])
    axes[0, 0].set_title("Sample Patches (40x40 pixels)")
    axes[0, 0].set_xlabel("Scale →")
    axes[0, 0].set_xticks([25, 75, 125, 175])
    axes[0, 0].set_xticklabels([str(s) for s in scales])

    # Plot 2: Histograms
    for _, (noise_np, scale, color) in enumerate(zip(all_noise, scales, colors, strict=False)):
        axes[0, 1].hist(noise_np.flatten(), bins=50, alpha=0.7, color=color, label=f"Scale {scale}", density=True)
    axes[0, 1].set_title("Value Distributions")
    axes[0, 1].set_xlabel("Noise Value")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Statistics comparison
    metrics = ["mean", "std", "min", "max"]
    x_pos = np.arange(len(scales))
    width = 0.2

    for i, metric in enumerate(metrics):
        values = [stat[metric] for stat in stats]
        axes[1, 0].bar(x_pos + i * width, values, width, label=metric.upper(), alpha=0.8)

    axes[1, 0].set_title("Statistical Measures")
    axes[1, 0].set_xlabel("Scale")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_xticks(x_pos + width * 1.5)
    axes[1, 0].set_xticklabels([str(s) for s in scales])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Power spectrum (frequency analysis)
    for noise_np, scale, color in zip(all_noise, scales, colors, strict=False):
        # Compute 2D FFT and power spectrum
        fft_2d = np.fft.fft2(noise_np)
        power_spectrum = np.abs(fft_2d) ** 2

        # Compute radial average
        h, w = power_spectrum.shape
        center = (h // 2, w // 2)
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        r = r.astype(int)

        # Bin the power spectrum by radius
        tbin = np.bincount(r.ravel(), power_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr

        # Plot only the meaningful part
        freq_range = min(50, len(radialprofile))
        axes[1, 1].loglog(
            range(1, freq_range),
            radialprofile[1:freq_range],
            color=color,
            label=f"Scale {scale}",
            alpha=0.8,
        )

    axes[1, 1].set_title("Power Spectrum (Radial Average)")
    axes[1, 1].set_xlabel("Spatial Frequency")
    axes[1, 1].set_ylabel("Power")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(
            "/home/devuser/workspace/code/Anomalib/dev/anomalib/workspace_dir/perlin_statistics.png",
            dpi=150,
            bbox_inches="tight",
        )
        print("Saved plot: perlin_statistics.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("Generating Perlin noise visualizations...")

    # Plot 1: Various scales
    plot_perlin_noise_scales()

    # Plot 2: Comparison
    plot_perlin_noise_comparison()

    # Plot 3: Statistics
    plot_perlin_statistics()

    print("Visualization complete!")
