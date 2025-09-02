"""Script for testing device performance with PyTorch tensors."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import time

import torch


def run(device: str) -> None:
    """Run performance test on specified device.

    Args:
        device: Device type ('xpu' or 'cpu').
    """
    dev = torch.device("xpu") if device == "xpu" and torch.xpu.is_available() else torch.device("cpu")

    print(f"Running on: {dev}")

    # Create random tensors
    size = 4096
    a = torch.randn(size, size, device=dev)
    b = torch.randn(size, size, device=dev)

    # Warm-up (first run can be slower)
    _ = a @ b

    torch.xpu.synchronize() if dev.type == "xpu" else None

    # Measure
    start = time.time()
    for _ in range(10):
        _ = a @ b
    torch.xpu.synchronize() if dev.type == "xpu" else None
    end = time.time()

    print(f"Time taken on {dev}: {end - start:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "xpu"], default="cpu")
    args = parser.parse_args()

    run(args.device)
