import argparse
import time

import torch


def run(device: str):
    if device == "xpu" and torch.xpu.is_available():
        dev = torch.device("xpu")
    else:
        dev = torch.device("cpu")

    print(f"Running on: {dev}")

    # Create random tensors
    size = 4096
    a = torch.randn(size, size, device=dev)
    b = torch.randn(size, size, device=dev)

    # Warm-up (first run can be slower)
    c = a @ b

    torch.xpu.synchronize() if dev.type == "xpu" else None

    # Measure
    start = time.time()
    for _ in range(10):
        c = a @ b
    torch.xpu.synchronize() if dev.type == "xpu" else None
    end = time.time()

    print(f"Time taken on {dev}: {end - start:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "xpu"], default="cpu")
    args = parser.parse_args()

    run(args.device)
