import argparse
import time

import torch
from lightning import seed_everything

from anomalib.data import MVTecAD
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import get_model


def main():
    seed_everything(42, workers=True)
    parser = argparse.ArgumentParser(description="Train and test an Anomalib model.")
    parser.add_argument('--model', type=str, required=True, help='Model name to use (e.g., patchcore, padim, etc.)')
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    datamodule = MVTecAD(category="toothbrush")

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
    results = engine.test(datamodule=datamodule, model=model)
    testing_time = time.time() - tic
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Testing time: {testing_time:.2f} seconds")


if __name__ == "__main__":
    main()
