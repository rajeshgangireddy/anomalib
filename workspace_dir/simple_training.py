import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import Patchcore

cuda_available = torch.cuda.is_available()
# Initialize components
datamodule = MVTecAD(category="toothbrush")

model = Patchcore()
if not cuda_available:
    # If CUDA is not available, use XPU strategy
    print("CUDA is not available. Using XPU strategy.")
    print(f"XPU Avialable: {torch.xpu.is_available()}")

    engine = Engine(
        strategy=SingleXPUStrategy(),
        accelerator=XPUAccelerator(),
        )

else:
    print("CUDA is available. Using default strategy.")
    # If CUDA is available, use the default strategy
    engine = Engine()
# Train the model
engine.fit(datamodule=datamodule, model=model)
results = engine.test(datamodule=datamodule, model=model)
