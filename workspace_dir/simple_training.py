from anomalib.data import MVTecAD
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import Patchcore

import torch

cuda_available = torch.cuda.is_available()
# Initialize components
datamodule = MVTecAD(category="toothbrush")

model = Patchcore()
if not cuda_available:
    # If CUDA is not available, use XPU strategy
    engine = Engine(
        strategy=SingleXPUStrategy(),
        accelerator=XPUAccelerator(),
        )
else:
    # If CUDA is available, use the default strategy
    engine = Engine()
# Train the model
engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
