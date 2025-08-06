from anomalib.data import MVTecAD
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import Patchcore

# Initialize components
datamodule = MVTecAD()
model = Patchcore()
engine = Engine(
        strategy=SingleXPUStrategy(),
        accelerator=XPUAccelerator(),
        )

# Train the model
engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
