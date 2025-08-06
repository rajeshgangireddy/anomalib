from anomalib.data import MVTecAD
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import PatchCore

# Initialize components
datamodule = MVTecAD()
model = PatchCore()
engine = Engine(
        strategy=SingleXPUStrategy(),
        accelerator=XPUAccelerator(),
        )

# Train the model
engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
