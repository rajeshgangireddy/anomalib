from anomalib.data import MVTecAD
from anomalib.models import Padim
from anomalib.pipelines.benchmark.job import BenchmarkJob

# Initialize model, datamodule and job
model = Padim()
datamodule = MVTecAD(category="carpet")
job = BenchmarkJob(
    accelerator="gpu",
    model=model,
    datamodule=datamodule,
    seed=42,
    flat_cfg={"model.name": "padim"},
)

results = job.run()
