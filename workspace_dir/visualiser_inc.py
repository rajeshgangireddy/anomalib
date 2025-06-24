
import psutil
import os
from anomalib.data.utils import read_image
from anomalib.deploy import TorchInferencer
from pathlib import Path
from anomalib.visualization import visualize_anomaly_map

os.environ["TRUST_REMOTE_CODE"] = "1"  # Required for TorchInferencer to work with custom models
# has to be adapted
result_path = "/home/rgangire/workspace/code/repos/Geti-Labs/Anomalib/repo/anomalib/workspace_dir/results/Patchcore/MVTecAD/bottle/v7"
dataset_path = "/home/rgangire/workspace/code/repos/Geti-Labs/Anomalib/repo/anomalib/workspace_dir/datasets/MVTecAD/bottle/test/broken_small"
img_path = os.path.join(dataset_path, '000.png')
file_path = Path('.')

inferencer = TorchInferencer(
    path=result_path + '/weights/torch/model.pt',
    #device="CPU",  # We would like to run it on an Intel CPU.
)

image = read_image(img_path)
predictions = inferencer.predict(image)

process = psutil.Process(os.getpid())


for i in range(0,2000):
    output_image = visualize_anomaly_map(predictions.anomaly_map)
    output_image.save(file_path / f"output_{i:04d}.png")
    memory_usage = process.memory_info()
    if i % 100 == 0:
        print(f"Loop {i}: Process memory used: {memory_usage.rss / (1024 * 1024):.2f} MB")