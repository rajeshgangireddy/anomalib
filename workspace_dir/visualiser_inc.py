from anomalib.visualization import ImageVisualizer
import psutil
import os
from anomalib.data.utils import read_image
from anomalib.deploy import TorchInferencer
from pathlib import Path

# has to be adapted
result_path = 'path to your model'
img_path = 'path to a image'
file_path = Path('.')

inferencer = TorchInferencer(
    path=result_path + '/weights/torch/model.pt',
    #device="CPU",  # We would like to run it on an Intel CPU.
)

image = read_image(img_path)
predictions = inferencer.predict(image)

process = psutil.Process(os.getpid())

visualizer = ImageVisualizer()
for i in range(0,2000):
    output_image = visualizer.visualize_image(predictions)
    visualizer.save(file_path / 'test.png', output_image)
    memory_usage = process.memory_info()
    if i % 100 == 0:
        print(f"Loop {i}: Process memory used: {memory_usage.rss / (1024 * 1024):.2f} MB")