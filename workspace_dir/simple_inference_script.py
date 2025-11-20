# simple script to load and predict an image


import os

from anomalib.deploy import TorchInferencer
from anomalib.models import Dinomaly as ModelClass

os.environ["TRUST_REMOTE_CODE"] = "1"

example_image = "datasets/MVTecAD/bottle/train/good/000.png"  # Good image
model_ckpt = "results/Dinomaly/MVTecAD/bottle/v7/weights/torch/Dinomaly_bottle_50_epochs.pt"

model = ModelClass()
inferencer = TorchInferencer(path=model_ckpt)
predictions = inferencer.predict(image=example_image)

print(f"Predictions: {predictions}")
