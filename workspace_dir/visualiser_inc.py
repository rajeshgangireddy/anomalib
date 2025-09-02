# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Visualizer script for anomaly detection inference."""

import os
from pathlib import Path

import psutil

from anomalib.data.utils import read_image
from anomalib.deploy import TorchInferencer
from anomalib.visualization import visualize_anomaly_map

os.environ["TRUST_REMOTE_CODE"] = "1"  # Required for TorchInferencer to work with custom models
# has to be adapted
result_path = (
    "/home/rgangire/workspace/code/repos/Geti-Labs/Anomalib/repo/anomalib/"
    "workspace_dir/results/Patchcore/MVTecAD/bottle/v7"
)
dataset_path = (
    "/home/rgangire/workspace/code/repos/Geti-Labs/Anomalib/repo/anomalib/"
    "workspace_dir/datasets/MVTecAD/bottle/test/broken_small"
)
img_path = Path(dataset_path) / "000.png"
file_path = Path()

inferencer = TorchInferencer(
    path=result_path + "/weights/torch/model.pt",
)

image = read_image(img_path)
predictions = inferencer.predict(image)

process = psutil.Process(os.getpid())


for i in range(2000):
    output_image = visualize_anomaly_map(predictions.anomaly_map)
    output_image.save(file_path / f"output_{i:04d}.png")
    memory_usage = process.memory_info()
    if i % 100 == 0:
        print(f"Loop {i}: Process memory used: {memory_usage.rss / (1024 * 1024):.2f} MB")
