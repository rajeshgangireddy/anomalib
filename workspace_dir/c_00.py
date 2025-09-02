"""Script for testing anomaly detection models with Patchcore and OpenVINO inference."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
from anomalib.deploy import ExportType, OpenVINOInferencer
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import Patchcore

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(description="Anomalib Test")
parser.add_argument("--train", action="store_true", help="Enable training of model")
parser.add_argument("--score", action="store_true", help="Enable F1 scoring of test data")

dataset_folder = "/home/ubuntu/workspace/datasets/IndustryBiscuitLotus"
args = parser.parse_args()
enable_training = args.train
enable_scores = args.score

# Initialize components
datamodule = Folder(
    name="IndustryBiscuitLotus",  # A unique name for your dataset
    root=dataset_folder,  # Root directory containing 'normal' and 'abnormal' folders
    normal_dir="train/good",  # Name of the folder containing normal images
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=8,
    normal_split_ratio=0.8,
    test_split_mode=TestSplitMode.SYNTHETIC,
)
datamodule.setup()

model = Patchcore()
engine = Engine(
    strategy=SingleXPUStrategy(),
    accelerator=XPUAccelerator(),
)

if enable_training:
    # Train the model
    engine.fit(datamodule=datamodule, model=model)

    # Test the model
    test_results = engine.test(model=model, datamodule=datamodule)

    # Export the model in OpenVINO format
    openvino_model_path = engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        export_root="patch_core_optimized_biscuit_lotus",
    )

inferencer = OpenVINOInferencer(
    path="./patch_core_optimized_biscuit_lotus/weights/openvino/model.xml",  # Path to the OpenVINO IR model.
    device="GPU",
)

folder_path = Path("./datasets/schneider/IndustryBiscuitLotus/test/mix")

y_true = []
y_pred = []
infer_time = []

for entry_path in folder_path.iterdir():
    if entry_path.is_file():
        t1 = time.time()
        predictions = inferencer.predict(
            image=str(entry_path),
        )
        t2 = time.time()
        infer_time_ms = (t2 - t1) * 1000
        infer_time.append(infer_time_ms)

        # Set ground truth
        if entry_path.name.startswith("NOK"):
            y_true.append(1)
        else:
            y_true.append(0)

        # Access the results
        if predictions is not None:
            for prediction in predictions:
                image_path = prediction.image_path
                anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
                pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
                pred_score = prediction.pred_score  # Image-level anomaly score

                y_pred.append(1 if pred_label else 0)

                print(f"{entry_path} -> {pred_label}/{pred_score:.2f}/{infer_time_ms:.2f} ms")

if enable_scores:
    print(f"Target predictions: {y_true}")
    print(f"Actual predictions: {y_pred}")
    print("-----------------")
    print(f"f1_score is: {f1_score(y_true, y_pred):.3f}")
    print(f"accuracy score is: {accuracy_score(y_true, y_pred):.3f}")

avg_infer_time = sum(infer_time) / len(infer_time)
print(f"Inference time average: {avg_infer_time:.2f} ms, min: {min(infer_time):.2f} ms, max: {max(infer_time):.2f} ms")
