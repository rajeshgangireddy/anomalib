import argparse
import logging
import os
import pathlib
import time

from sklearn.metrics import accuracy_score, f1_score

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
from anomalib.deploy import ExportType, OpenVINOInferencer
from anomalib.engine import Engine
from anomalib.models import Patchcore

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(description='Anomalib Test')
parser.add_argument('--train', action="store_true",
                    help='Enable training of model')
parser.add_argument('--score', action="store_true",
                    help='Enable F1 scoring of test data')
parser.add_argument('--dataset_path', type=str,
                    default="/home/devuser/workspace/datasets/customer_scheider/datasets/schneider/IndustryBiscuitLotus")

args = parser.parse_args()
enable_training = args.train
enable_scores = args.score

dataset_path = "/home/devuser/workspace/datasets/customer_scheider/datasets/schneider/IndustryBiscuitLotus"
# Initialize components
datamodule = Folder(
    name="IndustryBiscuitLotus",  # A unique name for your dataset
    root=dataset_path,  # Root directory containing 'normal' and 'abnormal' folders
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
    # strategy=SingleXPUStrategy(),
    # accelerator=XPUAccelerator(),
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

# folder_path = "/home/ubuntu/anomalib/datasets/schneider/IndustryBiscuitLotus/test/NOK"
# folder_path = "/home/ubuntu/anomalib/datasets/schneider/IndustryBiscuitLotus/train/good"
folder_path = "./datasets/schneider/IndustryBiscuitLotus/test/mix"

y_true = []
y_pred = []
infer_time = []

for entry_name in os.listdir(folder_path):
    full_path = os.path.join(folder_path, entry_name)
    if pathlib.Path(full_path).is_file():
        t1 = time.time()
        predictions = inferencer.predict(
            image=full_path,
        )
        t2 = time.time()
        infer_time_ms = (t2 - t1) * 1000
        infer_time.append(infer_time_ms)

        # Set ground truth
        if entry_name.startswith("NOK"):
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

                y_pred.append(1 if pred_label == True else 0)

                print(f"{full_path} -> {pred_label}/{pred_score:.2f}/{infer_time_ms:.2f} ms")

if enable_scores:
    print(f"Target predictions: {y_true}")
    print(f"Actual predictions: {y_pred}")
    print("-----------------")
    print(f"f1_score is: {f1_score(y_true, y_pred):.3f}")
    print(f"accuracy score is: {accuracy_score(y_true, y_pred):.3f}")

avg_infer_time = sum(infer_time)/len(infer_time)
print(f"Inference time average: {avg_infer_time:.2f} ms, min: {min(infer_time):.2f} ms, max: {max(infer_time):.2f} ms")
