"""Simple Script.

Just a simple script to run the training and testing of a
model for debugging.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# Initialize components
datamodule = MVTecAD()
model = Patchcore()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
engine.export(model=engine.model, export_type="torch")

# run predict on all images from the folder
path_to_test_folder = Path("./datasets/MVTecAD/bottle/test/good")
print(f"Predicting on images in: {path_to_test_folder}")
# Ensure the path exists
if not path_to_test_folder.exists():
    print(f"Path {path_to_test_folder} does not exist.")

# iterate and call engine.predict on each image
for image_path in path_to_test_folder.glob("*.png"):
    tic = time.time()
    results = engine.predict(model=model, data_path=image_path)
    toc = time.time()
    print(f"Predicted in {toc - tic:.2f} seconds")
