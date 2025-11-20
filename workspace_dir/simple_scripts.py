"""Simple Script.

Just a simple script to run the training and testing of a
model for debugging.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Dinomaly as ModelClass

# Initialize components
MAX_EPOCHS = 50
datamodule = MVTecAD(category="bottle", num_workers=0)
model = ModelClass()
engine = Engine(max_epochs=50)

# Train the model
engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
filename = f"{ModelClass.__name__}_bottle_{MAX_EPOCHS}_epochs"
engine.export(model=model, export_type="torch", model_file_name=filename)
