"""Simple Script.

Just a simple script to run the training and testing of a
model for debugging.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# Initialize components
datamodule = MVTecAD(num_workers=0, category="toothbrush")
model = Patchcore()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
engine.export(model=engine.model, export_type="openvino")
