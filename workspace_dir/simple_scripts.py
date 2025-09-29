"""Simple Script.

Just a simple script to run the training and testing of a
model for debugging.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import torch


# Force deterministic behavior on macOS
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim 



# Initialize components
datamodule = MVTecAD(category="toothbrush",num_workers=0)
model = Padim() 
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
# engine.export(model=engine.model, export_type="torch")
