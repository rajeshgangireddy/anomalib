"""Simple Script.

Just a simple script to run the training and testing of a
model for debugging.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Cflow as ModelClass
from anomalib.models import Dinomaly as ModelClass

# Initialize components
datamodule = MVTecAD(num_workers=0, category="carpet")
model = ModelClass()
# engine = Engine(DDPStrategy(find_unused_parameters=True))
engine = Engine(accelerator="gpu", devices=1, max_steps=5000)
# engine = Engine(max_epochs=3)

tic = time.time()
# Train the model
engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
# engine.export(model=engine.model, export_type=ExportType.TORCH)
# engine.export(model=engine.model, export_type=ExportType.OPENVINO)

toc = time.time()
print(f"Elapsed {toc} seconds")
