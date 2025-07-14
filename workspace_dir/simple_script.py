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
from anomalib.deploy import ExportType

# Initialize components
datamodule = MVTecAD(num_workers=0, category="carpet")
model = ModelClass()
# engine = Engine(DDPStrategy(find_unused_parameters=True))
# engine = Engine(accelerator="gpu", devices=1, max_steps=50)
engine = Engine(accelerator="gpu", devices=1, max_steps=50)
# engine = Engine(max_epochs=3)

tic = time.time()
# Train the model
engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
# engine.export(model=engine.model, export_type=ExportType.TORCH)
# opset_to_try = [21, 22, 23, 24, 25]
# supported_opsets = []
# for opset in opset_to_try:
#     try:
#         print(f"Exporting model to ONNX with opset {opset}")
#         engine.export(model=engine.model, export_type=ExportType.OPENVINO, onnx_kwargs={"opset_version": opset})
#         supported_opsets.append(opset)
#     except Exception as e:
#         print(f"Failed to export with opset {opset}: {e}")
#
# print(f"Supported opsets: {supported_opsets}")
# engine.export(model=engine.model, export_type=ExportType.ONNX, onnx_kwargs={"opset_version": 20})

toc = time.time() - tic
print(f"Elapsed {toc} seconds")
