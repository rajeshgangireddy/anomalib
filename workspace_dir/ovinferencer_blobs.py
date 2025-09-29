"""Simple script to test OpenVINO inferencer blob caching."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.data import MVTecAD
from anomalib.deploy import ExportType, OpenVINOInferencer
from anomalib.engine import Engine
from anomalib.models import Padim

datamodule = MVTecAD(category="toothbrush", num_workers=0)
model = Padim()
engine = Engine()
engine.fit(datamodule=datamodule, model=model)
ov_path = engine.export(
    model=model,
    export_type=ExportType.OPENVINO,
)

# Inference
for _ in range(10):  # for example, this creates 10 cache folders
    inf = OpenVINOInferencer(ov_path, device="CPU")
