"""Script for training EfficientAD model on MVTec LOCO dataset."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.data import MVTecLOCO
from anomalib.engine import Engine
from anomalib.models import EfficientAd

datamodule = MVTecLOCO(train_batch_size=1)
model = EfficientAd()

engine = Engine(max_epochs=200)

engine.fit(datamodule=datamodule, model=model)
