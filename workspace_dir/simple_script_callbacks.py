"""Simple Script.

Just a simple script to run the training and testing of a
model for debugging.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from anomalib.callbacks import ModelCheckpoint
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Fre as ModelClass

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MAX_EPOCHS = 15
SAVE_CHECKPOINTS = True

# Initialize components
datamodule = MVTecAD(category="toothbrush", num_workers=0)
model = ModelClass()
if not SAVE_CHECKPOINTS:
    # Disable saving checkpoints completely
    checkpoint_cb = ModelCheckpoint(
        save_top_k=0,
    )
    engine = Engine(max_epochs=MAX_EPOCHS, callbacks=[checkpoint_cb])
else:
    engine = Engine(max_epochs=MAX_EPOCHS)


tic = time.time()
# Train the model
engine.fit(datamodule=datamodule, model=model)
toc = time.time()
print(f"Save Checkpoints: {SAVE_CHECKPOINTS}, Training Time for {MAX_EPOCHS} epochs: {toc - tic} seconds")
engine.test(datamodule=datamodule, model=model)
