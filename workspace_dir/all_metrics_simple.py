# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Simple metrics testing script for Anomalib.

This script demonstrates how to use multiple metrics with Anomalib models.
It sets up comprehensive evaluation metrics including AUROC, F1Score, and AUPRO
for both image-level and pixel-level anomaly detection.
"""

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, Evaluator, F1Score
from anomalib.models import Padim

# Test metrics (more comprehensive)
test_metrics = [
    AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
    AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
    F1Score(fields=["pred_label", "gt_label"], prefix="image_", strict=False),
    F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
    AUPRO(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
]

# Create evaluator with both sets
evaluator = Evaluator(
    test_metrics=test_metrics,
)

# Use with model
model = Padim(evaluator=evaluator)

datamodule = MVTecAD(category="toothbrush")

engine = Engine()
engine.fit(model=model, datamodule=datamodule)
results = engine.test(model, datamodule=datamodule)
print(results)
