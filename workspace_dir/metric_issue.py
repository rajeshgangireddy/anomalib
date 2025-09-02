"""Script for testing anomaly detection metrics."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, F1Score
from anomalib.metrics.evaluator import Evaluator
from anomalib.models import Patchcore


@dataclass
class Batch:
    """Batch data structure for predictions and targets."""

    preds: torch.Tensor
    target: torch.Tensor


labels = torch.randint(0, 2, (10,)).long()
scores = torch.rand_like(labels, dtype=torch.float)

batch = Batch(preds=scores, target=labels)
metric = AUROC(fields=["preds", "target"])
auroc_score = metric(batch)
print(auroc_score)

image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)
pixel_aupro = AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)

datamodule = MVTecAD(num_workers=0, category="toothbrush")

test_metrics = [image_auroc, image_f1score, pixel_auroc, pixel_aupro]
evaluator = Evaluator(test_metrics=test_metrics)


model = Patchcore(
    evaluator=evaluator,
)

engine = Engine()

engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
