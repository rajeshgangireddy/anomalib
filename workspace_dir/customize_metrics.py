"""Testing the AnomalibMetric.

Code fragments to check AnomalibMetrics.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from torchmetrics import Accuracy
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification.accuracy import BinaryAccuracy
from torchmetrics.classification.auroc import AUROC

from anomalib.metrics import AnomalibMetric


class AnomalibAccuracy(AnomalibMetric, Accuracy):
    """Wrapper to add AnomalibMetric functionality to Accuracy metric."""


accuracy = AnomalibAccuracy(task="binary", fields=["pred_label", "gt_label"])


class F1Score(AnomalibMetric, BinaryF1Score):
    """Wrapper to add AnomalibMetric functionality to F1Score metric."""


image_f1 = F1Score(
    fields=["pred_label", "gt_label"],
    prefix="image_",
)


class AnomalibAuroc(AnomalibMetric, AUROC):
    """Wrapper to add AnomalibMetric functionality to AUROC metric."""


auroc = AnomalibAuroc(
    task="binary",
    fields=["pred_label", "gt_label"],
    prefix="image_",
)


class AnomalibBinaryAccuracy(AnomalibMetric, BinaryAccuracy):
    """Wrapper to add AnomalibMetric functionality to BinaryAccuracy metric."""


binary_accuracy = AnomalibBinaryAccuracy(
    fields=["pred_label", "gt_label"],
)
