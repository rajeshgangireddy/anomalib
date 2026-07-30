# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Measure the blob-area distribution of generated masks.

P4 (hybrid) only differs from P3 (poisson) for components below ``area_threshold``.
This quantifies how many components would actually be re-routed, per dataset.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from anomalib.data import MVTecAD, Visa
from anomalib.data.utils.generators import SyntheticAnomalyGenerator

THRESHOLD = 2500
SAMPLES = 40
torch.manual_seed(0)

DATASETS = {
    "mvtec": (MVTecAD, "./datasets/MVTecAD", ["bottle", "hazelnut", "screw", "carpet"]),
    "visa": (Visa, "./datasets/visa", ["candle", "macaroni1", "pcb1", "cashew"]),
}

gen = SyntheticAnomalyGenerator.from_preset("self_hybrid", probability=1.0)

print(f"threshold = {THRESHOLD} px | components below it get alpha instead of Poisson\n")
header = f"{'dataset':8s} {'category':10s} {'blobs':>6s} {'median':>7s} {'p90':>6s} {'%n<thr':>7s} {'%AREA<thr':>9s}"
print(header)
for name, (cls, root, cats) in DATASETS.items():
    totals = []
    for cat in cats:
        dm = cls(root=root, category=cat, train_batch_size=1, eval_batch_size=1, num_workers=0)
        dm.prepare_data()
        dm.setup()
        images = [dm.train_data[i].image for i in range(min(SAMPLES, len(dm.train_data)))]
        areas, covs = [], []
        for img in images:
            _, mask = gen(img)
            binary = (mask.squeeze(0).numpy() > 0).astype(np.uint8)
            covs.append(float(binary.mean()))
            count, labels = cv2.connectedComponents(binary)
            sizes = np.bincount(labels.ravel())[1:]
            areas += sizes.tolist()
        if not areas:
            continue
        arr = np.array(areas)
        totals += areas
        pct_n = 100.0 * (arr < THRESHOLD).mean()
        pct_area = 100.0 * arr[arr < THRESHOLD].sum() / arr.sum()
        print(
            f"{name:8s} {cat:10s} {len(arr):6d} {int(np.median(arr)):7d} "
            f"{int(np.percentile(arr, 90)):6d} {pct_n:6.1f}% {pct_area:8.1f}% {np.mean(covs):7.3f}",
        )
    if totals:
        arr = np.array(totals)
        pct_area = 100.0 * arr[arr < THRESHOLD].sum() / arr.sum()
        print(f"{name:8s} {'ALL':10s} {len(arr):6d} {int(np.median(arr)):7d} "
              f"{int(np.percentile(arr, 90)):6d} {100.0 * (arr < THRESHOLD).mean():6.1f}% {pct_area:8.1f}%\n")
