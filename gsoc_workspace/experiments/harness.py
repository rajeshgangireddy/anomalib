# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Core harness for the synthetic-anomaly experiments.

Builds datamodules for the three evaluation arms, constructs models with a rich
evaluator, runs a single fit/test cycle and returns a flat result row.

Arms (identical common test set across A and B for a given seed):
    A: real anomalies (carved from test) used for validation/threshold.
    B: synthetic validation from held-out train normals; real common test.
    C: synthetic validation and synthetic test (optimistic diagnostic).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from torchmetrics.classification import BinaryPrecision, BinaryRecall

import anomalib
from anomalib.data import MVTecAD, Visa
from anomalib.data.utils import ValSplitMode, random_split
from anomalib.data.utils.generators import SyntheticAnomalyGenerator
from anomalib.data.utils.synthetic import SyntheticAnomalyDataset
from anomalib.engine import Engine
from anomalib.metrics import AUPR, AUPRO, AUROC, Evaluator, F1Score, create_anomalib_metric
from anomalib.models import AnomalyDINO, Dinomaly, Draem, EfficientAd, Padim, Patchcore

Precision = create_anomalib_metric(BinaryPrecision)
Recall = create_anomalib_metric(BinaryRecall)

# Dataset registry: name -> (datamodule class, root, categories).
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
DATASETS = {
    "mvtec": (MVTecAD, "./datasets/MVTecAD", MVTEC_CATEGORIES),
    "visa": (Visa, "./datasets/visa", VISA_CATEGORIES),
}

# Pipeline registry: id -> (preset, extra source kwargs).
PIPELINES = {
    "P1": ("texture_alpha", {"texture_path": "./datasets/dtd"}),
    "P2": ("self_alpha", {}),
    "P3": ("self_poisson", {}),
}

# Per-model trainer settings and train batch size.
MODEL_TRAINER = {
    "patchcore": {"max_epochs": 1},
    "padim": {"max_epochs": 1},
    "efficient_ad": {"max_epochs": 20},
    "draem": {"max_epochs": 15},
    "dinomaly": {"max_epochs": 10},
    "anomaly_dino": {"max_epochs": 1},
}
MODEL_BATCH = {"efficient_ad": 1}
DEFAULT_BATCH = 8
HELD_OUT_NORMAL_RATIO = 0.3


@dataclass
class RunConfig:
    """Configuration for a single experiment run."""

    phase: str
    dataset: str
    category: str
    model: str
    arm: str
    pipeline: str = "-"
    seed: int = 1

    @property
    def key(self) -> str:
        """Unique, filesystem-safe identifier for this run."""
        return f"{self.phase}_{self.dataset}_{self.category}_{self.model}_{self.pipeline}_{self.arm}_s{self.seed}"


def build_evaluator() -> Evaluator:
    """Return an evaluator with image and pixel metrics (AUROC/F1/AUPR/PR + AUPRO)."""
    image = {"prefix": "image_"}
    pixel = {"prefix": "pixel_", "strict": False}
    val_metrics = [
        AUROC(fields=["pred_score", "gt_label"], **image),
        AUROC(fields=["anomaly_map", "gt_mask"], **pixel),
    ]
    test_metrics = [
        AUROC(fields=["pred_score", "gt_label"], **image),
        F1Score(fields=["pred_label", "gt_label"], **image),
        AUPR(fields=["pred_score", "gt_label"], **image),
        Precision(fields=["pred_label", "gt_label"], **image),
        Recall(fields=["pred_label", "gt_label"], **image),
        AUROC(fields=["anomaly_map", "gt_mask"], **pixel),
        F1Score(fields=["pred_mask", "gt_mask"], **pixel),
        AUPRO(fields=["anomaly_map", "gt_mask"], **pixel),
    ]
    return Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)


def build_model(name: str) -> object:
    """Instantiate a model by name with the rich evaluator."""
    evaluator = build_evaluator()
    factories = {
        "patchcore": lambda: Patchcore(evaluator=evaluator),
        "padim": lambda: Padim(evaluator=evaluator),
        "efficient_ad": lambda: EfficientAd(evaluator=evaluator),
        "draem": lambda: Draem(evaluator=evaluator),
        "dinomaly": lambda: Dinomaly(evaluator=evaluator),
        "anomaly_dino": lambda: AnomalyDINO(evaluator=evaluator),
    }
    return factories[name]()


def make_generator(pipeline: str) -> SyntheticAnomalyGenerator:
    """Build the synthetic-anomaly generator for a pipeline id."""
    preset, source_kwargs = PIPELINES[pipeline]
    overrides = {"probability": 1.0}
    if source_kwargs:
        overrides["source_kwargs"] = source_kwargs
    return SyntheticAnomalyGenerator.from_preset(preset, **overrides)


def build_datamodule(config: RunConfig) -> object:
    """Construct a datamodule wired for the requested evaluation arm.

    A fixed ``FROM_TEST`` split (seeded) yields an identical common test set for
    arms A and B, isolating the effect of the validation source.
    """
    dataset_cls, root, _ = DATASETS[config.dataset]
    train_batch = MODEL_BATCH.get(config.model, DEFAULT_BATCH)
    datamodule = dataset_cls(
        root=root,
        category=config.category,
        train_batch_size=train_batch,
        eval_batch_size=DEFAULT_BATCH,
        num_workers=4,
        val_split_mode=ValSplitMode.FROM_TEST,
        val_split_ratio=0.5,
        seed=config.seed,
    )
    datamodule.prepare_data()
    datamodule.setup()

    if config.arm != "A":
        # Arms B/C: synthetic validation from held-out train normals.
        train_data, val_normals = random_split(datamodule.train_data, HELD_OUT_NORMAL_RATIO, seed=config.seed)
        datamodule.train_data = train_data
        datamodule.val_data = SyntheticAnomalyDataset.from_dataset(
            val_normals,
            augmenter=make_generator(config.pipeline),
        )

        if config.arm == "C":
            train_data, test_normals = random_split(
                datamodule.train_data,
                HELD_OUT_NORMAL_RATIO,
                seed=config.seed + 100,
            )
            datamodule.train_data = train_data
            datamodule.test_data = SyntheticAnomalyDataset.from_dataset(
                test_normals,
                augmenter=make_generator(config.pipeline),
            )

    # Freeze the splits so the trainer does not re-run setup and discard our injection.
    datamodule._is_setup = True  # noqa: SLF001
    return datamodule


def _thresholds(model: object) -> tuple[float | None, float | None]:
    """Best-effort extraction of the fitted (raw, normalized) image thresholds."""
    try:
        post_processor = model.post_processor
        return float(post_processor.image_threshold), float(post_processor.normalized_image_threshold)
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return None, None


def run_single(config: RunConfig) -> dict:
    """Run one fit/test cycle and return a flat result row."""
    model = build_model(config.model)
    datamodule = build_datamodule(config)
    engine = Engine(
        accelerator="gpu",
        devices=1,
        logger=False,
        **MODEL_TRAINER[config.model],
    )

    start = time.time()
    engine.fit(model=model, datamodule=datamodule)
    fit_seconds = round(time.time() - start, 2)

    start = time.time()
    results = engine.test(model=model, datamodule=datamodule)
    test_seconds = round(time.time() - start, 2)

    metrics = {k: float(v) for k, v in (results[0] if results else {}).items()}
    image_threshold, normalized_threshold = _thresholds(model)
    return {
        **asdict(config),
        **metrics,
        "n_train": len(datamodule.train_data),
        "n_val": len(datamodule.val_data),
        "n_test": len(datamodule.test_data),
        "image_threshold": image_threshold,
        "normalized_image_threshold": normalized_threshold,
        "fit_seconds": fit_seconds,
        "test_seconds": test_seconds,
        "anomalib_version": anomalib.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
