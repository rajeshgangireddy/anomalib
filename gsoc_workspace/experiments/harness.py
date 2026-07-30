# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Core harness for the synthetic-anomaly experiments.

A single job trains one model **once** on all train normals and then derives every
evaluation arm from that identical model, so the arms differ only in the threshold
source (never in the underlying weights). This removes the model-variance confound
that separate per-arm training introduced.

Arms (all share one trained model; full official test for A and B):
    A: oracle. Threshold from the real test (fit-time val = SAME_AS_TEST) -> oracle F1-max.
    B: proxy. Threshold refit on {real test normals + N synthetic anomalies}
       (N = real anomaly count), then transferred to the full real test. Differs from
       A only in real->synthetic anomalies (same normals, same model).
    C: diagnostic. Threshold refit on, and evaluated on, the synthetic set itself
       (is the synthetic set too easy/hard relative to the real test?).
"""

from __future__ import annotations

import copy
import tempfile
import time
from dataclasses import dataclass

import pandas as pd
from torchmetrics.classification import BinaryPrecision, BinaryRecall

import anomalib
from anomalib.data import MVTecAD, Visa
from anomalib.data.utils import Split, ValSplitMode, split_by_label
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

# Pipeline registry: id -> (preset, generator overrides).
PIPELINES = {
    "P1": ("texture_alpha", {"source_kwargs": {"texture_path": "./datasets/dtd"}}),
    "P2": ("self_alpha", {}),
    "P3": ("self_poisson", {}),
    # Area-routed blend: alpha keeps small components visible, Poisson keeps large
    # ones seamless. Threshold is in pixels at the model's input resolution.
    "P4": ("self_hybrid", {"blend_kwargs": {"area_threshold": 2500}}),
    # Threshold ablation around P4. Lower values route more area through Poisson
    # (approaching P3); higher values route more through alpha (approaching P2).
    "P4a": ("self_hybrid", {"blend_kwargs": {"area_threshold": 500}}),
    "P4b": ("self_hybrid", {"blend_kwargs": {"area_threshold": 1000}}),
    "P4c": ("self_hybrid", {"blend_kwargs": {"area_threshold": 6000}}),
    "P4d": ("self_hybrid", {"blend_kwargs": {"area_threshold": 12000}}),
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


@dataclass
class JobConfig:
    """Configuration for a single training job.

    One job trains a model once and emits several result rows: arm A (oracle) plus
    arm B for each pipeline (and arm C for each pipeline when ``include_c`` is set).

    Args:
        phase (str): Sweep phase this job belongs to.
        dataset (str): Dataset key (``"mvtec"`` or ``"visa"``).
        category (str): Category within the dataset.
        model (str): Model name.
        seed (int): Random seed for the datamodule and synthetic sampling.
        pipelines (tuple[str, ...]): Synthetic-anomaly pipeline ids to evaluate for arm B/C.
        include_c (bool): Whether to also emit the diagnostic arm C.
    """

    phase: str
    dataset: str
    category: str
    model: str
    seed: int = 1
    pipelines: tuple[str, ...] = ("P1", "P2", "P3")
    include_c: bool = False

    @property
    def key(self) -> str:
        """Unique, filesystem-safe identifier for this job."""
        return f"{self.phase}_{self.dataset}_{self.category}_{self.model}_s{self.seed}"


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
    preset, overrides = PIPELINES[pipeline]
    return SyntheticAnomalyGenerator.from_preset(preset, probability=1.0, **overrides)


def _take_normals(dataset: object, count: int, seed: int) -> object:
    """Return a shallow copy of a normal dataset limited to ``count`` random rows."""
    subset = copy.copy(dataset)
    n = min(count, len(dataset.samples))
    subset.samples = dataset.samples.sample(n, random_state=seed).reset_index(drop=True)
    return subset


def _synthetic_eval_set(
    negatives: object,
    source_normals: object,
    n_anomalies: int,
    augmenter: SyntheticAnomalyGenerator,
    seed: int,
) -> SyntheticAnomalyDataset:
    """Build an eval dataset of ``{real normal negatives + N synthetic anomalies}``.

    The negatives are used as-is (e.g. the real test normals) so the set differs from
    the real test only in the anomaly source. Synthetic anomalies are generated from
    ``source_normals`` (train normals); the generated normal rows are discarded.
    """
    source = _take_normals(source_normals, max(2 * n_anomalies, 4), seed)
    synthetic = SyntheticAnomalyDataset.from_dataset(source, augmenter=augmenter)
    anomalies = synthetic.samples[synthetic.samples.label_index == 1].head(n_anomalies).copy()
    negative_samples = negatives.samples.copy()
    negative_samples["split"] = Split.VAL
    anomalies["split"] = Split.VAL
    synthetic.samples = pd.concat([negative_samples, anomalies], ignore_index=True)
    synthetic.samples.attrs["task"] = "segmentation"
    return synthetic


def build_datamodule(job: JobConfig) -> object:
    """Construct the base datamodule (all train normals; real test as val and test).

    The model trains on all official train normals and validates on the full official
    test set (SAME_AS_TEST), so the fit-time threshold is the oracle F1-max. Synthetic
    calibration sets for arms B/C are injected later in :func:`run_job`.
    """
    dataset_cls, root, _ = DATASETS[job.dataset]
    train_batch = MODEL_BATCH.get(job.model, DEFAULT_BATCH)
    datamodule = dataset_cls(
        root=root,
        category=job.category,
        train_batch_size=train_batch,
        eval_batch_size=DEFAULT_BATCH,
        num_workers=4,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        seed=job.seed,
    )
    datamodule.prepare_data()
    datamodule.setup()
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


def _metrics(results: list | None) -> dict[str, float]:
    """Flatten the first test-result dict into ``{metric: float}``."""
    return {k: float(v) for k, v in (results[0] if results else {}).items()}


def _reset_metrics(model: object) -> None:
    """Reset evaluator metrics so successive validate/test calls do not accumulate."""
    for metric in (*model.evaluator.val_metrics, *model.evaluator.test_metrics):
        metric.reset()


def _row(
    job: JobConfig,
    arm: str,
    pipeline: str,
    metrics: dict[str, float],
    sizes: tuple[int, int, int],
    thresholds: tuple[float | None, float | None],
    timings: tuple[float, float],
) -> dict:
    """Assemble a single flat result row."""
    n_train, n_val, n_test = sizes
    image_threshold, normalized_threshold = thresholds
    fit_seconds, test_seconds = timings
    return {
        "phase": job.phase,
        "dataset": job.dataset,
        "category": job.category,
        "model": job.model,
        "pipeline": pipeline,
        "arm": arm,
        "seed": job.seed,
        **metrics,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "image_threshold": image_threshold,
        "normalized_image_threshold": normalized_threshold,
        "fit_seconds": fit_seconds,
        "test_seconds": test_seconds,
        "anomalib_version": anomalib.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_job(job: JobConfig) -> list[dict]:
    """Train one model once and evaluate every arm from that identical model.

    Returns one row for arm A (oracle threshold), one row for arm B per pipeline
    (synthetic threshold transferred to the real test) and, when ``job.include_c``
    is set, one row for arm C per pipeline (synthetic threshold on the synthetic set).
    """
    model = build_model(job.model)
    datamodule = build_datamodule(job)
    n_train = len(datamodule.train_data)
    real_test = datamodule.test_data
    n_real_test = len(real_test)

    with tempfile.TemporaryDirectory(prefix="anomalib_run_") as scratch:
        engine = Engine(
            accelerator="gpu",
            devices=1,
            logger=False,
            default_root_dir=scratch,
            **MODEL_TRAINER[job.model],
        )

        # Train once; fit-time validation on the real test sets the oracle threshold.
        start = time.time()
        engine.fit(model=model, datamodule=datamodule)
        fit_seconds = round(time.time() - start, 2)

        rows: list[dict] = []

        # Arm A: oracle threshold (from the fit-time real-test validation).
        _reset_metrics(model)
        start = time.time()
        results_a = engine.test(model=model, datamodule=datamodule, verbose=False)
        test_seconds = round(time.time() - start, 2)
        rows.append(
            _row(job, "A", "-", _metrics(results_a), (n_train, n_real_test, n_real_test),
                 _thresholds(model), (fit_seconds, test_seconds)),
        )

        test_normals, test_anomalies = split_by_label(real_test)
        n_anomalies = len(test_anomalies)

        keep_alive: list[SyntheticAnomalyDataset] = []  # hold temp dirs until the job ends
        for pipeline in job.pipelines:
            calibration = _synthetic_eval_set(
                negatives=test_normals,
                source_normals=datamodule.train_data,
                n_anomalies=n_anomalies,
                augmenter=make_generator(pipeline),
                seed=job.seed,
            )
            keep_alive.append(calibration)
            n_calib = len(calibration)

            # Refit the threshold on the synthetic calibration set (arm B/C threshold).
            datamodule.val_data = calibration
            datamodule.test_data = real_test
            _reset_metrics(model)
            engine.validate(model=model, datamodule=datamodule, verbose=False)

            # Arm B: synthetic-derived threshold transferred to the real test.
            _reset_metrics(model)
            start = time.time()
            results_b = engine.test(model=model, datamodule=datamodule, verbose=False)
            test_seconds = round(time.time() - start, 2)
            rows.append(
                _row(job, "B", pipeline, _metrics(results_b), (n_train, n_calib, n_real_test),
                     _thresholds(model), (fit_seconds, test_seconds)),
            )

            if job.include_c:
                # Arm C: same synthetic threshold, evaluated on the synthetic set itself.
                datamodule.test_data = calibration
                _reset_metrics(model)
                start = time.time()
                results_c = engine.test(model=model, datamodule=datamodule, verbose=False)
                test_seconds = round(time.time() - start, 2)
                rows.append(
                    _row(job, "C", pipeline, _metrics(results_c), (n_train, n_calib, n_calib),
                         _thresholds(model), (fit_seconds, test_seconds)),
                )
                datamodule.test_data = real_test

    return rows
