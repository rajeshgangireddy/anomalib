"""Barebones training mode example.

This example demonstrates how to use barebones mode for fast training/testing
without checkpoint overhead while still capturing all metrics.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim

# %%
# Barebones Mode
# ==============
#
# Barebones mode disables checkpointing for faster training/testing.
# Useful for rapid prototyping, benchmarking, and CI/CD pipelines.

# Create engine with barebones mode enabled
engine = Engine(
    barebones=True,  # No checkpoint saving
)

# Initialize model and data
model = Padim()
datamodule = MVTecAD(category="bottle")

# Train without checkpoints
print("Training without checkpoints...")
engine.fit(model=model, datamodule=datamodule)

# Metrics are still captured and returned
print("\nTesting and retrieving metrics...")
results = engine.test(model=model, datamodule=datamodule)

# Print results
print("\nTest Results:")
for metric, value in results[0].items():
    print(f"{metric}: {value:.4f}")

# %%
# Compare with Normal Mode
# ========================
#
# Normal mode automatically saves checkpoints.

engine_normal = Engine(max_epochs=1)
engine_normal.fit(model=Padim(), datamodule=datamodule)
# Checkpoint files created in results/weights/lightning/

print("\nNormal mode: Checkpoints saved")
print(f"Checkpoint path: {engine_normal.best_model_path}")
