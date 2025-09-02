# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TensorBoard hyperparameter logging issue reproduction."""

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.models import Padim as ModelClass


class HParamsCallback(Callback):
    """Custom callback for logging hyperparameters to TensorBoard."""

    def __init__(self) -> None:
        super().__init__()
        self.hparams_logged = False

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log hyperparameters at the end of validation epoch."""
        # Log hparams only once with the latest metrics
        if not self.hparams_logged and trainer.current_epoch > 0:
            # Get TensorBoard logger
            tb_logger = None
            for logger in trainer.loggers:
                if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_hparams"):
                    tb_logger = logger.experiment
                    break

            if tb_logger:
                # Extract hyperparameters
                hparam_dict = {
                    "model_name": pl_module.__class__.__name__,
                    "learning_rate": pl_module.learning_rate if hasattr(pl_module, "learning_rate") else 0.001,
                    "batch_size": trainer.datamodule.train_batch_size if trainer.datamodule else 32,
                    "max_epochs": trainer.max_epochs,
                }

                # Get current metrics
                metrics = trainer.callback_metrics
                metric_dict = {}

                # Map metrics to hparams format
                for key, value in metrics.items():
                    if any(metric in key.lower() for metric in ["auroc", "f1", "loss", "auc"]):
                        # Convert tensor to float if needed
                        if hasattr(value, "item"):
                            metric_dict[key] = value.item()
                        else:
                            metric_dict[key] = float(value)

                if metric_dict:
                    tb_logger.add_hparams(hparam_dict, metric_dict)
                    self.hparams_logged = True


# Initialize components
datamodule = MVTecAD()
model = ModelClass()


# Train the model
logger = AnomalibTensorBoardLogger(
    save_dir="logs",
    name="padim_experiment",
)

hparams_callback = HParamsCallback()

# Create an engine with logger
engine = Engine(logger=logger, callbacks=[hparams_callback])


engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
engine.export(model=engine.model, export_type="torch")

# Configure TensorBoard logger with hparams
