# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference Entrypoint script."""

from jsonargparse import ActionConfigFile, Namespace
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import LightningArgumentParser
from torch.utils.data import DataLoader

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import AnomalibModule, get_model


def get_parser() -> LightningArgumentParser:
    """Get parser.

    Returns:
        LightningArgumentParser: The parser object.
    """
    parser = LightningArgumentParser(description="Inference on Anomaly models in Lightning format.")
    parser.add_lightning_class_args(AnomalibModule, "model", subclass_mode=True)
    parser.add_lightning_class_args(Callback, "--callbacks", subclass_mode=True, required=False)
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model weights")
    parser.add_class_arguments(PredictDataset, "--data", instantiate=False)
    parser.add_argument("--output", type=str, required=False, help="Path to save the output image(s).")
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="Show the visualized predictions on the screen.",
    )
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to a configuration file in json or yaml format.",
    )

    return parser


def infer(args: Namespace) -> None:
    """Run inference."""
    callbacks = None if not hasattr(args, "callbacks") else args.callbacks
    engine = Engine(
        default_root_dir=args.output,
        callbacks=callbacks,
        devices=1,
    )
    model = get_model(args.model)

    # create the dataset
    dataset = PredictDataset(**args.data)
    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, pin_memory=True)

    engine.predict(model=model, dataloaders=[dataloader], ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    infer(args)
