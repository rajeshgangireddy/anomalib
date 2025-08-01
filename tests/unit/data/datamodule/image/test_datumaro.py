# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests - Datumaro Datamodule."""

from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import Datumaro
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestDatumaro(_TestAnomalibImageDatamodule):
    """Datumaro Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path) -> Datumaro:
        """Create and return a Datumaro datamodule."""
        datamodule_ = Datumaro(
            root=dataset_path / "datumaro",
            train_batch_size=4,
            eval_batch_size=4,
            augmentations=Resize((256, 256)),
        )
        datamodule_.setup()

        return datamodule_

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "examples/configs/data/datumaro.yaml"
