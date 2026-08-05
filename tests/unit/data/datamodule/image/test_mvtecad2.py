# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests - MVTec AD 2 Datamodule."""

from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import MVTecAD2
from anomalib.data.datasets.image.mvtecad2 import TestType
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestMVTecAD2(_TestAnomalibImageDatamodule):
    """MVTec AD 2 Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path) -> MVTecAD2:
        """Create and return a MVTec AD 2 datamodule."""
        datamodule_ = MVTecAD2(
            root=dataset_path / "mvtecad2",
            category="dummy",
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
        return "examples/configs/data/mvtecad2.yaml"

    @staticmethod
    def test_test_types(datamodule: MVTecAD2) -> None:
        """Test that the datamodule can handle different test types."""
        # Test public test set
        public_loader = datamodule.test_dataloader(test_type=TestType.PUBLIC)
        assert public_loader is not None
        batch = next(iter(public_loader))
        assert batch.image.shape == (4, 3, 256, 256)

        # Test private test set
        private_loader = datamodule.test_dataloader(test_type=TestType.PRIVATE)
        assert private_loader is not None
        batch = next(iter(private_loader))
        assert batch.image.shape == (4, 3, 256, 256)

        # Test private mixed test set
        mixed_loader = datamodule.test_dataloader(test_type=TestType.PRIVATE_MIXED)
        assert mixed_loader is not None
        batch = next(iter(mixed_loader))
        assert batch.image.shape == (4, 3, 256, 256)

        # Test invalid test type
        with pytest.raises(ValueError, match=r"'invalid' is not a valid TestType"):
            datamodule.test_dataloader(test_type="invalid")

    @staticmethod
    def test_test_set_retains_normal_samples(datamodule: MVTecAD2) -> None:
        """Test that normal samples are retained in the public test set.

        Regression test: the datamodule previously did not pass ``test_split_mode``,
        so the base class defaulted to ``TestSplitMode.NONE``. ``_create_test_split``
        then split the normal images out of the test set without adding them back,
        yielding an anomalous-only test set on which image-level AUROC is undefined.
        """
        label_indices = set(datamodule.test_data.samples.label_index)
        assert label_indices == {0, 1}, f"Test set must contain both normal and anomalous samples, got {label_indices}"
        assert datamodule.test_data.has_normal
        assert datamodule.test_data.has_anomalous
