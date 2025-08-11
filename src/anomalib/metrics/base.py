# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base classes for metrics in Anomalib.

This module provides base classes for implementing metrics in Anomalib:

- ``AnomalibMetric``: Base class that makes torchmetrics compatible with Anomalib
- ``create_anomalib_metric``: Factory function to create Anomalib metrics

The ``AnomalibMetric`` class adds batch processing capabilities to any torchmetrics
metric. It allows metrics to be updated directly with ``Batch`` objects instead of
requiring individual tensors.

Example:
    Create a custom F1 score metric::

        >>> from torchmetrics.classification import BinaryF1Score
        >>> from anomalib.metrics import AnomalibMetric
        >>> from anomalib.data import ImageBatch
        >>> import torch
        >>>
        >>> class F1Score(AnomalibMetric, BinaryF1Score):
        ...     pass
        ...
        >>> # Create metric specifying batch fields to use
        >>> f1_score = F1Score(fields=["pred_label", "gt_label"])
        >>>
        >>> # Create sample batch
        >>> batch = ImageBatch(
        ...     image=torch.rand(4, 3, 256, 256),
        ...     pred_label=torch.tensor([0, 0, 0, 1]),
        ...     gt_label=torch.tensor([0, 0, 1, 1])
        ... )
        >>>
        >>> # Update metric with batch directly
        >>> f1_score.update(batch)
        >>> f1_score.compute()
        tensor(0.6667)

    Use factory function to create metric::

        >>> from anomalib.metrics import create_anomalib_metric
        >>> F1Score = create_anomalib_metric(BinaryF1Score)
        >>> f1_score = F1Score(fields=["pred_label", "gt_label"])

    Strict mode vs non-strict mode::

        >>> F1Score = create_anomalib_metric(BinaryF1Score)
        >>>
        >>> # create metric in strict mode (default), and non-strict mode
        >>> f1_score_strict = F1Score(fields=["pred_label", "gt_label"], strict=True)
        >>> f1_score_nonstrict = F1Score(fields=["pred_label", "gt_label"], strict=False)
        >>>
        >>> # create a batch in which 'pred_label' field is None
        >>> batch = ImageBatch(
        ...     image=torch.rand(4, 3, 256, 256),
        ...     gt_label=torch.tensor([0, 0, 1, 1])
        ... )
        >>>
        >>> f1_score_strict.update(batch)  # ValueError
        >>> f1_score_strict.compute()  # UserWarning, tensor(0.)
        >>>
        >>> f1_score_nonstrict.update(batch)  # No error
        >>> f1_score_nonstrict.compute()  # None
"""

from collections.abc import Sequence

import torch
from torchmetrics import Metric, MetricCollection

from anomalib.data import Batch


class AnomalibMetric:
    """Base class for metrics in Anomalib.

    Makes any torchmetrics metric compatible with the Anomalib framework by adding
    batch processing capabilities. Subclasses must inherit from both this class
    and a torchmetrics metric.

    The class enables updating metrics with ``Batch`` objects instead of
    individual tensors. It extracts the specified fields from the batch and
    passes them to the underlying metric's update method.

    Args:
        fields (Sequence[str] | None): Names of fields to extract from batch.
            If None, uses class's ``default_fields``. Required if no defaults.
        prefix (str): Prefix added to metric name. Defaults to "".
        strict (bool): Whether to raise an error if batch is missing fields.
        **kwargs: Additional arguments passed to parent metric class.

    Raises:
        ValueError: If no fields are specified and class has no defaults.

    Example:
        Create image and pixel-level F1 metrics::

            >>> from torchmetrics.classification import BinaryF1Score
            >>> class F1Score(AnomalibMetric, BinaryF1Score):
            ...     pass
            ...
            >>> # Image-level metric using pred_label and gt_label
            >>> image_f1 = F1Score(
            ...     fields=["pred_label", "gt_label"],
            ...     prefix="image_"
            ... )
            >>> # Pixel-level metric using pred_mask and gt_mask
            >>> pixel_f1 = F1Score(
            ...     fields=["pred_mask", "gt_mask"],
            ...     prefix="pixel_"
            ... )
    """

    default_fields: Sequence[str]

    def __init__(
            self,
            fields: Sequence[str] | None = None,
            prefix: str = "",
            strict: bool = True,
            **kwargs,
    ) -> None:
        fields = fields or getattr(self, "default_fields", None)
        if fields is None:
            msg = (
                f"Batch fields must be provided for metric {self.__class__}. "
                "Use the `fields` argument to specify which fields from the "
                "batch object should be used to update the metric."
            )
            raise ValueError(msg)
        self.fields = fields
        self.name = prefix + self.__class__.__name__
        self.strict = strict
        super().__init__(**kwargs)

    def __init_subclass__(cls, **kwargs) -> None:
        """Check that subclass implements torchmetrics.Metric interface."""
        del kwargs
        assert issubclass(
            cls,
            (Metric | MetricCollection),
        ), "AnomalibMetric must be a subclass of torchmetrics.Metric or torchmetrics.MetricCollection"

    def _device_safe_call(self, method_name: str, *args, **kwargs):
        """Execute method with device compatibility handling for Intel iGPU issues.

        This method handles the "UR error" that occurs on some Intel integrated GPUs
        when certain tensor operations are performed. It automatically falls back to
        CPU execution when this specific error is encountered.

        Args:
            method_name (str): Name of the parent method to call
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            The result of the method call
        """
        try:
            return getattr(super(), method_name)(*args, **kwargs)
        except RuntimeError as e:
            if "UR error" in str(e):
                # Handle Intel iGPU compatibility issue by moving to CPU temporarily
                cpu_args = []
                original_devices = []

                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        original_devices.append(arg.device)
                        cpu_args.append(arg.cpu())
                    else:
                        original_devices.append(None)
                        cpu_args.append(arg)

                cpu_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor):
                        cpu_kwargs[key] = value.cpu()
                    else:
                        cpu_kwargs[key] = value

                # Save original metric state and move to CPU
                original_state = self._save_and_move_state_to_cpu()

                try:
                    # Execute on CPU
                    result = getattr(super(), method_name)(*cpu_args, **cpu_kwargs)

                    # Move result back to original device if it's a tensor
                    original_device = next((d for d in original_devices if d is not None), None)
                    if isinstance(result, torch.Tensor) and original_device is not None:
                        result = result.to(original_device)

                    return result
                finally:
                    # Always restore original state
                    self._restore_state_from_cpu(original_state)
            else:
                raise

    def _save_and_move_state_to_cpu(self) -> dict:
        """Save current metric state and move it to CPU.

        Returns:
            dict: Dictionary containing original state information
        """
        original_state = {}

        # Handle common torchmetrics state attributes
        for attr_name in ['preds', 'target', 'confmat', 'tp', 'fp', 'tn', 'fn']:
            if hasattr(self, attr_name):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    original_state[attr_name] = {
                        'device': attr_value.device,
                        'value': attr_value,
                    }
                    setattr(self, attr_name, attr_value.cpu())
                elif isinstance(attr_value, list) and attr_value and isinstance(attr_value[0], torch.Tensor):
                    original_state[attr_name] = {
                        'device': attr_value[0].device,
                        'value': attr_value,
                    }
                    setattr(self, attr_name, [t.cpu() for t in attr_value])

        return original_state

    def _restore_state_from_cpu(self, original_state: dict) -> None:
        """Restore metric state from saved state information.

        Args:
            original_state (dict): Dictionary containing original state information
        """
        for attr_name, state_info in original_state.items():
            device = state_info['device']
            original_value = state_info['value']

            if isinstance(original_value, torch.Tensor):
                setattr(self, attr_name, original_value.to(device))
            elif isinstance(original_value, list):
                setattr(self, attr_name, [t.to(device) for t in original_value])

    def _move_state_to_device(self, device: torch.device) -> None:
        """Move metric internal state to specified device.

        This handles moving common torchmetrics state attributes back to the
        original device after CPU fallback execution.

        Args:
            device (torch.device): Target device to move state to
        """
        # Handle common torchmetrics state attributes
        for attr_name in ['preds', 'target', 'confmat', 'tp', 'fp', 'tn', 'fn']:
            if hasattr(self, attr_name):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(self, attr_name, attr_value.to(device))
                elif isinstance(attr_value, list) and attr_value and isinstance(attr_value[0], torch.Tensor):
                    setattr(self, attr_name, [t.to(device) for t in attr_value])

    def update(self, batch: Batch, *args, **kwargs) -> None:
        """Update metric with values from batch fields.

        Args:
            batch (Batch): Batch object containing required fields.
            *args: Additional positional arguments passed to parent update.
            **kwargs: Additional keyword arguments passed to parent update.

        Raises:
            ValueError: If batch is missing any required fields.
        """
        for key in self.fields:
            if getattr(batch, key, None) is None:
                # We cannot update the metric if the batch is missing required fields,
                # so we need to decrement the update count of the super class.
                self._update_count -= 1  # type: ignore[attr-defined]
                if not self.strict:
                    # If not in strict mode, skip updating the metric but don't raise an error
                    return
                # otherwise, raise an error
                if not hasattr(batch, key):
                    msg = (
                        f"Cannot update metric of type {type(self)}. Passed dataclass instance "
                        f"is missing required field: {key}"
                    )
                else:
                    msg = (
                        f"Cannot update metric of type {type(self)}. Passed dataclass instance "
                        f"does not have a value for field with name {key}."
                    )
                raise ValueError(msg)

        values = [getattr(batch, key) for key in self.fields]
        self._device_safe_call('update', *values, *args, **kwargs)

    def compute(self) -> torch.Tensor:
        """Compute the metric value.

        If the metric has not been updated, and metric is not in strict mode, return None.

        Returns:
            torch.Tensor: Computed metric value or None.
        """
        if self._update_count == 0 and not self.strict:  # type: ignore[attr-defined]
            return None
        return self._device_safe_call('compute')


def create_anomalib_metric(metric_cls: type) -> type:
    """Create an Anomalib version of a torchmetrics metric.

    Factory function that creates a new class inheriting from both
    ``AnomalibMetric`` and the input metric class. The resulting class has
    batch processing capabilities while maintaining the original metric's
    functionality.

    Args:
        metric_cls (type): torchmetrics metric class to wrap.

    Returns:
        type: New class inheriting from ``AnomalibMetric`` and input class.

    Raises:
        AssertionError: If input class is not a torchmetrics.Metric subclass.

    Example:
        Create F1 score metric:: 

            >>> from torchmetrics.classification import BinaryF1Score
            >>> F1Score = create_anomalib_metric(BinaryF1Score)
            >>> f1_score = F1Score(fields=["pred_label", "gt_label"])
            >>> f1_score.update(batch)  # Can update with batch directly
            >>> f1_score.compute()
            tensor(0.6667)
    """
    assert issubclass(metric_cls, Metric), "The wrapped metric must be a subclass of torchmetrics.Metric."
    return type(metric_cls.__name__, (AnomalibMetric, metric_cls), {})
