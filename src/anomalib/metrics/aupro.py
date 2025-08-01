# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Area Under Per-Region Overlap (AUPRO) metric.

This module provides the ``AUPRO`` class which computes the area under the
per-region overlap curve for evaluating anomaly segmentation performance.

The AUPRO score measures how well predicted anomaly maps overlap with ground truth
anomaly regions. It is computed by:

1. Performing connected component analysis on ground truth masks
2. Computing per-region ROC curves for each component
3. Averaging the curves and computing area under the curve up to a FPR limit

Example:
    >>> from anomalib.metrics import AUPRO
    >>> import torch
    >>> # Create sample data
    >>> labels = torch.randint(0, 2, (1, 10, 5))  # Binary segmentation masks
    >>> scores = torch.rand_like(labels)  # Anomaly segmentation maps
    >>> # Initialize and compute AUPRO
    >>> metric = AUPRO(fpr_limit=0.3)
    >>> aupro_score = metric(scores, labels)
    >>> aupro_score
    tensor(0.4321)

The metric can also be updated incrementally with batches:

    >>> for batch_scores, batch_labels in dataloader:
    ...     metric.update(batch_scores, batch_labels)
    >>> final_score = metric.compute()

Args:
    dist_sync_on_step (bool): Synchronize metric state across processes at each
        ``forward()`` before returning the value at the step.
        Defaults to ``False``.
    process_group (Any | None): Specify the process group on which
        synchronization is called. Defaults to ``None`` (entire world).
    dist_sync_fn (Callable | None): Callback that performs the allgather
        operation on the metric state. When ``None``, DDP will be used.
        Defaults to ``None``.
    fpr_limit (float): Limit for the false positive rate.
        Defaults to ``0.3``.
    num_thresholds (int | None): Number of thresholds to use for computing the
        ROC curve. When ``None``, uses thresholds from torchmetrics.
        Defaults to ``None``.

Note:
    The AUPRO score ranges from 0 to 1, with 1 indicating perfect overlap between
    predictions and ground truth regions.
"""

from collections.abc import Callable
from typing import Any

import torch
from matplotlib.figure import Figure
from torchmetrics import Metric
from torchmetrics.functional.classification import binary_roc
from torchmetrics.utilities.compute import auc
from torchmetrics.utilities.data import dim_zero_cat

from anomalib.metrics.pro import connected_components_cpu, connected_components_gpu

from .base import AnomalibMetric
from .binning import thresholds_between_0_and_1, thresholds_between_min_and_max
from .utils import plot_metric_curve


class _AUPRO(Metric):
    """Area under per region overlap (AUPRO) Metric.

    Args:
        dist_sync_on_step (bool): Synchronize metric state across processes at
            each ``forward()`` before returning the value at the step.
            Defaults to ``False``.
        process_group (Any | None): Specify the process group on which
            synchronization is called. Defaults to ``None`` (entire world).
        dist_sync_fn (Callable | None): Callback that performs the allgather
            operation on the metric state. When ``None``, DDP will be used.
            Defaults to ``None``.
        fpr_limit (float): Limit for the false positive rate.
            Defaults to ``0.3``.
        num_thresholds (int | None): Number of thresholds to use for computing
            the ROC curve. When ``None``, uses thresholds from torchmetrics.
            Defaults to ``None``.

    Examples:
        >>> import torch
        >>> from anomalib.metrics import AUPRO
        >>> # Create sample data
        >>> labels = torch.randint(0, 2, (1, 10, 5), dtype=torch.float32)
        >>> preds = torch.rand_like(labels)
        >>> # Initialize and compute
        >>> aupro = AUPRO(fpr_limit=0.3)
        >>> aupro(preds, labels)
        tensor(0.4321)

        Increasing the ``fpr_limit`` will increase the AUPRO value:

        >>> aupro = AUPRO(fpr_limit=0.7)
        >>> aupro(preds, labels)
        tensor(0.5271)
    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    preds: list[torch.Tensor]
    target: list[torch.Tensor]
    # When not None, the computation is performed in constant-memory by computing
    # the roc curve for fixed thresholds buckets/thresholds.
    # Warning: The thresholds are evenly distributed between the min and max
    # predictions if all predictions are inside [0, 1]. Otherwise, the thresholds
    # are evenly distributed between 0 and 1.
    # This warning can be removed when
    # https://github.com/Lightning-AI/torchmetrics/issues/1526 is fixed
    # and the roc curve is computed with deactivated formatting
    num_thresholds: int | None

    def __init__(
        self,
        dist_sync_on_step: bool = False,
        process_group: Any | None = None,  # noqa: ANN401
        dist_sync_fn: Callable | None = None,
        fpr_limit: float = 0.3,
        num_thresholds: int | None = None,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.register_buffer("fpr_limit", torch.tensor(fpr_limit))
        self.num_thresholds = num_thresholds

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with new values.

        Args:
            preds (torch.Tensor): predictions of the model
            target (torch.Tensor): ground truth targets
        """
        self.target.append(target)
        self.preds.append(preds)

    def perform_cca(self) -> torch.Tensor:
        """Perform the Connected Component Analysis on the self.target tensor.

        Raises:
            ValueError: ValueError is raised if self.target doesn't conform with
                requirements imposed by kornia for connected component analysis.

        Returns:
            Tensor: Components labeled from 0 to N.
        """
        target = dim_zero_cat(self.target)

        # check and prepare target for labeling via kornia
        if target.min() < 0 or target.max() > 1:
            msg = (
                "kornia.contrib.connected_components expects input to lie in the "
                f"interval [0, 1], but found [{target.min()}, {target.max()}]."
            )
            raise ValueError(
                msg,
            )
        target = target.unsqueeze(1)  # kornia expects N1HW format
        target = target.type(torch.float)  # kornia expects FloatTensor
        return connected_components_gpu(target) if target.is_cuda else connected_components_cpu(target)

    def compute_pro(
        self,
        cca: torch.Tensor,
        target: torch.Tensor,
        preds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the pro/fpr value-pairs until the fpr specified by fpr_limit.

        It leverages the fact that the overlap corresponds to the tpr, and thus
        computes the overall PRO curve by aggregating per-region tpr/fpr values
        produced by ROC-construction.

        Args:
            cca (torch.Tensor): Connected components tensor
            target (torch.Tensor): Ground truth tensor
            preds (torch.Tensor): Model predictions tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing final fpr and tpr
                values.
        """
        if self.num_thresholds is not None:
            # binary_roc is applying a sigmoid on the predictions before computing
            # the roc curve when some predictions are out of [0, 1], the binning
            # between min and max predictions cannot be applied in that case.
            # This can be removed when
            # https://github.com/Lightning-AI/torchmetrics/issues/1526 is fixed
            # and the roc curve is computed with deactivated formatting.

            if torch.all((preds >= 0) * (preds <= 1)):
                thresholds = thresholds_between_min_and_max(preds, self.num_thresholds, self.device)
            else:
                thresholds = thresholds_between_0_and_1(self.num_thresholds, self.device)

        else:
            thresholds = None

        # compute the global fpr-size
        fpr: torch.Tensor = binary_roc(
            preds=preds,
            target=target,
            thresholds=thresholds,
        )[0]  # only need fpr
        output_size = torch.where(fpr <= self.fpr_limit)[0].size(0)

        # compute the PRO curve by aggregating per-region tpr/fpr curves/values.
        tpr = torch.zeros(output_size, device=preds.device, dtype=torch.float)
        fpr = torch.zeros(output_size, device=preds.device, dtype=torch.float)
        new_idx = torch.arange(0, output_size, device=preds.device, dtype=torch.float)

        # Loop over the labels, computing per-region tpr/fpr curves, and
        # aggregating them. Note that, since the groundtruth is different for
        # every all to `roc`, we also get different/unique tpr/fpr curves
        # (i.e. len(_fpr_idx) is different for every call).
        # We therefore need to resample per-region curves to a fixed sampling
        # ratio (defined above).
        labels = cca.unique()[1:]  # 0 is background
        background = cca == 0
        fpr_: torch.Tensor
        tpr_: torch.Tensor
        for label in labels:
            interp: bool = False
            new_idx[-1] = output_size - 1
            mask = cca == label
            # Need to calculate label-wise roc on union of background & mask, as
            # otherwise we wrongly consider other label in labels as FPs.
            # We also don't need to return the thresholds
            fpr_, tpr_ = binary_roc(
                preds=preds[background | mask],
                target=mask[background | mask],
                thresholds=thresholds,
            )[:-1]

            # catch edge-case where ROC only has fpr vals > self.fpr_limit
            if fpr_[fpr_ <= self.fpr_limit].max() == 0:
                fpr_limit_ = fpr_[fpr_ > self.fpr_limit].min()
            else:
                fpr_limit_ = self.fpr_limit

            fpr_idx_ = torch.where(fpr_ <= fpr_limit_)[0]
            # if computed roc curve is not specified sufficiently close to
            # self.fpr_limit, we include the closest higher tpr/fpr pair and
            # linearly interpolate the tpr/fpr point at self.fpr_limit
            if not torch.allclose(fpr_[fpr_idx_].max(), self.fpr_limit):
                tmp_idx_ = torch.searchsorted(fpr_, self.fpr_limit)
                fpr_idx_ = torch.cat([fpr_idx_, tmp_idx_.unsqueeze_(0)])
                slope_ = 1 - ((fpr_[tmp_idx_] - self.fpr_limit) / (fpr_[tmp_idx_] - fpr_[tmp_idx_ - 1]))
                interp = True

            fpr_ = fpr_[fpr_idx_]
            tpr_ = tpr_[fpr_idx_]

            fpr_idx_ = fpr_idx_.float()
            fpr_idx_ /= fpr_idx_.max()
            fpr_idx_ *= new_idx.max()

            if interp:
                # last point will be sampled at self.fpr_limit
                new_idx[-1] = fpr_idx_[-2] + ((fpr_idx_[-1] - fpr_idx_[-2]) * slope_)

            tpr_ = self.interp1d(fpr_idx_, tpr_, new_idx)
            fpr_ = self.interp1d(fpr_idx_, fpr_, new_idx)
            tpr += tpr_
            fpr += fpr_

        # Actually perform the averaging
        tpr /= labels.size(0)
        fpr /= labels.size(0)
        return fpr, tpr

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the PRO curve.

        Perform the Connected Component Analysis first then compute the PRO curve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing final fpr and tpr
                values.
        """
        cca = self.perform_cca().flatten()
        target = dim_zero_cat(self.target).flatten()
        preds = dim_zero_cat(self.preds).flatten()

        return self.compute_pro(cca=cca, target=target, preds=preds)

    def compute(self) -> torch.Tensor:
        """First compute PRO curve, then compute and scale area under the curve.

        Returns:
            Tensor: Value of the AUPRO metric
        """
        fpr, tpr = self._compute()

        aupro = auc(fpr, tpr, reorder=True)
        return aupro / fpr[-1]  # normalize the area

    def generate_figure(self) -> tuple[Figure, str]:
        """Generate a figure containing the PRO curve and the AUPRO.

        Returns:
            tuple[Figure, str]: Tuple containing both the figure and the figure
                title to be used for logging
        """
        fpr, tpr = self._compute()
        aupro = self.compute()

        xlim = (0.0, float(self.fpr_limit.detach().cpu().numpy()))
        ylim = (0.0, 1.0)
        xlabel = "Global FPR"
        ylabel = "Averaged Per-Region TPR"
        loc = "lower right"
        title = "PRO"

        fig, _axis = plot_metric_curve(fpr, tpr, aupro, xlim, ylim, xlabel, ylabel, loc, title, metric_name="AUPRO")

        return fig, title

    @staticmethod
    def interp1d(old_x: torch.Tensor, old_y: torch.Tensor, new_x: torch.Tensor) -> torch.Tensor:
        """Interpolate a 1D signal linearly to new sampling points.

        Args:
            old_x (torch.Tensor): original 1-D x values (same size as y)
            old_y (torch.Tensor): original 1-D y values (same size as x)
            new_x (torch.Tensor): x-values where y should be interpolated at

        Returns:
            Tensor: y-values at corresponding new_x values.
        """
        # Compute slope
        eps = torch.finfo(old_y.dtype).eps
        slope = (old_y[1:] - old_y[:-1]) / (eps + (old_x[1:] - old_x[:-1]))

        # Prepare idx for linear interpolation
        idx = torch.searchsorted(old_x, new_x)

        # searchsorted looks for the index where the values must be inserted
        # to preserve order, but we actually want the preceeding index.
        idx -= 1
        # we clamp the index, because the number of intervals = old_x.size(0) -1,
        # and the left neighbour should hence be at most number of intervals -1,
        idx = torch.clamp(idx, 0, old_x.size(0) - 2)

        # perform actual linear interpolation
        return old_y[idx] + slope[idx] * (new_x - old_x[idx])


class AUPRO(AnomalibMetric, _AUPRO):  # type: ignore[misc]
    """Wrapper to add AnomalibMetric functionality to AUPRO metric."""
