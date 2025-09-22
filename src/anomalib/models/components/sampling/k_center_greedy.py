# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""k-Center Greedy Method.

Returns points that minimizes the maximum distance of any point to a center.

Reference:
    - https://arxiv.org/abs/1708.00489
"""

import logging
import time

import numpy as np
import torch
from tqdm import tqdm

from anomalib.models.components.dimensionality_reduction import SparseRandomProjection

logger = logging.getLogger(__name__)
# set logger level to INFO
logger.setLevel(logging.INFO)


class KCenterGreedy:
    """k-center-greedy method for coreset selection.

    This class implements the k-center-greedy method to select a coreset from an
    embedding space. The method aims to minimize the maximum distance between any
    point and its nearest center.

    Args:
        embedding (torch.Tensor): Embedding tensor extracted from a CNN.
        sampling_ratio (float): Ratio to determine coreset size from embedding size.

    Attributes:
        embedding (torch.Tensor): Input embedding tensor.
        coreset_size (int): Size of the coreset to be selected.
        model (SparseRandomProjection): Dimensionality reduction model.
        features (torch.Tensor): Transformed features after dimensionality reduction.
        min_distances (torch.Tensor): Minimum distances to cluster centers.
        n_observations (int): Number of observations in the embedding.

    Example:
        >>> import torch
        >>> embedding = torch.randn(219520, 1536)
        >>> sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.001)
        >>> sampled_idxs = sampler.select_coreset_idxs()
        >>> coreset = embedding[sampled_idxs]
        >>> coreset.shape
        torch.Size([219, 1536])
    """

    def __init__(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        self.embedding = embedding
        self.coreset_size = int(embedding.shape[0] * sampling_ratio)
        self.model = SparseRandomProjection(eps=0.9)

        self.features: torch.Tensor
        self.min_distances: torch.Tensor = None
        self.n_observations = self.embedding.shape[0]

    def reset_distances(self) -> None:
        """Reset minimum distances to None."""
        self.min_distances = None

    # @torch.compile(mode="reduce-overhead", backend="aot_eager")
    # def single_center_distance(self,features, centers):
    #     return torch.norm(features - centers, p=2, dim=1, keepdim=True)

    def update_distances(self, cluster_centers: list[int] | torch.Tensor) -> None:
        """Update minimum distances given cluster centers.

        Args:
            cluster_centers (list[int] | torch.Tensor): Indices of cluster centers.
        """
        if cluster_centers is not None and len(cluster_centers) > 0:
            # Handle both list[int] and tensor inputs
            if isinstance(cluster_centers, torch.Tensor):
                centers = self.features[cluster_centers]
            else:
                centers = self.features[cluster_centers]

            # Optimize for the single-center case
            if centers.dim() == 1 or (centers.dim() == 2 and centers.shape[0] == 1):
                # Single center case - use broadcast subtraction and norm
                centers = centers.squeeze(0) if centers.dim() == 2 else centers
                distances = torch.norm(self.features - centers, p=2, dim=1, keepdim=True)
                # distances = self.single_center_distance(self.features, centers)
                # distances = torch.linalg.norm(self.features - centers, ord=2, dim=1, keepdim=True) # same time as norm 12 seconds, test on longer?
                # distances = torch.cdist(self.features, centers.unsqueeze(0)) - slowe. 38 seconds
                # distances = torch.sum((self.features - centers) ** 2, dim=1, keepdim=True)  # squared L2 - 20 seconds
            else:
                # Batch processing for multiple centers
                distances = torch.cdist(self.features, centers, p=2.0)
                distances = distances.min(dim=1, keepdim=True).values

            if self.min_distances is None:
                self.min_distances = distances
            else:
                self.min_distances = torch.minimum(self.min_distances, distances)

    def get_new_idx(self) -> torch.Tensor:
        """Get index of the next sample based on maximum minimum distance.

        Returns:
            torch.Tensor: Index of the selected sample (tensor, not converted to int).

        Raises:
            TypeError: If `self.min_distances` is not a torch.Tensor.
        """
        if isinstance(self.min_distances, torch.Tensor):
            # idx = int(torch.argmax(self.min_distances).item())
            _, idx = torch.max(self.min_distances.squeeze(), dim=0)
            return idx  # Keep as tensor - no device transfer
        msg = f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}"
        raise TypeError(msg)

    def select_coreset_idxs(self, selected_idxs: list[int] | None = None) -> list[int]:
        """Greedily form a coreset to minimize maximum distance to cluster centers.

        Args:
            selected_idxs (list[int] | None, optional): Indices of pre-selected
                samples. Defaults to None.

        Returns:
            list[int]: Indices of samples selected to minimize distance to cluster
                centers.

        Raises:
            ValueError: If a newly selected index is already in `selected_idxs`.
        """
        if selected_idxs is None:
            selected_idxs = []

        if self.embedding.ndim == 2:
            self.model.fit(self.embedding)
            self.features = self.model.transform(self.embedding)
            self.reset_distances()
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
            self.update_distances(cluster_centers=selected_idxs)

        selected_coreset_idxs_tensor = []  # Keep as tensor list until the end
        update_times = []
        get_idx_times = []  # Add timing for get_new_idx
        tensor_indexing_times = []  # Time for self.min_distances[idx] = 0
        list_append_times = []  # Time for selected_coreset_idxs.append
        item_convert_times = []  # Time for .item() conversion
        tic = time.time()
        idx = torch.randint(high=self.n_observations, size=(1,)).squeeze()  # Keep as tensor

        # Convert selected_idxs to tensor for efficient collision detection
        if selected_idxs:
            selected_idxs_tensor = torch.tensor(selected_idxs, device=idx.device)
        else:
            selected_idxs_tensor = torch.empty(0, dtype=torch.long, device=idx.device)

        for _ in tqdm(range(self.coreset_size), desc="Selecting Coreset Indices."):
            update_tic = time.time()
            # Pass tensor directly to update_distances (no conversion needed)
            self.update_distances(cluster_centers=idx.unsqueeze(0))
            # Force XPU synchronization to get accurate timing
            torch.xpu.synchronize()
            update_toc = time.time()
            update_times.append(update_toc - update_tic)

            # Time the get_new_idx operation
            get_idx_tic = time.time()
            idx = self.get_new_idx()
            # Force XPU synchronization to get accurate timing
            torch.xpu.synchronize()
            get_idx_toc = time.time()
            get_idx_times.append(get_idx_toc - get_idx_tic)

            # Time tensor indexing operation using scatter_ (in-place, should be faster)
            tensor_idx_tic = time.time()
            # Use scatter_ for efficient in-place tensor modification
            # Need to match dimensions: min_distances is [N, 1], so index needs to be [1, 1]
            self.min_distances.scatter_(0, idx.unsqueeze(0).unsqueeze(1), 0.0)
            # Force XPU synchronization to get accurate timing
            torch.xpu.synchronize()
            tensor_idx_toc = time.time()
            tensor_indexing_times.append(tensor_idx_toc - tensor_idx_tic)

            # Check collision using tensor operations (no .item() needed)
            if selected_idxs_tensor.numel() > 0 and torch.any(selected_idxs_tensor == idx):
                msg = "New indices should not be in selected indices."
                raise ValueError(msg)

            # Time tensor append operation (keep as tensor)
            append_tic = time.time()
            selected_coreset_idxs_tensor.append(idx)
            append_toc = time.time()
            list_append_times.append(append_toc - append_tic)

        # Convert to CPU only once at the end
        # shape of selected_coreset_idxs_tensor
        logger.info(f"Shape of selected_coreset_idxs_tensor: {len(selected_coreset_idxs_tensor)}")
        item_tic = time.time()
        selected_coreset_idxs = [int(tensor_idx.item()) for tensor_idx in selected_coreset_idxs_tensor]
        item_toc = time.time()
        # print(item_toc-item_tic)
        item_convert_times.append(item_toc - item_tic)
        total_time = time.time() - tic

        update_times_np = np.array(update_times)
        get_idx_times_np = np.array(get_idx_times)
        tensor_indexing_times_np = np.array(tensor_indexing_times)
        list_append_times_np = np.array(list_append_times)
        item_convert_times_np = np.array(item_convert_times)

        logger.info(f"Coreset selection time: {total_time:.2f} seconds")
        logger.info(
            f"update_distances timings - mean: {update_times_np.mean():.6f}s, "
            f"sum: {update_times_np.sum():.2f}s, std: {update_times_np.std():.6f}s",
        )
        logger.info(
            f"get_new_idx timings - mean: {get_idx_times_np.mean():.6f}s, "
            f"sum: {get_idx_times_np.sum():.2f}s, std: {get_idx_times_np.std():.6f}s",
        )
        logger.info(
            f"tensor indexing (min_distances[idx]=0) - mean: {tensor_indexing_times_np.mean():.6f}s, "
            f"sum: {tensor_indexing_times_np.sum():.2f}s, std: {tensor_indexing_times_np.std():.6f}s",
        )
        logger.info(
            f".item() conversions (single batch at end) - total: {item_convert_times[0]:.6f}s",
        )
        logger.info(
            f"list.append() operations - mean: {list_append_times_np.mean():.6f}s, "
            f"sum: {list_append_times_np.sum():.2f}s, std: {list_append_times_np.std():.6f}s",
        )

        # Calculate unaccounted time
        accounted_time = (
            update_times_np.sum()
            + get_idx_times_np.sum()
            + tensor_indexing_times_np.sum()
            + list_append_times_np.sum()
            + item_convert_times_np.sum()
        )
        unaccounted_time = total_time - accounted_time
        logger.info(
            f"Unaccounted time (likely tqdm/overhead): {unaccounted_time:.2f}s ({unaccounted_time / total_time * 100:.1f}%)",
        )
        return selected_coreset_idxs

    def sample_coreset(self, selected_idxs: list[int] | None = None) -> torch.Tensor:
        """Select coreset from the embedding.

        Args:
            selected_idxs (list[int] | None, optional): Indices of pre-selected
                samples. Defaults to None.

        Returns:
            torch.Tensor: Selected coreset.

        Example:
            >>> import torch
            >>> embedding = torch.randn(219520, 1536)
            >>> sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.001)
            >>> coreset = sampler.sample_coreset()
            >>> coreset.shape
            torch.Size([219, 1536])
        """
        idxs = self.select_coreset_idxs(selected_idxs)
        return self.embedding[idxs]
