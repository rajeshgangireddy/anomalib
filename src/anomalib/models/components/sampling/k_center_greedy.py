# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""k-Center Greedy Method.

Returns points that minimizes the maximum distance of any point to a center.

Reference:
    - https://arxiv.org/abs/1708.00489
"""

import torch
from tqdm import tqdm

from anomalib.models.components.dimensionality_reduction import SparseRandomProjection


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
        idx = torch.randint(high=self.n_observations, size=(1,)).squeeze()  # Keep as tensor

        # Convert selected_idxs to tensor for efficient collision detection
        if selected_idxs:
            selected_idxs_tensor = torch.tensor(selected_idxs, device=idx.device)
        else:
            selected_idxs_tensor = torch.empty(0, dtype=torch.long, device=idx.device)

        selected_coreset_idxs_tensor = []
        for _ in tqdm(range(self.coreset_size), desc="Selecting Coreset Indices."):
            self.update_distances(cluster_centers=idx.unsqueeze(0))
            idx = self.get_new_idx()
            self.min_distances.scatter_(0, idx.unsqueeze(0).unsqueeze(1), 0.0)
            if selected_idxs_tensor.numel() > 0 and torch.any(selected_idxs_tensor == idx):
                msg = "New indices should not be in selected indices."
                raise ValueError(msg)
            selected_coreset_idxs_tensor.append(idx)

        selected_coreset_idxs = [int(tensor_idx.item()) for tensor_idx in selected_coreset_idxs_tensor]
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
