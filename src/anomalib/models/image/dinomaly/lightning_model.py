# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dinomaly: Vision Transformer-based Anomaly Detection with Feature Reconstruction.

This module implements the Dinomaly model for anomaly detection using a Vision Transformer
encoder-decoder architecture. The model leverages pre-trained DINOv2 features and employs
a reconstruction-based approach to detect anomalies by comparing encoder and decoder features.

Dinomaly extracts features from multiple intermediate layers of a DINOv2 Vision Transformer,
compresses them through a bottleneck MLP, and reconstructs them using a Vision Transformer
decoder. Anomaly detection is performed by computing cosine similarity between encoder
and decoder features at multiple scales.

The model is particularly effective for visual anomaly detection tasks where the goal is
to identify regions or images that deviate from normal patterns learned during training.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Dinomaly
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = Dinomaly()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP

Notes:
    - The model uses DINOv2 Vision Transformer as the backbone encoder
    - Features are extracted from intermediate layers (typically layers 2-9 for base models)
    - A bottleneck MLP compresses multi-layer features before reconstruction
    - Anomaly maps are computed using cosine similarity between encoder-decoder features
    - The model supports both unsupervised anomaly detection and localization

See Also:
    :class:`anomalib.models.image.dinomaly.torch_model.ViTill`:
        PyTorch implementation of the Dinomaly model.
"""

import logging
import math
import warnings
from functools import partial
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize, ToTensor

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.models.image.dinomaly.components.optimizer import StableAdamW
from anomalib.models.image.dinomaly.components.schedulers import WarmCosineScheduler
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import ViTill

logger = logging.getLogger(__name__)


DEFAULT_IMAGE_SIZE = 448
DEFAULT_CROP_SIZE = 392


class Dinomaly(AnomalibModule):
    """Dinomaly Lightning Module for Vision Transformer-based Anomaly Detection.

    This lightning module trains the Dinomaly anomaly detection model (ViTill).
    During training, the decoder learns to reconstruct normal features.
    During inference, the trained decoder is expected to successfully reconstruct normal
    regions of feature maps, but fail to reconstruct anomalous regions as
    it has not seen such patterns.

    Args:
        encoder_name (str): Name of the Vision Transformer encoder to use.
            Supports DINOv2 variants (small, base, large) with different patch sizes.
            Defaults to "dinov2reg_vit_base_14".
        bottleneck_dropout (float): Dropout rate for the bottleneck MLP layer.
            Helps prevent overfitting during feature compression. Defaults to 0.2.
        decoder_depth (int): Number of Vision Transformer decoder layers.
            More layers allow for more complex reconstruction. Defaults to 8.
        target_layers (list[int] | None): List of encoder layer indices to extract
            features from. If None, uses [2, 3, 4, 5, 6, 7, 8, 9] for base models
            and [4, 6, 8, 10, 12, 14, 16, 18] for large models.
        fuse_layer_encoder (list[list[int]] | None): Groupings of encoder layers
            for feature fusion. If None, uses [[0, 1, 2, 3], [4, 5, 6, 7]].
        fuse_layer_decoder (list[list[int]] | None): Groupings of decoder layers
            for feature fusion. If None, uses [[0, 1, 2, 3], [4, 5, 6, 7]].
        mask_neighbor_size (int): Size of neighborhood for attention masking in decoder.
            Set to 0 to disable masking. Defaults to 0.
        remove_class_token (bool): Whether to remove class token from features
            before processing. Defaults to False.
        encoder_require_grad_layer (list[int]): List of encoder layer indices
            that require gradients during training. Empty list freezes all encoder
            layers. Defaults to empty list.
        pre_processor (PreProcessor | bool, optional): Pre-processor instance or
            flag to use default. Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor instance
            or flag to use default. Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator instance or flag to use
            default. Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer instance or flag to
            use default. Defaults to ``True``.

    Example:
        >>> from anomalib.data import MVTecAD
        >>> from anomalib.models import Dinomaly
        >>>
        >>> # Basic usage with default parameters
        >>> model = Dinomaly()
        >>>
        >>> # Custom configuration
        >>> model = Dinomaly(
        ...     encoder_name="dinov2reg_vit_large_14",
        ...     decoder_depth=12,
        ...     bottleneck_dropout=0.1,
        ...     mask_neighbor_size=3
        ... )
        >>>
        >>> # Training with datamodule
        >>> datamodule = MVTecAD()
        >>> engine = Engine()
        >>> engine.fit(model, datamodule=datamodule)

    Note:
        The model requires significant GPU memory due to the Vision Transformer
        architecture. Consider using gradient checkpointing or smaller model
        variants for memory-constrained environments.
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        bottleneck_dropout: float = 0.2,
        decoder_depth: int = 8,
        target_layers=None,
        fuse_layer_encoder=None,
        fuse_layer_decoder=None,
        mask_neighbor_size=0,
        remove_class_token=False,
        encoder_require_grad_layer=[],
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        self.model: ViTill = ViTill(
            encoder_name=encoder_name,
            bottleneck_dropout=bottleneck_dropout,
            decoder_depth=decoder_depth,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            mask_neighbor_size=mask_neighbor_size,
            remove_class_token=remove_class_token,
            encoder_require_grad_layer=encoder_require_grad_layer,
        )

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        crop_size: int | None = None,
    ) -> PreProcessor:
        """Configure the default pre-processor for Dinomaly.

        Sets up image preprocessing pipeline including resizing, center cropping,
        and normalization with ImageNet statistics. The preprocessing is optimized
        for DINOv2 Vision Transformer models.

        Args:
            image_size (tuple[int, int] | None): Target size for image resizing
                as (height, width). Defaults to (448, 448).
            crop_size (int | None): Target size for center cropping (assumes square crop).
                Should be smaller than image_size. Defaults to 392.

        Returns:
            PreProcessor: Configured pre-processor with transforms for Dinomaly.

        Raises:
            ValueError: If crop_size is larger than the minimum dimension of image_size.

        Note:
            The default ImageNet normalization statistics are used:
            - Mean: [0.485, 0.456, 0.406]
            - Std: [0.229, 0.224, 0.225]
        """
        crop_size = crop_size or DEFAULT_CROP_SIZE
        image_size = image_size or (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

        # Validate inputs
        if crop_size > min(image_size):
            raise ValueError(f"Crop size {crop_size} cannot be larger than image size {image_size}")

        data_transforms = Compose([
            Resize(image_size),
            ToTensor(),
            CenterCrop(crop_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return PreProcessor(transform=data_transforms)

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Training step for the Dinomaly model.

        Performs a single training iteration by computing feature reconstruction loss
        between encoder and decoder features. Uses progressive cosine similarity loss
        with hardest mining to focus training on difficult examples.

        Args:
            batch (Batch): Input batch containing images and metadata.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Dictionary containing the computed loss value.

        Raises:
            ValueError: If model output doesn't contain required features during training.

        Note:
            The loss function uses progressive weight scheduling where the hardest
            mining percentage increases from 0 to 0.9 over 1000 steps, focusing
            on increasingly difficult examples as training progresses.
        """
        del args, kwargs  # These variables are not used.
        try:
            model_output = self.model(batch.image)
            if not isinstance(model_output, dict) or "encoder_features" not in model_output:
                raise ValueError("Model output should contain encoder_features during training")

            en = model_output["encoder_features"]
            de = model_output["decoder_features"]

            # Progressive loss weight - make this configurable
            p_final = 0.9
            p_schedule_steps = 1000
            p = min(p_final * self.global_step / p_schedule_steps, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_p_schedule", p, on_step=True, on_epoch=False)
            return {"loss": loss}

        except Exception as e:
            logger.error(f"Error in training step: {e}")
            raise

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Validation step for the Dinomaly model.

        Performs inference on the validation batch to compute anomaly scores
        and anomaly maps. The model operates in evaluation mode to generate
        predictions for anomaly detection evaluation.

        Args:
            batch (Batch): Input batch containing images and metadata.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Updated batch with pred_score (anomaly scores) and
                anomaly_map (pixel-level anomaly maps) predictions.

        Raises:
            Exception: If an error occurs during validation inference.

        Note:
            During validation, the model returns InferenceBatch with anomaly
            scores and maps computed from encoder-decoder feature comparisons.
        """
        del args, kwargs  # These variables are not used.
        try:
            predictions = self.model(batch.image)
            return batch.update(pred_score=predictions.pred_score, anomaly_map=predictions.anomaly_map)

        except Exception as e:
            logger.error(f"Error in validation step: {e}")
            raise

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and learning rate scheduler for Dinomaly training.

        Sets up the training configuration with frozen DINOv2 encoder and trainable
        bottleneck and decoder components. Uses StableAdamW optimizer with warm
        cosine learning rate scheduling.

        Returns:
            OptimizerLRScheduler: Tuple containing optimizer and scheduler configurations.

        Note:
            - DINOv2 encoder parameters are frozen to preserve pre-trained features
            - Only bottleneck MLP and decoder parameters are trained
            - Uses truncated normal initialization for Linear layers
            - Learning rate schedule: warmup (100 steps) + cosine decay (5000 total steps)
            - Base learning rate: 2e-3, final learning rate: 2e-4
        """
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze bottleneck and decoder
        for param in self.model.bottleneck.parameters():
            param.requires_grad = True
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        trainable = torch.nn.ModuleList([self.model.bottleneck, self.model.decoder])
        for m in trainable.modules():
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, torch.nn.Linear) and m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)

        optimizer = StableAdamW(
            [{"params": trainable.parameters()}], lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-8
        )
        lr_scheduler = WarmCosineScheduler(
            optimizer, base_value=2e-3, final_value=2e-4, total_iters=5000, warmup_iters=100
        )

        return [optimizer], [lr_scheduler]

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Dinomaly is an unsupervised anomaly detection model that learns normal
        data patterns without requiring anomaly labels during training.

        Returns:
            LearningType: Always returns LearningType.ONE_CLASS for unsupervised learning.

        Note:
            This property may be subject to change if supervised training support
            is introduced in future versions.
        """
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return Dinomaly-specific trainer arguments.

        Provides configuration arguments optimized for Dinomaly training,
        including strategies for distributed training when available.

        Returns:
            dict[str, Any]: Dictionary of trainer arguments with strategy
                configuration for optimal training performance.

        Note:
            Uses DDPStrategy when available for multi-GPU training to improve
            training efficiency for the Vision Transformer architecture.
        """
        # strategy=DDPStrategy(find_unused_parameters=True),
        return {"gradient_clip_val": 0.1, "num_sanity_val_steps": 0}


def global_cosine_hm_percent(a, b, p=0.9, factor=0.0):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1), b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def modify_grad(x, inds, factor=0.0):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x
