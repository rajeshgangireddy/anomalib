import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToTensor, CenterCrop
from .torch_model import ViTill
from .stable_adamw import StableAdamW, WarmCosineScheduler
import math
import warnings
from functools import partial

logger = logging.getLogger(__name__)


class Dinomaly(AnomalibModule):
    """Dinomaly anomaly detection model using Vision Transformer.

    This model implements the Dinomaly approach for anomaly detection using
    a Vision Transformer encoder-decoder architecture.

    Args:
        encoder_name: Name of the Vision Transformer encoder
        bottleneck_dropout: Dropout rate for bottleneck layer
        decoder_depth: Number of decoder layers
        target_layers: Encoder layers to extract features from
        fuse_layer_encoder: Encoder layer groupings for feature fusion
        fuse_layer_decoder: Decoder layer groupings for feature fusion
        mask_neighbor_size: Neighborhood masking size (0 to disable)
        remove_class_token: Whether to remove class token
        encoder_require_grad_layer: Encoder layers requiring gradients
        pre_processor: Pre-processor configuration
        post_processor: Post-processor configuration
        evaluator: Evaluator configuration
        visualizer: Visualizer configuration
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
            encoder_require_grad_layer=encoder_require_grad_layer
        )

    @classmethod
    def configure_pre_processor(
            cls,
            image_size: tuple[int, int] | None = None,
            crop_size: int | None = None
    ) -> PreProcessor:
        """Configure the default pre-processor for Dinomaly.

        Pre-processor resizes images, applies center cropping, and normalizes
        using ImageNet statistics.

        Args:
            image_size: Target size for resizing. Defaults to (448, 448).
            crop_size: Target size for center cropping. Defaults to 392.

        Returns:
            PreProcessor: Configured Dinomaly pre-processor
        """
        crop_size = crop_size or 392
        image_size = image_size or (448, 448)

        # Validate inputs
        if crop_size > min(image_size):
            raise ValueError(f"Crop size {crop_size} cannot be larger than image size {image_size}")

        data_transforms = Compose([
            Resize(image_size),
            ToTensor(),
            CenterCrop(crop_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return PreProcessor(transform=data_transforms)

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Training step for Dinomaly mode
        Args:
                batch: Input batch containing images and metada
                Returns:
                        Dictionary containing the computed loss
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
        """Validation step for Dinomaly model.

        Args:
            batch: Input batch containing images and metadata

        Returns:
            Updated batch with predictions
        """
        del args, kwargs  # These variables are not used.
        try:
            predictions = self.model(batch.image)
            # Log validation metrics if available
            if hasattr(predictions, 'pred_score'):
                self.log("val_pred_score_mean", predictions.pred_score.mean(),
                         on_step=False, on_epoch=True)

            return batch.update(pred_score=predictions.pred_score, anomaly_map=predictions.anomaly_map)

        except Exception as e:
            logger.error(f"Error in validation step: {e}")
            raise

    def configure_optimizers(self) -> OptimizerLRScheduler:
        trainable = torch.nn.ModuleList([self.model.bottleneck, self.model.decoder])
        for m in trainable.modules():
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, torch.nn.Linear) and m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)

        optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-8)
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=5000,
                                           warmup_iters=100)

        return [optimizer], [lr_scheduler]

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        This is subject to change in the future when support for supervised training is introduced.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return SuperSimpleNet trainer arguments."""
        return {"gradient_clip_val": 0.1, "num_sanity_val_steps": 0}


def global_cosine_hm_percent(a, b, p=0.9, factor=0.):
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

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

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
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x
