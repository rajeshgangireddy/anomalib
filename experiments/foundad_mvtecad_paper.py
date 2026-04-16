"""Reproduce FoundAD Table 7: MVTec-AD results (1/2/4-shot).

Faithfully reproduces the paper's training protocol:
  - ONE shared ManifoldProjector trained on ALL 15 MVTec-AD categories
  - k-shot: k images sampled per category → 15*k training images total
  - DINOv3 ViT-B/16 encoder (via timm, or DINOv2 fallback)
  - 4000 epochs over the combined dataset, AdamW lr=0.001 constant
  - Evaluation per-category with 4 metrics: I-AUROC, AUPR, P-AUROC, PRO

Configuration (from original params.yaml + train_dinov3.yaml):
    - Encoder: DINOv3 ViT-B/16 (dinov3_vit_base_16)
    - image_size: 512 (crop_size in original)
    - pred_depth: 6, pred_emb_dim: 384
    - n_layer: 3 (3rd from last encoder block)
    - top_k: 10 (K_top_mvtec)
    - dropout: 0.2
    - lr: 0.001 (constant, no scheduler)
    - weight_decay: 1e-4
    - epochs: 4000
    - batch_size: 8
    - seed: 42
    - Training augmentations: hflip, vflip, rotate90, color_jitter, gray, blur
    - CutPaste augmentation: color_jitter=0.5
    - feat_normed: False, use_pos_embed: False

Usage:
    # Full experiment (all 15 categories, 1/2/4-shot)
    python experiments/foundad_mvtecad_paper.py

    # Quick smoke test
    python experiments/foundad_mvtecad_paper.py --smoke-test

    # Specific shots
    python experiments/foundad_mvtecad_paper.py --shots 1 2

    # Use DINOv2 fallback
    python experiments/foundad_mvtecad_paper.py --encoder dinov2_vit_base_14 --image-size 518

    # Evaluate pre-trained checkpoints directly
    python experiments/foundad_mvtecad_paper.py --pretrained-dir datasets/MvTeCAD-FoundAD
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from anomalib.data import MVTecAD
from anomalib.data.dataclasses.torch.image import ImageBatch
from anomalib.metrics import AUPRO, AUPR, AUROC
from anomalib.models.components import GaussianBlur2d
from anomalib.models.components.dinov2 import DinoV2Loader
from anomalib.models.image.dinomaly.components import vision_transformer as dinomaly_vt
from anomalib.models.image.foundad.components.cutpaste import CutPasteUnion
from anomalib.models.image.foundad.components.manifold_projector import ManifoldProjector

# ──────────────────────────────────────────────────────────────────────
# Paper configuration constants
# ──────────────────────────────────────────────────────────────────────
SEED = 42
DATASET_ROOT = "./datasets/MVTecAD"
RESULTS_DIR = "./results/foundad_paper_mvtecad"

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

SHOTS = [1, 2, 4]

# Encoder
ENCODER_NAME = "dinov3_vit_base_16"
IMAGE_SIZE = 512  # crop_size in original (512/16 = 32 patches per side)

# Projector
PRED_DEPTH = 6
PRED_EMB_DIM = 384
N_LAYER = 3
TOP_K = 10  # K_top_mvtec
DROPOUT = 0.2

# Training
LR = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 4000
BATCH_SIZE = 8
COLOR_JITTER_STRENGTH = 0.5

# Evaluation
FPR_LIMIT = 0.3

# ──────────────────────────────────────────────────────────────────────
# Architecture configs
# ──────────────────────────────────────────────────────────────────────
ARCH_CONFIGS = {
    "small": {"embed_dim": 384, "num_heads": 6},
    "base": {"embed_dim": 768, "num_heads": 12},
    "large": {"embed_dim": 1024, "num_heads": 16},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce FoundAD Table 7 on MVTec-AD")
    parser.add_argument("--encoder", type=str, default=ENCODER_NAME)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--shots", type=int, nargs="+", default=SHOTS)
    parser.add_argument("--categories", type=str, nargs="+", default=MVTEC_CATEGORIES)
    parser.add_argument("--dataset-root", type=str, default=DATASET_ROOT)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick run: 2 categories, 1-shot, 50 epochs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pretrained-dir", type=str, default=None,
                        help="Load pre-trained projector checkpoint instead of training")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Print loss every N epochs")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_pre_transform(image_size: int) -> transforms.Compose:
    """Resize + ToTensor + Normalize for PIL images (matching original test pipeline)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_tensor_transform(image_size: int) -> torch.nn.Module:
    """Resize + Normalize for tensor images from anomalib dataloader.

    Anomalib returns tensors in [0,1] range at original resolution.
    We need to resize to image_size and normalize.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    class _EvalTransform(torch.nn.Module):
        def __init__(self, size, m, s):
            super().__init__()
            self.size = size
            self.register_buffer("mean", m)
            self.register_buffer("std", s)

        def forward(self, x):
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
            return (x - self.mean) / self.std

    return _EvalTransform(image_size, mean, std)


class _RandomRotate90or270:
    """Randomly rotate by 90 or 270 degrees (matching original FoundAD)."""

    def __init__(self, p: float = 1.0) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            angle = random.choice([90, 270])
            return TF.rotate(img, angle)
        return img


def build_train_transform_staged(image_size: int) -> transforms.Compose:
    """Build staged training augmentations matching original FoundAD exactly.

    The original uses build_train_transform_staged with:
      - Stage 1 (orientation): RandomChoice of [HFlip, VFlip, Rotate90or270], applied with p=0.3
      - Stage 2 (appearance): RandomChoice of [ColorJitter, Grayscale, GaussianBlur], applied with p=0.3
      - Then: Resize, ToTensor, Normalize

    All augmentations are at PIL level, before ToTensor.
    """
    ops = []

    # Stage 1: orientation augmentations (pick ONE with p=0.3)
    orient_candidates = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        _RandomRotate90or270(p=1.0),
    ]
    ops.append(
        transforms.RandomApply(
            [transforms.RandomChoice(orient_candidates)],
            p=0.3,
        )
    )

    # Stage 2: appearance augmentations (pick ONE with p=0.3)
    appear_candidates = [
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.RandomGrayscale(p=1.0),
        transforms.GaussianBlur(
            kernel_size=23 if image_size >= 384 else 11,
            sigma=(0.1, 2.0),
        ),
    ]
    ops.append(
        transforms.RandomApply(
            [transforms.RandomChoice(appear_candidates)],
            p=0.3,
        )
    )

    # Resize + ToTensor + Normalize
    ops.extend([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return transforms.Compose(ops)


class FewShotTrainDataset(Dataset):
    """Combined dataset of k images per category, pre-loaded into memory.

    Stores raw PIL images and applies augmentations at PIL level (before
    ToTensor/Normalize), matching the original FoundAD pipeline.
    """

    def __init__(
        self,
        categories: list[str],
        n_shot: int,
        dataset_root: str,
        image_size: int,
        seed: int,
    ) -> None:
        self.transform = build_train_transform_staged(image_size)
        self.pil_images: list[Image.Image] = []

        rng = random.Random(seed)

        for cat in categories:
            dm = MVTecAD(root=dataset_root, category=cat, train_batch_size=1, seed=seed)
            dm.setup(stage="fit")
            total = len(dm.train_data)
            indices = list(range(total))
            rng.shuffle(indices)
            selected = sorted(indices[:n_shot])
            for idx in selected:
                item = dm.train_data[idx]
                # Get the image file path from the datamodule item
                img_path = item.image_path
                pil_img = Image.open(img_path).convert("RGB")
                self.pil_images.append(pil_img)

        print(f"  FewShotTrainDataset: {len(self.pil_images)} images "
              f"({n_shot}-shot x {len(categories)} categories)")

    def __len__(self) -> int:
        return len(self.pil_images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(self.pil_images[idx])


# ──────────────────────────────────────────────────────────────────────
# Model components
# ──────────────────────────────────────────────────────────────────────

def build_encoder(encoder_name: str) -> torch.nn.Module:
    """Build frozen encoder."""
    if encoder_name.startswith("dinov3"):
        encoder = DinoV2Loader().load(encoder_name)
    else:
        encoder = DinoV2Loader(vit_factory=dinomaly_vt).load(encoder_name)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def extract_features(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    n_layer: int = N_LAYER,
) -> torch.Tensor:
    """Extract patch features from frozen encoder, stripping prefix tokens.

    Returns: (B, num_patches, embed_dim)
    """
    with torch.no_grad():
        features = encoder.get_intermediate_layers(images, n=n_layer)[0]
        # TimmDinoV3Wrapper already strips prefix tokens; DINOv2 doesn't
        if not getattr(encoder, "prefix_stripped", False):
            num_prefix = 1 + encoder.num_register_tokens
            features = features[:, num_prefix:, :]
    return features


# ──────────────────────────────────────────────────────────────────────
# Shared-projector training
# ──────────────────────────────────────────────────────────────────────

def train_shared_projector(
    encoder: torch.nn.Module,
    categories: list[str],
    n_shot: int,
    args: argparse.Namespace,
) -> ManifoldProjector:
    """Train ONE shared ManifoldProjector on ALL categories simultaneously.

    This matches the paper's protocol: a single projector sees images from
    all categories, preventing overfitting on any single category.
    """
    device = next(encoder.parameters()).device

    # Get architecture info
    embed_dim = encoder.embed_dim
    # Compute actual num_patches from image_size and patch_size
    patch_size = getattr(encoder, "patch_size", 14)  # DINOv3=16, DINOv2=14
    num_patches = (args.image_size // patch_size) ** 2
    arch_name = args.encoder.split("_")[-2]
    num_heads = ARCH_CONFIGS[arch_name]["num_heads"]

    # Build projector
    projector = ManifoldProjector(
        num_patches=num_patches,
        embed_dim=embed_dim,
        predictor_embed_dim=PRED_EMB_DIM,
        depth=PRED_DEPTH,
        num_heads=num_heads,
        use_pos_embed=False,
        feat_normed=False,
    ).to(device)

    dropout = torch.nn.Dropout(DROPOUT)
    cutpaste = CutPasteUnion(color_jitter=COLOR_JITTER_STRENGTH)

    # Build combined dataset
    dataset = FewShotTrainDataset(
        categories=categories,
        n_shot=n_shot,
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        seed=args.seed,
    )

    effective_batch_size = min(args.batch_size, len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=True,  # Match original (DistributedSampler + drop_last=True)
        num_workers=0,  # Images are already in memory
        pin_memory=False,
    )

    # Optimizer (constant lr, no scheduler — matching original lr_config: const)
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Training loop
    projector.train()
    total_steps = 0
    t0 = time.time()

    print(f"\n  Training shared projector: {args.epochs} epochs, "
          f"batch_size={args.batch_size}, {len(dataset)} images")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_images in dataloader:
            batch_images = batch_images.to(device)

            # Extract target features (from original images)
            target = extract_features(encoder, batch_images, N_LAYER)

            # Apply CutPaste augmentation
            augmented = cutpaste(batch_images)

            # 50% chance: projector sees augmented features; else normal
            if random.random() < 0.5:
                context = extract_features(encoder, augmented, N_LAYER)
            else:
                context = target

            # Forward through projector
            predicted = projector(dropout(context))
            loss = F.mse_loss(target, predicted)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1
            total_steps += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch + 1}/{args.epochs} | "
                  f"loss={avg_loss:.6f} | "
                  f"steps={total_steps} | "
                  f"time={elapsed:.0f}s")

    train_time = time.time() - t0
    print(f"  Training complete: {total_steps} steps in {train_time:.1f}s")

    return projector


def load_pretrained_projector(
    pretrained_dir: str,
    n_shot: int,
    encoder: torch.nn.Module,
    device: torch.device,
    image_size: int = IMAGE_SIZE,
) -> ManifoldProjector:
    """Load a pre-trained projector checkpoint from the original FoundAD repo."""
    arch_name = "base"
    ckpt_path = Path(pretrained_dir) / f"mvtec_{n_shot}shot" / "dinov3_pretrained" / "pretrained.pth.tar"

    if not ckpt_path.exists():
        msg = f"Pre-trained checkpoint not found: {ckpt_path}"
        raise FileNotFoundError(msg)

    embed_dim = encoder.embed_dim
    patch_size = getattr(encoder, "patch_size", 14)
    num_patches = (image_size // patch_size) ** 2
    num_heads = ARCH_CONFIGS[arch_name]["num_heads"]

    projector = ManifoldProjector(
        num_patches=num_patches,
        embed_dim=embed_dim,
        predictor_embed_dim=PRED_EMB_DIM,
        depth=PRED_DEPTH,
        num_heads=num_heads,
        use_pos_embed=False,
        feat_normed=False,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # The original checkpoint includes `mask_token` which is only used during
    # masked training (not inference). Our ManifoldProjector doesn't have it,
    # so we filter it out before loading.
    state_dict = {k: v for k, v in ckpt["predictor"].items() if k != "mask_token"}
    projector.load_state_dict(state_dict, strict=True)
    print(f"  Loaded pre-trained projector from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    return projector


# ──────────────────────────────────────────────────────────────────────
# Per-category evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_category(
    encoder: torch.nn.Module,
    projector: ManifoldProjector,
    category: str,
    args: argparse.Namespace,
) -> dict:
    """Evaluate the shared projector on a single category's test set."""
    device = next(encoder.parameters()).device
    projector.eval()

    eval_transform = build_eval_tensor_transform(args.image_size).to(device)
    gaussian_blur = GaussianBlur2d(sigma=4.0, channels=1, kernel_size=5).to(device)

    # Load test data via anomalib's datamodule (handles collation)
    dm = MVTecAD(root=args.dataset_root, category=category, eval_batch_size=32,
                 num_workers=args.num_workers, seed=args.seed)
    dm.setup(stage="test")

    # Metrics
    i_auroc = AUROC(fields=["pred_score", "gt_label"])
    i_aupr = AUPR(fields=["pred_score", "gt_label"])
    p_auroc = AUROC(fields=["anomaly_map", "gt_mask"])
    p_aupro = AUPRO(fields=["anomaly_map", "gt_mask"], fpr_limit=FPR_LIMIT)

    test_loader = dm.test_dataloader()

    with torch.no_grad():
        for batch in test_loader:
            images = eval_transform(batch.image.to(device))
            gt_label = batch.gt_label.to(device)
            gt_mask = batch.gt_mask  # Stay on CPU for metrics

            # Extract features and predict
            target = extract_features(encoder, images, N_LAYER)
            predicted = projector(target)

            # Per-patch MSE
            patch_mse = F.mse_loss(target, predicted, reduction="none").mean(dim=2)

            # Image-level score: mean of top-K patches
            k = min(TOP_K, patch_mse.shape[1])
            pred_score = torch.topk(patch_mse, k, dim=1).values.mean(dim=1)

            # Pixel-level anomaly map — resize to original image size for fair eval
            h = w = int(math.sqrt(patch_mse.shape[1]))
            anomaly_map = patch_mse.view(-1, 1, h, w)
            if gt_mask is not None:
                # gt_mask may be (B, H, W) or (B, 1, H, W)
                if gt_mask.ndim == 3:
                    orig_size = (gt_mask.shape[1], gt_mask.shape[2])
                else:
                    orig_size = (gt_mask.shape[2], gt_mask.shape[3])
            else:
                orig_size = (batch.image.shape[2], batch.image.shape[3])
            anomaly_map = F.interpolate(anomaly_map, size=orig_size,
                                        mode="bilinear", align_corners=False)
            anomaly_map = gaussian_blur(anomaly_map)

            # Update metrics via ImageBatch
            metric_batch = ImageBatch(
                image=batch.image,
                gt_label=gt_label.cpu(),
                gt_mask=gt_mask,
            ).update(
                pred_score=pred_score.cpu(),
                anomaly_map=anomaly_map.cpu(),
            )

            i_auroc.update(metric_batch)
            i_aupr.update(metric_batch)
            if gt_mask is not None:
                p_auroc.update(metric_batch)
                p_aupro.update(metric_batch)

    results = {
        "category": category,
        "image_AUROC": i_auroc.compute().item(),
        "image_AUPR": i_aupr.compute().item(),
        "pixel_AUROC": p_auroc.compute().item(),
        "pixel_AUPRO": p_aupro.compute().item(),
    }
    return results


# ──────────────────────────────────────────────────────────────────────
# Table formatting and saving
# ──────────────────────────────────────────────────────────────────────

def format_table(all_results: list[dict], shots: list[int], categories: list[str]) -> str:
    """Format results as a table matching Table 7."""
    by_cat: dict[str, dict[int, dict]] = {}
    for r in all_results:
        by_cat.setdefault(r["category"], {})[r["n_shot"]] = r

    header_parts = ["Class"]
    for shot in shots:
        header_parts.extend([
            f"{shot}s I-AUC", f"{shot}s AUPR",
            f"{shot}s P-AUC", f"{shot}s PRO",
        ])
    sep = " | "
    header = sep.join(f"{h:>12s}" for h in header_parts)
    divider = "-" * len(header)

    lines = [divider, header, divider]

    avgs: dict[int, dict[str, list]] = {
        s: {"i_auroc": [], "i_aupr": [], "p_auroc": [], "p_aupro": []}
        for s in shots
    }

    for cat in categories:
        if cat not in by_cat:
            continue
        row = [f"{cat:>12s}"]
        for shot in shots:
            r = by_cat.get(cat, {}).get(shot, {})
            vals = {
                "i_auroc": r.get("image_AUROC", 0) * 100,
                "i_aupr": r.get("image_AUPR", 0) * 100,
                "p_auroc": r.get("pixel_AUROC", 0) * 100,
                "p_aupro": r.get("pixel_AUPRO", 0) * 100,
            }
            row.extend([f"{v:12.1f}" for v in vals.values()])
            for k, v in vals.items():
                avgs[shot][k].append(v)
        lines.append(sep.join(row))

    lines.append(divider)
    avg_row = [f"{'Average':>12s}"]
    for shot in shots:
        for key in ["i_auroc", "i_aupr", "p_auroc", "p_aupro"]:
            vals = avgs[shot][key]
            avg = sum(vals) / len(vals) if vals else 0
            avg_row.append(f"{avg:12.1f}")
    lines.append(sep.join(avg_row))
    lines.append(divider)

    return "\n".join(lines)


def save_results(all_results: list[dict], results_dir: str) -> None:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "results.json"
    with json_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)

    csv_path = out_dir / "table7_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Shot", "I-AUROC(%)", "AUPR(%)", "P-AUROC(%)", "PRO(%)"])
        for r in all_results:
            writer.writerow([
                r["category"], r["n_shot"],
                f"{r.get('image_AUROC', 0) * 100:.1f}",
                f"{r.get('image_AUPR', 0) * 100:.1f}",
                f"{r.get('pixel_AUROC', 0) * 100:.1f}",
                f"{r.get('pixel_AUPRO', 0) * 100:.1f}",
            ])
    print(f"  Results saved to {json_path} and {csv_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.smoke_test:
        args.categories = ["bottle", "toothbrush"]
        args.shots = [1]
        args.epochs = 50
        args.log_interval = 10
        print("*** SMOKE TEST: 2 categories, 1-shot, 50 epochs ***")

    print(f"\nFoundAD Table 7 Reproduction")
    print(f"  Encoder: {args.encoder}")
    print(f"  Image size: {args.image_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Categories: {args.categories}")
    print(f"  Shots: {args.shots}")
    print(f"  Seed: {args.seed}")

    # Seed everything
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build encoder once (shared across all experiments)
    print(f"\nLoading encoder: {args.encoder}...")
    encoder = build_encoder(args.encoder)
    device = torch.device(args.device)
    encoder = encoder.to(device)
    print(f"  embed_dim={encoder.embed_dim}, "
          f"num_patches={encoder.patch_embed.num_patches}, "
          f"register_tokens={encoder.num_register_tokens}")

    all_results = []

    for n_shot in args.shots:
        print(f"\n{'=' * 70}")
        print(f"  {n_shot}-SHOT EXPERIMENT")
        print(f"{'=' * 70}")

        # Train or load shared projector
        if args.pretrained_dir:
            projector = load_pretrained_projector(
                args.pretrained_dir, n_shot, encoder, device, args.image_size,
            )
        else:
            projector = train_shared_projector(encoder, args.categories, n_shot, args)

        # Save projector checkpoint
        ckpt_dir = Path(args.results_dir) / f"{n_shot}shot"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "projector.pth"
        torch.save(projector.state_dict(), ckpt_path)
        print(f"  Projector saved to {ckpt_path}")

        # Evaluate per category
        projector = projector.to(device)
        for i, category in enumerate(args.categories, 1):
            print(f"\n  [{i}/{len(args.categories)}] Evaluating {category}...")
            results = evaluate_category(encoder, projector, category, args)
            results["n_shot"] = n_shot

            i_auroc = results["image_AUROC"] * 100
            i_aupr = results["image_AUPR"] * 100
            p_auroc = results["pixel_AUROC"] * 100
            p_aupro = results["pixel_AUPRO"] * 100
            print(f"    I-AUROC={i_auroc:.1f}  AUPR={i_aupr:.1f}  "
                  f"P-AUROC={p_auroc:.1f}  PRO={p_aupro:.1f}")

            all_results.append(results)

        # Save intermediate results
        save_results(all_results, args.results_dir)

    # Print final table
    print(f"\n\n{'=' * 80}")
    print(f"  FoundAD Table 7 — MVTec-AD")
    print(f"  Encoder: {args.encoder}, Epochs: {args.epochs}, Seed: {args.seed}")
    print(f"{'=' * 80}\n")
    print(format_table(all_results, args.shots, args.categories))

    save_results(all_results, args.results_dir)


if __name__ == "__main__":
    main()
