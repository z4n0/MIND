#!/usr/bin/env python3
"""
simclr_pretrain.py — Self-supervised SimCLR backbone training for CINECA.
INSPIRED BY: https://docs.lightly.ai/self-supervised-learning/examples/simclr.html

Run
----
srun ... python simclr_pretrain.py --yaml configs/simclr_d121
"""
from __future__ import annotations
# ───────────────────── std-lib
import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ───────────────────── third-party
import tifffile as tiff
import pytorch_lightning as pl
from monai.transforms import (
    Compose,
    Resize,
    RandFlip,
    RandGaussianSmooth,
    RandGaussianNoise,
    RandAdjustContrast,
    RandHistogramShift,
)
from monai.utils.misc import set_determinism
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Lightly: keep the SimCLR building blocks
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
# ───────────────────── project path
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

# ───────────────────── project imports
from configs.ConfigLoader import ConfigLoader
from classes.ModelManager import ModelManager


# ───────────────────── CLI
def parse() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="SimCLR YAML path")
    return p.parse_args()


# ───────────────────── transforms & dataset
class RandomK90:
    """Random 0/90/180/270° rotation on (C, H, W) tensors."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            k = random.randint(0, 3)
            x = torch.rot90(x, k=k, dims=(1, 2))
        return x


def make_domain_aware_transform(
    out_size: Tuple[int, int]
) -> Compose:
    """
    Build a microscopy-friendly augmentation pipeline which:
      • preserves channels (no RGB/color jitter),
      • perturbs intensity/contrast/histograms,
      • adds mild blur/noise,
      • uses flips and 90° rotations.
    """
    return Compose(
        [
            Resize(spatial_size=out_size),
            RandFlip(prob=0.5, spatial_axis=1),  # H
            RandFlip(prob=0.5, spatial_axis=2),  # W
            RandomK90(p=0.5),
            RandGaussianSmooth(sigma_x=(0.6, 1.2), prob=0.3),
            RandGaussianNoise(prob=0.3, mean=0.0, std=0.01),
            RandAdjustContrast(gamma=(0.8, 1.2), prob=0.5),
            RandHistogramShift(num_control_points=5, prob=0.3),
        ]
    )


class TwoView:
    """Apply the same transform twice to create (view1, view2)."""

    def __init__(self, base_t):
        self.base_t = base_t

    def __call__(self, x: torch.Tensor):
        return self.base_t(x.clone()), self.base_t(x.clone())


class TiffPairDataset(Dataset):
    """
    TIFF-aware dataset which:
      • reads 8/16-bit TIFFs (2D or multi-page),
      • ensures (C, H, W) channel-first,
      • scales per-channel to [0, 1] float32,
      • returns two augmented views.
    """

    def __init__(self, root: Path, transform: Compose):
        self.paths = sorted([p for p in Path(root).rglob("*.tif*")])
        if not self.paths:
            raise FileNotFoundError(f"No TIFFs under: {root}")
        self.transform = TwoView(transform)

    def __len__(self) -> int:
        return len(self.paths)

    def _load_tiff(self, p: Path) -> torch.Tensor:
        arr = tiff.imread(str(p))  # np.ndarray
        # Handle shapes: (H, W, C) -> (C, H, W), (C, H, W) unchanged.
        if arr.ndim == 3 and arr.shape[-1] in (3, 4, 5, 6):
            arr = np.moveaxis(arr, -1, 0)
        elif arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim > 3:
            # If a Z-stack sneaks in, perform MIP across the first spatial axis
            # heuristically; then ensure (C, H, W).
            # Assume shapes like (Z, H, W, C) or (C, Z, H, W).
            if arr.shape[0] > 6:  # likely (Z, H, W[, C])
                arr = arr.max(axis=0)
                if arr.ndim == 2:
                    arr = arr[None, ...]
                elif arr.ndim == 3 and arr.shape[-1] <= 6:
                    arr = np.moveaxis(arr, -1, 0)
            else:  # likely (C, Z, H, W)
                arr = arr.max(axis=1)

        arr = arr.astype(np.float32)
        # Robust per-channel min-max scaling to [0, 1].
        eps = 1e-6
        c = arr.shape[0]
        flat = arr.reshape(c, -1)
        cmins = flat.min(axis=1)[:, None, None]
        cmaxs = flat.max(axis=1)[:, None, None]
        arr = (arr - cmins) / (cmaxs - cmins + eps)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        x = self._load_tiff(p)  # (C, H, W), float32 in [0, 1]
        v1, v2 = self.transform(x)
        # Keep tuple structure compatible with your training_step
        return (v1, v2), 0, str(p)


# ───────────────────── utility functions
def reproducibility(seed: int = 42) -> None:
    """Set seeds and deterministic flags for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed)
    cudnn.deterministic, cudnn.benchmark = True, False


def strip_linear_head(backbone: nn.Module) -> None:
    """Replace common classification heads with identity."""
    if hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()
    if hasattr(backbone, "classifier"):
        backbone.classifier = nn.Identity()


@torch.no_grad()
def infer_outdim(backbone: nn.Module, input_shape: Tuple[int, int, int]) -> int:
    """
    Infer flattened feature dimension via a dry forward pass on device.
    Ensures we don't rely on brittle class-name heuristics.
    """
    backbone.eval()
    device = next(backbone.parameters()).device
    dummy = torch.zeros(2, *input_shape, device=device)
    y = backbone(dummy)
    if y.ndim > 2:
        y = torch.flatten(y, 1)
    return int(y.shape[1])


# ───────────────────── LightningModule
class SimCLRModule(pl.LightningModule):
    """
    SimCLR Lightning module with:
      • explicit L2-normalized projections,
      • AdamW + warmup→cosine schedule,
      • optional memory bank and distributed gathering in NT-Xent.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_outdim: int,
        proj_hidden_dim: int = 512,
        proj_out_dim: int = 128,
        temperature: float = 0.07,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        memory_bank_negatives: int = 0,
        gather_distributed: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone

        # Remove probe heads if present
        strip_linear_head(self.backbone)

        self.projection_head = SimCLRProjectionHead(
            input_dim=backbone_outdim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_out_dim,
        )

        # Configure NT-Xent: queue if memory_bank_negatives > 0
        mb = (
            (int(memory_bank_negatives), int(proj_out_dim))
            if memory_bank_negatives > 0
            else 0
        )
        self.criterion = NTXentLoss(
            temperature=temperature,
            memory_bank_size=mb,
            gather_distributed=gather_distributed,
        )

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    @property
    def encoder(self) -> nn.Module:
        return self.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        z = self.projection_head(feats)
        # L2-normalize for cosine similarities in NT-Xent
        return F.normalize(z, dim=1)

    def training_step(self, batch, batch_idx: int):
        (x0, x1), _, _ = batch
        z0 = self(x0)
        z1 = self(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # 10-epoch linear warmup, then cosine to epoch max
        warm_epochs = min(10, max(1, getattr(self.trainer, "max_epochs", 100) // 10))
        warm = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, total_iters=warm_epochs
        )
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs - warm_epochs
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warm, cos], milestones=[warm_epochs]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }


# ───────────────────── main
def main() -> None:
    args = parse()
    cfg = ConfigLoader(Path(args.yaml))

    reproducibility(42)

    # ---------- unlabeled data root ------------------------------------
    data_root = Path(os.environ["DATA_ROOT"])
    ssl_dir = data_root / cfg.dataset["unlabeled_subdir"]
    if not ssl_dir.is_dir():
        raise FileNotFoundError(f"Unlabeled dir not found: {ssl_dir}")

    # ---------- transforms & dataset -----------------------------------
    out_h, out_w = tuple(cfg.data_augmentation["resize_spatial_size"])
    base_t = make_domain_aware_transform(out_size=(out_h, out_w))
    dataset = TiffPairDataset(root=ssl_dir, transform=base_t)

    num_workers = cfg.get_num_workers()
    loader = DataLoader(
        dataset,
        batch_size=cfg.get_batch_size(),
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    print(f"Found {len(dataset):,} unlabeled images")

    # ---------- backbone ------------------------------------------------
    manager = ModelManager(cfg, library=cfg.model["backbone_library"])
    backbone, device = manager.setup_model(
        num_classes=cfg.model["proj_hidden_dim"],  # dummy head; removed below
        pretrained_weights=None,
    )
    in_ch = cfg.get_in_channels()
    backbone.input_shape = (in_ch, out_h, out_w)
    backbone = backbone.to(device)

    # Remove probe, then infer feature dim by dry forward pass
    strip_linear_head(backbone)
    feat_dim = infer_outdim(backbone, backbone.input_shape)

    # ---------- lightning module ---------------------------------------
    module = SimCLRModule(
        backbone=backbone,
        backbone_outdim=feat_dim,
        proj_hidden_dim=cfg.model["proj_hidden_dim"],
        proj_out_dim=cfg.model["proj_output_dim"],
        temperature=cfg.training["temperature"],
        lr=float(cfg.training["lr"]),
        weight_decay=float(cfg.training.get("weight_decay", 1e-4)),
        memory_bank_negatives=int(cfg.training.get("memory_bank_negatives", 0)),
        gather_distributed=bool(cfg.training.get("gather_distributed", False)),
    )

    # ---------- trainer -------------------------------------------------
    ckpt_dir = Path(cfg.output["weights_path"]).expanduser().parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="simclr-{epoch:03d}",
        save_last=True,
        save_top_k=0,
        every_n_epochs=5,
    )
    lrmon_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,  # set to "auto" and strategy="ddp" on CINECA for multi-GPU
        precision="16-mixed",
        max_epochs=cfg.training["num_epochs"],
        log_every_n_steps=10,
        callbacks=[ckpt_cb, lrmon_cb],
    )

    print("⚙  SimCLR pre-training …")
    t0 = time.time()
    trainer.fit(module, loader)
    print(f"✓ finished in {(time.time() - t0) / 60:.1f} min")

    # ---------- save encoder weights -----------------------------------
    w_path = Path(cfg.output["weights_path"]).expanduser()
    w_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.encoder.state_dict(), w_path)
    print("✓ Encoder saved:", w_path)


if __name__ == "__main__":
    main()
