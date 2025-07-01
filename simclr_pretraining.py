#!/usr/bin/env python3
"""
simclr_pretrain.py
==================
Self-supervised SimCLR backbone pre-training – ready for CINECA.

Usage
-----
sbatch … python simclr_pretrain.py --yaml configs/simclr_densenet_3c.yaml

Environment variables (export them in the `.slurm` file):
    DATA_ROOT   : absolute path of your dataset root
                  e.g.  $WORK/lzanotto/data
"""

# ───────────────────────── std libs & settings ────────────────────────────
import argparse, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from monai.utils.misc import set_determinism
from torch import nn
from torch.utils.data import DataLoader

# ───────────────────────── project path ───────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))           # enable local imports

# ───────────────────────── project imports ────────────────────────────────
from configs.ConfigLoader import ConfigLoader
from classes.ModelManager import ModelManager

# lightly / lightning
import pytorch_lightning as pl
from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead

# ───────────────────────── argparse ───────────────────────────────────────
def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True,
                    help="Path to the SimCLR YAML (relative or absolute)")
    return ap.parse_args()

# ───────────────────────── Lightning module ───────────────────────────────
class SimCLRModule(pl.LightningModule):
    """Encoder + projection head + NT-Xent loss."""

    def __init__(
        self,
        backbone: nn.Module,
        proj_hidden_dim: int,
        proj_out_dim: int,
        temperature: float = 0.07,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone          # encoder to train
        self.criterion = NTXentLoss(temperature=temperature)

        # -- strip any classification head the backbone may have
        self._remove_linear_probe_head()
        backbone_outdim = self._infer_feature_dim()

        # projection head
        self.projector = SimCLRProjectionHead(
            input_dim=backbone_outdim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_out_dim,
        )

    # -------------------------- util helpers ------------------------------
    def _remove_linear_probe_head(self) -> None:
        if hasattr(self.backbone, "fc") and isinstance(self.backbone.fc, nn.Module):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "classifier") and isinstance(
            self.backbone.classifier, nn.Module
        ):
            self.backbone.classifier = nn.Identity()

    def _infer_feature_dim(self) -> int:
        """Forward a dummy batch to discover encoder output dim."""
        device = next(self.backbone.parameters()).device
        dummy = torch.zeros(
            1,
            self.hparams.in_channels,
            self.hparams.img_h,
            self.hparams.img_w,
            device=device,
        )
        with torch.no_grad():
            feats = self.backbone(dummy).flatten(1)
        return feats.shape[1]

    # -------------------------- Lightning API -----------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x).flatten(1)
        return self.projector(feats)

    def training_step(self, batch, _):
        (x0, x1), _, _ = batch       # lightly's SimCLR collate
        z0, z1 = self(x0), self(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-6
        )

# ───────────────────────── main ───────────────────────────────────────────
def main() -> None:
    args = parse()
    cfg = ConfigLoader(Path(args.yaml))

    # ------------ reproducibility
    SEED = 42
    random.seed(SEED);  np.random.seed(SEED);  torch.manual_seed(SEED)
    set_determinism(SEED)
    cudnn.deterministic, cudnn.benchmark = True, False

    # ------------ paths & env
    DATA_ROOT = Path(os.environ["DATA_ROOT"])
    ssl_subdir = Path(cfg.dataset["unlabeled_subdir"])   # e.g. PRETRAINING_MSA_VS_PD
    ssl_dir = DATA_ROOT / ssl_subdir
    if not ssl_dir.is_dir():
        raise FileNotFoundError(f"{ssl_dir} not found – check DATA_ROOT or YAML.")

    # ------------ transforms & dataset
    transform = SimCLRTransform(
        input_size=cfg.model["input_size"],
        gaussian_blur=0.5,
        cj_strength=0.4,
        vf_prob=0.5,
        hf_prob=0.5,
        rr_prob=0.5,
        rr_degrees=180,
    )
    dataset = LightlyDataset(input_dir=ssl_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg.training["num_workers"],
        pin_memory=True,
        persistent_workers=cfg.training["num_workers"] > 0,
    )
    print(f"Found {len(dataset):,} unlabeled images → batch {cfg.training['batch_size']}")

    # ------------ backbone
    manager = ModelManager(cfg, library=cfg.model["backbone_library"])
    encoder, device = manager.setup_model(
        num_classes=cfg.model["proj_hidden_dim"],       # dummy (ignored)
        pretrained_weights=None,
        remove_linear_head=True,
    )

    # ------------ Lightning wrapper
    module = SimCLRModule(
        backbone=encoder.to(device),
        proj_hidden_dim=cfg.model["proj_hidden_dim"],
        proj_out_dim=cfg.model["proj_output_dim"],
        temperature=cfg.training["temperature"],
        lr=cfg.training["lr"],
    )
    # store for dummy-shape inference
    module.hparams.update(
        in_channels=cfg.model["in_channels"],
        img_h=cfg.model["input_size"],
        img_w=cfg.model["input_size"],
    )

    # ------------ trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=cfg.training["num_epochs"],
        precision="16-mixed",
        log_every_n_steps=10,
    )

    print("⚙  Starting SimCLR pre-training …")
    t0 = time.time()
    trainer.fit(module, loader)
    print(f"✓ Finished in {(time.time() - t0)/60:.1f} min")

    # ------------ save weights
    weights_path = Path(cfg.output["weights_path"]).expanduser()
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.encoder.state_dict(), weights_path)
    print("✓ Encoder weights saved to:", weights_path)


if __name__ == "__main__":
    main()
