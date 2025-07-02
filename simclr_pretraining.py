#!/usr/bin/env python3
"""
simclr_pretrain.py ─ Self-supervised SimCLR backbone training for CINECA.

Run
----
srun … python simclr_pretrain.py --yaml configs/simclr_densenet_3c.yaml
"""

# ───────────────────── std-lib
from __future__ import annotations
import argparse, os, random, sys, time
from pathlib import Path
import numpy as np, torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from monai.utils.misc import set_determinism

# ───────────────────── project path
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

# ───────────────────── project imports
from configs.ConfigLoader import ConfigLoader
from classes.ModelManager import ModelManager

# ───────────────────── lightly / lightning
import pytorch_lightning as pl
from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead

# ───────────────────── CLI
def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="SimCLR YAML path")
    return p.parse_args()


# ───────────────────── LightningModule
class SimCLRModule(pl.LightningModule):
    """Lightning wrapper around an arbitrary CNN backbone + SimCLR projection head."""
    def __init__(
        self,
        backbone: nn.Module,
        proj_hidden_dim: int,
        proj_out_dim: int,
        temperature: float,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone

        # —— remove any classification head the backbone might have
        if hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()

        # —— detect encoder output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, *self.backbone.input_shape).to(self.device)
            feat_dim = self.backbone(dummy).flatten(1).shape[1]

        self.projection_head = SimCLRProjectionHead(
            input_dim=feat_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_out_dim,
        )
        self.criterion = NTXentLoss(temperature=temperature)

    # Lightning API ---------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x).flatten(1)
        return self.projection_head(z)

    def training_step(self, batch, _):
        (x0, x1), _, _ = batch           # lightly returns (views, label, fname)
        z0, z1 = self(x0), self(x1)
        loss   = self.criterion(z0, z1)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-6
        )


# ───────────────────── helpers
def reproducibility(seed: int = 42) -> None:
    random.seed(seed);  np.random.seed(seed);  torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed)
    cudnn.deterministic, cudnn.benchmark = True, False


# ───────────────────── main
def main() -> None:
    args = parse()
    cfg  = ConfigLoader(Path(args.yaml))

    reproducibility(42)

    # ---------- find unlabeled images ------------------------------------
    data_root = Path(os.environ["DATA_ROOT"])
    ssl_dir   = data_root / cfg.dataset["unlabeled_subdir"]
    if not ssl_dir.is_dir():
        raise FileNotFoundError(f"Unlabeled dir not found: {ssl_dir}")

    # lightly dataset ------------------------------------------------------
    transform = SimCLRTransform(
        input_size=cfg.data_augmentation["resize_spatial_size"][0],
        gaussian_blur=0.5,
        cj_strength=0.4,
        vf_prob=0.5, hf_prob=0.5, rr_prob=0.5
    )
    dataset = LightlyDataset(input_dir=str(ssl_dir), transform=transform)
    loader  = DataLoader(
        dataset,
        batch_size=cfg.data_loading["batch_size"],
        shuffle=True,  drop_last=True,
        num_workers=cfg.data_loading["num_workers"],
        pin_memory=True, persistent_workers=True,
    )
    print(f"Found {len(dataset):,} unlabeled images")

    # ---------- backbone --------------------------------------------------
    manager          = ModelManager(cfg, library=cfg.model["backbone_library"])
    backbone, device = manager.setup_model(
        num_classes=cfg.model["proj_hidden_dim"],  # dummy head, removed later
        pretrained_weights=None,
    )
    backbone.input_shape = (
        cfg.model["in_channels"],
        *cfg.data_augmentation["resize_spatial_size"],
    )
    backbone = backbone.to(device)

    # ---------- lightning -------------------------------------------------
    module = SimCLRModule(
        backbone=backbone,
        proj_hidden_dim=cfg.model["proj_hidden_dim"],
        proj_out_dim   =cfg.model["proj_output_dim"],
        temperature    =cfg.training["temperature"],
        lr             =cfg.training["lr"],
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        max_epochs=cfg.training["num_epochs"],
        log_every_n_steps=10,
    )

    print("⚙  SimCLR pre-training …");  t0 = time.time()
    trainer.fit(module, loader)
    print(f"✓ finished in {(time.time() - t0)/60:.1f} min")

    # ---------- save encoder weights -------------------------------------
    w_path = Path(cfg.output["weights_path"]).expanduser()
    w_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.encoder.state_dict(), w_path)
    print("✓ Encoder saved:", w_path)


if __name__ == "__main__":
    main()
