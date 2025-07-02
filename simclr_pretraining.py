#!/usr/bin/env python3
"""
simclr_pretrain.py ─ Self-supervised SimCLR backbone training for CINECA.

Run
----
srun … python simclr_pretrain.py --yaml configs/simclr_d121
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
    def __init__(self, backbone, proj_hidden_dim=512, proj_out_dim=128, temperature=0.07, lr=3e-4):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.temperature = temperature
        self.lr = lr
        
        # Strip encoder linear probe head and get feature dimension
        backbone_outdim = self._get_projection_head_input_dim()
        self._remove_linear_probe_head()
        
        self.projection_head = SimCLRProjectionHead(
            input_dim=backbone_outdim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_out_dim,
        )

        self.criterion = NTXentLoss(temperature=self.temperature)
    
    @property
    def encoder(self):
        return self.backbone
        
    def _remove_linear_probe_head(self):
        if hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()
        else:
            # For torchvision models, this is normal - they might not have these attributes
            pass
    
    def _get_projection_head_input_dim(self):
        """
        Get the feature dimension without doing a forward pass.
        Uses known architectures to avoid device mismatch issues.
        """
        # Check if we have a custom LinearProbeHead with stored dimensions
        if hasattr(self.backbone, "fc") and hasattr(self.backbone.fc, "in_dim"):
            if self.backbone.fc.__class__.__name__ == "LinearProbeHead":
                return self.backbone.fc.in_dim
        elif hasattr(self.backbone, "classifier") and hasattr(self.backbone.classifier, "in_dim"):
            if self.backbone.classifier.__class__.__name__ == "LinearProbeHead":
                return self.backbone.classifier.in_dim
        
        # Check if ModelManager set in_dim attribute
        if hasattr(self.backbone, "in_dim"):
            return self.backbone.in_dim
            
        # Fallback: Use known architecture dimensions
        model_name = self.backbone.__class__.__name__.lower()
        
        if "densenet121" in model_name:
            return 1024
        elif "densenet169" in model_name:
            return 1664
        elif "densenet201" in model_name:
            return 1920
        elif "resnet18" in model_name:
            return 512
        elif "resnet34" in model_name:
            return 512
        elif "resnet50" in model_name:
            return 2048
        elif "resnet101" in model_name:
            return 2048
        elif "resnet152" in model_name:
            return 2048
        else:
            # Last resort: inspect the classifier/fc layer
            if hasattr(self.backbone, "classifier") and hasattr(self.backbone.classifier, "in_features"):
                return self.backbone.classifier.in_features
            elif hasattr(self.backbone, "fc") and hasattr(self.backbone.fc, "in_features"):
                return self.backbone.fc.in_features
            else:
                # Final fallback - assume DenseNet121 since that's what you're using
                print("Warning: Could not determine feature dimension, assuming DenseNet121 (1024)")
                return 1024
        
    def forward(self, x):
        feats = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(feats)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch  # x0,x1 are tensor of shape (B,C,H,W)
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)
        return optim


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
        proj_out_dim=cfg.model["proj_output_dim"],
        temperature=cfg.training["temperature"],
        lr=float(cfg.training["lr"]),  # Ensure it's a float
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
