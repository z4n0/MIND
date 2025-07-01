#!/usr/bin/env python3
"""
simclr_pretrain.py  –  self-supervised SimCLR encoder training.

Run
----
sbatch … python simclr_pretrain.py --yaml configs/simclr_densenet_3c.yaml
"""

# ───────────────── std libs
from json import encoder
import argparse, os, sys, random, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from monai.utils.misc import set_determinism

# ───────────────── project path
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

# ───────────────── project imports
from configs.ConfigLoader import ConfigLoader
from classes.ModelManager import ModelManager
import utils.transformations_functions as tf

# lightly (SimCLR utilities)
import time
import torchvision.transforms.v2 as T
from lightly.data import LightlyDataset
from lightly.data.collate import SimCLRCollateFunction
import pytorch_lightning as pl
from lightly.loss import NTXentLoss
import torch.nn as nn
from pytorch_lightning import Trainer
from lightly.transforms import SimCLRTransform
from lightly.models.modules import SimCLRProjectionHead
from utils.train_functions import LinearProbeHead 

# ───────────────── CLI
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="path to YAML with SimCLR block")
    return p.parse_args()


# ───────────────── helper
def list_unlabeled_images(root: Path, class_names: list[str] = None) -> list[Path]:
    mip_dir = root / "PRETRAINING_MSA_VS_PD" if "MSA-P" not in class_names else root / "PRETRAINING_MSAP_VS_PD"
    if subdir:
        img_dir = mip_dir / subdir
        paths   = list(img_dir.glob("*.tif"))
    else:  # scan every class folder
        paths = [p for p in mip_dir.rglob("*.tif") if p.is_file()]
    if not paths:
        raise FileNotFoundError(f"No .tif found under {mip_dir}")
    return paths


# ───────────────── SimCLR LightningModule
class SimCLRModule(pl.LightningModule):
    def __init__(self, backbone, emb_dim=64, proj_hidden_dim=512, temperature=0.07, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.temperature = temperature
        self.lr = lr
        
        ##STRIP ENCODER LINEAR PROBE HEAD
        backbone_outdim = self._get_projection_head_input_dim()
        # print(self.backbone)
        self._remove_linear_probe_head()
        
        self.projection_head = SimCLRProjectionHead(
            input_dim=backbone_outdim,
            hidden_dim=proj_hidden_dim,
            output_dim=emb_dim,
        )

        self.criterion = NTXentLoss(temperature=self.temperature)
    
    @property
    def encoder(self):
        return self.backbone
        
    def _remove_linear_probe_head(self):
        # if isinstance(self.backbone.fc, LinearProbeHead):
        # print(self.backbone.fc.__class__.__name__)
        if hasattr(self.backbone, "fc"):
            if self.backbone.fc.__class__.__name__ == "LinearProbeHead":
                self.backbone.fc = nn.Identity()
            
        elif hasattr(self.backbone, "classifier"):
            if self.backbone.classifier.__class__.__name__ == "LinearProbeHead":
                self.backbone.classifier = nn.Identity()
        else:
            raise RuntimeError(
                    "Could not remove linear probe head"
                )
    
    def _get_projection_head_input_dim(self):
        """
        Return the dimensionality of the **feature vector that comes right
        before the classifier** (the value you must feed into a projection
        or linear-probe head).
        
        """
        print(f"backbone: {self.backbone}")
        print(f"hasattr(self.backbone, 'fc'): {hasattr(self.backbone, 'fc')}")
        print(f"hasattr(self.backbone, 'classifier'): {hasattr(self.backbone, 'classifier')}")
        print(f"hasattr(self.backbone, 'in_features'): {hasattr(self.backbone.fc, 'in_features')}")
        
        if hasattr(self.backbone, "fc") and hasattr(self.backbone.fc, "in_dim"):
            print(f"backbone.fc.in_features: {self.backbone.fc.in_dim}")
            if self.backbone.fc.__class__.__name__ == "LinearProbeHead":
                return self.backbone.fc.in_dim
            # return self.backbone.fc.in_features
        elif hasattr(self.backbone, "classifier"):
            if self.backbone.classifier.__class__.__name__ == "LinearProbeHead":
                return self.backbone.classifier.in_dim
            
        print(f" backbone input dim: {self.backbone.in_dim}")
        return self.backbone.in_dim
        
    def forward(self, x):
        feats = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(feats)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1),_,_ = batch #x0,x1 are tensor of shape (B,C,H,W)
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # optim = torch.optim.SGD(self.parameters(), lr=self.lr) 
        optim = torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=1e-6)
        return optim



STABLE_BACKBONE_PATH = PROJ_ROOT + f"simclr_encoder_weights_{num_input_channels}c_{encoder.__class__.__name__}.pth"
# ───────────────── main
def main():
    args = parse()
    cfg  = ConfigLoader(PROJ_ROOT / args.yaml)
    
    print(f"Using configuration: {args.yaml}")
    class_names        = cfg.get_class_names()
    print(f"Class names: {class_names}")
    if class_names is None:
        raise ValueError("class_names returned None. Please check your configuration.")
    num_channels       = cfg.get_model_input_channels()
    pretrained_weights = cfg.get_pretrained_weights()
    num_epochs         = cfg.get_num_epochs()
    num_workers        = cfg.get_num_workers()
    batch_size         = cfg.get_batch_size()
    num_folds          = cfg.get_num_folds()
    model_library      = cfg.get_model_library()

    print(f"Number of channels: {num_channels}")
    print(f"Pretrained weights: {pretrained_weights}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Number of workers: {num_workers}")
    print(f"Batch size: {batch_size}")
    print(f"Number of folds: {num_folds}")
    print(f"Model library: {model_library}")

    # reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    set_determinism(SEED)
    cudnn.deterministic, cudnn.benchmark = True, False

    # images ----------------------------------------------------------------
    data_root = Path(os.environ["DATA_ROOT"])
    img_paths = list_unlabeled_images(
        data_root, class_names=class_names
    )
    
    print(f"Found {len(img_paths):,} unlabeled images")

    # dataset / dataloader --------------------------------------------------
    transform = SimCLRTransform(
        input_size=256,
        gaussian_blur=0.5,  # probability of applying blur
        random_gray_scale=0.2,  # probability of grayscale
        cj_strength=0.4,  # color jitter strength
        vf_prob=0.5,
        hf_prob=0.5,
        rr_prob=0.5,
        rr_degrees=180,
    )
    
    dataset = LightlyDataset(
        input_dir=ssl_input_dir,
        transform=transform
    )

    # collate_fn = SimCLRCollateFunction(input_size=256)
    print("len of dataset")
    print(len(dataset)) 

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = len(dataset),
        shuffle=True,
        # collate_fn=collate_fn,
        drop_last=True,
        num_workers=2
    )

    # backbone architecture definition --------------------------------------------------------------
    encoder, device = model_manager.setup_model(num_classes=num_classes, pretrained_weights=pretrained_weights)
    # backbone, device = model_manager.setup_model(
    #     num_classes=mdl_cfg["proj_hidden_dim"],  # dummy
    #     pretrained_weights=None,
    #     remove_linear_head=True,                 # keep pure feature extractor
    # )
    
    backbone = backbone.to(device)

    # detect encoder output dim
    with torch.no_grad():
        dummy  = torch.zeros(1, mdl_cfg["in_channels"], 256, 256).to(device)
        feat_dim = backbone(dummy).flatten(1).shape[1]

    # lightning module + trainer -------------------------------------------
    module = SimCLRModule(
        backbone=backbone,
        in_dim=feat_dim,
        proj_hidden_dim=mdl_cfg["proj_hidden_dim"],
        proj_out_dim=mdl_cfg["proj_output_dim"],
        lr=tr_cfg["lr"],
        temperature=tr_cfg["temperature"],
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=tr_cfg["num_epochs"],
        precision="16-mixed",
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    print("⚙  Starting SimCLR pre-training …")
    t0 = time.time()
    trainer.fit(module, loader)
    print(f"✓ Finished in {(time.time()-t0)/60:.1f} minutes")

    # save encoder ----------------------------------------------------------
    weights_path = Path(out_cfg["weights_path"]).expanduser()
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.encoder.state_dict(), weights_path)
    print(f"✓ Encoder weights saved to: {weights_path}")

if __name__ == "__main__":
    main()
