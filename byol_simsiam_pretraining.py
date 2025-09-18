#!/usr/bin/env python3
"""
byol_simsiam_pretraining.py ─ Self-supervised BYOL/SimSiam backbone training for CINECA.

Run
----
srun … python byol_simsiam_pretraining.py --yaml configs/byol_resnet18_3c.yaml
"""

# ───────────────────── std-lib
from __future__ import annotations
import argparse
import os
import random
import sys
import time
import glob
from pathlib import Path

# ───────────────────── scientific computing
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from numpy import stack

# ───────────────────── monai
from monai.utils.misc import set_determinism
from monai.transforms.croppad.array import RandSpatialCropSamples
from monai.data.grid_dataset import PatchDataset
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    Resized,
    ScaleIntensityd,
    RandFlip,                    
    RandRotate90,  
    EnsureTyped,              
)
from monai.transforms.utility.dictionary import LambdaD

# ───────────────────── torchvision
from torchvision import models

# ───────────────────── lightly / byol
from byol_pytorch import BYOL

# ───────────────────── project path
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

# ───────────────────── project imports
from configs.ConfigLoader import ConfigLoader
from classes.ModelManager import ModelManager
from classes.CustomTiffFileReader import CustomTiffFileReader
from classes.PrintShapeTransform import PrintShapeTransform
from classes.MonaiDictTransformAdapter import MonaiDictTransformAdapter
from utils import transformations_functions as tf
from utils.transformations_functions import from_GBR_to_RGB
from utils.ssl.byol_transformation_functions import get_byol_transforms_dict
from utils.train_functions import make_unlabeled_loader


# ───────────────────── CLI
def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="BYOL YAML path")
    return p.parse_args()


# ───────────────────── helpers
def reproducibility(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed)
    cudnn.deterministic, cudnn.benchmark = True, False


def dict_collate_fn(batch):
    """Custom collate function that returns dictionaries for SSL training."""
    stacked_images = torch.stack(batch, 0)
    dummy_labels = torch.zeros(stacked_images.size(0), dtype=torch.long)
    return {"image": stacked_images, "label": dummy_labels}


# ───────────────────── main
def main() -> None:
    args = parse()
    cfg = ConfigLoader(Path(args.yaml))
    cfg.set_freezed_layer_index(None)
    
    # ---------- yaml parsing/param extraction + reproducibility ---------------
    num_classes = cfg.get_num_classes()
    class_names = cfg.get_class_names()
    color_transforms = cfg.get_use_color_transforms()
    pretrained_weights = cfg.get_pretrained_weights() if cfg.get_transfer_learning() else None
    model_library = cfg.get_model_library()
    BATCH_SIZE = cfg.get_batch_size()

    reproducibility(42)

    # ---------- find unlabeled image folder ------------------------------------
    data_root = Path(os.environ["DATA_ROOT"])    
    ssl_dir = data_root / cfg.dataset["unlabeled_subdir"]
    
    if not ssl_dir.is_dir():
        raise FileNotFoundError(f"Unlabeled dir not found: {ssl_dir}")
    
    # -------- load a model and its transformations
    train_transforms, val_transforms, test_transforms = tf.get_transforms(cfg, color_transforms=color_transforms)
    model_manager = ModelManager(cfg, library=model_library)
    
    # Verify the number of unique labels in the dataset
    print(f"Number of classes in the dataset: {num_classes}")
    
    # Ensure the model's output matches the number of classes
    model, device = model_manager.setup_model(num_classes=num_classes, pretrained_weights=pretrained_weights)
    
    # -------- extract images paths
    ssl_images_paths = glob.glob(os.path.join(ssl_dir, "*.tif"))
    ssl_images_paths_np = np.array(ssl_images_paths)
    print("Number of images in ALL folder:", len(ssl_images_paths))

    # dataset /dataloaders ------------------------------------------------------
    # a) 1‑per‑image dicts with a path under key "image"
    dataset_ssl = [{"image": path} for path in ssl_images_paths_np]
    
    # b) image‑level transforms 
    img_tf = Compose([
        CustomTiffFileReader(keys=["image"]),
        EnsureTyped(keys=["image"], data_type="tensor", dtype=torch.float32),
        LambdaD(keys="image", func=from_GBR_to_RGB),
        Resized(keys="image", spatial_size=(512, 512)),
        ScaleIntensityd(keys="image"),
    ])

    # c) patch sampler : 8 random 256×256 crops — pure (non‑dict) version
    sampler = RandSpatialCropSamples(
        roi_size=(256, 256),
        num_samples=8,
        random_center=True,
        random_size=False,
    )

    # d) patch‑level augments
    patch_tf = Compose([
        RandFlip(spatial_axis=0, prob=0.5),  
        RandFlip(spatial_axis=1, prob=0.5),
        RandRotate90(max_k=3, prob=0.5),
    ])

    # e) PatchDataset – extract the Tensor, sample, then augment
    patch_ds = PatchDataset(
        data=dataset_ssl,
        samples_per_image=8,
        patch_func=lambda d: sampler(img_tf(d)["image"]),
        transform=patch_tf,
    )
    
    # Testing purposes
    first_batch = next(iter(patch_ds))
    print(f"Batch type: {type(first_batch)}")
    print(f"Images shape: {first_batch.shape}")
    
    base_transforms = train_transforms

    # Create the patch-based DataLoader
    train_loader_patch_ds = DataLoader(
        patch_ds,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
        collate_fn=dict_collate_fn
    )

    # Create the full-image DataLoader
    train_loader_full_ds = make_unlabeled_loader(
        image_paths=ssl_images_paths_np,
        transforms=base_transforms,
        cfg=cfg,
        shuffle=True,
    )
    
    # Loader selection
    if cfg.data_loading["loader_type"] == "patch":
        train_loader = train_loader_patch_ds
    elif cfg.data_loading["loader_type"] == "full_image":
        train_loader = train_loader_full_ds 
    
    # Check the type of the train_loader and shape of the returned batches
    print(type(train_loader))
    for batch in train_loader:
        print(f"Batch type: {type(batch)}")
        print(f"Batch keys: {batch.keys()}")
        print(f"Images shape: {batch['image'].shape}")
        assert batch['image'].shape == torch.Size([BATCH_SIZE, 3, 256, 256]), "Batch size mismatch!"
        break

    # ----------- define BYOL AUGMENTATIONS -------------
    normalize_intensity = False
    monai_view1_dict_transform, monai_view2_dict_transform = get_byol_transforms_dict(cfg, normalize_intensity=normalize_intensity)
    
    # Wrap the MONAI Compose objects
    wrapped_monai_view1_transform = MonaiDictTransformAdapter(monai_view1_dict_transform, image_key="image")
    wrapped_monai_view2_transform = MonaiDictTransformAdapter(monai_view2_dict_transform, image_key="image")

    # Determine the device
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Move the wrapped MONAI transforms to the device
    augment_fn = wrapped_monai_view1_transform.to(current_device)
    augment_fn2 = wrapped_monai_view2_transform.to(current_device)
    
    # ---------- backbone setup --------------------------------------------------
    model, device = model_manager.setup_model(num_classes=num_classes, pretrained_weights=pretrained_weights)
    
    simsiam_flag = cfg.training["simsiam_flag"]
    print(f"SimSiam flag is set to: {simsiam_flag}")

    if not simsiam_flag:
        print("BYOL")
        learner = BYOL(
            net=model,
            image_size=256,
            hidden_layer='avgpool',
            augment_fn=augment_fn,
            augment_fn2=augment_fn2,
            projection_size=256,
            projection_hidden_size=4096,
            moving_average_decay=0.99
        )
    else:
        print("SimSiam")
        learner = BYOL(
            net=model,
            image_size=256,
            hidden_layer=-2,
            augment_fn=augment_fn,
            augment_fn2=augment_fn2,
            use_momentum=False,
        ) 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    learner = learner.to(device)
    optimizer = optim.Adam(learner.parameters(), lr=1e-4)
    scaler = GradScaler()  # Fixed: removed "cuda" argument

    start_time = time.time()
    
    # Encoder Pre-training loop
    patience = 25          
    best_train_loss = float("inf")
    epochs_no_improve = 0
    best_state_dict = None

    num_epochs = 120
    for epoch in range(1, num_epochs + 1):
        learner.train()
        total_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device)

            with torch.autocast(device_type=device, dtype=torch.float16):
                loss = learner(imgs)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if not simsiam_flag:
                learner.update_moving_average()

            total_loss += loss.item() * imgs.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:3d} | train loss {train_loss:.4f}")

        # Early-stopping on training loss
        if train_loss < best_train_loss - 1e-6:
            best_train_loss = train_loss
            epochs_no_improve = 0
            best_state_dict = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n Stopping: train loss hasn't improved for {patience} epochs.")
                break

    # Save best encoder weights
    if best_state_dict is not None:
        learner.net.load_state_dict(best_state_dict)

    # Create dynamic filename based on configuration
    base_name = Path(cfg.output["weights_path"]).stem  # Gets filename without extension
    base_dir = Path(cfg.output["weights_path"]).parent # Gets the directory of the weights path
    extension = Path(cfg.output["weights_path"]).suffix # Gets the file extension

    # Get configuration details for filename
    method = "simsiam" if simsiam_flag else "byol"
    loader_type = cfg.data_loading["loader_type"]  # "patch" or "full_image"

    # Determine dataset type based on class names
    if "MSA" in class_names and "MSA-P" not in class_names:
        dataset_type = "MSA_VS_PD"
    elif "MSA-P" in class_names:
        dataset_type = "MSAP_VS_PD"
    else:
        raise ValueError("Unknown class names in config. Please specify the correct dataset type.")

    # Create new filename with method, dataset type, and loader type
    new_filename = f"{base_name}_{method}_{dataset_type}_{loader_type}{extension}"
    w_path = base_dir / new_filename

    w_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(learner.state_dict(), w_path)
    print(f"✓ Encoder weights saved to: {w_path}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
