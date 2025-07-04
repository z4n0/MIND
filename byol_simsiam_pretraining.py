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
    p.add_argument("--yaml", required=True, help="BYOL YAML path")
    return p.parse_args()


# ───────────────────── LightningModule

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
    cfg.set_freezed_layer_index(None)
    
    # ---------- yaml parsing/param extraction + reproducibility ---------------
    # num_input_channels = cfg.get_model_input_channels()
    num_classes = cfg.get_num_classes()
    class_names = cfg.get_class_names()
    color_transforms = cfg.data_augmentation.get("use_color_transforms", False)
    pretrained_weights = cfg.get_pretrained_weights() if cfg.get_transfer_learning() else None
    model_library = cfg.get_model_library()
    BATCH_SIZE = cfg.get_batch_size() # Define your batch size

    reproducibility(42)

    # ---------- find unlabeled image folder ------------------------------------
    data_root = Path(os.environ["DATA_ROOT"])
    if "MSA" in class_names:
        ssl_dir = data_root / "PRETRAINING_MSA_VS_PD"
    elif "MSA-P" in class_names:
        ssl_dir = data_root / "PRETRAINING_MSAP_VS_PD"
    else:
        raise ValueError("Unknown class names in config. Please specify the correct dataset path.")
    if not ssl_dir.is_dir():
        raise FileNotFoundError(f"Unlabeled dir not found: {ssl_dir}")
    
    # -------- load a model and its transformations
    from utils import transformations_functions as tf
    train_transforms, val_transforms, test_transforms = tf.get_transforms(cfg,color_transforms=color_transforms)
    model_manager = ModelManager(cfg, library=model_library)
    # Verify the number of unique labels in the dataset
    print(f"Number of classes in the dataset: {num_classes}")
    # Ensure the model's output matches the number of classes
    model, device = model_manager.setup_model(num_classes=num_classes, pretrained_weights=pretrained_weights)
    
    # -------- extract images paths
    import glob
    ssl_images_paths = glob.glob(os.path.join(ssl_dir, "*.tif"))
    ssl_images_paths_np = np.array(ssl_images_paths)
    print("Number of images in ALL folder:", len(ssl_images_paths))

    # dataset /dataloaders ------------------------------------------------------
    # ------------------------------------------------------------------
    # a) 1‑per‑image dicts with a path under key "image"
    # ------------------------------------------------------------------
    dataset_ssl = [{"image": path} for path in ssl_images_paths_np]
    
    from monai.transforms.croppad.array import RandSpatialCropSamples  # pure sampler
    from monai.data.grid_dataset import PatchDataset
    import matplotlib.pyplot as plt
    import torch, itertools, math, numpy as np

    # b) image‑level transforms 
    # ------------------------------------------------------------------
    from monai.transforms import (
        Compose,
        # EnsureChannelFirstd,
        Resized,
        ScaleIntensityd,
        RandFlip,                    
        RandRotate90,  
        EnsureTyped,              
    )
    from monai.transforms.utility.dictionary import LambdaD
    from utils.transformations_functions import from_GBR_to_RGB
    from classes.CustomTiffFileReader import CustomTiffFileReader

    img_tf = Compose([
        CustomTiffFileReader(keys=["image"]), # loads the image from the path as a numpy array (C,H,W) in GBR format
        EnsureTyped(keys=["image"], data_type="tensor", dtype=torch.float32), # Ensure image is a tensor
        LambdaD(keys="image", func=from_GBR_to_RGB),
        # EnsureChannelFirstd(keys="image"),
        Resized(keys="image", spatial_size=(512, 512)),
        ScaleIntensityd(keys="image"),
    ])

    # ------------------------------------------------------------------
    # c) patch sampler : 8 random 256×256 crops — pure (non‑dict) version
    # ------------------------------------------------------------------
    sampler = RandSpatialCropSamples(
        roi_size=(256, 256),
        num_samples=8,
        random_center=True,
        random_size=False,
    )

    # ------------------------------------------------------------------
    # d) patch‑level augments
    # ------------------------------------------------------------------
    patch_tf = Compose([
        # RandChannelShift(max_shift=15, prob=0.70),
        RandFlip(spatial_axis=0, prob=0.5),  
        RandFlip(spatial_axis=1, prob=0.5),
        RandRotate90(max_k=3, prob=0.5),
    ])

    # ------------------------------------------------------------------
    # e) PatchDataset – extract the Tensor, sample, then augment
    # ------------------------------------------------------------------
    patch_ds = PatchDataset(
        data=dataset_ssl,                                             # use named arg `data`
        # dataset=img_ds,                                             # use named arg `dataset`
        samples_per_image=8,                                        # produces 8 patches per image
        patch_func=lambda d: sampler(img_tf(d)["image"]),           # <-- pull out d["image"]
        transform=patch_tf,                                         # augment each pure Tensor
    )
    #testing porpouses
    first_batch = next(iter(patch_ds))
    print(f"Batch type: {type(first_batch)}") #Batch type: <class 'monai.data.meta_tensor.MetaTensor'>
    print(f"Images shape: {first_batch.shape}") # Batch shape: torch.Size([8, 3, 256, 256])
    
    from monai.data import Dataset, DataLoader
    from numpy import stack
    from classes.PrintShapeTransform import PrintShapeTransform
    from classes.CustomTiffFileReader import CustomTiffFileReader

    base_transforms = train_transforms

    # Wrap your patch_ds with a custom collate function that returns dictionaries
    # it tells the DataLoader how to combine the individual patches into a batch of dictionaries
    # alternative to this is just to not use dictionary transform but array transforms 
    def dict_collate_fn(batch):
        #stack the images in the batch and return a dictionary
        stacked_images = torch.stack(batch, 0)
        dummy_labels = torch.zeros(stacked_images.size(0), dtype=torch.long) # Dummy label, not used in SSL
        return {"image": stacked_images, "label": dummy_labels} # This assumes each item in the batch is a dictionary with an "image" key.


    # Create the patch-based DataLoader
    train_loader_patch_ds = DataLoader(
        patch_ds,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
        collate_fn=dict_collate_fn  #tells the DataLoader how to combine the individual patches into a batch of dictionaries
    )


    # Create the full-image DataLoader
    from utils.train_functions import make_unlabeled_loader
    
    train_loader_full_ds = make_unlabeled_loader(
        image_paths=ssl_images_paths_np,
        transforms=base_transforms,
        cfg=cfg,
        shuffle=True,
    )
    
    #!TODO remember to put this section in the yaml
    ##--------loader selection ------------------
    if cfg.data_loading["loader_type"] == "patch":
        train_loader = train_loader_patch_ds
    elif cfg.data_loading["loader_type"] == "full_image":
        train_loader = train_loader_full_ds 
    
    #---------------check ----------------------
    #should give 
    # <class 'monai.data.dataloader.DataLoader'>
    # Batch type: <class 'dict'>
    # Batch keys: dict_keys(['image', 'label'])
    # Images shape: torch.Size([16, 3, 256, 256])
    # Check the type of the train_loader and shape of the returned batches
    print(type(train_loader))  # Should be a DataLoader
    for batch in train_loader:
        print(f"Batch type: {type(batch)}")  # Should be a dictionary
        print(f"Batch keys: {batch.keys()}")  # Should contain 'image' and possibly 'label'
        print(f"Images shape: {batch['image'].shape}")  # Should be [BATCH_SIZE, C, H, W]
        assert batch['image'].shape == torch.Size([BATCH_SIZE, 3, 256, 256]), "Batch size mismatch!"
        break  # Just show the first batch for now
    
    # ----------- define BYOL AUGMENTATIONS -------------
    from utils.ssl.byol_transformation_functions import get_byol_transforms_dict
    from classes.MonaiDictTransformAdapter import MonaiDictTransformAdapter

    # Get the MONAI Compose objects for dictionary-based transforms
    normalize_intensity = False  # Set this based on your needs
    monai_view1_dict_transform, monai_view2_dict_transform = get_byol_transforms_dict(cfg, normalize_intensity=normalize_intensity)
    type_of_transform = "monai_dict"
    # Wrap the MONAI Compose objects
    # You might need to inspect monai_view1_dict_transform to confirm the exact image key if it's not "image"
    # For example, if your get_byol_transforms_dict creates transforms that use a different key, adjust "image" accordingly.
    wrapped_monai_view1_transform = MonaiDictTransformAdapter(monai_view1_dict_transform, image_key="image")
    wrapped_monai_view2_transform = MonaiDictTransformAdapter(monai_view2_dict_transform, image_key="image")

    # Determine the device
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Move the wrapped MONAI transforms to the device
    augment_fn = wrapped_monai_view1_transform.to(current_device)
    augment_fn2 = wrapped_monai_view2_transform.to(current_device)
    
    # ---------- backbone  --------------------------------------------------
    from byol_pytorch import BYOL
    from torchvision import models
    import time
    # resnet = models.resnet50(weights = None)  # or pretrained=True if you like
    model, device = model_manager.setup_model(num_classes=num_classes, pretrained_weights=pretrained_weights)
    #usage Simply plugin your neural network, specifying 
    # (1) the image dimensions
    # (2) the name (or index) of the hidden layer, 
    # whose output is used as the latent representati
    # this library will use the augmentations from the SimCLR paper (which is also used in the BYOL paper).
    # learner = BYOL(
    #     resnet,
    #     image_size = 256,          # the height & width of your input images
    #     augment_fn = custom_augment_fn,
    #     augment_fn2 = custom_augment_fn,  # or define a second pipeline if desired
    #     hidden_layer = 'avgpool',  # output layer used for representation
    #     # projection_size = 256,     # defaults to 256
    #     # projection_hidden_size = 4096,
    #     # moving_average_decay = 0.99
    # )
    
    simsiam_flag = cfg.training["simsiam_flag"]
    print(f"SimSiam flag is set to: {simsiam_flag}")

    if not simsiam_flag:
        print("BYOL")
        learner = BYOL(
            net=model,
            image_size=256,           # must match your input size
            hidden_layer='avgpool',   # layer to extract embeddings from in ResNet
            augment_fn=augment_fn,   # our custom random transforms
            augment_fn2=augment_fn2,  # can define a second if you want different augmentations
            projection_size=256,
            projection_hidden_size=4096,
            moving_average_decay=0.99
        )
    else:
        print("SimSiam")
        learner = BYOL(
            net=model,
            image_size=256,  # must match your input size
            hidden_layer=-2, #or "avgpool"   # layer to extract embeddings from in ResNet
            augment_fn=augment_fn,   # our custom random transforms
            augment_fn2=augment_fn2,  # can define a second if you want different augmentations
            # projection_size=256,
            # projection_hidden_size=4096,
            # moving_average_decay=0.99
            use_momentum=False, # This is the main difference from byol
        ) 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    learner = learner.to(device)
    optimizer = optim.Adam(learner.parameters(), lr=1e-4)
    scaler = torch.GradScaler("cuda")  # for mixed precision training

    start_time = time.time()
    #Encoder Pre-training loop
    patience          = 25          
    best_train_loss   = float("inf")
    epochs_no_improve = 0
    best_state_dict   = None

    num_epochs = 120
    for epoch in range(1, num_epochs + 1):
        learner.train()
        total_loss = 0.0
        for batch in train_loader:
            #load the images from the batch ()
            imgs = batch["image"].to(device)
            # print(f"Batch shape: {imgs.shape}")  # Should be [batch_size, 3, 256, 256]

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

        # ---------- early-stopping on training loss ----------
        if train_loss < best_train_loss - 1e-6:     # 1e-6 = small min_delta
            best_train_loss   = train_loss
            epochs_no_improve = 0
            best_state_dict   = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n Stopping: train loss hasn’t improved for {patience} epochs.")
                break

    # -------------- save best encoder weights -------------------------------
    if best_state_dict is not None:
        learner.net.load_state_dict(best_state_dict)

    w_path = Path(cfg.output["weights_path"]).expanduser()
    w_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(learner.net.state_dict(), w_path)
    print(f"✓ Encoder weights saved to: {w_path}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    

if __name__ == "__main__":
    main()
