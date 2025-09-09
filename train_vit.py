#!/usr/bin/env python3
"""
train_vit.py – single-flag launcher for CINECA ViT experiments.

Env-vars (export in run_train.slurm)
    DATA_ROOT              = dataset root (e.g. $WORK/lzanotto/data)
    MLFLOW_TRACKING_URI    = file store   (e.g. file:$WORK/lzanotto/mlruns)
    MLFLOW_EXPERIMENT_NAME = optional experiment name
Run:
    python train_vit.py --yaml configs/vit.yaml
"""

# ──────────────────────── std libs ─────────────────────────────────────────
import time
import argparse, os, sys, pathlib, random, re, glob
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, recall_score, precision_score, matthews_corrcoef
from pathlib import Path
import torch.backends.cudnn as cudnn
from monai.utils.misc import set_determinism
import shutil

# ──────────────────────── PYTHONPATH ───────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent.parent # Go up one level from /notebooks
sys.path.insert(0, str(PROJ_ROOT))

# ──────────────────────── project imports ─────────────────────────────────
from configs.ConfigLoader import ConfigLoader
from monai.data import Dataset, DataLoader
from monai.networks.nets import ViT
from utils.reproducibility_functions import set_global_seed
import utils.transformations_functions as tf

# ───────────────────── CLI (one flag) ──────────────────────────────────────
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="path inside configs/")
    return p.parse_args()

# ───────────────────── helpers ─────────────────────────────────────────────
def extract_patient_id(path: str) -> str:
    """Extracts patient ID from a file path."""
    m = re.search(r'(\d{4})', path)
    return m.group(1) if m else "UNKNOWN"

# ───────────────────── ViT Training & Validation Functions ─────────────────
def train_epoch_vit(model, loader, optimizer, loss_function, device):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        images_batch, labels_batch = batch["image"].to(device), batch["label"].to(device).long()
        optimizer.zero_grad()
        outputs, _ = model(images_batch)
        loss = loss_function(outputs, labels_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels_batch).sum().item()
        total += labels_batch.size(0)
    return epoch_loss / len(loader), correct / total

def val_epoch_vit(model, loader, loss_function, device):
    """
    Validate the ViT model for one epoch.
    This version is aligned with the standard `val_epoch` function.

    Args:
        model (torch.nn.Module): The ViT model to validate.
        loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        loss_function (torch.nn.Module): Loss function to use.
        device (torch.device): Device to run the validation on.

    Returns:
        A tuple containing: val_loss, accuracy, precision, recall, f1, balanced_acc, roc_auc, mcc.
    """
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probs = []
    num_valid_val_batches = 0

    with torch.no_grad():
        for batch in loader:
            images_batch, labels_batch = batch["image"].to(device), batch["label"].to(device).long()
            
            # ViT model returns (outputs, hidden_states), we only need outputs here
            outputs, _ = model(images_batch)

            loss = loss_function(outputs, labels_batch)

            if not (torch.isnan(loss).any() or torch.isinf(loss).any()):
                epoch_loss += loss.item()
                num_valid_val_batches += 1
            else:
                print(f"Warning: NaN/Inf loss encountered in val_epoch_vit, skipping batch.")

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, dim=1)

            correct += (predicted == labels_batch).sum().item()
            total += labels_batch.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            
            n_classes = outputs.shape[1]
            if n_classes == 2:
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                all_probs.append(probs.cpu().float().numpy())

    if num_valid_val_batches > 0:
        val_loss = epoch_loss / num_valid_val_batches
    else:
        print("Warning: No valid batches in val_epoch_vit; returning NaN loss.")
        val_loss = float('nan')

    if total == 0:
        print("Warning: No samples processed in val_epoch_vit.")
        return val_loss, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    accuracy = correct / total
    unique_labels = set(all_labels)
    avg_mode = 'binary' if len(unique_labels) == 2 else 'weighted'

    precision = precision_score(all_labels, all_predictions, average=avg_mode, zero_division=0)
    recall = recall_score(all_labels, all_predictions, average=avg_mode, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average=avg_mode, zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    mcc = matthews_corrcoef(all_labels, all_predictions)
    
    roc_auc = 0.0
    try:
        if len(unique_labels) == 2:
            if len(all_probs) == len(all_labels):
                roc_auc = roc_auc_score(all_labels, all_probs)
        elif len(unique_labels) > 2:
            all_probs_stacked = np.vstack(all_probs)
            roc_auc = roc_auc_score(all_labels, all_probs_stacked, multi_class='ovr', average='macro')
    except ValueError as e:
        print(f"Could not compute ROC AUC in val_epoch_vit: {e}")

    return val_loss, accuracy, precision, recall, f1, balanced_acc, roc_auc, mcc


# ───────────────────── Data & Model Setup ──────────────────────────────────
def get_data_directory(num_input_channels: int) -> Path:
    base = Path(os.environ.get("DATA_ROOT", ""))
    if not base.exists():
        raise EnvironmentError("DATA_ROOT not set or path does not exist.")
    sub = {3: "3c_MIP", 4: "4c_MIP"}.get(num_input_channels)
    if sub is None:
        raise ValueError("num_input_channels must be 3 or 4")
    data_dir = base / sub
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    return data_dir

def setup_vit_model(cfg: ConfigLoader, num_classes: int, device: torch.device):
    """Initializes the Vision Transformer model."""
    model = ViT(
        in_channels=cfg.get_model_input_channels(),
        img_size=cfg.model['img_size'],
        patch_size=cfg.model['patch_size'],
        hidden_size=cfg.model['hidden_size'],
        mlp_dim=cfg.model['mlp_dim'],
        num_layers=cfg.model['num_layers'],
        num_heads=cfg.model['num_heads'],
        classification=True,
        num_classes=num_classes,
        save_attn=True, # Crucial for attention maps
        spatial_dims=2
    )
    return model.to(device)

# ───────────────────── main ────────────────────────────────────────────────
def main():
    args = parse()
    start_time = time.time()

    # ---------- reproducibility -------------------------------------------
    SEED = 42
    set_global_seed(SEED)
    set_determinism(seed=SEED)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # ---------- configuration ---------------------------------------------
    cfg = ConfigLoader(str(PROJ_ROOT / args.yaml))
    print(f"Torch version: {torch.__version__}")
    print(f"Using configuration: {args.yaml}")

    class_names = cfg.get_class_names()
    num_channels = cfg.get_model_input_channels()
    num_workers = cfg.get_num_workers()
    batch_size = cfg.get_batch_size()
    DATA_ROOT = get_data_directory(num_channels)

    print(f"Class names: {class_names}")
    print(f"Number of channels: {num_channels}")
    print(f"Number of workers: {num_workers}")
    print(f"Batch size: {batch_size}")
    print(f"Data directory: {DATA_ROOT}")

    # ---------- dataset ----------------------------------------------------
    images, labels = [], []
    for lab, cname in enumerate(class_names):
        class_dir = DATA_ROOT / cname
        if not class_dir.exists():
            print(f"Warning: Directory not found for class '{cname}': {class_dir}")
            continue
        for p in class_dir.glob("*.tif"):
            if "vaso" in p.name.lower():
                continue
            images.append(str(p))
            labels.append(lab)

    if len(images) == 0:
        raise FileNotFoundError(f"No images found in {DATA_ROOT} for classes {class_names}. Check your dataset.")

    # Create a DataFrame for easier handling
    df = pd.DataFrame({
        "image_path": images,
        "label": labels,
        "patient_id": [extract_patient_id(p) for p in images]
    })

    # Get unique patients and their labels for stratified splitting
    pat_df = df.groupby("patient_id").first().reset_index()
    unique_pats = pat_df["patient_id"].values
    pat_labels = pat_df["label"].values

    # Split patients into train/val and test sets
    train_val_pats, test_pats, _, _ = train_test_split(
        unique_pats, pat_labels,
        test_size=cfg.get_test_ratio(),
        stratify=pat_labels,
        random_state=cfg.get_random_seed()
    )

    # Get the full data for each split
    train_val_df = df[df['patient_id'].isin(train_val_pats)]
    test_df = df[df['patient_id'].isin(test_pats)]

    print(f"Total patients: {len(unique_pats)}. Train/Val patients: {len(train_val_pats)}. Test patients: {len(test_pats)}.")
    print(f"Total images: {len(df)}. Train/Val images: {len(train_val_df)}. Test images: {len(test_df)}.")

    # ---------- transforms -------------------------------------------------
    train_transforms, val_transforms, test_transforms = tf.get_transforms(
        cfg, color_transforms=True
    )

    # ---------- device and model setup -------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA not available, running on CPU. This will be slow.")

    model = setup_vit_model(cfg, len(class_names), device)
    print(f"Model '{model.__class__.__name__}' initialized and moved to {device}.")

    # ---------- Next Steps -------------------------------------------------
    print("\nScript foundation is ready.")
    print("Next steps will involve implementing:")
    print("1. Nested cross-validation loop using the train_val_df.")
    print("2. Final evaluation on the test_df.")
    print("3. Attention map generation and saving.")
    print("4. Comprehensive MLflow logging.")

if __name__ == "__main__":
    main()