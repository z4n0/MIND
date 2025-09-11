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
PROJ_ROOT = Path(__file__).resolve().parent # Adjusted path assuming script is in root
sys.path.insert(0, str(PROJ_ROOT))

# ──────────────────────── project imports ─────────────────────────────────
from classes.ModelManager import ModelManager
from configs.ConfigLoader import ConfigLoader
# from monai.data import Dataset, DataLoader
# from monai.networks.nets import ViT
from utils.reproducibility_functions import set_global_seed
import utils.transformations_functions as tf
from classes.NestedCVStratifiedByPatient import NestedCVStratifiedByPatient
from utils.mlflow_functions import log_SSL_run_to_mlflow

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

def best_fold_idx(results, metric="test_balanced_acc") -> int:
    return int(np.argmax([r[metric] for r in results]))

# ───────────────────── ViT Training & Validation Functions ─────────────────
def train_epoch_vit(model, loader, optimizer, loss_function, device, print_batch_stats=False):
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

        if print_batch_stats:
            print(f"Batch loss: {loss.item():.4f}, Accuracy: {correct / total:.4f}")
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

# ───────────────────── main ────────────────────────────────────────────────
def main():
    args = parse()
    start_time = time.time()

    # ---------- reproducibility -------------------------------------------
    SEED = 42
    set_global_seed(SEED)
    set_determinism(seed=SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # ---------- configuration ---------------------------------------------
    cfg = ConfigLoader(str(PROJ_ROOT / args.yaml))
    print(f"Torch version: {torch.__version__}")
    print(f"Using configuration: {args.yaml}")
    print(f"Torch version: {torch.__version__}")
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
    DATA_ROOT          = get_data_directory(num_channels)

    print(f"Number of channels: {num_channels}")
    print(f"Pretrained weights: {pretrained_weights}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Number of workers: {num_workers}")
    print(f"Batch size: {batch_size}")
    print(f"Number of folds: {num_folds}")
    print(f"Model library: {model_library}")
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
    
    images, labels = np.array(images), np.array(labels)

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

    # ---------- transforms -------------------------------------------------
    train_transforms, val_transforms, test_transforms = tf.get_transforms(
        cfg, color_transforms=True
    )

    # ---------- device setup -----------------------------------------------
    model_manager = ModelManager(cfg, library=cfg.get_model_library())
    model, device = model_manager.setup_model(len(class_names))
    if device.type != "cuda":
        raise RuntimeError(
            "This script is intended to run on a CUDA-enabled GPU. "
            "Please ensure you have a compatible GPU and the necessary drivers installed."
        )
    # ---------- experiment -------------------------------------------------
    job_id = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    run_tag = f"{Path(args.yaml).stem}_{cfg.get_model_input_channels()}c"
    RUN_DIR = (PROJ_ROOT / "runs" / f"{run_tag}_{job_id}").resolve()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {RUN_DIR}")

    experiment = NestedCVStratifiedByPatient(
        df=df,
        cfg=cfg,
        labels_np=np.array(labels),
        pat_labels=pat_labels,
        unique_pat_ids=unique_pats,
        class_names=class_names,
        model_manager=model_manager,
        num_folds=num_folds,
        output_dir=str(RUN_DIR),
        train_fn=train_epoch_vit,
        val_fn=val_epoch_vit,
        pretrained_weights=None # ViT from MONAI is not pretrained
    )

    train_metrics, test_results = experiment.run_experiment()
    execution_time = time.time() - start_time

    # Get counts for logging
    train_counts, val_counts = experiment.get_early_stopping_split_counts()
    test_counts = {f"fold_{i}": len(pats) for i, pats in experiment.test_pat_ids_per_fold.items()}


    print("\n--- Experiment Finished ---")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print("Test results per fold:", test_results)

    # ---------- MLflow logging --------------------------------------------
    # 1. Find the best model from the cross-validation run
    best_idx = best_fold_idx(test_results)
    best_model_path = RUN_DIR / f"best_model_fold_{best_idx}.pth"

    if best_model_path.exists():
        print(f"Loading best model from fold {best_idx} for MLflow logging...")
        # Load the best model state
        best_model, device = model_manager.setup_model(len(class_names))
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_model.to(device)

        # Get the test data and transforms for the best fold
        test_pats_for_best_fold = experiment.get_test_patient_ids_for_fold(best_idx)
        if test_pats_for_best_fold is None:
            raise ValueError(f"Could not retrieve test patient IDs for the best fold ({best_idx}).")

        test_df_for_best_fold = df[df['patient_id'].isin(test_pats_for_best_fold)]
        te_imgs = test_df_for_best_fold['image_path'].values
        te_y = test_df_for_best_fold['label'].values
        
        # Note: We need the transforms for the specific fold, but get_current_fold_transforms
        # only holds the last fold's transforms. For logging, using the general val_transforms
        # is a reasonable approximation if normalization is consistent (e.g., ImageNet stats).
        _, val_transforms, _ = tf.get_transforms(cfg, color_transforms=False)


        log_SSL_run_to_mlflow(
            device=device,
            cfg=cfg,
            model=best_model,
            class_names=class_names,
            fold_results=test_results,
            per_fold_metrics=train_metrics,
            test_transforms=val_transforms, # Using val_transforms as a proxy
            test_images_paths_np=te_imgs,
            test_true_labels_np=te_y,
            yaml_path=str(PROJ_ROOT / args.yaml),
            color_transforms=True, # ViT uses color transforms
            model_library=model_library,
            pretrained_weights=pretrained_weights,
            execution_time=execution_time,
            train_counts=train_counts,
            val_counts=val_counts,
            test_counts=test_counts,
            output_dir=str(RUN_DIR),
        )
    else:
        print("Could not find the best model file to log artifacts.")
    # ---------- cleanup ----------------------------------------------------
    if os.environ.get("KEEP_RUN_DIR", "0").lower() not in ("1", "true", "yes"):
        try:
            print(f"Cleaning up run directory: {RUN_DIR}")
            shutil.rmtree(RUN_DIR, ignore_errors=True)
        except Exception as e:
            print(f"Warning: failed to remove {RUN_DIR}: {e}")


if __name__ == "__main__":
    main()