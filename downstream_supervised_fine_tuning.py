#!/usr/bin/env python3
"""
downstream_supervised_fine_tuning.py – SSL fine-tuning launcher for CINECA

Env-vars (export in run_train.slurm)
    DATA_ROOT              = dataset root (e.g. $WORK/lzanotto/data)
    MLFLOW_TRACKING_URI    = file store   (e.g. file:$WORK/lzanotto/mlruns)
    MLFLOW_EXPERIMENT_NAME = optional experiment name
    
Run:
    python downstream_supervised_fine_tuning.py --yaml configs/ssl/byol_resnet18_3c.yaml --encoder_path /path/to/encoder.pth
"""

# ──────────────────────── std libs ─────────────────────────────────────────
import time
import argparse
import os
import sys
import random
import re
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from monai.utils.misc import set_determinism

# ──────────────────────── PYTHONPATH ───────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

# ──────────────────────── project imports ─────────────────────────────────
from configs.ConfigLoader import ConfigLoader
from classes.ModelManager import ModelManager
from classes.NestedCVStratifiedByPatient import NestedCVStratifiedByPatient
from utils.reproducibility_functions import set_global_seed
from utils.mlflow_functions import log_SSL_run_to_mlflow
from utils.train_functions import remove_projection_head, SSLClassifierModule, solve_cuda_oom
import utils.transformations_functions as tf

# ───────────────────── CLI ──────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="SSL config path (e.g., configs/ssl/byol_resnet18_3c.yaml)")
    p.add_argument("--encoder_path", required=False, help="Path to pretrained SSL encoder weights (not needed if randomly_initialized=true)")
    p.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder during fine-tuning")
    return p.parse_args()

# ───────────────────── helpers ─────────────────────────────────────────────
def extract_patient_id(path: str) -> str:
    m = re.search(r'(\d{4})', path)
    return m.group(1) if m else "UNKNOWN"

def get_best_fold_idx(outer_fold_test_results, metric="test_balanced_acc"):
    """Get the index of the best fold based on a specified metric."""
    best_fold_idx = np.argmax([r[metric] for r in outer_fold_test_results])
    best_fold_result = outer_fold_test_results[best_fold_idx]
    print(f"Best {metric} Fold Result: {best_fold_result}")
    return best_fold_result["fold"]

def get_data_directory(num_input_channels: int) -> Path:
    """Get data directory based on number of channels."""
    base = Path(os.environ.get("DATA_ROOT", ""))
    if not base.exists():
        raise EnvironmentError("DATA_ROOT not set or path doesn't exist.")
    
    sub = {3: "3c_MIP", 4: "4c_MIP"}.get(num_input_channels)
    if sub is None:
        raise ValueError("num_input_channels must be 3 or 4")
    
    data_dir = base / sub
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    
    return data_dir

def _remove_linear_probe_head(backbone):
    """Remove linear probe head from backbone if it exists."""
    if hasattr(backbone, "fc"):
        if backbone.fc.__class__.__name__ == "LinearProbeHead":
            print("Removing linear probe head from fc")
            backbone.fc = nn.Identity()
    elif hasattr(backbone, "classifier"):
        if backbone.classifier.__class__.__name__ == "LinearProbeHead":
            print("Removing linear probe head from classifier")
            backbone.classifier = nn.Identity()
    else:
        print("No linear probe head found to remove")

def load_ssl_encoder(model_manager, encoder_path, num_classes, pretrained_weights, device, cfg):
    """Load SSL pretrained encoder and prepare it for fine-tuning."""
    
    # Check if encoder path contains the model name
    encoder_filename = Path(encoder_path).stem.lower()
    model_name = model_manager.model_name.lower()
    
    if model_name not in encoder_filename:
        raise ValueError(
            f"Encoder path mismatch: Expected model '{model_manager.model_name}' "
            f"but encoder path '{encoder_path}' does not contain '{model_name}'. "
            f"Please ensure you're using the correct pretrained encoder for this model architecture."
        )
    
    print(f"✓ Encoder path validation passed: '{model_name}' found in '{encoder_filename}'")
    
    # Create base model
    model, _ = model_manager.setup_model(num_classes=num_classes, pretrained_weights=pretrained_weights)
    
    # Load SSL weights
    try:
        state_dict = torch.load(encoder_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load SSL weights from '{encoder_path}': {str(e)}. "
            f"Please check if the encoder was trained with the same model architecture."
        )
    
    model = model.to(device)
    
    # Remove projection head and get feature dimension
    model_without_projection = remove_projection_head(copy.deepcopy(model))
    
    # Get input shape from configuration
    num_channels = cfg.get_model_input_channels()
    spatial_size = cfg.get_spatial_size()  # Returns (height, width)
    
    # Test to get feature dimension
    with torch.no_grad():
        test_input = torch.zeros(1, num_channels, spatial_size[0], spatial_size[1], device=device)
        feats = model_without_projection(test_input).flatten(start_dim=1)
        feature_dim = feats.shape[1]
    
    print(f"Detected encoder output dimension: {feature_dim}")
    print(f"Test input shape used: {test_input.shape}")
    return model, feature_dim

def create_baseline_encoder(model_manager, num_classes, pretrained_weights, device, cfg):
    """Create baseline encoder without SSL pretraining."""
    model, _ = model_manager.setup_model(num_classes=num_classes, pretrained_weights=pretrained_weights)
    _remove_linear_probe_head(model)
    model = model.to(device)
    
    # Get input shape from configuration
    num_channels = cfg.get_model_input_channels()
    spatial_size = cfg.get_spatial_size()  # Returns (height, width)
    
    # Get feature dimension
    with torch.no_grad():
        test_input = torch.zeros(1, num_channels, spatial_size[0], spatial_size[1], device=device)
        feats = model(test_input).flatten(start_dim=1)
        feature_dim = feats.shape[1]
    
    print(f"Baseline encoder output dimension: {feature_dim}")
    print(f"Test input shape used: {test_input.shape}")
    return model, feature_dim

# ───────────────────── main ────────────────────────────────────────────────
def main():
    args = parse()
    start_time = time.time()
    
    # ---------- environment setup -----------------------------------------
    MLFLOW_URI = os.environ["MLFLOW_TRACKING_URI"]
    
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
    
    # Set class names for downstream task
    new_class_names = ["MSA-P", "PD"]  # Adjust as needed
    cfg.set_class_names(new_class_names)
    
    print(f"Using SSL configuration: {args.yaml}")
    print(f"Encoder path: {args.encoder_path}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    
    # Extract configuration
    class_names = cfg.get_class_names()
    num_channels = cfg.get_model_input_channels()
    pretrained_weights = cfg.get_pretrained_weights()
    model_library = cfg.get_model_library()
    DATA_ROOT = get_data_directory(num_channels)
    
    # Check if using randomly initialized weights
    use_random_init = cfg.training.get("randomly_initialized", False)
    print(f"Using randomly initialized encoder: {use_random_init}")
    # Validate encoder path requirement
    if not use_random_init and not args.encoder_path:
        raise ValueError("--encoder_path is required when randomly_initialized=false")
    
    if use_random_init and args.encoder_path:
        print("Warning: --encoder_path provided but will be ignored due to randomly_initialized=true")
    
    print(f"Class names: {class_names}")
    print(f"Number of channels: {num_channels}")
    print(f"Data directory: {DATA_ROOT}")

    # ---------- dataset preparation ---------------------------------------
    images, labels = [], []
    for lab, cname in enumerate(class_names):
        for p in (DATA_ROOT / cname).glob("*.tif"):
            if "vaso" in p.name.lower():
                continue
            images.append(str(p))
            labels.append(lab)
    
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in {DATA_ROOT}")

    images, labels = np.array(images), np.array(labels)

    # Create test split
    tr_imgs, te_imgs, tr_y, te_y = train_test_split(
        images, labels,
        test_size=cfg.get_test_ratio(),
        stratify=labels, 
        random_state=42
    )

    # Create patient-level dataframe
    df = pd.DataFrame({
        "image_path": images,
        "label": labels,
        "patient_id": [extract_patient_id(p) for p in images]
    })
    pat_df = df.groupby("patient_id").first().reset_index()
    unique_pat_ids = pat_df["patient_id"].values
    pat_labels = pat_df["label"].values

    # ---------- model setup -----------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script requires CUDA")

    model_manager = ModelManager(cfg, library=model_library)
    
    # ---------- encoder selection -----------------------------------------
    freeze_status = "frozen" if args.freeze_encoder else "unfrozen"
    
    if use_random_init:
        # Use randomly initialized encoder
        print("Creating randomly initialized encoder...")
        encoder, feature_dim = create_baseline_encoder(
            model_manager, len(class_names), pretrained_weights, device, cfg
        )
        encoder_type = f"RandomInit_{freeze_status}"
        pretrained_backbone_path = None  # No pretrained path for random init
        
    else:
        # Use SSL pretrained encoder
        print("Loading SSL pretrained encoder...")
        
        # Determine SSL method from encoder path
        encoder_filename = Path(args.encoder_path).stem
        if "simsiam" in encoder_filename.lower():
            ssl_method = "SimSiam"
        elif "byol" in encoder_filename.lower():
            ssl_method = "BYOL"
        else:
            ssl_method = "SSL"  # Generic SSL
        
        encoder, feature_dim = load_ssl_encoder(
            model_manager, args.encoder_path, len(class_names), 
            pretrained_weights, device, cfg
        )
        encoder_type = f"{ssl_method}_{freeze_status}"
        pretrained_backbone_path = args.encoder_path

    # ---------- run experiment --------------------------------------------
    print(f"\n{'='*60}")
    print(f"Running experiment: {encoder_type}")
    print(f"{'='*60}")
    
    # Define model factory for the experiment
    def model_factory(lr: float) -> torch.nn.Module:
        """Model factory for creating SSL classifier modules."""
        fresh_encoder = copy.deepcopy(encoder)
        return SSLClassifierModule(
            encoder=fresh_encoder,
            num_classes=len(class_names),
            freeze_encoder=args.freeze_encoder,
            lr=cfg.get_learning_rate(),
            backbone_output_dim=feature_dim,
        )

    # Create experiment
    experiment = NestedCVStratifiedByPatient(
        df=df,
        cfg=cfg,
        labels_np=labels,
        pat_labels=pat_labels,
        unique_pat_ids=unique_pat_ids,
        pretrained_weights=pretrained_weights,
        class_names=class_names,
        model_factory=model_factory,
        num_folds=cfg.get_num_folds()
    )

    # Run experiment
    per_fold_metrics, test_results = experiment.run_experiment()

    # Load best model for logging
    best_fold_idx = get_best_fold_idx(test_results, metric="test_balanced_acc")
    best_model, _ = experiment._get_model_and_device()
    best_model.eval()
    
    try:
        best_model.load_state_dict(
            torch.load(f"best_model_fold_{best_fold_idx}.pth")
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find best_model_fold_{best_fold_idx}.pth")

    # Get transforms for logging
    train_transforms, val_transforms, _ = experiment.get_current_fold_transforms()
    
    # Get counts for MLflow logging
    train_counts, val_counts = experiment.get_early_stopping_split_counts()
    test_counts = experiment.num_outer_images

    # ---------- MLflow logging --------------------------------------------
    experiment_name = f"ssl_fine_tuning_{num_channels}c_{encoder_type}"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

    execution_time = time.time() - start_time
    
    log_SSL_run_to_mlflow(
        cfg=cfg,
        model=best_model,
        class_names=class_names,
        fold_results=test_results,
        per_fold_metrics=per_fold_metrics,
        test_transforms=val_transforms,
        test_images_paths_np=te_imgs,
        test_true_labels_np=te_y,
        yaml_path=str(PROJ_ROOT / args.yaml),
        color_transforms=False,
        model_library=model_library,
        pretrained_weights=pretrained_backbone_path,
        ssl=not use_random_init,
        encoder_type=encoder_type,
        freeze_encoder=args.freeze_encoder,
        execution_time=execution_time,
        train_counts=train_counts,
        val_counts=val_counts,
        test_counts=test_counts,
    )

    print(f"\nTotal execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()