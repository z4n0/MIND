#!/usr/bin/env python3
"""
train.py – single-flag launcher for CINECA

Env-vars (export in run_train.slurm)
    DATA_ROOT              = dataset root (e.g. $WORK/lzanotto/data)
    MLFLOW_TRACKING_URI    = file store   (e.g. file:$WORK/lzanotto/mlruns)
    MLFLOW_EXPERIMENT_NAME = optional experiment name
Run:
    python train.py --yaml configs/4c/densenet121.yaml
"""

# ──────────────────────── std libs ─────────────────────────────────────────
import argparse, os, sys, pathlib, random, re, glob
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch.backends.cudnn as cudnn
from monai.utils.misc import set_determinism

# ──────────────────────── PYTHONPATH ───────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

# ──────────────────────── project imports ─────────────────────────────────
from configs.ConfigLoader import ConfigLoader
from classes.ModelManager import ModelManager
from classes.NestedCVStratifiedByPatient import NestedCVStratifiedByPatient
from utils.reproducibility_functions import set_global_seed
from utils.mlflow_functions import log_run_to_mlflow
import utils.transformations_functions as tf   # <-- transforms factory

# ───────────────────── CLI (one flag) ──────────────────────────────────────
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="path inside configs/")
    return p.parse_args()

# ───────────────────── helpers ─────────────────────────────────────────────
def extract_patient_id(path: str) -> str:
    m = re.search(r'(\d{4})', path)
    return m.group(1) if m else "UNKNOWN"

def best_fold_idx(results, metric="test_balanced_acc") -> int:
    return int(np.argmax([r[metric] for r in results]))

# ───────────────────── main ────────────────────────────────────────────────
def main():
    args = parse()

    # ---------- env vars ---------------------------------------------------
    DATA_ROOT  = Path(os.environ["DATA_ROOT"])
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
    class_names        = cfg.get_class_names()
    if class_names is None:
        raise ValueError("class_names returned None. Please check your configuration.")
    num_channels       = cfg.get_model_input_channels()
    pretrained_weights = cfg.get_pretrained_weights()
    num_epochs         = cfg.get_num_epochs()
    num_workers        = cfg.get_num_workers()
    batch_size         = cfg.get_batch_size()
    num_folds          = cfg.get_num_folds()
    model_library     = cfg.get_model_library()

    # ---------- dataset ----------------------------------------------------
    images, labels = [], []
    for lab, cname in enumerate(class_names):
        for p in (DATA_ROOT / cname).glob("*.tif"):
            if "vaso" in p.name.lower():
                continue
            images.append(str(p)); labels.append(lab)

    images, labels = np.array(images), np.array(labels)

    tr_imgs, te_imgs, tr_y, te_y = train_test_split(
        images, labels,
        test_size=cfg.get_test_ratio(),
        stratify=labels, random_state=42)

    df = pd.DataFrame({"image_path": images,
                       "label": labels,
                       "patient_id": [extract_patient_id(p) for p in images]})
    pat_df      = df.groupby("patient_id").first().reset_index()
    unique_pats = pat_df["patient_id"].values
    pat_labels  = pat_df["label"].values

    # ---------- transforms (as in notebook) --------------------------------
    train_transforms, val_transforms, test_transforms = tf.get_transforms(
        cfg, color_transforms=False
    )

    # ---------- model ------------------------------------------------------
    model_manager = ModelManager(cfg, library=cfg.get_model_library())
    model, device = model_manager.setup_model(len(class_names), pretrained_weights)

    # ---------- experiment -------------------------------------------------
    experiment = NestedCVStratifiedByPatient(
        df=df,
        cfg=cfg,
        labels_np=labels,
        pat_labels=pat_labels,
        unique_pat_ids=unique_pats,
        pretrained_weights=pretrained_weights,
        class_names=class_names,
        model_manager=model_manager,
        num_folds=num_folds,
    )

    train_metrics, test_results = experiment.run_experiment()

    # ---------- MLflow logging --------------------------------------------
    EXPERIMENT_NAME = f"supervised_learning_{num_channels}c"    
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME

    best_idx   = best_fold_idx(test_results)
    best_model, _ = experiment._get_model_and_device()
    best_model.load_state_dict(
        torch.load(f"best_model_fold_{best_idx}.pth", map_location=device))
    best_model.eval()

    log_run_to_mlflow(
        cfg=cfg,
        model=best_model,
        class_names=class_names,
        fold_results=test_results,
        per_fold_metrics=train_metrics,
        test_transforms=val_transforms,             # ← here!
        test_images_paths_np=te_imgs,
        test_true_labels_np=te_y,
        yaml_path=str(PROJ_ROOT / args.yaml),
        color_transforms=False,
        model_library=model_library,
        pretrained_weights=pretrained_weights,
        test_pat_ids_per_fold=experiment.test_pat_ids_per_fold,
        best_fold_idx=best_idx,
    )

if __name__ == "__main__":
    main()
