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
import time
import argparse, os, sys, pathlib, random, re, glob
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch.backends.cudnn as cudnn
from monai.utils.misc import set_determinism
import shutil  # <-- added

# ──────────────────────── PYTHONPATH ───────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

# ──────────────────────── project imports ─────────────────────────────────
from configs.ConfigLoader import ConfigLoader
from classes.ModelManager import ModelManager
from classes.NestedCVStratifiedByPatient import NestedCVStratifiedByPatient
from utils.reproducibility_functions import set_global_seed
from utils.mlflow_functions import log_SSL_run_to_mlflow
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

def get_data_directory(num_input_channels: int) -> Path:
    """
    Restituisce la cartella che contiene le immagini a seconda
    del numero di canali (3 → 3c_MIP, 4 → 4c_MIP).

    Richiede che la variabile d'ambiente DATA_ROOT sia già definita.
    """
    base = Path( os.environ.get("DATA_ROOT", "") )
    if not base.exists():
        raise EnvironmentError(
            "DATA_ROOT non impostata o path inesistente. "
            "Esegui `export DATA_ROOT=...` oppure modifica il tuo SLURM."
        )

    sub = {3: "3c_MIP", 4: "4c_MIP"}.get(num_input_channels)
    if sub is None:
        raise ValueError("num_input_channels deve essere 3 o 4")

    data_dir = base / sub
    if not data_dir.exists():
        raise FileNotFoundError(f"Cartella dati non trovata: {data_dir}")

    return data_dir

# ───────────────────── main ────────────────────────────────────────────────
def main():
    args = parse()
    start_time = time.time()
    # ---------- env vars ---------------------------------------------------
    # DATA_ROOT  = Path(os.environ["DATA_ROOT"])
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
    #NOTE: if you want to change the classes in classification use 
    # new_class_names =  ["MSA-P", "MSA-C"] #["MSA-P", "MSA-C"],["MSA-P", "PD"],["PD", "MSA-P", "MSA-C"]
    # new_class_names = ["MSA-P", "PD"] # ["MSA-P", "MSA-C"] #["MSA-P", "MSA-C"],["MSA-P", "PD"],["PD", "MSA-P", "MSA-C"]
    # cfg.set_class_names(new_class_names) 
    #NOTE if you want to change the number of channels use
    # cfg.set_model_input_channels(4) # or 4
    #NOTE if you want to change the pretrained weights use
    # cfg.set_pretrained_weights("torchvision") # or "monai"
    #NOTE if you want to change the number of epochs use
    # cfg.set_num_epochs(100) # or any other number
    #NOTE if you want to change the number of folds use
    # cfg.set_num_folds(7) # or any other number
    #NOTE if you want to change the model library use
    # cfg.set_model_library("monai") # or "torchvision"
    #NOTE if you want to use pretrained models use
    # cfg.set_transfer_learning(True) # or False
    
    # cfg.set_class_names(new_class_names)
    #print torch version
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
        for p in (DATA_ROOT / cname).glob("*.tif"):
            if "vaso" in p.name.lower():
                continue
            images.append(str(p)); labels.append(lab)
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in {DATA_ROOT}. Check your dataset.")

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
        cfg, color_transforms=True
    )

    # ---------- model ------------------------------------------------------
    model_manager = ModelManager(cfg, library=cfg.get_model_library())
    model, device = model_manager.setup_model(len(class_names), pretrained_weights)
    if device.type != "cuda":
        raise RuntimeError(
            "This script is intended to run on a CUDA-enabled GPU. "
            "Please ensure you have a compatible GPU and the necessary drivers installed."
        )
    # ---------- experiment -------------------------------------------------
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or str(os.getpid())
    run_tag = f"{Path(args.yaml).stem}_{cfg.get_model_input_channels()}c" #eg densenet121_3c
    # such folder will be: <PROJ_ROOT>/runs/densenet121_<job_id> where densenet121 is taken from the .yaml file stem
    RUN_DIR = (PROJ_ROOT / "runs" / f"{run_tag}_{job_id}").resolve()
    RUN_DIR.mkdir(parents=True, exist_ok=True) #creates the folder if it doesn't exist
    print(f"Run directory: {RUN_DIR}")
    
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
        output_dir=str(RUN_DIR),
    )

    train_metrics, test_results = experiment.run_experiment()
    execution_time = time.time() - start_time
    
    train_counts, val_counts = experiment.get_early_stopping_split_counts()
    test_counts = experiment.num_outer_images

    print("Train counts per fold:", train_counts)
    print("Validation counts per fold:", val_counts)
    print("Test count:", test_counts)
    # ---------- MLflow logging --------------------------------------------
    best_idx   = best_fold_idx(test_results)
    best_model_path = RUN_DIR / f"best_model_fold_{best_idx}.pth"
    best_model, _ = experiment._get_model_and_device()
    best_model.load_state_dict(torch.load(str(best_model_path), map_location=device))
    best_model.eval()

    log_SSL_run_to_mlflow(
        cfg=cfg,
        model=best_model,
        class_names=class_names,
        fold_results=test_results,
        per_fold_metrics=train_metrics,
        test_transforms=val_transforms,            
        test_images_paths_np=te_imgs,
        test_true_labels_np=te_y,
        yaml_path=str(PROJ_ROOT / args.yaml),
        color_transforms=False,
        model_library=model_library,
        pretrained_weights=pretrained_weights,
        execution_time=execution_time,
        train_counts=train_counts,
        val_counts=val_counts,
        test_counts=test_counts,
        output_dir=str(RUN_DIR),
    )

    # ---------- cleanup ----------------------------------------------------
    # Remove the per-run directory (learning_curves, confusion_matrices, checkpoints) after logging.
    # Set KEEP_RUN_DIR=1 to retain it.
    if os.environ.get("KEEP_RUN_DIR", "0").lower() not in ("1", "true", "yes"):
        try:
            print(f"Cleaning up run directory: {RUN_DIR}")
            shutil.rmtree(RUN_DIR, ignore_errors=True)
        except Exception as e:
            print(f"Warning: failed to remove {RUN_DIR}: {e}")

if __name__ == "__main__":
    main()
