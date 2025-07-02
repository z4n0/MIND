#!/usr/bin/env python3
"""
train.py  –  unified launcher for (supervised or SSL-finetune) training on CINECA.

How to run
----------
sbatch your_slurm_file.slurm    # the slurm file exports DATA_ROOT / MLFLOW_*
or, interactively:
srun ... python train.py --yaml configs/4c/densenet121_with_ssl.yaml

YAML layout
-----------
The YAML may inherit from a base config via the usual `_base_:` key and can
optionally append an `ssl:` block, e.g.

    _base_: base.yaml

    # ↓ standard blocks (override whatever you want)
    model:
      model_name: "Densenet121"
      pretrained_weights: imagenet
    training:
      num_epochs: 125

    # ↓ optional, triggers SSL path
    ssl:
      encoder_weights:  "/path/to/simclr_encoder.pth"
      freeze_encoder:   true        # true = linear-probe, false = fine-tune
      lr:               3e-4
      proj_hidden_dim:  512
"""

# ───────────────────── std libs ───────────────────────────────────────────
from __future__ import annotations
import argparse, os, sys, time, random, re, glob
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
import torch.backends.cudnn as cudnn
from monai.utils.misc import set_determinism

# ────────────────────── PYTHONPATH helper ────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

# ────────────────────── project imports ──────────────────────────────────
from configs.ConfigLoader import ConfigLoader            # your existing class
from classes.ModelManager import ModelManager
from classes.NestedCVStratifiedByPatient import NestedCVStratifiedByPatient
from utils.reproducibility_functions import set_global_seed
from utils.mlflow_functions import log_SSL_run_to_mlflow
import utils.transformations_functions as tf


# ────────────────────── CLI (single flag) ────────────────────────────────
def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--yaml",
        required=True,
        help="YAML with _base_ plus optional ssl block (relative to configs/)",
    )
    return p.parse_args()


# ────────────────────── misc helpers ─────────────────────────────────────
def extract_patient_id(path: str) -> str:
    m = re.search(r"(\d{4})", path)
    return m.group(1) if m else "UNKNOWN"


def best_fold_idx(results: list[dict], metric: str = "test_balanced_acc") -> int:
    return int(np.argmax([r[metric] for r in results]))


def get_data_directory(num_input_channels: int) -> Path:
    root = Path(os.environ["DATA_ROOT"])
    sub  = {3: "3c_MIP", 4: "4c_MIP"}[num_input_channels]
    path = root / sub
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {path}")
    return path


# ────────────────────── main ─────────────────────────────────────────────
def main() -> None:
    args = parse()
    t0   = time.time()

    # reproducibility
    SEED = 42
    set_global_seed(SEED)
    set_determinism(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    cudnn.deterministic, cudnn.benchmark = True, False

    # ---------------------------------------------------------------- CFG
    cfg = ConfigLoader(PROJ_ROOT / args.yaml)
    ssl_cfg: dict | None = getattr(cfg, "ssl", None)   # may not exist

    # --------------- convenience overrides (edit / comment as needed) ----
    # cfg.set_class_names(["MSA-P", "PD"])
    # cfg.set_model_library("torchvision")
    # cfg.set_transfer_learning(True)

    # ---------------------------------------------------------------- I/O paths
    data_dir = get_data_directory(cfg.get_model_input_channels())

    # ---------------------------------------------------------------- dataset
    images, labels = [], []
    for lab, cname in enumerate(cfg.get_class_names()):
        for p in (data_dir / cname).glob("*.tif"):
            if "vaso" in p.name.lower():
                continue
            images.append(str(p)); labels.append(lab)

    if not images:
        raise RuntimeError(f"No images found under {data_dir}")

    images, labels = map(np.array, (images, labels))

    (train_img, test_img,
     train_lab, test_lab) = train_test_split(
         images, labels,
         test_size=cfg.get_test_ratio(),
         stratify=labels, random_state=SEED)

    df = pd.DataFrame({
        "image_path": images,
        "label":      labels,
        "patient_id": [extract_patient_id(p) for p in images]
    })
    pat_df      = df.groupby("patient_id").first().reset_index()
    unique_pat  = pat_df["patient_id"].values
    pat_labels  = pat_df["label"].values

    # ---------------------------------------------------------------- transforms
    tr_tf, val_tf, test_tf = tf.get_transforms(cfg, color_transforms=False)

    # ---------------------------------------------------------------- model / manager
    manager      = ModelManager(cfg, library=cfg.get_model_library())

    if ssl_cfg:   # ---------- SSL path  ---------------------------------
        encoder_path   = Path(ssl_cfg["encoder_weights"])
        freeze_enc     = bool(ssl_cfg.get("freeze_encoder", True))
        print(f"⚙  Using SSL encoder: {encoder_path}  (frozen={freeze_enc})")

        model, device = manager.setup_ssl_model(
            encoder_path=encoder_path,
            num_classes=len(cfg.get_class_names()),
            freeze_encoder=freeze_enc,
        )
    else:         # ---------- normal supervised path --------------------
        model, device = manager.setup_model(
            num_classes=len(cfg.get_class_names()),
            pretrained_weights=cfg.get_pretrained_weights()
        )

    # GPU check
    if device.type != "cuda":
        raise RuntimeError("CUDA device not detected – check your Slurm gres line!")

    # ---------------------------------------------------------------- experiment (nested CV)
    experiment = NestedCVStratifiedByPatient(
        df=df,
        cfg=cfg,
        labels_np=labels,
        pat_labels=pat_labels,
        unique_pat_ids=unique_pat,
        pretrained_weights=cfg.get_pretrained_weights(),
        class_names=cfg.get_class_names(),
        model_manager=manager,
        num_folds=cfg.get_num_folds(),
    )

    tr_metrics, te_results = experiment.run_experiment()
    exec_time = time.time() - t0
    tr_counts, val_counts = experiment.get_early_stopping_split_counts()
    te_count = experiment.num_outer_images

    # ---------------------------------------------------------------- MLflow
    os.environ["MLFLOW_EXPERIMENT_NAME"] = (
        f"{'ssl' if ssl_cfg else 'supervised'}_{cfg.get_model_input_channels()}c"
    )
    os.environ["MLFLOW_TRACKING_URI"] = os.environ["MLFLOW_TRACKING_URI"]  # pass-through

    best_idx = best_fold_idx(te_results)
    best_model, _ = experiment._get_model_and_device()
    best_model.load_state_dict(
        torch.load(f"best_model_fold_{best_idx}.pth", map_location=device)
    )
    best_model.eval()

    log_SSL_run_to_mlflow(
        cfg=cfg,
        model=best_model,
        class_names=cfg.get_class_names(),
        fold_results=te_results,
        per_fold_metrics=tr_metrics,
        test_transforms=val_tf,
        test_images_paths_np=test_img,
        test_true_labels_np=test_lab,
        yaml_path=str(Path(args.yaml).resolve()),
        color_transforms=False,
        model_library=cfg.get_model_library(),
        pretrained_weights=cfg.get_pretrained_weights(),
        execution_time=exec_time,
        train_counts=tr_counts,
        val_counts=val_counts,
        test_counts=te_count,
    )


if __name__ == "__main__":
    main()
