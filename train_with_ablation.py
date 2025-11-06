#!/usr/bin/env python3
"""
train_with_ablation.py – Ablation study launcher for channel importance analysis

This script performs systematic channel ablation experiments by:
1. Iterating over each input channel (0 to n_channels-1)
2. Training a model with that specific channel zeroed out
3. Comparing performance degradation to identify critical channels

Scientific rationale:
- Channels with larger performance drops when ablated are more informative
- Helps interpret which microscopy channels (G-B-Gr-R) contain discriminative biomarkers
- Provides evidence for optimal channel subset selection in resource-constrained settings

Usage:
    python train_with_ablation.py --yaml configs/4c/densenet121.yaml
"""

# ──────────────────────── std libs ─────────────────────────────────────────
import time
import argparse, os, sys, pathlib, random, re, glob
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch.cuda
import torch.backends.cudnn as cudnn
from monai.utils.misc import set_determinism
import shutil
import warnings

# Suppress specific UserWarnings from PyTorch AMP
warnings.filterwarnings("ignore", message=".*`torch.cuda.amp.autocast(args...)` is deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*`torch.cuda.amp.GradScaler(args...)` is deprecated.*", category=UserWarning)

# ──────────────────────── PYTHONPATH ───────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

# ──────────────────────── project imports ─────────────────────────────────
from configs.ConfigLoader import ConfigLoader
from classes.ModelManager import ModelManager
from classes.NestedCVStratifiedByPatient import NestedCVStratifiedByPatient
from utils.reproducibility_functions import set_global_seed
from utils.mlflow_functions import log_run_to_mlflow
import utils.transformations_functions as tf

# ───────────────────── CLI helpers ─────────────────────────────────────────
from utils.training_helpers import *

# ───────────────────── main ────────────────────────────────────────────────
def main():
    args = parse()
    start_time = time.time()
    
    # ---------- env vars ---------------------------------------------------
    MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    
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
    
    # NOTE: Configuration overrides (uncomment as needed)
    # cfg.set_class_names(["MSA-P", "PD"])
    # cfg.set_num_input_channels(3)  # or 4
    # cfg.set_pretrained_weights("torchvision")  # or "monai"
    # cfg.set_num_epochs(100)
    # cfg.set_num_folds(7)
    # cfg.set_model_library("monai")  # or "torchvision"
    # cfg.set_transfer_learning(True)
    
    print(f"Torch version: {torch.__version__}")
    print(f"Using configuration: {args.yaml}")
    
    class_names        = cfg.get_class_names()
    num_channels       = cfg.get_model_input_channels()
    pretrained_weights = cfg.get_pretrained_weights()
    num_epochs         = cfg.get_num_epochs()
    num_workers        = cfg.get_num_workers()
    batch_size         = cfg.get_batch_size()
    num_folds          = cfg.get_num_folds()
    model_library      = cfg.get_model_library()
    cfg.set_use_ablation(True) # Enable ablation mode
    DATA_ROOT          = get_data_directory(num_channels)
    
    print(f"Class names: {class_names}")
    if class_names is None:
        raise ValueError("class_names returned None. Please check your configuration.")
    
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
            images.append(str(p))
            labels.append(lab)
    
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in {DATA_ROOT}. Check your dataset.")

    df = pd.DataFrame({
        "image_path": images,
        "label": labels,
        "patient_id": [extract_patient_id(p) for p in images]
    })
    
    print(f"Total patients in dataset: {df['patient_id'].nunique()}")
    print(f"All detected patient IDs: {df['patient_id'].unique()}")
    print(f"Total images in dataset: {len(df)}")

    pat_df      = df.groupby("patient_id").first().reset_index()
    print(f"Total unique patients for CV stratification: {len(pat_df)}")
    unique_pats = pat_df["patient_id"].values
    
    pat_labels  = pat_df["label"].values
    
    # ---------- Ablation study loop ----------------------------------------
    # Scientific approach: systematically ablate each channel independently
    # to measure its contribution to classification performance
    
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or str(os.getpid())
    run_tag = f"{Path(args.yaml).stem}_{cfg.get_model_input_channels()}c_ablation"
    BASE_RUN_DIR = (PROJ_ROOT / "runs" / f"{run_tag}_{job_id}").resolve()
    # BASE_RUN_DIR.mkdir(parents=True, exist_ok=True)
    
    # Store results for all ablation configurations
    all_ablation_results = {}
    
    print(f"\n{'='*80}")
    print(f"STARTING CHANNEL ABLATION STUDY")
    print(f"Total channels to ablate: {num_channels}")
    print(f"Base output directory: {BASE_RUN_DIR}")
    print(f"{'='*80}\n")

    for channel_idx in range(3,num_channels):
        ablation_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"ABLATION ITERATION {channel_idx + 1}/{num_channels}")
        print(f"Ablating channel index: {channel_idx}")
        print(f"{'='*80}\n")
        
        # Create subdirectory for this ablation configuration
        RUN_DIR = BASE_RUN_DIR / f"ablate_ch{channel_idx}"
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        
        # Configure ablation for current channel
        cfg.set_channels_to_ablate([channel_idx])  # ← Use loop variable
        
        # CRITICAL: Recreate transforms for each ablation configuration
        # This ensures the ChannelAblationD transform uses the updated config
        print(f"Creating transforms with channel {channel_idx} ablation enabled")
        
        # Validation: Check if ablation config is properly set
        ablated_channels = cfg.get_channels_to_ablate()
        if channel_idx not in ablated_channels:
            raise RuntimeError(
                f"Ablation configuration error: expected channel {channel_idx} "
                f"to be in ablated channels list, got {ablated_channels}"
            )
        
        print(f"⚠️  Channel {channel_idx} will be ZEROED during training")
        print(f"   → Fold-specific statistics will be recomputed with ablation applied")
        
        train_transforms, val_transforms, test_transforms = tf.get_transforms(cfg)

        # Recreate model for each ablation run to ensure clean state
        print(f"Initializing fresh model for ablation iteration {channel_idx}")
        model_manager = ModelManager(cfg, library=cfg.get_model_library())
        model, device = model_manager.setup_model(len(class_names), pretrained_weights)
        
        if device.type != "cuda":
            raise RuntimeError(
                "Please ensure you have a compatible GPU and the necessary drivers installed."
            )
        
        # Run nested CV experiment for this ablation configuration
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
        ablation_time = time.time() - ablation_start_time
        
        train_counts, val_counts = experiment.get_early_stopping_split_counts()
        test_counts = experiment.num_outer_images

        print(f"\n--- Ablation Ch{channel_idx} Results ---")
        print(f"Train counts per fold: {train_counts}")
        print(f"Validation counts per fold: {val_counts}")
        print(f"Test count: {test_counts}")
        print(f"Execution time: {ablation_time:.2f}s")
        
        # Store results for comparison
        all_ablation_results[channel_idx] = {
            "test_results": test_results,
            "train_metrics": train_metrics,
            "execution_time": ablation_time,
            "output_dir": str(RUN_DIR)
        }
        
        # ---------- MLflow logging for this ablation configuration ---------
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI

        best_idx = best_fold_idx(test_results)
        best_model_path = RUN_DIR / f"best_model_fold_{best_idx}.pth"
        best_model, _ = experiment._get_model_and_device()
        best_model.load_state_dict(torch.load(str(best_model_path), map_location=device))
        best_model.eval()

        best_fold_test_pats = experiment.get_test_patient_ids_for_fold(best_idx)
        if best_fold_test_pats is None:
            raise ValueError(f"Could not retrieve test patient IDs for the best fold ({best_idx}).")

        best_fold_test_df = df[df['patient_id'].isin(best_fold_test_pats)]
        te_imgs = best_fold_test_df['image_path'].values
        te_y = best_fold_test_df['label'].values

        # Log to MLflow with ablation metadata
        log_run_to_mlflow(
            cfg=cfg,
            model=best_model,
            class_names=class_names,
            fold_results=test_results,
            per_fold_metrics=train_metrics,
            test_transforms=val_transforms,
            test_images_paths_np=te_imgs,
            test_true_labels_np=te_y,
            yaml_path=str(PROJ_ROOT / args.yaml),
            model_library=model_library,
            pretrained_weights=pretrained_weights,
            execution_time=ablation_time,
            train_counts=train_counts,
            val_counts=val_counts,
            test_counts=test_counts,
            output_dir=str(RUN_DIR),
            test_pat_ids_per_fold=experiment.test_pat_ids_per_fold,
            best_fold_idx=best_idx,
            # Add ablation-specific metadata
            ablation_config={
                "ablated_channel": channel_idx,
                "total_channels": num_channels,
                "iteration": f"{channel_idx + 1}/{num_channels}"
            }
        )
        
        print(f"\n✓ Completed ablation for channel {channel_idx}")
        print(f"  Results saved to: {RUN_DIR}")
        print(f"  Time elapsed: {ablation_time:.2f}s\n")
    
    # ---------- Summary comparison across all ablations --------------------
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"ABLATION STUDY COMPLETED")
    print(f"Total execution time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    print(f"{'='*80}\n")
    
    # Generate summary comparison
    print("\n--- Performance Summary Across Ablations ---")
    summary_path = BASE_RUN_DIR / "ablation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Channel Ablation Study Summary\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Configuration: {args.yaml}\n")
        f.write(f"Model: {cfg.get_model_name()}\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"Total channels: {num_channels}\n")
        f.write(f"Number of CV folds: {num_folds}\n\n")
        
        f.write("Channel Ablation Results (mean ± std across CV folds):\n")
        f.write("-" * 80 + "\n\n")
        
        for ch_idx in sorted(all_ablation_results.keys()):
            results = all_ablation_results[ch_idx]
            test_res = results["test_results"]
            
            # Compute mean and std for each metric across folds
            bal_acc_vals = [r["test_balanced_acc"] for r in test_res]
            f1_vals = [r["test_f1"] for r in test_res]
            auc_vals = [r.get("test_auc", 0) for r in test_res]
            mcc_vals = [r.get("test_mcc", 0) for r in test_res]
            
            avg_bal_acc = np.mean(bal_acc_vals)
            std_bal_acc = np.std(bal_acc_vals)
            
            avg_f1 = np.mean(f1_vals)
            std_f1 = np.std(f1_vals)
            
            avg_auc = np.mean(auc_vals)
            std_auc = np.std(auc_vals)
            
            avg_mcc = np.mean(mcc_vals)
            std_mcc = np.std(mcc_vals)
            
            summary = (
                f"Channel {ch_idx} ablated:\n"
                f"  Balanced Accuracy:  {avg_bal_acc:.4f} ± {std_bal_acc:.4f}\n"
                f"  F1 Score:           {avg_f1:.4f} ± {std_f1:.4f}\n"
                f"  AUC-ROC:            {avg_auc:.4f} ± {std_auc:.4f}\n"
                f"  MCC:                {avg_mcc:.4f} ± {std_mcc:.4f}\n"
                f"  Execution time:     {results['execution_time']:.2f}s ({results['execution_time']/60:.1f}min)\n"
                f"  Output directory:   {results['output_dir']}\n\n"
            )
            print(summary)
            f.write(summary)
        
        f.write("-" * 80 + "\n")
        f.write(f"Total study time: {total_time:.2f}s ({total_time/3600:.2f}h)\n")
        f.write(f"\nInterpretation Guide:\n")
        f.write("- Channels with larger performance drops when ablated are more critical\n")
        f.write("- Compare balanced accuracy across ablations to identify discriminative channels\n")
        f.write("- High std values indicate unstable performance across folds\n")
    
    print(f"\nSummary report saved to: {summary_path}")
    
    # ---------- cleanup ----------------------------------------------------
    # Note: Keep run directories for ablation studies to enable post-hoc analysis
    if os.environ.get("KEEP_RUN_DIR", "1").lower() not in ("1", "true", "yes"):
        print("\n Warning: KEEP_RUN_DIR=0 detected, but preserving ablation results.")
        print(" Set KEEP_RUN_DIR=0 only if you've logged everything to MLflow.")

if __name__ == "__main__":
    main()
