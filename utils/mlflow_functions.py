import mlflow
from narwhals import Unknown
from .directory_functions import get_tracking_uri
import os
import torch
from typing import Dict, List, Optional, Union, Any
import torch.nn as nn
from monai.transforms import Compose
import numpy as np
from configs.ConfigLoader import ConfigLoader

#!TODO delete this you already have get_tracking_uri
def get_mlrun_base_folder(gdrive, kaggle,linux):
    if kaggle:
        print("you are in kaggle so there's a problem")
        return "/kaggle/input/rgb-tif/"
    elif gdrive:
        print("you are in google drive")
        return "/content/drive/MyDrive/TESI/colab_mlruns"
    elif linux:
        print("you are in linux")
        return "/home/zano/Documents/TESI/mlruns"
    else:
        uri = get_tracking_uri(gdrive, kaggle)
        print("tracking_uri =", uri)
        base_folder = uri.replace("file:///", "")
        return base_folder

# 3. Function to Get Experiment ID from Name
def get_experiment_id_byName(experiment_name):
    """Retrieves the experiment ID from the experiment name."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    return experiment.experiment_id

def get_run_name_by_id(run_id, tracking_uri, experiment_id=None):
    """
    Prints the name of an MLflow run given its ID.

    Args:
        run_id: The ID of the MLflow run.
        tracking_uri: The MLflow tracking URI.
        experiment_id: The ID of the experiment (optional). If not provided,
                       it will attempt to find the experiment based on the run ID.
    """
    mlflow.set_tracking_uri(tracking_uri)
    try:
        run = mlflow.get_run(run_id)
        if run:
          return run.data.tags.get('mlflow.runName', 'N/A')  # Get the run name, or 'N/A' if it doesn't exist
        else:
            print(f"Run with ID '{run_id}' not found.")

    except Exception as e:
        print(f"An error occurred: {e}")

#SHOW RUNS ID AND NAME FOR A SPECIFIC EXPERIMENT
def print_run_ids_and_names(experiment_name, tracking_uri):
    """
    Prints the Run IDs and their associated run names (mlflow.runName tag)
    for a given experiment name.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        tracking_uri (str): The MLflow tracking URI.
    """
    mlflow.set_tracking_uri(tracking_uri)

    try:
        # Get the experiment by name
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            print(f"Experiment '{experiment_name}' not found.")
            return

        # Search for all runs within the experiment
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if runs.empty:
            print(f"No runs found for experiment '{experiment_name}'.")
            return

        print(f"Runs for experiment '{experiment_name}':")
        for index, run in runs.iterrows():
            run_id = run["run_id"]
            run_name = run.get("tags.mlflow.runName", "Unnamed Run")  # Handle missing run name
            print(f"  Run ID: {run_id}, Run Name: {run_name}")

    except Exception as e:
        print(f"An error occurred: {e}")


def get_transform_params(transform_list):
    """Extract parameters from any MONAI transform in the composition"""
    params = {}

    for transform in transform_list.transforms:
        transform_name = transform.__class__.__name__
        
        # Get all public attributes of the transform
        transform_params = {
            k: v for k, v in transform.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }

        # Clean and format the parameters
        for param_name, param_value in transform_params.items():
            # Convert numpy arrays, tensors, etc., to lists for JSON serialization
            if hasattr(param_value, 'tolist'):
                param_value = param_value.tolist()

            # Handle other non-serializable types
            try:
                params[f"{transform_name}.{param_name}"] = param_value
            except Exception as e:
                print(f"Skipping {transform_name}.{param_name} due to serialization issue: {e}")

    return params

def load_mlflow_model(tracking_uri: str, experiment_name: str, gdrive: bool = False, kaggle: bool = False, linux: bool = True) -> torch.nn.Module:
    """
    Load a PyTorch model from MLflow given the tracking URI and experiment name.
    
    Args:
        tracking_uri (str): The MLflow tracking URI
        experiment_name (str): Name of the experiment
        gdrive (bool): Flag for Google Drive environment
        kaggle (bool): Flag for Kaggle environment
        linux (bool): Flag for Linux environment
        
    Returns:
        torch.nn.Module: Loaded PyTorch model
    """
    # Set up MLflow tracking
    mlflow.set_tracking_uri(tracking_uri)
    
    # Get experiment ID
    experiment_id = get_experiment_id_byName(experiment_name)
    print(f"Experiment '{experiment_name}' has ID: {experiment_id}")
    
    # Display available runs
    print_run_ids_and_names(experiment_name, tracking_uri)
    
    # Get run ID from user
    run_id = input("Enter the run ID you see in the above output: ")
    
    # Set up paths
    model_path = "model"
    base_folder = get_mlrun_base_folder(gdrive, kaggle, linux)
    local_artifact_path = os.path.join(base_folder, experiment_id, run_id)
    test_path = os.path.join(local_artifact_path, "artifacts", model_path)
    normalize_path = os.path.normpath(test_path)
    
    # Load the model
    try:
        if os.path.exists(normalize_path):
            model = mlflow.pytorch.load_model(
                os.path.join(local_artifact_path, "artifacts", model_path),
                map_location=torch.device('cpu')
            )
        else:
            print("using empty path")
            model = mlflow.pytorch.load_model(
                local_artifact_path,
                artifact_path=model_path,
                map_location=torch.device('cpu')
            )
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise RuntimeError("Failed to load the model.")
    
    
# def log_fold_results_metrics(fold_results, prefix='val'):
#     """
#     Log metrics from fold results to MLflow with a flexible prefix.
    
#     Args:
#         fold_results: List of dictionaries containing metrics from each fold
#         prefix: Metric prefix ('val' or 'test' or any other prefix)
#     """
#     if fold_results is not None:
#         mlflow.log_param("fold_results_list", str(fold_results))
        
#         # Define base metrics and their corresponding MLflow names
#         metrics_mapping = {
#             'loss': f"mean_{prefix}_loss",
#             'acc': f"mean_{prefix}_accuracy",
#             'f1': f"mean_{prefix}_f1", 
#             'balanced_acc': f"mean_{prefix}_balanced_accuracy",
#             'auc': f"mean_{prefix}_auc"
#         }
        
#         # Calculate and log mean values for available metrics
#         for base_metric, mlflow_name in metrics_mapping.items():
#             metric_key = f"{prefix}_{base_metric}"
            
#             # Check if this metric exists in any fold result
#             if any(metric_key in fold for fold in fold_results):
#                 values = [fold[metric_key] for fold in fold_results if metric_key in fold]
#                 if values:
#                     mean_value = sum(values) / len(values)
#                     mlflow.log_metric(mlflow_name, mean_value)
#                     print(f"Logged {mlflow_name}: {mean_value:.4f}")

import pandas as pd

def log_folds_results_to_csv(fold_results, prefix='val'):
    if not fold_results:
        print("No fold results to log.")
        return

    # Create a DataFrame
    df = pd.DataFrame(fold_results)

    # Compute summary (mean, std, min, max) for selected metrics
    metric_keys = ['loss', 'acc', 'f1', 'balanced_acc', 'auc', 'mcc']
    for key in metric_keys:
        metric_name = f"{prefix}_{key}"
        if metric_name in df.columns:
            mean = df[metric_name].mean()
            std = df[metric_name].std()
            min_ = df[metric_name].min()
            max_ = df[metric_name].max()

            mlflow.log_metric(f"mean_{metric_name}", mean)
            mlflow.log_metric(f"std_{metric_name}", std)
            mlflow.log_metric(f"min_{metric_name}", min_)
            mlflow.log_metric(f"max_{metric_name}", max_)

            print(f"{metric_name} | mean: {mean:.4f}, std: {std:.4f}, min: {min_:.4f}, max: {max_:.4f}")

    # Log the full table as CSV artifact
    csv_path = f"{prefix}_fold_metrics.csv"
    df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)
    # Clean up local file
    os.remove(csv_path)
    

def log_kfold_epoch_metrics(per_fold_metrics, prefix="val"):
    """
    Logs per-epoch metrics using per-fold metric names like 'val_fold_0/val_accuracy',
    which show up as separate charts in MLflow UI.
    """
    for metric_name, fold_dict in per_fold_metrics.items():
        for fold_idx, values in fold_dict.items():
            for epoch, value in enumerate(values):
                # print(f"Logging {prefix}_fold_{fold_idx}/{metric_name}")
                mlflow.log_metric(f"{prefix}_fold_{fold_idx}/{metric_name}", value, step=epoch)



import os
import torch
import mlflow

def load_model_for_run(run_id: str, experiment_id: str, base_folder: str, device: torch.device) -> torch.nn.Module:
    """
    Load a PyTorch model for a given MLflow run.

    Args:
        run_id (str): The MLflow run ID.
        experiment_id (str): The experiment ID.
        base_folder (str): Base folder where artifacts are stored.
        device (torch.device): The compute device.

    Returns:
        torch.nn.Module: The loaded model if successful, or None if an error occurs.
    """
    model_path = "model"
    local_artifact_path = os.path.join(base_folder, experiment_id, run_id)
    artifact_full_path = os.path.join(local_artifact_path, "artifacts", model_path)
    normalized_path = os.path.normpath(artifact_full_path)

    try:
        if os.path.exists(normalized_path):
            print(f"Loading model from: {normalized_path}")
            model = mlflow.pytorch.load_model(normalized_path, map_location=device)
        else:
            print(f"Artifact path not found at {normalized_path}, attempting alternate loading...")
            model = mlflow.pytorch.load_model(local_artifact_path, artifact_path=model_path, map_location=device)

        model.to(device)
        model.eval()
        print("Model loaded successfully.")
        return model

    except Exception as e:
        print(f"Error loading model for run ID {run_id}: {e}")
        return None


def create_run_name(cfg=None, color_transforms=False, model_library="", pretrained_weights=None):
    import datetime
    model_name = cfg.model["model_name"]
    components = [model_name]

    if cfg.training["mixup_alpha"] > 0:
        components.append(f"mixup{cfg.training['mixup_alpha']}")

    if cfg.training["oversample"]:
        components.append("oversamp")

    if cfg.training["undersample"]:
        components.append("undersamp")

    if cfg.training["weighted_loss"]:
        components.append("weighted")

    if cfg.training["transfer_learning"]:
        components.append("TL")

    if cfg.training["pretrained"] and pretrained_weights is not None:
        components.append("pretrained:"+ str(pretrained_weights))

    if cfg.get_freezed_layer_index() is not None:
        components.append(f"freeze:{cfg.get_freezed_layer_index()}")

    run_name = "_".join(components)
    run_name = run_name + "_" + model_library + "_" +f"color_transforms:{str(color_transforms)}_" + str(datetime.datetime.now().strftime("%m-%d_at:%H-%M-%S"))
    print("Run name:", run_name)
    return run_name


def find_last_conv_layer(model):
    last_conv_name, last_conv = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_name, last_conv = name, module
    return last_conv_name, last_conv

import os, subprocess, shutil, sys

def start_mlflow_ui(port: int = 6006) -> None:
    """
    Launch the MLflow UI on the login node *in background*.
    It reads MLFLOW_TRACKING_URI from the environment.

    • On Leonardo you run it from the **login node**, NOT inside a job.
    • Access it via SSH tunnel from your laptop:
        ssh -N -L 6006:localhost:6006 lzanotto@login.leonardo.cineca.it
    """
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI not set in the environment")

    log_file = "mlflow_ui.log"
    cmd = ["mlflow", "ui", "--backend-store-uri", uri,
           "--host", "127.0.0.1", "--port", str(port)]

    print(f"Starting MLflow UI on port {port} …  logs → {log_file}")
    with open(log_file, "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, close_fds=True)
    

import mlflow, os, numpy as np, torch, shutil, yaml, pandas as pd
from typing import Any, Dict, List, Optional
from monai.transforms.compose import Compose
from torch import nn
from pathlib import Path

# ---------- utilities already in your project ------------------------------
from utils.mlflow_functions import (
    log_folds_results_to_csv, log_kfold_epoch_metrics, create_run_name
)

from utils.explainability_functions import (
    generate_and_save_gradcam_batch, process_and_save_batch_gradcam_and_Overlay,
)

from utils.data_visualization_functions import (
    plot_learning_curves, plot_confusion_matrix, generate_cv_results_figure
)
from utils.train_functions import make_loader
from utils.data_visualization_functions import min_max_normalization

# ---------------------------------------------------------------------------
def log_SSL_run_to_mlflow(
    cfg,
    model: nn.Module,
    class_names: List[str],
    fold_results: List[Dict[str, Any]],
    per_fold_metrics: Dict[str, Any],
    test_transforms: Compose,
    test_images_paths_np: np.ndarray,
    test_true_labels_np: np.ndarray,
    yaml_path: str,
    *,
    color_transforms: bool = False,
    model_library: str = "torchvision",
    pretrained_weights: Optional[str] = None,
    ssl: bool = False,
    encoder_type: Optional[str] = None,
    freeze_encoder: bool = False,
    execution_time: float = 0.0,
    train_counts: int = 0,
    val_counts: int = 0,
    test_counts: int = 0,
) -> None:
    """Log parameters, metrics and artifacts to MLflow (file store)."""

    # ------------------------------------------------------------------ URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI env-var missing")
    mlflow.set_tracking_uri(tracking_uri)
    # sanity-check
    resolved_uri = mlflow.get_tracking_uri()
    if resolved_uri.rstrip("/") != tracking_uri.rstrip("/"):
        raise RuntimeError(f"MLflow tracking URI mismatch: '{resolved_uri}'")

    # ---------------------------------------------------------------- run / exp
    run_name = create_run_name(
        cfg,
        color_transforms=color_transforms,
        model_library=model_library,
        pretrained_weights=pretrained_weights,
    )
    exp_suffix = "SSL" if ssl else "supervised"
    exp_name = f"{'_vs_'.join(class_names)}_{cfg.get_model_input_channels()}c_{exp_suffix}"
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=run_name):
        # --------------------- PARAMS
        mlflow.log_params(
            {
                "epochs": cfg.training["num_epochs"],
                "batch_size": cfg.get_batch_size(),
                "transfer_learning": cfg.training["transfer_learning"],
                "fine_tuning": cfg.training["fine_tuning"],
                "pretrained": pretrained_weights,
                "weight_decay": cfg.get_weight_decay(),
                "dropout_rate": cfg.get_dropout_prob(),
                "model_name": cfg.get_model_name(),
                "model_library": cfg.get_model_library(),
                "n_folds": cfg.get_num_folds(),
                "train_counts": train_counts,
                "val_counts": val_counts,
                "test_counts": test_counts,
                "color_transforms": color_transforms,
                "freezed_layer_index": cfg.get_freezed_layer_index(),
            }
        )
        mlflow.log_param(
            "total_params", int(sum(p.numel() for p in model.parameters()))
        )
        mlflow.log_param(
            "trainable_params",
            int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        )

        # --------------------- METRICS
        log_kfold_epoch_metrics(per_fold_metrics, prefix="val")
        per_fold = np.asarray(
            [
                (
                    r["test_loss"],
                    r["test_acc"],
                    r["test_balanced_acc"],
                    r["test_f1"],
                    r.get("test_auc", 0.0),
                    r.get("test_mcc", 0.0),
                    r.get("test_precision", 0.0),
                    r.get("test_recall", 0.0),
                )
                for r in fold_results
            ]
        )
        mlflow.log_metrics(
            {
                "mean_test_loss": float(per_fold[:, 0].mean()),
                "std_test_loss": float(per_fold[:, 0].std()),
                "mean_test_accuracy": float(per_fold[:, 1].mean()),
                "std_test_accuracy": float(per_fold[:, 1].std()),
                "mean_test_balanced_acc": float(per_fold[:, 2].mean()),
                "std_test_balanced_acc": float(per_fold[:, 2].std()),
                "mean_test_f1": float(per_fold[:, 3].mean()),
                "std_test_f1": float(per_fold[:, 3].std()),
                "mean_test_auc": float(per_fold[:, 4].mean()),
                "std_test_auc": float(per_fold[:, 4].std()),
                "mean_test_mcc": float(per_fold[:, 5].mean()),
                "std_test_mcc": float(per_fold[:, 5].std()),
                "mean_test_precision": float(per_fold[:, 6].mean()),
                "std_test_precision": float(per_fold[:, 6].std()),
                "mean_test_recall": float(per_fold[:, 7].mean()),
                "std_test_recall": float(per_fold[:, 7].std()),
                "exec_time_min": execution_time / 60.0,
            }
        )

        # --------------------- ARTIFACTS
        # 1. model (always push CPU version)
        import cloudpickle
        with torch.no_grad():
            mlflow.pytorch.log_model(
                model.cpu(), "model", pickle_module=cloudpickle
            )
        # 2. config YAML
        mlflow.log_artifact(yaml_path, artifact_path="config")

        # 3. CV box-plot
        fig_box = generate_cv_results_figure(fold_results, "test")
        mlflow.log_figure(fig_box, "fold_box_plot.png")

        # 4. learning curves (if produced)
        lc_dir = Path.cwd() / "learning_curves"
        if lc_dir.is_dir():
            mlflow.log_artifacts(str(lc_dir), artifact_path="learning_curves")

        # 5. train.py backup
        if Path("train.py").is_file():
            mlflow.log_artifact("train.py", artifact_path="scripts")

        # 6. Grad-CAM / attention maps
        try:
            if "vit" in cfg.get_model_name().lower():
                if not ssl:
                    log_attention_maps_to_mlflow(
                        model,
                        test_images_paths_np,
                        test_true_labels_np,
                        test_transforms,
                        cfg,
                    )
            else:
                if not ssl:
                    tmp_dir = Path.cwd() / "tmp_gradcam"
                    tmp_dir.mkdir(exist_ok=True)
                    log_gradcam_to_mlflow(
                        model,
                        test_images_paths_np,
                        test_true_labels_np,
                        test_transforms,
                        class_names,
                        cfg,
                        ssl=ssl,
                        run_name=run_name,
                        experiment_name=exp_name,
                        base_dir=tmp_dir,
                    )
        finally:
            # ensure temp dirs are cleaned even if log_artifacts() fails
            shutil.rmtree(Path.cwd() / "tmp_gradcam", ignore_errors=True)

    # restore
    cfg.set_freezed_layer_index(None)

def log_attention_maps_to_mlflow(
    model: nn.Module,
    test_images_paths_np: np.ndarray,
    test_true_labels_np: np.ndarray,
    test_transforms: Compose,
    cfg,
):
    import shutil
    if "vit" not in cfg.get_model_name().lower():
        raise RuntimeError(f"you can't generate attention maps for {cfg.get_model_name()} because it's not a ViT or you are doing SSL")
        
    from utils.train_functions import make_loader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define test_loader before using it
    test_loader = make_loader(
        test_images_paths_np,
        test_true_labels_np,
        transforms=test_transforms,
        cfg=cfg,
        shuffle=False,
    )
    
    from utils.vit_explanation_functions import save_attention_overlays_side_by_side
    activation_overlay_output_path = "activation_overlays"
    save_attention_overlays_side_by_side(test_loader, model, activation_overlay_output_path, device, overlay_alpha=0.5)
    mlflow.log_artifacts(activation_overlay_output_path, artifact_path="activation_overlays_test")
    # Delete the local directory after logging to MLflow
    shutil.rmtree(activation_overlay_output_path)

def log_gradcam_to_mlflow(
    model: nn.Module,
    test_images_paths_np: np.ndarray,
    test_true_labels_np: np.ndarray,
    test_transforms: Compose,
    class_names: List[str],
    cfg,  # Add cfg parameter
    ssl: bool = False,  # Add ssl parameter
    run_name: Optional[str] = None,  # Add run_name parameter
    experiment_name: Optional[str] = None,  # Add experiment_name parameter
    base_dir: Optional[Path] = None,  # Add base_dir parameter
):
    # ----------------- saving gradcams with threshold ---------------------
    from utils.explainability_functions import process_and_save_batch_gradcam_and_Overlay
    from utils.data_visualization_functions import min_max_normalization
    from utils.train_functions import make_loader
    
    model_name = cfg.get_model_name()
    if "vit" in model_name.lower() or ssl:
        raise RuntimeError(f"you can't generate gradCAMs for {model_name} because it's not a CNN or you are doing SSL")
    
    from utils.explainability_functions import generate_and_save_gradcam_batch
    import shutil
    from monai.visualize import GradCAM, GradCAMpp
    
    # Or GradCAM++
    target_layer, target_layer_type = find_last_conv_layer(model)
    if target_layer is None:
        print("No convolutional layer found in the model. Cannot apply GradCAM.")
        return
    
    model.eval().cpu()
    
    try:
        gradcampp = GradCAMpp(
            nn_module=model,
            target_layers=[target_layer],
            register_backward=True
        )
    except Exception as e:
        print("no gradcam can be defined over this model")
        return  # Add return to exit early if GradCAM fails
    
    # Create test_loader here, before using it
    
    test_loader = make_loader(
        test_images_paths_np,
        test_true_labels_np,
        transforms=test_transforms,
        cfg=cfg,
        shuffle=False,
    )
    
    #------- saving test images gradcams -------------------------------
    # try: 
    #     gradcam_folder = generate_and_save_gradcam_batch(
    #         model=model,
    #         loader=test_loader,
    #         gradcam_obj=gradcampp,
    #         output_dir=base_dir,
    #         class_names=class_names,
    #         run_name=run_name,
    #         experiment_name=experiment_name
    #     )
        
    #     mlflow.log_artifacts(str(gradcam_folder), artifact_path="test_gradcam_images")
    # except Exception as e:
    #     print(f"Error generating GradCAM for test images: {e}")
    # finally:
    #     # Clean up the local directory after logging to MLflow
    #     shutil.rmtree(gradcam_folder, ignore_errors=True)
        
    try:
        thresholded_gradcam_folder = process_and_save_batch_gradcam_and_Overlay(
            model=model,
            test_loader=test_loader,
            gradcam_obj=gradcampp,
            base_dir=base_dir,
            class_names=class_names,
            min_max_rescale_for_display=min_max_normalization,
            threshold=0.0,
            run_name=run_name,
            experiment_name=experiment_name
        )
        mlflow.log_artifacts(thresholded_gradcam_folder, artifact_path="gradcam_images_with_threshold")
        shutil.rmtree(thresholded_gradcam_folder, ignore_errors=True)
    except Exception as e:
        print(f"Error generating GradCAM with threshold: {e}")
        