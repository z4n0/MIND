import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import optuna
import re
import os
import matplotlib.pyplot as plt # Assuming plt is used by plotting functions
# Assume these are in your utils files:
from utils.data_visualization_functions import plot_learning_curves, plot_confusion_matrix
from classes.ModelManager import ModelManager
from utils.test_functions import evaluate_model
# Also assume make_loader, train_epoch, val_epoch, oversample_minority,
# undersample_majority, freeze_layers_up_to, print_model_summary are available
# from utils or other modules. For brevity, I won't redefine them here.
from utils.train_functions import train_epoch, val_epoch, freeze_layers_up_to, train_epoch_mixUp, oversample_minority, undersample_majority, make_loader
# from utils.data_utils import make_loader, oversample_minority, undersample_majority
# from utils.model_utils import print_model_summary
from utils.transformations_functions import get_transforms, compute_dataset_mean_std
from configs.ConfigLoader import ConfigLoader
from pathlib import Path # Added for Path operations
from optuna.exceptions import TrialPruned

class NestedCVStratifiedByPatient:
    """
    NestedCVStratifiedByPatient is a class that performs nested cross-validation on a dataset stratified by patient.
    It uses a model factory or model manager to create and manage models.
    
    Args:
        df (pd.DataFrame): DataFrame containing image paths and labels.
        cfg (dict): Configuration dictionary.
        labels_np (np.ndarray): Array of labels.
        pat_labels (np.ndarray): Array of patient labels.
        unique_pat_ids (np.ndarray): Array of unique patient IDs.
        train_transforms (transform): Transformations applied to the train set.
        
    NOTE: values are set from cfg unless overridden by kwargs passed to the constructor
        
    Usage:
        #    model = torchvision.models.resnet18(pretrained=True) # or your custom model
        #    model.fc = nn.Linear(model.fc.in_features, num_classes_var) # num_classes_var needs to be accessible
        #    # This factory needs to know num_classes, or it's passed to setup_model in manager
        #    return model

        # experiment = NestedCVStratifiedByPatient(
        #     df, cfg, labels_np, pat_labels, unique_pat_ids,
        #     train_transforms, val_transforms, pretrained_weights,
        #     class_names, model_factory=my_model_factory # or model_manager=my_manager
        # )
        # per_fold_training_metrics, outer_fold_test_results = experiment.run_experiment()
    """
    def __init__(self, df, cfg: ConfigLoader, labels_np, pat_labels, unique_pat_ids, pretrained_weights,
                 class_names, train_transforms=None, val_transforms=None, model_factory=None, model_manager=None, num_folds=None, compute_custom_normalization=False, output_dir: str | None = None,
                 train_fn=None, val_fn=None):

        self.df = df
        self._cfg = cfg
        self.labels_np = labels_np
        self.pat_labels = pat_labels
        self.unique_pat_ids = unique_pat_ids
        self.pretrained_weights = pretrained_weights
        self.best_lr = None
        self.class_names = class_names
        self.model_factory = model_factory # e.g., a function: def create_model(lr): ...
        self.model_manager: ModelManager = model_manager # e.g., an object with setup_model method
        self.train_fn = train_fn if train_fn is not None else train_epoch
        self.val_fn = val_fn if val_fn is not None else val_epoch
        self.num_folds = num_folds if num_folds is not None else self.cfg.get_num_folds()
        self.oversample = self.cfg.get_oversample()
        self.undersample = self.cfg.get_undersample()
        self.num_classes = self._determine_num_classes()
        self.val_set_size = self.cfg.get_val_set_size()
        self.num_epochs = self.cfg.get_num_epochs()
        self.lr_discovery_folds = self.cfg.get_lr_discovery_folds()
        self.is_supported_by_torchvision = self.pretrained_weights is not None
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.random_seed = self.cfg.get_random_seed()
        #store current fold's transforms
        self.current_fold_train_transforms = None
        self.current_fold_val_transforms = None
        self.current_fold_test_transforms = None
        self.per_fold_val_images_paths = {} #type: ignore store validation set image paths for each fold
        self.compute_custom_normalization = compute_custom_normalization
        self.x_outer_len = None
        self.x_outer_len = None
        self.train_image_counts_per_fold = {} #type: ignore
        self.val_image_counts_per_fold = {} #type: ignore
        self._num_outer_images = None
        self.output_dir = Path(output_dir or ".").resolve()
        self.cm_dir = str(self.output_dir / "confusion_matrices")
        self.cm_patient_dir = str(self.output_dir / "confusion_matrices" / "patient")
        self.learning_dir = str(self.output_dir / "learning_curves")
        self._test_pat_ids_per_fold = {} #<-- to store test patient ids for each fold

        print(f"Detected {self.num_classes} unique classes.")

        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Metrics for each fold, keyed by metric name and fold index
        self.per_fold_metrics: dict[str, dict[int, float]] = {
            'train_loss': {},
            'val_loss': {},
            'train_accuracy': {},
            'val_accuracy': {},
            'val_f1': {},
            'val_balanced_accuracy': {},
            'val_precision': {},
            'val_recall': {},
            'val_auc': {},
            'val_mcc': {}
        }

        # Stores results for each fold (e.g., dicts of metrics or results)
        self.fold_results: list[dict] = []
        self._setup_directories()
    
    def get_best_lr(self):
        return self.best_lr
    
    def set_best_lr(self, best_lr):
        self.best_lr = best_lr
    
    def _setup_directories(self):
        """Create output subdirectories for metrics and plots for this run.

        Creates `confusion_matrices/` and `learning_curves/` inside
        the configured `output_dir` so concurrent runs don't clash.
        """
        os.makedirs(self.cm_dir, exist_ok=True)
        os.makedirs(self.cm_patient_dir, exist_ok=True)
        os.makedirs(self.learning_dir, exist_ok=True)
        

    def _compute_patient_level_metrics(self, X_test_outer, y_test_outer, eval_results, fold_idx: int) -> dict:
        """
        Aggregate per-image predictions to patient-level via:
          - majority vote over predicted classes
          - soft vote via mean of per-image probability vectors

        Saves patient-level confusion matrices and prints metrics.

        Returns a dict with patient-level metrics for both schemes.
        metrics are: accuracy, balanced_accuracy, f1
        """
        from collections import defaultdict, Counter
        from utils.test_functions import calculate_classification_metrics

        try:
            # Extract patient ids aligned with dataloader order (shuffle=False)
            patient_ids = np.array([self._extract_patient_id(p) for p in X_test_outer])
            preds_img = np.array(eval_results.get('predictions'))          # [N]
            probs_img = eval_results.get('probs')
            probs_img = np.array(probs_img) if probs_img is not None else None  # [N, C]
            true_img  = np.array(y_test_outer)                             # [N]

            # Group indices by patient
            pat_to_indices: dict[str, list[int]] = defaultdict(list)
            for i, pid in enumerate(patient_ids):
                pat_to_indices[pid].append(i)

            # Majority vote and soft-vote predictions per patient
            pat_true: dict[str, int] = {}
            pat_pred_major: dict[str, int] = {}
            pat_pred_soft: dict[str, int] = {}
            for pid, idxs in pat_to_indices.items():
                # Patient true label as majority of slice labels
                pat_true[pid] = int(Counter(true_img[idxs]).most_common(1)[0][0])
                # Majority vote on slice predictions
                pat_pred_major[pid] = int(Counter(preds_img[idxs]).most_common(1)[0][0])
                # Soft vote: mean probabilities across slices
                if probs_img is not None and len(probs_img) == len(true_img):
                    mean_probs = probs_img[idxs].mean(axis=0) #media delle probabilità
                    pat_pred_soft[pid] = int(mean_probs.argmax())
                else:
                    pat_pred_soft[pid] = pat_pred_major[pid]

            # Build arrays in stable patient order
            ordered_pids = sorted(pat_true.keys())
            y_pat_true  = np.array([pat_true[pid] for pid in ordered_pids])
            y_pat_major = np.array([pat_pred_major[pid] for pid in ordered_pids])
            y_pat_soft  = np.array([pat_pred_soft[pid] for pid in ordered_pids])

            # Compute patient-level metrics
            metrics_major = calculate_classification_metrics(y_pat_true, y_pat_major, class_names=self.class_names)
            metrics_soft  = calculate_classification_metrics(y_pat_true, y_pat_soft,  class_names=self.class_names)

            print(f" [FOLD {fold_idx} PATIENT] Majority: Acc={metrics_major['accuracy']:.4f} | BalAcc={metrics_major['balanced_accuracy']:.4f} | F1={metrics_major['f1']:.4f}")
            print(f" [FOLD {fold_idx} PATIENT] Soft    : Acc={metrics_soft['accuracy']:.4f} | BalAcc={metrics_soft['balanced_accuracy']:.4f} | F1={metrics_soft['f1']:.4f}")

            # Save patient-level confusion matrices
            cm_fig_major = plot_confusion_matrix(metrics_major['confusion_matrix'], self.class_names, f'Patient CM (Majority) Fold {fold_idx}')
            cm_path_major = os.path.join(self.cm_patient_dir, f"confusion_matrix_patient_majority_fold_{fold_idx}.png")
            cm_fig_major.savefig(cm_path_major, dpi=100, bbox_inches='tight')
            plt.close(cm_fig_major)

            cm_fig_soft = plot_confusion_matrix(metrics_soft['confusion_matrix'], self.class_names, f'Patient CM (Soft) Fold {fold_idx}')
            cm_path_soft = os.path.join(self.cm_patient_dir, f"confusion_matrix_patient_soft_fold_{fold_idx}.png")
            cm_fig_soft.savefig(cm_path_soft, dpi=100, bbox_inches='tight')
            plt.close(cm_fig_soft)

            return {
                # Balanced accuracy (requested primary)
                "patient_major_bal_acc": float(metrics_major['balanced_accuracy']),
                "patient_soft_bal_acc": float(metrics_soft['balanced_accuracy']),
                # AUC/MCC (if available)
                "patient_major_auc": float(metrics_major.get('auc')) if metrics_major.get('auc') is not None else None,
                "patient_soft_auc": float(metrics_soft.get('auc')) if metrics_soft.get('auc') is not None else None,
                "patient_major_mcc": float(metrics_major.get('mcc')) if metrics_major.get('mcc') is not None else None,
                "patient_soft_mcc": float(metrics_soft.get('mcc')) if metrics_soft.get('mcc') is not None else None,
                # Precision/Recall
                "patient_major_precision": float(metrics_major['precision']),
                "patient_soft_precision": float(metrics_soft['precision']),
                "patient_major_recall": float(metrics_major['recall']),
                "patient_soft_recall": float(metrics_soft['recall']),
            }
        except Exception as e:
            print(f"Warning: patient-level aggregation failed: {e}")
            return {
                "patient_major_bal_acc": None,
                "patient_soft_bal_acc": None,
                "patient_major_auc": None,
                "patient_soft_auc": None,
                "patient_major_mcc": None,
                "patient_soft_mcc": None,
                "patient_major_precision": None,
                "patient_soft_precision": None,
                "patient_major_recall": None,
                "patient_soft_recall": None,
            }
    
    @property
    def cfg(self):
        return self._cfg

    @property
    def test_pat_ids_per_fold(self):
        """Returns the patient IDs used for the outer test set in all folds."""
        return self._test_pat_ids_per_fold
    
    @property
    def num_outer_images(self):
        """Returns the number of images in the outer test fold."""
        return self._num_outer_images # CORRECT: The property reads from the internal attribute
    
    def get_early_stopping_split_counts(self):
        """
        Returns the number of train and validation images used for early stopping in each fold.
        
        Returns:
            tuple: A tuple containing two dictionaries:
                (train_image_counts_per_fold, val_image_counts_per_fold)
        """
        return self.train_image_counts_per_fold, self.val_image_counts_per_fold
    
    

    def get_current_fold_transforms(self):
        """Return the train/val/test transforms for the current outer fold.

        Returns:
            tuple: (train_transforms, val_transforms, test_transforms)
        """
        return self.current_fold_train_transforms, self.current_fold_val_transforms, self.current_fold_test_transforms

    def get_test_patient_ids_for_fold(self, fold_idx: int) -> np.ndarray:
        """Returns the patient IDs used for the outer test set in a given fold 
        arg: fold_idx (int): The index of the fold for which to get the test patient IDs array
        return: np.ndarray: The patient IDs used for the outer test set in the given fold decided by index arg
        ."""
        return self._test_pat_ids_per_fold.get(fold_idx)
    
    
    def _determine_num_classes(self):
        """Infer the number of classes from the provided labels array."""
        return len(np.unique(self.labels_np))

    def _setup_directories(self):
        """Create (idempotently) the confusion-matrix and learning-curve dirs.

        Note: This overload must mirror the earlier definition to also
        create the patient subfolder to avoid race/time-of-use errors.
        """
        os.makedirs(self.cm_dir, exist_ok=True)
        # Ensure patient subfolder exists for patient-level CMs
        if hasattr(self, 'cm_patient_dir') and self.cm_patient_dir:
            os.makedirs(self.cm_patient_dir, exist_ok=True)
        os.makedirs(self.learning_dir, exist_ok=True)

    def _extract_patient_id(self, image_path):
        """Extract a patient identifier from an image path using regex.

        Args:
            image_path (str): Full path to the image file.

        Returns:
            str: Extracted patient identifier.
        """
        match = re.search(r'(\d{4})', image_path) # Example regex
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Could not extract patient ID from {image_path}")

    def _get_model_and_device(self, learning_rate_for_factory=None):
        """
        Set up and return the model and device for training/evaluation.
        
        This method handles model creation based on either the model manager or
        model factory provided during class initialization. It also applies transfer
        learning or fine-tuning settings if configured.
        
        Args:
            learning_rate_for_factory (float, optional): Learning rate to pass to the model factory.
                Only used when using a model_factory instead of model_manager. If None and using
                model_factory, falls back to the learning rate from config.
                
        Returns:
            tuple: A tuple containing:
                - model (nn.Module): The instantiated model moved to the appropriate device
                - device (torch.device): The device to use for computation
                
        Raises:
            ValueError: If neither model_manager nor model_factory was provided during initialization
            
        Note:
            When using model_factory, the learning rate is passed to the factory function.
            When using model_manager, the model setup is delegated to the manager's setup_model method.
        """
        torch.manual_seed(self.cfg.get_random_seed())
        
        if self.model_manager is not None:
            # print(f"Using model manager: {self.model_manager}") # For debugging
            model, device = self.model_manager.setup_model(
                num_classes=self.num_classes,
                pretrained_weights=self.pretrained_weights
            )
            
        elif self.model_factory is not None:
            # print(f"Using passed model factory.") # For debugging
            device = self.device
            if learning_rate_for_factory is None:
                # This case might need a default LR from cfg or error if not for Optuna
                if "lr" not in self.cfg.optimizer:
                    raise ValueError("lr not found in optimizer config")
                lr = self.cfg.optimizer["lr"]# Get a default if not for Optuna
                print(f"Warning: model_factory used without specific LR, using default/cfg LR: {lr}")
            else:
                lr = learning_rate_for_factory
            model = self.model_factory(lr).to(device) # Pass LR to factory
        else:
            raise ValueError("Either model_manager or model_factory must be provided.")

        if self.cfg.training.get("transfer_learning"):
            freeze_layers_up_to(model, self.cfg)
        
        return model, device
    
    def get_val_image_paths_per_fold(self, fold_idx):
        """ 
        Returns the image paths for the validation set of a given fold.
        Args:
            fold_idx (int): The index of the fold.
        Returns:
            list: The image paths for the validation set of the given fold.
        """
        return self.per_fold_val_images_paths[fold_idx]

    def _get_loss_function(self, y_train_data_for_weighting):
        """Return a CrossEntropyLoss, optionally weighted by class imbalance.

        If `training.weighted_loss` is True and no over/undersampling is used,
        compute class weights on the provided training labels.
        """
        if (self.cfg.training.get("weighted_loss", False) and
            not (self.cfg.training.get("oversample", False) or self.cfg.training.get("undersample", False))):
            unique_labels = np.unique(y_train_data_for_weighting)
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=unique_labels,
                y=y_train_data_for_weighting
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            return nn.CrossEntropyLoss()

    def _get_optimizer(self, model, learning_rate, opt_overrides=None):
        """
        Build the optimizer (Adam/AdamW), allowing per-trial overrides.

        Args:
            model: nn.Module
            learning_rate (float): base LR
            opt_overrides (dict | None): optional keys:
                - weight_decay (float)
                - betas (tuple[float, float])
                - eps (float)
                - backbone_lr_mult (float)
                - no_decay_on_norm_bias (bool)
                - optimizer_name (str): 'Adam' or 'AdamW' (default: cfg)
        """
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found.")

        ov = opt_overrides or {}
        optimizer_name = ov.get("optimizer_name", self.cfg.get_optimizer_name())

        weight_decay = float(
            ov.get("weight_decay", float(self.cfg.optimizer["weight_decay"]))
        )
        betas = ov.get("betas", (0.9, 0.999))
        eps = float(ov.get("eps", 1e-8))
        backbone_lr_mult = float(ov.get("backbone_lr_mult", 1.0))
        no_decay_on_norm_bias = bool(ov.get("no_decay_on_norm_bias", True))

        param_groups = self._build_param_groups(
            model=model,
            base_lr=float(learning_rate),
            weight_decay=weight_decay,
            backbone_lr_mult=backbone_lr_mult,
            no_decay_on_norm_bias=no_decay_on_norm_bias,
        )

        if optimizer_name == "Adam":
            return optim.Adam(param_groups, lr=learning_rate,
                            betas=betas, eps=eps)
        elif optimizer_name == "AdamW":
            return optim.AdamW(param_groups, lr=learning_rate,
                            betas=betas, eps=eps)
        elif optimizer_name == "SGD":
            # you said to ignore SGD; keeping for completeness
            return optim.SGD(param_groups, lr=learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")

    def _get_scheduler(self, optimizer):
        """Return LR scheduler and a flag indicating epoch-stepped scheduling.

        Returns:
            tuple[torch.optim.lr_scheduler._LRScheduler, bool]:
                (scheduler, using_cosine_scheduler)
        """
        if self.cfg.get_model_name().lower() == "vit": 
            print("Using CosineAnnealingWarmRestarts scheduler")
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            return CosineAnnealingWarmRestarts(optimizer, T_0=7, T_mult=2, eta_min=5e-6), True
        else:
            try:
                patience = self.cfg.optimizer.get("patience")
            except:
                raise ValueError(f"patience not found in cfg for {self.cfg.get_model_name()}")
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5), False

    # Aggiungi questo metodo all'interno della tua classe NestedCVStratifiedByPatient

    def find_best_lr_grid_search(self,
                        #  lr_candidates: list[float] = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                           num_epochs_per_lr: int = 10,
                           n_splits_lr: int = 3): # New: Number of splits for mini-CV
        """
            Performs a more robust Grid Search for the learning rate using a mini cross-validation.
            This method is executed ONCE before the main cross-validation.
            it uses all the patients in the training set to find the best learning rate once and then reuse it for all the folds.
            

            Args:
                num_epochs_per_lr (int): Number of epochs to train for each LR on each split.
                n_splits_lr (int): Number of splits to use for the internal cross-validation.

            Returns:
                float: The learning rate with the best average validation loss across all splits.
        """
        print(f"--- Starting Robust Grid Search for Learning Rate ---")
        print(f"Internal CV splits: {n_splits_lr}, Epochs per LR: {num_epochs_per_lr}")
        
        lr_map = {
                    'resnet50': [5e-5, 1e-4, 3e-4, 5e-4],
                    'resnet101': [1e-5, 5e-5, 1e-4, 2e-4],
                    'densenet121': [5e-5, 1e-4, 3e-4, 5e-4],
                    'densenet169': [1e-5, 5e-5, 1e-4, 2e-4]
                }
                
        current_model_name = self.cfg.get_model_name().lower() 

                # 3. Select the appropriate LR list, with a fallback to a default
        default_lrs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        lr_candidates_for_this_model = lr_map.get(current_model_name, default_lrs)
        print(f"Using LR candidates for '{current_model_name}': {lr_candidates_for_this_model}")

                # Chiama la nuova funzione UNA VOLTA prima di iniziare la CV
        # lr_candidates = lr_candidates_for_this_model
        lr_performance = { lr: [] for lr in lr_candidates_for_this_model }

        # Improvement 1: Mini cross-validation for the grid search
        skf = StratifiedKFold(n_splits=n_splits_lr, shuffle=True, random_state=self.random_seed)

        # Improvement 2: Consistent model initialization
        # Initialize the model once to get its starting weights
        initial_model, device = self._get_model_and_device()
        initial_state_dict = initial_model.state_dict()
        del initial_model # Free up memory

        for fold_idx, (train_pat_idx, val_pat_idx) in enumerate(skf.split(self.unique_pat_ids, self.pat_labels)):
            print(f"\n--- LR Search - Split {fold_idx + 1}/{n_splits_lr} ---")
            train_pats = self.unique_pat_ids[train_pat_idx]
            val_pats = self.unique_pat_ids[val_pat_idx]

            # Get image paths for this split
            train_mask = self.df["patient_id"].isin(train_pats)
            val_mask = self.df["patient_id"].isin(val_pats)
            # Get image paths and labels for this split take all paths with such mask
            X_train, y_train = self.df.loc[train_mask, "image_path"].values, self.df.loc[train_mask, "label"].values
            X_val, y_val = self.df.loc[val_mask, "image_path"].values, self.df.loc[val_mask, "label"].values

            # Handle balancing (oversampling/undersampling)
            if self.oversample:
                X_train_bal, y_train_bal = oversample_minority(X_train, y_train, self.random_seed)
            elif self.undersample:
                X_train_bal, y_train_bal = undersample_majority(X_train, y_train, self.random_seed)
            else:
                X_train_bal, y_train_bal = X_train, y_train

            # Create DataLoaders
            train_stats = None
            val_stats = None
            if self.train_transforms is None:
                train_stats = compute_dataset_mean_std(X_train_bal, self.cfg, is_supported_by_torchvision=self.is_supported_by_torchvision)
            if self.val_transforms is None:
                val_stats = compute_dataset_mean_std(X_train_bal, self.cfg, is_supported_by_torchvision=self.is_supported_by_torchvision)

            train_transforms = self.train_transforms or get_transforms(self.cfg, fold_specific_stats=train_stats)[0]
            val_transforms = self.val_transforms or get_transforms(self.cfg, fold_specific_stats=val_stats)[1]
            train_loader = make_loader(X_train_bal, y_train_bal, train_transforms, self.cfg, shuffle=True)
            val_loader = make_loader(X_val, y_val, val_transforms, self.cfg, shuffle=False)

            # Iterate through each learning rate for this split
            for lr in lr_candidates_for_this_model:
                print(f"  Testing LR: {lr:.6f}")
                # Re-create the model and load the same initial weights
                model, device = self._get_model_and_device()
                model.load_state_dict(initial_state_dict)

                loss_function = self._get_loss_function(y_train_bal)
                optimizer = self._get_optimizer(model, lr)
                
                epoch_val_losses = []
                for epoch in range(num_epochs_per_lr):
                    self.train_fn(model, train_loader, optimizer, loss_function, device)
                    val_loss, _, _, _, _, _, _, _ = self.val_fn(model, val_loader, loss_function, device)
                    epoch_val_losses.append(val_loss)

                # Improvement 3: Evaluate on average of last few epochs
                # Use last 3 epochs, or all if fewer than 3 were run
                num_epochs_to_average = min(3, num_epochs_per_lr)
                avg_last_val_loss = np.mean(epoch_val_losses[-num_epochs_to_average:])
                lr_performance[lr].append(avg_last_val_loss)
                print(f"    Avg Val Loss (last {num_epochs_to_average} epochs): {avg_last_val_loss:.4f}")

        # Calculate final average performance across all splits
        avg_lr_performance = {lr: np.mean(losses) for lr, losses in lr_performance.items()}
        
        # Select the best LR based on the overall average
        best_lr = min(avg_lr_performance, key=avg_lr_performance.get)

        print(f"\n--- Grid Search Concluded ---")
        print("Average Validation Loss per LR:")
        for lr, avg_loss in avg_lr_performance.items():
            print(f"  LR: {lr:.6f} -> Avg Loss: {avg_loss:.4f}")
        print(f"\nBest Learning Rate found: {best_lr:.6f}")
        
        return best_lr
    
    def _build_param_groups(self, model, base_lr, weight_decay,
                        backbone_lr_mult=1.0, no_decay_on_norm_bias=True):
        """
        Create parameter groups:
        - apply weight decay to 'decay' params
        - set weight decay = 0 to biases and normalization layers
        - (optional) use a lower LR for the backbone

        Returns:
            list[dict]: parameter groups for the optimizer.
        """
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_norm = any(k in name.lower()
                        for k in ["bn", "gn", "ln", "norm", "layernorm"])
            is_bias = name.endswith("bias")
            if no_decay_on_norm_bias and (is_norm or is_bias):
                no_decay.append(param)
            else:
                decay.append(param)

        groups = [
            {"params": decay, "weight_decay": float(weight_decay), "lr": base_lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": base_lr},
        ]

        # Optional discriminative LR: scale backbone layers
        if backbone_lr_mult != 1.0:
            head_names = ("fc", "classifier", "head")
            head_params, bb_params = [], []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if any(h in n.lower() for h in head_names):
                    head_params.append(p)
                else:
                    bb_params.append(p)
            if head_params and bb_params:
                groups = [
                    {"params": bb_params, "weight_decay": float(weight_decay),
                    "lr": base_lr * float(backbone_lr_mult)},
                    {"params": head_params, "weight_decay": float(weight_decay),
                    "lr": base_lr},
                ]
                # Preserve the no-decay split inside each group if you wish;
                # to keep it simple, we keep weight_decay on head/backbone groups.

        return groups
    
    def _tune_optim_hparams(self, X_train_outer, y_train_outer):
        """
        Perform hyperparameter optimization for the optimizer using Optuna.

        This function tunes optimizer-related hyperparameters (learning rate, weight decay,
        beta2, epsilon, backbone learning rate multiplier, etc.) on the outer training set,
        using an inner cross-validation strategy. The best hyperparameters are determined
        by minimizing the mean (1 - balanced accuracy) across inner folds, as evaluated 
        by the objective function.

        Args:
            X_train_outer (np.ndarray or list[str]): Training image paths for the outer fold.
            y_train_outer (np.ndarray or list[int]): Training labels (aligned to image paths) for the outer fold.

        Returns:
            dict: A dictionary of the best optimizer hyperparameters found, containing:
                - 'lr': Optimized learning rate (float)
                - 'weight_decay': Optimized weight decay coefficient (float)
                - 'betas': Tuple of beta values for Adam/AdamW optimizer (tuple(float, float))
                - 'eps': Optimized epsilon value for optimizer stabilization (float)
                - 'backbone_lr_mult': Backbone learning rate multiplier (float)
                - 'optimizer_name': Optimizer type as defined in config (str)
                - 'no_decay_on_norm_bias': Whether to avoid weight decay on normalization and bias params (bool)

        Note:
            - Uses a TPESampler with a fixed random seed for reproducibility.
            - Uses a MedianPruner for early stopping of bad trials.
            - The number of Optuna trials is drawn from the config (default: 8).
            - Prints and returns the best optimizer hyperparameter set found.
        """
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        )
        study.optimize(
            lambda t: self._objective(t, X_train_outer, y_train_outer),
            n_trials=self.cfg.training.get("n_trials", 4),
        )
        
        best = study.best_trial.params # this is a dictionary containing the best hyperparameters
        # Ensure presence of all keys with sensible defaults
        best_opt = {
            "lr": float(best["lr"]),
            "weight_decay": float(best.get("weight_decay", self.cfg.optimizer["weight_decay"])),
            "betas": (0.9, float(best.get("beta2", 0.999))),
            "eps": float(best.get("eps", 1e-8)),
            "backbone_lr_mult": float(best.get("backbone_lr_mult", 1.0)),
            "optimizer_name": self.cfg.get_optimizer_name(),
            "no_decay_on_norm_bias": True,
        }
        print(f"Best inner-CV hparams: {best_opt}")
        return best_opt
    
    def _run_combined_hparam_tuning(self, X_train_outer, y_train_outer):
        """Run a single Optuna study to retrieve LR and optimizer hyperparameters."""
        opt_cfg = self._tune_optim_hparams(X_train_outer, y_train_outer)
        best_lr = float(opt_cfg.get("lr", self.cfg.get_learning_rate()))
        print(f"  Best LR from combined inner CV = {best_lr:.6f}")
        return best_lr, opt_cfg
    
    def _objective(self, trial, X_train_outer_fold, y_train_outer_fold):
        """
        Optuna objective: per-trial inner CV over outer-train patients.
        Jointly tunes LR + optimizer hyper-parameters, and *minimizes*
        (1 - patient-level balanced accuracy) averaged across inner folds.

        Args:
            trial (optuna.Trial): Current Optuna trial.
            X_train_outer_fold (np.ndarray | list[str]): Image paths for outer-train.
            y_train_outer_fold (np.ndarray | list[int]): Labels aligned to paths.

        Returns:
            float: Mean (1 - BalAcc) over inner folds (lower is better).
        """
        # ── 0) Sample hyper-parameters for THIS trial ────────────────────────────
        lr = trial.suggest_float("lr", 5e-6, 2e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        beta2 = trial.suggest_float("beta2", 0.95, 0.999)  # optional but cheap
        eps = trial.suggest_float("eps", 1e-9, 1e-7, log=True)  # optional
        backbone_lr_mult = trial.suggest_categorical(
            "backbone_lr_mult", [0.1, 0.3, 1.0]
        )

        opt_overrides = {
            "optimizer_name": self.cfg.get_optimizer_name(),  # 'AdamW' or 'Adam'
            "weight_decay": float(weight_decay),
            "betas": (0.9, float(beta2)),
            "eps": float(eps),
            "backbone_lr_mult": float(backbone_lr_mult),
            "no_decay_on_norm_bias": True,
        }

        # ── 1) Build an inner CV over PATIENTS (grouped) ────────────────────────
        df_outer = pd.DataFrame(
            {"image_path": X_train_outer_fold, "label": y_train_outer_fold}
        )
        df_outer["patient_id"] = df_outer["image_path"].apply(self._extract_patient_id)

        # One row per patient with a representative label for stratification
        df_pat = df_outer.groupby("patient_id", as_index=False)["label"].first()
        inner_pat_ids = df_pat["patient_id"].values
        inner_pat_labels = df_pat["label"].values

        inner_skf = StratifiedKFold(
            n_splits=self.lr_discovery_folds, shuffle=True,
            random_state=self.random_seed
        )

        inner_scores = []  # store 1 - BalAcc per inner fold

        # How many epochs to run per inner fold (short; just for selection)
        inner_epochs = int(self.cfg.training.get("num_inner_epochs_hyp", 3))

        # ── 2) Loop inner folds ─────────────────────────────────────────────────
        for inner_fold_idx, (tr_pat_idx, va_pat_idx) in enumerate(
            inner_skf.split(inner_pat_ids, inner_pat_labels)
        ):
            # Patients for this inner split
            tr_pats = inner_pat_ids[tr_pat_idx]
            va_pats = inner_pat_ids[va_pat_idx]

            # Select images/labels by patient
            tr_mask = df_outer["patient_id"].isin(tr_pats)
            va_mask = df_outer["patient_id"].isin(va_pats)

            X_tr = df_outer.loc[tr_mask, "image_path"].values
            y_tr = df_outer.loc[tr_mask, "label"].values
            X_va = df_outer.loc[va_mask, "image_path"].values
            y_va = df_outer.loc[va_mask, "label"].values

            # Optional resampling on inner-train
            if self.oversample:
                X_tr_bal, y_tr_bal = oversample_minority(X_tr, y_tr, self.random_seed)
            elif self.undersample:
                X_tr_bal, y_tr_bal = undersample_majority(X_tr, y_tr, self.random_seed)
            else:
                X_tr_bal, y_tr_bal = X_tr, y_tr

            # Build loaders using transforms already set for the OUTER fold
            train_loader = make_loader(
                image_paths=np.asarray(X_tr_bal, dtype=str),
                labels=np.asarray(y_tr_bal, dtype=int),
                transforms=self.current_fold_train_transforms,
                cfg=self.cfg,
                shuffle=True,
            )
            val_loader = make_loader(
                image_paths=np.asarray(X_va, dtype=str),
                labels=np.asarray(y_va, dtype=int),
                transforms=self.current_fold_val_transforms,
                cfg=self.cfg,
                shuffle=False,
            )

            # Model + optimizer for THIS trial
            model, device = self._get_model_and_device(
                learning_rate_for_factory=lr
            )
            
            loss_fn = self._get_loss_function(y_tr_bal)
            optimizer = self._get_optimizer(
                model, lr, opt_overrides=opt_overrides
            )

            # Short inner training
            best_val_loss = float("inf")
            for epoch in range(inner_epochs):
                train_loss_e, _ = self.train_fn(
                    model, train_loader, optimizer, loss_fn, device
                )
                val_loss_e, *_ = self.val_fn(
                    model, val_loader, loss_fn, device
                )

                # Track best (optional; keeps trial robust to noise)
                if val_loss_e < best_val_loss:
                    best_val_loss = val_loss_e

                # Enable pruning
                trial.report(val_loss_e, step=(
                    inner_fold_idx * max(1, inner_epochs) + epoch
                ))
                if trial.should_prune():
                    raise TrialPruned()

            # ── 3) Patient-level aggregation on the INNER-VAL split ────────────
            # We mirror your outer evaluation: predict on val_loader and
            # aggregate per patient with majority/soft voting.
            eval_res = evaluate_model(
                model=model,
                dataloader=val_loader,
                class_names=self.class_names,
                return_misclassified=False,
            )
            # Extract patient ids aligned with X_va order
            va_patient_ids = np.array([self._extract_patient_id(p) for p in X_va])
            preds_img = np.array(eval_res.get("predictions"))
            probs_img = eval_res.get("probs")
            probs_img = np.array(probs_img) if probs_img is not None else None
            true_img = np.array(y_va)

            # Group indices by patient
            from collections import defaultdict, Counter
            pat_to_indices = defaultdict(list)
            for i, pid in enumerate(va_patient_ids):
                pat_to_indices[pid].append(i)

            # Majority and (if available) soft voting per patient
            pat_true, pat_pred_major, pat_pred_soft = {}, {}, {}
            for pid, idxs in pat_to_indices.items():
                pat_true[pid] = int(Counter(true_img[idxs]).most_common(1)[0][0])
                pat_pred_major[pid] = int(
                    Counter(preds_img[idxs]).most_common(1)[0][0]
                )
                if probs_img is not None and len(probs_img) == len(true_img):
                    mean_probs = probs_img[idxs].mean(axis=0)
                    pat_pred_soft[pid] = int(np.argmax(mean_probs))
                else:
                    pat_pred_soft[pid] = pat_pred_major[pid]

            # Stable patient order
            ordered_pids = sorted(pat_true.keys())
            y_pat_true = np.array([pat_true[p] for p in ordered_pids])
            y_pat_major = np.array([pat_pred_major[p] for p in ordered_pids])
            y_pat_soft = np.array([pat_pred_soft[p] for p in ordered_pids])

            # Compute patient-level balanced accuracy; prefer soft vote
            from sklearn.metrics import balanced_accuracy_score
            bal_acc_major = balanced_accuracy_score(y_pat_true, y_pat_major)
            bal_acc_soft = balanced_accuracy_score(y_pat_true, y_pat_soft)
            bal_acc = float(bal_acc_soft)  # use soft as primary

            # We minimize (1 - BalAcc)
            inner_scores.append(1.0 - bal_acc)

        # ── 4) Return mean (1 - BalAcc) across inner folds ─────────────────────
        return float(np.mean(inner_scores))


    # def _objective(self, trial, X_train_outer_fold, y_train_outer_fold):
    #     """Optuna objective: inner CV to evaluate candidate learning rates.
    #     Args:
    #         trial (optuna.Trial): The trial object.
    #         X_train_outer_fold (np.ndarray): The training set.
    #         y_train_outer_fold (np.ndarray): The training labels.
    #     Returns:
    #         float: The average validation loss.
    #     """
    #     candidate_lr = trial.suggest_float("lr", 5e-6, 2e-3, log=True)

    #     df_outer_train_fold = pd.DataFrame({"image_path": X_train_outer_fold, "label": y_train_outer_fold})
    #     # Ensure patient_id extraction is robust or pre-calculated
    #     df_outer_train_fold["patient_id"] = df_outer_train_fold["image_path"].apply(self._extract_patient_id)
        
    #     df_pat_inner = df_outer_train_fold.groupby("patient_id", as_index=False)["label"].first()
    #     inner_pat_ids = df_pat_inner["patient_id"].values
    #     inner_pat_labels = df_pat_inner["label"].values

    #     inner_skf = StratifiedKFold(
    #         n_splits = self.lr_discovery_folds, #or Get from cfg
    #         shuffle=True,
    #         random_state=self.random_seed
    #     )
    #     inner_val_losses = []

    #     for inner_train_pat_idx, inner_val_pat_idx in inner_skf.split(inner_pat_ids, inner_pat_labels):
    #         these_train_pats = inner_pat_ids[inner_train_pat_idx]
    #         these_val_pats = inner_pat_ids[inner_val_pat_idx]

    #         train_mask_inner = df_outer_train_fold["patient_id"].isin(these_train_pats)
    #         val_mask_inner = df_outer_train_fold["patient_id"].isin(these_val_pats)

    #         X_train_inner = df_outer_train_fold.loc[train_mask_inner, "image_path"].values
    #         y_train_inner = df_outer_train_fold.loc[train_mask_inner, "label"].values
    #         X_val_inner = df_outer_train_fold.loc[val_mask_inner, "image_path"].values
    #         y_val_inner = df_outer_train_fold.loc[val_mask_inner, "label"].values

    #         # Handle over/undersampling (assuming functions are imported)
    #         if self.oversample:
    #             # from utils.data_utils import oversample_minority
    #             X_train_inner_bal, y_train_inner_bal = oversample_minority(
    #                 X_train_inner, y_train_inner, random_seed=self.cfg.get_random_seed()
    #             )
    #         elif self.undersample:
    #             # from utils.data_utils import undersample_majority
    #             X_train_inner_bal, y_train_inner_bal = undersample_majority(
    #                 X_train_inner, y_train_inner, random_seed=self.cfg.get_random_seed()
    #             )
    #         else:
    #             X_train_inner_bal, y_train_inner_bal = X_train_inner, y_train_inner

    #         # from utils.data_utils import make_loader
    #         train_loader_inner = make_loader(
    #             image_paths=np.asarray(X_train_inner_bal, dtype=str),
    #             labels=np.asarray(y_train_inner_bal, dtype=int),
    #             transforms=self.current_fold_train_transforms,
    #             cfg=self.cfg,
    #             shuffle=True
    #         )
            
    #         val_loader_inner = make_loader(
    #             image_paths=np.asarray(X_val_inner, dtype=str),
    #             labels=np.asarray(y_val_inner, dtype=int),
    #             transforms=self.current_fold_val_transforms,
    #             cfg=self.cfg,
    #             shuffle=False
    #         )

    #         model_inner, device_inner = self._get_model_and_device(learning_rate_for_factory=candidate_lr)
    #         loss_function_inner = self._get_loss_function(y_train_inner_bal) # Pass y_train_inner_bal for weighting
    #         optimizer_inner = self._get_optimizer(model_inner, candidate_lr)

    #         inner_num_epochs = self.cfg.training.get("num_inner_epochs_hyp", 3) # Get from cfg
    #         best_inner_loss = float("inf")

    #         for epoch in range(inner_num_epochs):
    #             # from utils.training_utils import train_epoch, val_epoch
    #             train_loss_e, _ = self.train_fn(model_inner, train_loader_inner, optimizer_inner, loss_function_inner, device_inner)
    #             val_loss_e, *_ = self.val_fn(model_inner, val_loader_inner, loss_function_inner, device_inner)

    #             # update best and report current status
    #             if val_loss_e < best_inner_loss:
    #                 best_inner_loss = val_loss_e

    #             # enable pruning
    #             trial.report(val_loss_e, step=epoch)
    #             if trial.should_prune():
    #                 raise TrialPruned()

    #         inner_val_losses.append(best_inner_loss)
    #     return float(np.mean(inner_val_losses))

    def _tune_learning_rate(self, X_train_outer, y_train_outer):
        """
        Run Optuna to select the best learning rate (LR) using inner cross-validation
        on the training patients.

        This method performs hyperparameter optimization for the learning rate by
        running an Optuna study. The study uses a TPE sampler for efficient search
        and a median pruner to terminate unpromising trials early. The objective
        function is evaluated using inner cross-validation on the provided training
        data, and the best learning rate is selected based on the lowest validation
        loss.

        Args:
            X_train_outer (np.ndarray or list): Array or list of image paths for the
                outer training set (used for inner CV).
            y_train_outer (np.ndarray or list): Array or list of labels corresponding
                to X_train_outer.

        Returns:
            float: The best learning rate found by Optuna inner CV.

        Notes:
            - The number of learning rate candidates (n_trials) is currently set to 2.
            - The pruner will start pruning after 2 startup trials.
            - The objective function is defined in self._objective.
        """
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.cfg.get_random_seed()),
            # Pruner to avoid wasting time on trials that will not improve the best result.
            # Prunes if a trial’s metric is worse than the median of completed trials at the same step.
            # n_startup_trials: The number of trials that must be completed before the pruning starts.
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2)
        )
        # Use a lambda to pass the extra arguments to the objective function.
        study.optimize(
            lambda trial: self._objective(trial, X_train_outer, y_train_outer),
            n_trials=2,  # Number of LR candidates tested in inner CV.
        )
        best_lr = study.best_params["lr"]
        print(f"  Best LR from inner CV = {best_lr:.6f}")
        return best_lr
    
    def _append_fold_metrics(self, fold_idx, train_loss, val_loss, train_acc, val_acc, val_f1, val_bal_acc, val_prec, val_recall, val_auc, val_mcc):
                """Append a single epoch's metrics into the per-fold history."""
                self.per_fold_metrics['train_loss'][fold_idx].append(train_loss)
                self.per_fold_metrics['val_loss'][fold_idx].append(val_loss)
                self.per_fold_metrics['train_accuracy'][fold_idx].append(train_acc)
                self.per_fold_metrics['val_accuracy'][fold_idx].append(val_acc)
                self.per_fold_metrics['val_f1'][fold_idx].append(val_f1)
                self.per_fold_metrics['val_balanced_accuracy'][fold_idx].append(val_bal_acc)
                self.per_fold_metrics['val_precision'][fold_idx].append(val_prec)
                self.per_fold_metrics['val_recall'][fold_idx].append(val_recall)
                self.per_fold_metrics['val_auc'][fold_idx].append(val_auc)
                self.per_fold_metrics['val_mcc'][fold_idx].append(val_mcc)

    def _train_model_with_early_stopping(self, fold_idx, X_train_outer, y_train_outer, opt_cfg):
        """
        Train on outer-train with an early-stopping split and return best model path.
        `opt_cfg` is a dict with keys: lr, weight_decay, betas, eps, backbone_lr_mult, ...
        """
        X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
            X_train_outer, y_train_outer,
            test_size=self.val_set_size,
            stratify=y_train_outer,
            random_state=self.random_seed
        )
        self.train_image_counts_per_fold[fold_idx] = len(X_train_es)
        self.val_image_counts_per_fold[fold_idx] = len(X_val_es)
        self.per_fold_val_images_paths[fold_idx] = X_val_es

        if self.oversample:
            X_train_es_bal, y_train_es_bal = oversample_minority(X_train_es, y_train_es, self.cfg.data_splitting["random_seed"])
        elif self.undersample:
            X_train_es_bal, y_train_es_bal = undersample_majority(X_train_es, y_train_es, self.cfg.data_splitting["random_seed"])
        else:
            X_train_es_bal, y_train_es_bal = X_train_es, y_train_es

        train_loader_es = make_loader(X_train_es_bal, y_train_es_bal, self.current_fold_train_transforms, self.cfg, shuffle=True)
        val_loader_es   = make_loader(X_val_es, y_val_es, self.current_fold_val_transforms, self.cfg, shuffle=False)

        lr_for_fold = float(opt_cfg["lr"])
        model, device = self._get_model_and_device(learning_rate_for_factory=lr_for_fold)
        from utils.train_functions import print_model_summary
        print_model_summary(model)

        loss_function = self._get_loss_function(y_train_es_bal)
        optimizer = self._get_optimizer(model, lr_for_fold, opt_overrides=opt_cfg)
        scheduler, using_cosine_scheduler = self._get_scheduler(optimizer)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        model_save_path = str(self.output_dir / f"best_model_fold_{fold_idx}.pth")

        for epoch in range(self.num_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            if self.cfg.get_mixup_alpha() > 0:
                train_loss, train_acc = train_epoch_mixUp(model, train_loader_es, optimizer, loss_function, device, self.cfg.get_mixup_alpha())
            else:
                train_loss, train_acc = self.train_fn(model, train_loader_es, optimizer, loss_function, device)

            val_loss, val_acc, val_prec, val_recall, val_f1, val_bal_acc, val_roc_auc, val_mcc = self.val_fn(model, val_loader_es, loss_function, device)

            if using_cosine_scheduler:
                scheduler.step(epoch)
            else:
                scheduler.step(val_loss)

            self._append_fold_metrics(fold_idx, train_loss, val_loss, train_acc, val_acc, val_f1, val_bal_acc, val_prec, val_recall, val_roc_auc, val_mcc)

            print(f" Fold {fold_idx} Epoch {epoch+1}/{self.num_epochs}: "
                f"Tr L: {train_loss:.4f}, Tr Acc: {train_acc:.4f}, "
                f"Val L: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Val Bal Acc: {val_bal_acc:.4f}, Val Roc AUC: {val_roc_auc:.4f}, "
                f"Val_mcc: {val_mcc:.4f}, Val F1: {val_f1:.4f} lr: {current_lr:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve == self.cfg.training["early_stopping_patience"]:
                    print(f"Early stopping triggered at epoch {epoch+1} for fold {fold_idx}")
                    break

        return model_save_path, loss_function

    def _evaluate_fold_on_test_set(self, fold_idx, model_path, loss_function_for_eval, X_test_outer, y_test_outer, best_lr_for_fold):
        """Load best checkpoint for the fold and evaluate on the outer test split.
        return final_test_loss, final_test_acc, final_test_prec, final_test_recall, final_test_f1, final_test_balanced_acc, test_auc, test_mcc, patient_metrics
        """
        model, device = self._get_model_and_device() # Get a fresh model instance
        model.load_state_dict(torch.load(str(model_path), map_location=device)) # Load the best model for the fold
        model.eval() # Set the model to evaluation mode

        test_loader = make_loader(X_test_outer, y_test_outer, self.current_fold_val_transforms, self.cfg, shuffle=False)

        print(f"Test set class counts for fold {fold_idx}: {pd.Series(y_test_outer).value_counts().to_dict()}")
        print(f"percentage of classes in test set: {pd.Series(y_test_outer).value_counts() / len(y_test_outer)}")

        # Using val_epoch for evaluation metrics
        (final_test_loss, final_test_acc, final_test_prec, final_test_recall,
         final_test_f1, final_test_balanced_acc, test_auc, test_mcc) = self.val_fn(model, test_loader, loss_function_for_eval, device)

        print(f" [FOLD {fold_idx} FINAL] Test Loss: {final_test_loss:.4f} | Test Acc: {final_test_acc:.4f} | test Balanced Acc: {final_test_balanced_acc:.4f} | test F1: {final_test_f1:.4f} | Test AUC: {test_auc:.4f} | Test MCC: {test_mcc:.4f}")

        # Assuming evaluate_model
        eval_results = evaluate_model(model=model, dataloader=test_loader, class_names=self.class_names, return_misclassified=True)

        # Patient-level aggregation via dedicated helper
        patient_metrics = self._compute_patient_level_metrics(
            X_test_outer=X_test_outer,
            y_test_outer=y_test_outer,
            eval_results=eval_results,
            fold_idx=fold_idx,
        )
        
        # Plot learning curves for the fold
        fig_learning_curves = plot_learning_curves(
            self.per_fold_metrics['train_loss'][fold_idx], self.per_fold_metrics['val_loss'][fold_idx],
            self.per_fold_metrics['train_accuracy'][fold_idx], self.per_fold_metrics['val_accuracy'][fold_idx],
            ignore_first_epoch_loss=True
        )
        
        lc_path = os.path.join(self.learning_dir, f"learning_curves_fold_{fold_idx}.png")
        fig_learning_curves.savefig(lc_path, dpi=100, bbox_inches='tight')
        plt.close(fig_learning_curves)

        # Plot confusion matrix
        cm_fig = plot_confusion_matrix(eval_results['metrics']['confusion_matrix'], self.class_names, f'Test CM Fold {fold_idx}')
        cm_path = os.path.join(self.cm_dir, f"confusion_matrix_fold_{fold_idx}.png")
        cm_fig.savefig(cm_path, dpi=100, bbox_inches='tight')
        plt.close(cm_fig)

        self.fold_results.append({
            "fold": fold_idx, "test_loss": final_test_loss, "test_acc": final_test_acc,
            "test_f1": final_test_f1, "test_balanced_acc": final_test_balanced_acc,
            "test_auc": test_auc, "test_precision": final_test_prec,
            "test_recall": final_test_recall, "test_mcc": test_mcc, "best_lr": best_lr_for_fold,
            #--------- Patient-level metrics (balanced acc, auc, mcc, precision, recall) ---------
            "patient_major_bal_acc": patient_metrics.get("patient_major_bal_acc"),
            "patient_soft_bal_acc": patient_metrics.get("patient_soft_bal_acc"),
            "patient_major_auc": patient_metrics.get("patient_major_auc"),
            "patient_soft_auc": patient_metrics.get("patient_soft_auc"),
            "patient_major_mcc": patient_metrics.get("patient_major_mcc"),
            "patient_soft_mcc": patient_metrics.get("patient_soft_mcc"),
            "patient_major_precision": patient_metrics.get("patient_major_precision"),
            "patient_soft_precision": patient_metrics.get("patient_soft_precision"),
            "patient_major_recall": patient_metrics.get("patient_major_recall"),
            "patient_soft_recall": patient_metrics.get("patient_soft_recall"),
        })
        
        
    def run_experiment(self):
        """Execute outer CV with optional LR tuning per fold and collect metrics.
        
            return self.per_fold_metrics, self.fold_results
        """
        outer_skf = StratifiedKFold(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=self.cfg.get_random_seed()
        )
        
        lr_discovery_method = self.cfg.get_lr_discovery_method() # Es: 'pre_split', 'nested', 'fixed'

        # tune the learning rate once with grid search if grid search is used and the best lr is not set
        # NOTE: Grid search is not used anymore since it introduces optimistic bias since it uses data that will be tested as data on which to tune the lr
        if lr_discovery_method == 'grid_search':
            print(f"--- Starting Grid Search for Learning Rate ---")
            print("REMEMBER: grid search introduce optimistic bias since it uses data that will be tested as data on which to tune the lr")
                # Chiama la nuova funzione UNA VOLTA prima di iniziare la CV
            best_lr = self.find_best_lr_grid_search()
        elif lr_discovery_method == 'fixed':
            best_lr = self.cfg.get_manual_lr()
        
        # Determine if color transforms should be used based on cfg
        # use_color_transforms = self.cfg.data_augmentation.get("use_color_transforms", True)
        for fold_idx_actual, (train_pat_idx, test_pat_idx) in enumerate(outer_skf.split(self.unique_pat_ids, self.pat_labels)):
            fold_display_idx = fold_idx_actual + 1
            print(f"\n===== OUTER FOLD {fold_display_idx} / {self.num_folds} =====")
            # initialize the per fold metrics
            for key in self.per_fold_metrics:
                self.per_fold_metrics[key][fold_idx_actual] = []

            train_patients = self.unique_pat_ids[train_pat_idx]
            test_patients = self.unique_pat_ids[test_pat_idx]
            self._test_pat_ids_per_fold[fold_idx_actual] = test_patients # store test patient ids

            train_mask = self.df["patient_id"].isin(train_patients)
            test_mask = self.df["patient_id"].isin(test_patients)

            X_train_outer = self.df.loc[train_mask, "image_path"].values
            y_train_outer = self.df.loc[train_mask, "label"].values
            X_test_outer = self.df.loc[test_mask, "image_path"].values
            y_test_outer = self.df.loc[test_mask, "label"].values
            
            print(f"Outer Train images: {len(X_train_outer)} | Outer Test images: {len(X_test_outer)}")

            self._num_outer_images = len(X_test_outer)
            # --- Calculate fold-specific normalization statistics ---
            fold_stats = None
            # if not pretrained or compute_custom_normalization is forced via flag
            if not self.cfg.is_pretrained() or self.compute_custom_normalization:
                print(f"--- Calculating normalization stats for Fold {fold_display_idx} Training Data ---")
                fold_stats = compute_dataset_mean_std(X_train_outer, self.cfg, is_supported_by_torchvision=self.is_supported_by_torchvision)
                print(f"Fold {fold_display_idx} stats: {fold_stats}")
            else:
                print("Using pretrained model; ImageNet normalization will be applied by torchvision transforms.")

            # --- Get fold-specific transforms we do it here ---
            print(f"--- Generating data transforms for Fold {fold_display_idx} ---")
            if self.train_transforms is None or self.val_transforms is None:
                #if the transformations are not set, generate them
                self.current_fold_train_transforms, self.current_fold_val_transforms, self.current_fold_test_transforms = get_transforms(
                self.cfg,
                fold_specific_stats=fold_stats # note that is None if pretrained=True
                )
                print(f"Transforms generated for Fold {fold_display_idx}.")
            else:
                # If transforms are already set, use them
                print("Using the train and val transform passed as arguments")
                # Set the current fold transforms
                self.current_fold_train_transforms = self.train_transforms
                self.current_fold_val_transforms = self.val_transforms
                self.current_fold_test_transforms = self.val_transforms # Assuming test uses the same as val

            # 1. Tune Learning Rate
            best_lr_for_all_folds = self.best_lr
            opt_cfg = None
            
            if lr_discovery_method == 'nested':
                print(f"--- Starting Nested Hyperparameter Tuning for Fold {fold_display_idx} ---")
                # best_lr = self._tune_learning_rate(X_train_outer, y_train_outer)
                # opt_cfg = self._tune_optim_hparams(X_train_outer, y_train_outer)
                best_lr, opt_cfg = self._run_combined_hparam_tuning(
                    X_train_outer, y_train_outer
                )
            else:
                # Usa il valore pre-calcolato o fisso
                best_lr = best_lr_for_all_folds # type: ignore
                print(f"Utilizzando il Learning Rate pre-determinato per il Fold {fold_display_idx}: {best_lr:.6f}")

            # 2. Train Final Model for the Fold
            self.set_best_lr(best_lr)
            self.cfg.set_learning_rate(best_lr)
            # self.cfg.set_learning_rate(best_lr)
            print(f"--- Starting Final Model Training for Fold {fold_display_idx} with LR={best_lr:.6f} ---")
            #  best_model_path, loss_func_for_eval = self._train_model_with_early_stopping(fold_idx_actual, X_train_outer, y_train_outer, best_lr)
            best_model_path, loss_func_for_eval = self._train_model_with_early_stopping(
                    fold_idx_actual, X_train_outer, y_train_outer, opt_cfg
                )
            # 3. Evaluate on Test Set
            print(f"--- Evaluating Fold {fold_display_idx} on Outer Test Set ---")
            self._evaluate_fold_on_test_set(fold_idx_actual, best_model_path, loss_func_for_eval, X_test_outer, y_test_outer, best_lr)

        self._print_summary() # Ensure this method exists and is correctly implemented
        return self.per_fold_metrics, self.fold_results

    
    def _print_summary(self):
        """Print per-fold and aggregate results for the completed CV run."""
        print("\n-------------------------------------------------")
        print("Cross-validation results (outer folds):")
        for res in self.fold_results:
            print(f"  Fold {res['fold']}: Test Loss={res['test_loss']:.4f}, Acc={res['test_acc']:.4f}, "
                  f"F1={res['test_f1']:.4f}, Bal Acc={res['test_balanced_acc']:.4f}, AUC={res['test_auc']:.4f}, MCC={res['test_mcc']:.4f} (Best LR={res['best_lr']:.6f})")

        # Aggregate and print mean/std if desired
        if self.fold_results:
            avg_acc = np.mean([r['test_acc'] for r in self.fold_results])
            std_acc = np.std([r['test_acc'] for r in self.fold_results])
            avg_f1 = np.mean([r['test_f1'] for r in self.fold_results])
            std_f1 = np.std([r['test_f1'] for r in self.fold_results])
            avg_auc = np.mean([r['test_auc'] for r in self.fold_results])
            std_auc = np.std([r['test_auc'] for r in self.fold_results])
            avg_balanced_acc = np.mean([r['test_balanced_acc'] for r in self.fold_results])
            std_balanced_acc = np.std([r['test_balanced_acc'] for r in self.fold_results])
            avg_precision = np.mean([r['test_precision'] for r in self.fold_results])
            std_precision = np.std([r['test_precision'] for r in self.fold_results])
            avg_recall = np.mean([r['test_recall'] for r in self.fold_results])
            std_recall = np.std([r['test_recall'] for r in self.fold_results])
            avg_mcc = np.mean([r['test_mcc'] for r in self.fold_results])
            std_mcc = np.std([r['test_mcc'] for r in self.fold_results])
            print("\n--- Aggregate Results ---")
            print(f"Avg Test Accuracy: {avg_acc:.4f} +/- {std_acc:.4f}")
            print(f"Avg Test F1-Score: {avg_f1:.4f} +/- {std_f1:.4f}")
            # print(f"Avg Test AUC:      {avg_auc:.4f} +/- {std_auc:.4f}")
            print(f"Avg Test Balanced Acc: {avg_balanced_acc:.4f} +/- {std_balanced_acc:.4f}")
            print(f"Avg Test Precision: {avg_precision:.4f} +/- {std_precision:.4f}")
            print(f"Avg Test Recall: {avg_recall:.4f} +/- {std_recall:.4f}")
            print(f"Avg Test MCC: {avg_mcc:.4f} +/- {std_mcc:.4f}")
        print("-------------------------------------------------")

