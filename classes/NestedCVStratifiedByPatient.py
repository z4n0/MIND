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
from utils.test_functions import evaluate_model
# Also assume make_loader, train_epoch, val_epoch, oversample_minority,
# undersample_majority, freeze_layers_up_to, print_model_summary are available
# from utils or other modules. For brevity, I won't redefine them here.
from utils.train_functions import train_epoch, val_epoch, freeze_layers_up_to, train_epoch_mixUp, oversample_minority, undersample_majority, make_loader

# from utils.data_utils import make_loader, oversample_minority, undersample_majority
# from utils.model_utils import print_model_summary
from utils.transformations_functions import get_transforms, compute_dataset_mean_std
from pathlib import Path # Added for Path operations

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
    def __init__(self, df, cfg, labels_np, pat_labels, unique_pat_ids, pretrained_weights,
                 class_names, train_transforms=None, val_transforms=None, model_factory=None, model_manager=None, num_folds=None, compute_custom_normalization=False, output_dir: str | None = None):

        self.df = df
        self._cfg = cfg
        self.labels_np = labels_np
        self.pat_labels = pat_labels
        self.unique_pat_ids = unique_pat_ids
        self.pretrained_weights = pretrained_weights
        self.class_names = class_names
        self.model_factory = model_factory # e.g., a function: def create_model(lr): ...
        self.model_manager = model_manager # e.g., an object with setup_model method
        self.num_folds = num_folds if num_folds is not None else self.cfg.data_splitting['num_folds']
        self.oversample = self.cfg.training.get("oversample", False)
        self.undersample = self.cfg.training.get("undersample", False)
        self.num_classes = self._determine_num_classes()
        self.val_set_size = self.cfg.data_splitting.get("val_set_size", 0.15)
        self.num_epochs = self.cfg.training.get("num_epochs", 100)
        self.is_supported_by_torchvision = self.pretrained_weights is not None
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        #store current fold's transforms
        self.current_fold_train_transforms = None
        self.current_fold_val_transforms = None
        self.current_fold_test_transforms = None
        self.per_fold_val_images_paths = {}
        self.compute_custom_normalization = compute_custom_normalization
        self.x_outer_len = None
        self.x_outer_len = None
        self.train_image_counts_per_fold = {}
        self.val_image_counts_per_fold = {}
        self._num_outer_images = None
        self.output_dir = Path(output_dir or ".").resolve()
        self.cm_dir = str(self.output_dir / "confusion_matrices")
        self.learning_dir = str(self.output_dir / "learning_curves")
        self._setup_directories()

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
    
    
    def _setup_directories(self):
        os.makedirs(self.cm_dir, exist_ok=True)
        os.makedirs(self.learning_dir, exist_ok=True)
    
    @property
    def cfg(self):
        return self._cfg
    
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
        return self.current_fold_train_transforms, self.current_fold_val_transforms, self.current_fold_test_transforms

    def _determine_num_classes(self):
        return len(np.unique(self.labels_np))

    def _setup_directories(self):
        os.makedirs(self.cm_dir, exist_ok=True)
        os.makedirs(self.learning_dir, exist_ok=True)

    def _extract_patient_id(self, image_path):
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
        torch.manual_seed(self.cfg.data_splitting["random_seed"])
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
                lr = self.cfg.optimizer.get("lr", 0.001) # Get a default if not for Optuna
                print(f"Warning: model_factory used without specific LR, using default/cfg LR: {lr}")
            else:
                lr = learning_rate_for_factory
            model = self.model_factory(lr).to(device) # Pass LR to factory
        else:
            raise ValueError("Either model_manager or model_factory must be provided.")

        if self.cfg.training.get("transfer_learning") or self.cfg.training.get("fine_tuning"):
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

    def _get_optimizer(self, model, learning_rate):
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=float(self.cfg.optimizer["weight_decay"])
        )

    def _get_scheduler(self, optimizer):
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

    def _objective(self, trial, X_train_outer_fold, y_train_outer_fold):
        candidate_lr = trial.suggest_float("lr", 5e-6, 2e-3, log=True)

        df_outer_train_fold = pd.DataFrame({"image_path": X_train_outer_fold, "label": y_train_outer_fold})
        # Ensure patient_id extraction is robust or pre-calculated
        df_outer_train_fold["patient_id"] = df_outer_train_fold["image_path"].apply(self._extract_patient_id)
        
        df_pat_inner = df_outer_train_fold.groupby("patient_id", as_index=False)["label"].first()
        inner_pat_ids = df_pat_inner["patient_id"].values
        inner_pat_labels = df_pat_inner["label"].values

        inner_skf = StratifiedKFold(
            n_splits = 2, #or Get from cfg
            shuffle=True,
            random_state=self.cfg.data_splitting["random_seed"]
        )
        inner_val_losses = []

        for inner_train_pat_idx, inner_val_pat_idx in inner_skf.split(inner_pat_ids, inner_pat_labels):
            these_train_pats = inner_pat_ids[inner_train_pat_idx]
            these_val_pats = inner_pat_ids[inner_val_pat_idx]

            train_mask_inner = df_outer_train_fold["patient_id"].isin(these_train_pats)
            val_mask_inner = df_outer_train_fold["patient_id"].isin(these_val_pats)

            X_train_inner = df_outer_train_fold.loc[train_mask_inner, "image_path"].values
            y_train_inner = df_outer_train_fold.loc[train_mask_inner, "label"].values
            X_val_inner = df_outer_train_fold.loc[val_mask_inner, "image_path"].values
            y_val_inner = df_outer_train_fold.loc[val_mask_inner, "label"].values

            # Handle over/undersampling (assuming functions are imported)
            if self.oversample:
                # from utils.data_utils import oversample_minority
                X_train_inner_bal, y_train_inner_bal = oversample_minority(
                    X_train_inner, y_train_inner, random_seed=self.cfg.data_splitting["random_seed"]
                )
            elif self.undersample:
                # from utils.data_utils import undersample_majority
                X_train_inner_bal, y_train_inner_bal = undersample_majority(
                    X_train_inner, y_train_inner, random_seed=self.cfg.data_splitting["random_seed"]
                )
            else:
                X_train_inner_bal, y_train_inner_bal = X_train_inner, y_train_inner

            # from utils.data_utils import make_loader
            train_loader_inner = make_loader(
                image_paths=np.asarray(X_train_inner_bal, dtype=str),
                labels=np.asarray(y_train_inner_bal, dtype=int),
                transforms=self.current_fold_train_transforms,
                cfg=self.cfg,
                shuffle=True
            )
            
            val_loader_inner = make_loader(
                image_paths=np.asarray(X_val_inner, dtype=str),
                labels=np.asarray(y_val_inner, dtype=int),
                transforms=self.current_fold_val_transforms,
                cfg=self.cfg,
                shuffle=False
            )

            model_inner, device_inner = self._get_model_and_device(learning_rate_for_factory=candidate_lr)
            loss_function_inner = self._get_loss_function(y_train_inner_bal) # Pass y_train_inner_bal for weighting
            optimizer_inner = self._get_optimizer(model_inner, candidate_lr)

            inner_num_epochs = self.cfg.training.get("num_inner_epochs_hyp", 3) # Get from cfg
            best_inner_loss = float("inf")

            for _ in range(inner_num_epochs):
                # from utils.training_utils import train_epoch, val_epoch
                train_loss_e, _ = train_epoch(model_inner, train_loader_inner, optimizer_inner, loss_function_inner, device_inner, print_batch_stats=False)
                val_loss_e, _, _, _, _, _, _, _ = val_epoch(model_inner, val_loader_inner, loss_function_inner, device_inner)
                if val_loss_e < best_inner_loss:
                    best_inner_loss = val_loss_e
            inner_val_losses.append(best_inner_loss)
        return float(np.mean(inner_val_losses))

    def _tune_learning_rate(self, X_train_outer, y_train_outer):
        sampler = optuna.samplers.TPESampler(seed=self.cfg.data_splitting["random_seed"])
        study = optuna.create_study(direction="minimize", sampler=sampler)
        # Use a lambda to pass the extra arguments to objective
        study.optimize(
            lambda trial: self._objective(trial, X_train_outer, y_train_outer),
            n_trials= 2 # Get from cfg
        )
        best_lr = study.best_params["lr"]
        print(f"  Best LR from inner CV = {best_lr:.6f}")
        return best_lr
    
    def _append_fold_metrics(self, fold_idx, train_loss, val_loss, train_acc, val_acc, val_f1, val_bal_acc, val_prec, val_recall, val_auc, val_mcc):
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

    def _train_model_with_early_stopping(self, fold_idx, X_train_outer, y_train_outer, best_lr):
        X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
            X_train_outer, y_train_outer,
            test_size=self.val_set_size,
            stratify=y_train_outer, # Stratify by original labels of this subset
            random_state=self.cfg.data_splitting["random_seed"]
        )
        self.train_image_counts_per_fold[fold_idx] = len(X_train_es)
        self.val_image_counts_per_fold[fold_idx] = len(X_val_es)
        print(f"X_train_es: {X_train_es.shape} | X_val_es: {X_val_es.shape}")
        print(f"Early stopping split: Train images: {len(X_train_es)}, Validation images: {len(X_val_es)}")
        # Handle over/undersampling (assuming functions are imported)
        self.per_fold_val_images_paths[fold_idx] = X_val_es
        
        if self.oversample:
            X_train_es_bal, y_train_es_bal = oversample_minority(X_train_es, y_train_es, self.cfg.data_splitting["random_seed"])
        elif self.undersample:
            X_train_es_bal, y_train_es_bal = undersample_majority(X_train_es, y_train_es, self.cfg.data_splitting["random_seed"])
        else:
            X_train_es_bal, y_train_es_bal = X_train_es, y_train_es

        train_loader_es = make_loader(X_train_es_bal, y_train_es_bal, self.current_fold_train_transforms, self.cfg, shuffle=True)
        val_loader_es = make_loader(X_val_es, y_val_es, self.current_fold_val_transforms, self.cfg, shuffle=False)

        model, device = self._get_model_and_device(learning_rate_for_factory=best_lr) # Use best_lr for factory
        # from utils.model_utils import print_model_summary
        from utils.train_functions import print_model_summary
        print_model_summary(model) 

        loss_function = self._get_loss_function(y_train_es_bal) # Pass y_train_es_bal for weighting
        optimizer = self._get_optimizer(model, best_lr)
        scheduler, using_cosine_scheduler = self._get_scheduler(optimizer)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        model_save_path = str(self.output_dir / f"best_model_fold_{fold_idx}.pth")

        for epoch in range(self.num_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            # from utils.training_utils import train_epoch_mixUp (if used)
            if self.cfg.training.get("mixup_alpha", 0) > 0:
                train_loss, train_acc = train_epoch_mixUp(model, train_loader_es, optimizer, loss_function, device, self.cfg.training["mixup_alpha"])
            else:
                train_loss, train_acc = train_epoch(model, train_loader_es, optimizer, loss_function, device, print_batch_stats=False)

            val_loss, val_acc, val_prec, val_recall, val_f1, val_bal_acc, val_roc_auc, val_mcc = val_epoch(model, val_loader_es, loss_function, device)

            if using_cosine_scheduler: scheduler.step(epoch)
            else: scheduler.step(val_loss)

            # Store metrics for this fold
            self._append_fold_metrics(fold_idx, train_loss, val_loss, train_acc, val_acc, val_f1, val_bal_acc, val_prec, val_recall, val_roc_auc, val_mcc)

            print(f" Fold {fold_idx} Epoch {epoch+1}/{self.num_epochs}: "
                  f"Tr L: {train_loss:.4f}, Tr Acc: {train_acc:.4f}, "
                  f"Val L: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Bal Acc: {val_bal_acc:.4f}, Val Roc AUC: {val_roc_auc:.4f}, Val_mcc: {val_mcc:.4f}, Val F1: {val_f1:.4f} lr: {current_lr:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve == self.cfg.training["early_stopping_patience"]:
                    print(f"Early stopping triggered at epoch {epoch+1} for fold {fold_idx}")
                    break
        return model_save_path, loss_function # Return loss_function for test eval

    def _evaluate_fold_on_test_set(self, fold_idx, model_path, loss_function_for_eval, X_test_outer, y_test_outer, best_lr_for_fold):
        model, device = self._get_model_and_device() # Get a fresh model instance
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.eval()

        test_loader = make_loader(X_test_outer, y_test_outer, self.current_fold_val_transforms, self.cfg, shuffle=False)

        print(f"Test set class counts for fold {fold_idx}: {pd.Series(y_test_outer).value_counts().to_dict()}")
        print(f"percentage of classes in test set: {pd.Series(y_test_outer).value_counts() / len(y_test_outer)}")

        # Using val_epoch for evaluation metrics
        (final_test_loss, final_test_acc, final_test_prec, final_test_recall,
         final_test_f1, final_test_balanced_acc, test_auc, test_mcc) = val_epoch(model, test_loader, loss_function_for_eval, device)

        print(f" [FOLD {fold_idx} FINAL] Test Loss: {final_test_loss:.4f} | Test Acc: {final_test_acc:.4f} | test Balanced Acc: {final_test_balanced_acc:.4f} | test F1: {final_test_f1:.4f} | Test AUC: {test_auc:.4f} | Test MCC: {test_mcc:.4f}")

        # Assuming evaluate_model is general enough
        eval_results = evaluate_model(model=model, dataloader=test_loader, class_names=self.class_names, return_misclassified=True)
        
        # Plot learning curves for the fold
        fig_learning_curves = plot_learning_curves(
            self.per_fold_metrics['train_loss'][fold_idx], self.per_fold_metrics['val_loss'][fold_idx],
            self.per_fold_metrics['train_accuracy'][fold_idx], self.per_fold_metrics['val_accuracy'][fold_idx],
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
            "test_recall": final_test_recall, "test_mcc": test_mcc, "best_lr": best_lr_for_fold
        })

    def run_experiment(self):
        outer_skf = StratifiedKFold(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=self.cfg.data_splitting["random_seed"]
        )

        # Determine if color transforms should be used based on cfg
        use_color_transforms = self.cfg.data_augmentation.get("use_color_transforms", True)
        for fold_idx_actual, (train_pat_idx, test_pat_idx) in enumerate(outer_skf.split(self.unique_pat_ids, self.pat_labels)):
            fold_display_idx = fold_idx_actual + 1
            print(f"\n===== OUTER FOLD {fold_display_idx} / {self.num_folds} =====")
            
            for key in self.per_fold_metrics:
                self.per_fold_metrics[key][fold_idx_actual] = []

            train_patients = self.unique_pat_ids[train_pat_idx]
            test_patients = self.unique_pat_ids[test_pat_idx]

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
            if not self.cfg.training.get("pretrained", False) or self.compute_custom_normalization:
                print(f"--- Calculating normalization stats for Fold {fold_display_idx} Training Data ---")
                fold_stats = compute_dataset_mean_std(X_train_outer, self.cfg, is_supported_by_torchvision=self.is_supported_by_torchvision)
                print(f"Fold {fold_display_idx} stats: {fold_stats}")
            else:
                print("Using pretrained model; ImageNet normalization will be applied by torchvision transforms.")

            # --- Get fold-specific transforms ---
            print(f"--- Generating data transforms for Fold {fold_display_idx} ---")
            if self.train_transforms is None or self.val_transforms is None:
                self.current_fold_train_transforms, self.current_fold_val_transforms, self.current_fold_test_transforms = get_transforms(
                self.cfg,
                color_transforms=use_color_transforms,
                fold_specific_stats=fold_stats # Pass None if pretrained=True
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
            print(f"--- Starting Hyperparameter Tuning for Fold {fold_display_idx} ---")
            best_lr = self._tune_learning_rate(X_train_outer, y_train_outer)

            # 2. Train Final Model for the Fold
            print(f"--- Starting Final Model Training for Fold {fold_display_idx} with LR={best_lr:.6f} ---")
            best_model_path, loss_func_for_eval = self._train_model_with_early_stopping(fold_idx_actual, X_train_outer, y_train_outer, best_lr)
            
            # 3. Evaluate on Test Set
            print(f"--- Evaluating Fold {fold_display_idx} on Outer Test Set ---")
            self._evaluate_fold_on_test_set(fold_idx_actual, best_model_path, loss_func_for_eval, X_test_outer, y_test_outer, best_lr)

        self._print_summary() # Ensure this method exists and is correctly implemented
        return self.per_fold_metrics, self.fold_results

    
    def _print_summary(self):
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



