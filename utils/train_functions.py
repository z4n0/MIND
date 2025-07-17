import random
import sys
from typing import Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from sklearn.utils import resample
import matplotlib.pyplot as plt
from configs.ConfigLoader import ConfigLoader
from torch.amp import GradScaler, autocast
from torch.amp.autocast_mode import autocast


def train_epoch(
    model, 
    loader, 
    optimizer, 
    loss_function, 
    device, 
    clip_value=1.0, 
    print_batch_stats=False,
    enable_amp: bool = True  # New parameter to control AMP
):
    """
    Trains the model for one epoch.
    Optionally uses Automatic Mixed Precision (AMP) if enable_amp is True and on CUDA.
    This version is based on the user-provided snippet.
    Includes gradient clipping and NaN checks.

    Args:
        model (torch.nn.Module): The model to be trained.
        loader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        loss_function (callable): Loss function to compute the loss.
        device (torch.device): Device on which to perform computations (e.g., 'cpu' or 'cuda').
        clip_value (float): Maximum norm for gradient clipping. Default is 1.0.
        print_batch_stats (bool): Whether to print stats for the first batch.
        enable_amp (bool): Whether to enable Automatic Mixed Precision. Default is True.

    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average loss over valid batches in the epoch.
            - avg_accuracy (float): The average accuracy over all processed samples in the epoch.
    """
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    num_valid_batches = 0 # Track batches without NaN loss

    # Determine if AMP should actually be used based on enable_amp and device type
    # This replaces the original fixed: use_amp = device.type == 'cuda'
    amp_active = enable_amp and (device.type == 'cuda')
    
    scaler = None # Initialize scaler
    if amp_active:
        scaler = GradScaler()
        if print_batch_stats: # Only print if other batch stats are also being printed
            print("AMP is active for training this epoch.")
    elif enable_amp and device.type != 'cuda':
        if print_batch_stats:
             print("AMP was enabled, but CUDA is not available. Training in float32.")
    elif not enable_amp:
        if print_batch_stats:
            print("AMP is disabled by user. Training in float32.")


    for i, batch in enumerate(loader): # Added enumerate for batch index logging
        images_batch = batch["image"].to(device)
        labels_batch = batch["label"].to(device).long()
        
        if i == 0 and print_batch_stats: # Check first batch only
            print(f"\n--- Input Batch Stats (Batch {i}) ---")
            print(f"  Images shape: {images_batch.shape}, dtype: {images_batch.dtype}")
            print(f"  Images min: {images_batch.min()}, max: {images_batch.max()}, mean: {images_batch.mean()}")
            print(f"  Images has NaN: {torch.isnan(images_batch).any()}, has Inf: {torch.isinf(images_batch).any()}")
            print(f"  Labels shape: {labels_batch.shape}, dtype: {labels_batch.dtype}")
            print(f"  Labels unique: {torch.unique(labels_batch)}")
            print(f"  Images dtype: {images_batch.dtype}, device: {images_batch.device}")
            # Add check for label range if num_classes is known
            # num_classes = model.classifier_model.classifier[-1].out_features # Example way to get num_classes
            # if (labels_batch < 0).any() or (labels_batch >= num_classes).any():
            #      print(f"!!! WARNING: Labels out of range [0, {num_classes-1}) detected: {torch.unique(labels_batch)}")
            print("-------------------------------------\n")
        
        optimizer.zero_grad()

        # --- Automatic Mixed Precision Context (or regular context if AMP is not active) ---
        with autocast(device_type=device.type, enabled=amp_active):
            outputs = model(images_batch) # Directly use images_batch as in the provided version
            # Ensure labels_batch is compatible if loss expects float (unlikely for CrossEntropy)
            loss = loss_function(outputs, labels_batch)

        # --- Check for NaN/Inf Loss BEFORE backward ---
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected at training batch index {i}. Loss: {loss.item()}. Skipping batch update.")
            # Optional: Investigate why loss is NaN here (e.g., print outputs, inputs)
            # print(f"  Outputs sample: {outputs.flatten()[:10]}")
            continue # Skip backward and optimizer step for this batch

        # --- Perform backward pass and optimizer step (conditionally using scaler) ---
        if amp_active and scaler: # If AMP is active (implies CUDA and scaler is initialized)
            scaler.scale(loss).backward()
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            # Clip gradients after unscaling
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            # Optimizer step (internally checks for inf/nans in gradients)
            scaler.step(optimizer)
            # Update scaler for next iteration
            scaler.update()
        else: # If AMP is not active (either on CPU or enable_amp=False)
            loss.backward()
            # Clip gradients directly
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            optimizer.step()

        # --- Accumulate metrics for valid batches ---
        epoch_loss += loss.item()
        num_valid_batches += 1

        # Get predictions (works for both binary and multi-class)
        # Ensure outputs are not NaN before calculating accuracy
        if not torch.isnan(outputs).any():
             with torch.no_grad(): # Ensure no gradients calculated here
                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == labels_batch).sum().item()
        else:
            print(f"Warning: NaN outputs detected at training batch index {i} during accuracy calculation.")

        total += labels_batch.size(0) # Count total samples processed regardless of NaN loss

    # --- Calculate average metrics ---
    if num_valid_batches > 0:
        avg_loss = epoch_loss / num_valid_batches
    else:
        print("Warning: No valid batches processed in this epoch (all resulted in NaN/Inf loss).")
        avg_loss = float('nan') # Indicate failure

    if total > 0:
        avg_accuracy = correct / total
    else:
        print("Warning: No samples processed in this epoch.")
        avg_accuracy = 0.0

    return avg_loss, avg_accuracy


def val_epoch(model, loader, loss_function, device):
    """
    Validate the model for one epoch (WITHOUT Automatic Mixed Precision).

    Args:
        model (torch.nn.Module): The model to validate.
        loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        loss_function (torch.nn.Module): Loss function to use.
        device (torch.device): Device to run the validation on (e.g., 'cpu' or 'cuda').

    Returns in order: val_loss, accuracy, precision, recall, f1, balanced_acc, roc_auc, mcc
        tuple: A tuple containing the following metrics:
            - val_loss (float): Average loss over all batches.
            - accuracy (float): Accuracy of the model on the validation data.
            - precision (float): Precision score.
            - recall (float): Recall score.
            - f1 (float): F1 score (using binary averaging for 2 classes, weighted for multi-class).
            - balanced_acc (float): Balanced accuracy score.
            - roc_auc (float): ROC AUC score.
            - mcc (float): Matthews Correlation Coefficient.
    """
    model.eval() # Set model to evaluation mode so that it doesn't update batch norm or dropout layers
    epoch_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    # For storing probabilities:
    all_probs = [] # Note: Still needed if ROC AUC calculation is re-enabled
    num_valid_val_batches = 0 # Initialize counter for valid batches
    
    with torch.no_grad(): # Disable gradient calculations for validation
        for i, batch in enumerate(loader): # Added enumerate for potential debugging
            images_batch = batch["image"].to(device)
            true_labels_batch = batch["label"].to(device).long() # Ensure labels are LongTensor

            # --- REMOVED AMP CONTEXT ---
            # Perform forward pass and loss calculation in default precision (likely float32)
            outputs = model(images_batch)

            # Ensure outputs are float32 before loss calculation if model might be in another default dtype
            # Usually unnecessary if model is standard nn.Module, but safe to include.
            loss = loss_function(outputs, true_labels_batch)
            # --- DEBUG: Check for NaN/Inf in Loss (Still useful) ---
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"!!! NaN/Inf found in loss WITHOUT AMP at val batch index {i} !!!")
                print(f"    Loss value: {loss.item()}")
                # Optional: Add more debugging like printing outputs or inputs
                # raise ValueError(f"NaN loss detected at val batch {i}") # Uncomment to stop execution
            # --- END DEBUG ---

            # Accumulate loss (check if loss is valid first if needed)
            if not (torch.isnan(loss).any() or torch.isinf(loss).any()):
                 epoch_loss += loss.item()
                 num_valid_val_batches += 1 # Increment valid batch counter
            else:
                 # Decide how to handle NaN loss - skip batch, count as high loss?
                 # Skipping is simple, but might skew average if it happens often.
                 print(f"Warning: NaN/Inf loss encountered at val batch {i}, skipping batch for avg loss calculation.")


            # Calculate predictions and store for metrics
            probs = torch.softmax(outputs, dim=1) # Still useful for potential AUC
            _, predicted = torch.max(outputs, dim=1) # Get predicted class indices

            correct += (predicted == true_labels_batch).sum().item()
            total += true_labels_batch.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(true_labels_batch.cpu().numpy())

            # Storing probabilities for ROC AUC (logic remains the same)
            n_classes = outputs.shape[1]
            if n_classes == 2:
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                # For multi-class, need to handle potentially large arrays
                # Consider alternatives if memory becomes an issue
                all_probs.append(probs.cpu().float().numpy())


    # Ensure we don't divide by zero if the loader was empty or all batches had NaN loss
    if len(loader) == 0:
         print("Warning: Validation loader was empty.")
         val_loss = 0.0 # Or handle as appropriate
    else:
        # Calculate average loss ONLY over batches that didn't result in NaN/Inf
        # Note: A better approach might be needed if NaNs are frequent
        # num_valid_batches = sum(1 for batch_loss in # Pseudocode - need actual tracking if skipping NaNs
        #                          [loss_function(model(b["image"].to(device)).float(), b["label"].to(device).long()).item()
        #                           for b in loader if not torch.isnan(loss_function(model(b["image"].to(device)).float(), b["label"].to(device).long())).any()]
        #                        )
        if num_valid_val_batches > 0: # Use the new counter
             val_loss = epoch_loss / num_valid_val_batches
        else:
             print("Warning: All validation batches resulted in NaN/Inf loss or loader was empty.")
             val_loss = float('nan') # Indicate failure clearly


    # --- Metric Calculations---
    if total == 0: # Handle case where loader might be empty or all labels invalid
         print("Warning: No valid samples found for metric calculation in validation.")
         accuracy = 0.0
         precision = 0.0
         recall = 0.0
         f1 = 0.0
         balanced_acc = 0.0
         roc_auc = 0.0
         mcc = 0.0
    else:
        accuracy = correct / total

        # Check for valid predictions/labels before calculating sklearn metrics
        if len(all_labels) > 0 and len(all_predictions) == len(all_labels):
            # Choose averaging based on number of classes in true labels.
            unique_labels = set(all_labels)
            if len(unique_labels) == 2:
                avg_mode = 'binary'
            else:
                avg_mode = 'weighted' # Or 'macro' depending on preference

            # Use zero_division=0 to prevent errors and return 0 if a class has no predictions/labels
            precision = precision_score(all_labels, all_predictions, average=avg_mode, zero_division=0)
            recall = recall_score(all_labels, all_predictions, average=avg_mode, zero_division=0)
            f1 = f1_score(all_labels, all_predictions, average=avg_mode, zero_division=0)
            balanced_acc = balanced_accuracy_score(all_labels, all_predictions) # Handles multi-class directly
            mcc = matthews_corrcoef(all_labels, all_predictions)

            # ROC AUC Calculation (add back if needed, ensure all_probs is correctly formatted)
            if len(unique_labels) == 2 and len(all_probs) == len(all_labels):
                try:
                    roc_auc = roc_auc_score(all_labels, all_probs)
                except ValueError as e:
                    print(f"Could not compute ROC AUC: {e}")
                    roc_auc = 0.0 # Or NaN
            elif len(unique_labels) > 2 and len(all_probs) > 0:
                try:
                    # Need to correctly stack probabilities for multi-class
                    all_probs_stacked = np.vstack(all_probs)
                    # Ensure probabilities sum to 1 if needed (softmax should handle this)
                    roc_auc = roc_auc_score(all_labels, all_probs_stacked, multi_class='ovr', average='macro') # Or 'weighted'
                except ValueError as e:
                     print(f"Could not compute ROC AUC: {e}")
                     roc_auc = 0.0 # Or NaN
            else:
                roc_auc = 0.0 # Placeholder
        else:
            print("Warning: No valid labels/predictions for sklearn metrics.")
            precision, recall, f1, balanced_acc, mcc = 0.0, 0.0, 0.0, 0.0, 0.0
            roc_auc = 0.0

    # roc_auc_placeholder = 1

    return val_loss, accuracy, precision, recall, f1, balanced_acc, roc_auc, mcc



def print_model_summary(model):
    print("===========================")
    """Print model architecture details"""
    print("Model Architecture:")
    print("==================")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("===========================")


def plot_cv_results(fold_results, prefix='val'):
    
    """
    Plot cross-validation results with flexible metric prefixes.
    
    Args:
        fold_results: List of dictionaries containing metrics from each fold
        prefix: Metric prefix ('val' or 'test' or any other prefix)
        
    Returns:
        fig: Matplotlib figure object
    """
    base_metrics = ['loss', 'acc', 'f1', 'balanced_acc']
    metrics = [f"{prefix}_{metric}" for metric in base_metrics]
    
    # Check if the specified metrics exist in fold_results
    available_metrics = []
    for metric in metrics:
        if any(metric in fold.keys() for fold in fold_results):
            available_metrics.append(metric)
    
    if not available_metrics:
        print(f"No metrics with prefix '{prefix}_' found in fold_results.")
        print(f"Available keys: {list(fold_results[0].keys())}")
        return
    
    # Calculate subplot grid dimensions
    n_metrics = len(available_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
    
    # Handle single subplot case
    if n_metrics == 1:
        axes = np.array([axes])
    
    axes = axes.ravel()

    for idx, metric in enumerate(available_metrics):
        values = [fold[metric] for fold in fold_results if metric in fold]
        if values:
            axes[idx].boxplot(values)
            axes[idx].set_title(f'Distribution of {metric} across folds')
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].grid(True)
    
    # Hide unused subplots if any
    for idx in range(len(available_metrics), len(axes)):
        fig.delaxes(axes[idx])
        
    
    plt.tight_layout()
    # Store the figure reference before showing it
    if 'ipykernel' in sys.modules:
        # For Jupyter notebooks, display but keep reference
        plt.show()
    else:
        # For non-interactive contexts, don't display automatically
        pass
        
    # Also print numeric summary
    print(f"{prefix.upper()} Metrics Summary:")
    for metric in available_metrics:
        values = [fold[metric] for fold in fold_results if metric in fold]
        if values:
            print(f"{metric}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
                f"min={np.min(values):.4f}, max={np.max(values):.4f}")
                
    # Return the figure for the caller to use
    return fig


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """
    Returns mixed inputs, pairs of targets, and lambda for mixup.

    Args:
        x (torch.Tensor): Batch of input images of shape (B, C, H, W).
        y (torch.Tensor): Batch of labels of shape (B,).
        alpha (float): Mixup hyperparameter (Beta distribution).
        device (str): "cpu" or "cuda" device.

    Returns:
        mixed_x (torch.Tensor): Mixup-ed images.
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Labels from the shuffled batch.
        lam (float): Mixup coefficient (lambda).
    """
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(loss_function, outputs, y_a, y_b, lam):
    """
    Compute the mixup loss given the network outputs and the two sets of labels.
    """
    return lam * loss_function(outputs, y_a) + (1 - lam) * loss_function(outputs, y_b)


def train_epoch_mixUp(model, loader, optimizer, loss_function, device, mixup_alpha=0.2):
    """
    Train for one epoch with Mixup and AMP mixed precision.

    Args:
        model: Your PyTorch model (DenseNet, etc.).
        loader: DataLoader with your training data.
        optimizer: The optimizer (Adam, etc.).
        loss_function: Typically CrossEntropyLoss (or a wrapped MONAI loss).
        device: "cpu" or "cuda".
        mixup_alpha: If > 0, enable Mixup with Beta(alpha, alpha). 0 means "no mixup".

    Returns:
        Tuple of (mean_epoch_loss, epoch_accuracy).
    """
    model.train()
    epoch_loss = 0.0
    correct = 0.0
    total = 0
    
    # Initialize gradient scaler for AMP
    scaler = torch.amp.GradScaler()

    for batch in loader:
        images_batch = batch["image"].to(device)
        labels_batch = batch["label"].to(device).long()

        # --- Mixup Step ---
        if mixup_alpha > 0:
            # Create a mixup version of the batch
            images_batch, labels_a, labels_b, lam = mixup_data(
                images_batch, labels_batch, alpha=mixup_alpha, device=device
            )
        else:
            lam = 1.0
            labels_a, labels_b = labels_batch, labels_batch

        # Forward pass with AMP
        optimizer.zero_grad()
        
        with torch.amp.autocast():
            outputs = model(images_batch)
            # Mixup loss
            loss = mixup_criterion(loss_function, outputs, labels_a, labels_b, lam)

        # Backprop with AMP
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()

        # For accuracy, we typically take argmax of outputs
        _, predicted = torch.max(outputs, dim=1)

        # Approximate mixup accuracy (blend of labels_a and labels_b)
        if mixup_alpha > 0:
            correct += lam * (predicted == labels_a).sum().item() + (1 - lam) * (predicted == labels_b).sum().item()
        else:
            correct += (predicted == labels_batch).sum().item()

        total += labels_batch.size(0)

    avg_loss = epoch_loss / len(loader)
    avg_acc = correct / total
    return avg_loss, avg_acc


#print all the layers of the loaded model
def print_layers(model):
    for name, _ in model.named_modules():
        print(name)


# oversampling and undersampling functions
import numpy as np
from sklearn.utils import resample

def oversample_minority(train_x, train_y, random_seed=42):
    """
    Oversamples each class so that all classes have the same number of samples as the majority.
    This function works for binary as well as multi-class datasets.
    
    Args:
        train_x: Feature array (can be list or numpy array)
        train_y: Labels array (can be list or numpy array)
        random_seed: Seed for reproducibility
        
    Returns:
        balanced_x: Oversampled feature array
        balanced_y: Oversampled labels array
    """
    # Convert to numpy arrays if not already
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    
    # Get unique classes and counts
    unique_classes, class_counts = np.unique(train_y, return_counts=True)
    
    # Determine the target count: maximum number of samples among classes
    target_count = np.max(class_counts)
    
    # Container lists for oversampled data
    oversampled_x_list = []
    oversampled_y_list = []
    
    # Oversample each class independently
    for cls in unique_classes:
        cls_indices = np.where(train_y == cls)[0]
        # Resample with replacement to get target_count samples for this class
        cls_oversampled_x = resample(
            train_x[cls_indices],
            replace=True,
            n_samples=target_count,
            random_state=random_seed
        )
        cls_oversampled_y = np.full(target_count, cls)
        oversampled_x_list.append(cls_oversampled_x)
        oversampled_y_list.append(cls_oversampled_y)
    
    # Combine the oversampled data for all classes
    balanced_x = np.concatenate(oversampled_x_list)
    balanced_y = np.concatenate(oversampled_y_list)

    # Shuffle the combined dataset to avoid any ordering bias
    np.random.seed(random_seed)
    shuffle_indices = np.random.permutation(len(balanced_x))
    balanced_x = balanced_x[shuffle_indices]
    balanced_y = balanced_y[shuffle_indices]

    return balanced_x, balanced_y


def undersample_majority(train_x, train_y, random_seed=42):
    """
    Undersamples each class so that all classes have the same number of samples as the minority.
    This function works for binary as well as multi-class datasets.
    
    Args:
        train_x: Feature array (can be list or numpy array)
        train_y: Labels array (can be list or numpy array)
        random_seed: Seed for reproducibility
        
    Returns:
        balanced_x: Undersampled feature array
        balanced_y: Undersampled labels array
    """
    # Convert to numpy arrays if not already
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    
    # Get unique classes and counts
    unique_classes, class_counts = np.unique(train_y, return_counts=True)
    
    # Determine the target count: minimum number of samples among classes
    target_count = np.min(class_counts)
    
    undersampled_x_list = []
    undersampled_y_list = []
    
    # Undersample each class independently
    for cls in unique_classes:
        cls_indices = np.where(train_y == cls)[0]
        # Resample without replacement to get target_count samples for this class
        cls_undersampled_indices = resample(
            cls_indices,
            replace=False,
            n_samples=target_count,
            random_state=random_seed
        )
        undersampled_x_list.append(train_x[cls_undersampled_indices])
        undersampled_y_list.append(train_y[cls_undersampled_indices])
    
    # Combine the undersampled data for all classes
    balanced_x = np.concatenate(undersampled_x_list)
    balanced_y = np.concatenate(undersampled_y_list)

    # Shuffle the combined dataset to avoid any ordering bias
    np.random.seed(random_seed)
    shuffle_indices = np.random.permutation(len(balanced_x))
    balanced_x = balanced_x[shuffle_indices]
    balanced_y = balanced_y[shuffle_indices]

    return balanced_x, balanced_y


def train_epoch_vit(model, loader, optimizer, loss_function, device):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        images_batch, labels_batch = batch["image"].to(device), batch["label"].to(device).long()
        # pixel are correctly between 0 and 1
        optimizer.zero_grad()
        outputs, hidden_states_out = model(images_batch)
        #print(f"output(logits) {outputs}")
        loss = loss_function(outputs, labels_batch)
        loss.backward()
        # Debug: Check gradients
        # for name, param in model.named_parameters():
        #     if param.grad is not None and torch.all(param.grad == 0):
        #         print(f"Zero gradients in layer: {name}")
        # After loss.backward()
        total_grad = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
        #print(f"Gradient magnitude: {total_grad.item()}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Add this
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels_batch).sum().item()
        total += labels_batch.size(0)
    return epoch_loss / len(loader), correct / total

def val_epoch_vit(model, loader, loss_function, device):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    # For storing probabilities:
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            images_batch, labels_batch = batch["image"].to(device), batch["label"].to(device).long()
            outputs, hidden_states_out = model(images_batch)

            loss = loss_function(outputs, labels_batch)

            # Calculate standard metrics
            epoch_loss += loss.item()
            
            # Calculate probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, dim=1)

            correct += (predicted == labels_batch).sum().item()
            total += labels_batch.size(0)

            # Store predictions and labels for metric calculation
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            
            # Determine number of classes from model output shape
            n_classes = outputs.shape[1]
            if n_classes == 2:
                # For binary, store probability for positive class (class 1)
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                # For multi-class, store full probability distributions
                probs_np = probs.cpu().float().numpy()
                probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
                all_probs.append(probs_np)

    accuracy = correct / total

    # Choose averaging based on number of classes in true labels
    if len(set(all_labels)) == 2:
        f1 = f1_score(all_labels, all_predictions, average='binary')
        precision = precision_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')
    else:
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
    
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    
    # ROC AUC calculation
    if len(set(all_labels)) == 2:
        # roc_auc = 1 #NOTE this has been set to 1 since it gave error ValueError: Input contains NaN.
        roc_auc = roc_auc_score(all_labels, all_probs)
    else:
        all_probs = np.vstack(all_probs)
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

    val_loss = epoch_loss / len(loader)
    return val_loss, accuracy, precision, recall, f1, balanced_acc, roc_auc


def freeze_layers_up_to(model, cfg: ConfigLoader) -> int:
    """
    Print all layers of a model and freeze layers based on the configuration's freezed_layerIndex.
    Only prompts for input on the first fold.
    
    Enter -1 to not freeze any layers.
    """
    layers = [name for name, _ in model.named_parameters()]

    # Only print layers and ask for input if freezed_layerIndex is not set
    if cfg.get_freezed_layer_index() is None:
        # select interactivly the last layer to freeze
        print("Model Layers:")
        for i, layer in enumerate(layers):
            print(f"{i}: {layer}")
            
        sys.stdout.flush()
        
        # Add a small delay to ensure output is displayed
        from time import sleep
        sleep(0.5)

        while True:
            try:
                end_index = int(input(f"Enter the index of the last layer to freeze included (-1 for no freezing, 0 to {len(layers)-1}): "))
                if end_index == -1 or (0 <= end_index < len(layers)):
                    cfg.set_freezed_layer_index(end_index)
                    break
                else:
                    print("Invalid index. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    # Use the stored index for all folds
    end_index = cfg.get_freezed_layer_index()
    print(f"Freezing layers up to index: {end_index}")

    # Special case: if end_index is -1, don't freeze any layers
    if end_index == -1:
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Freeze layers up to and including the selected index
        for i, (name, param) in enumerate(model.named_parameters()):
            param.requires_grad = i > end_index
    
    return end_index


def freeze_layers_up_to_progressive_ft(model, cfg: ConfigLoader, layerwise_lrs=None) -> list:
    """
    Freeze layers based on cfg.freezed_layerIndex and optionally set layer-wise learning rates.

    Args:
        model: The PyTorch model.
        cfg: Config object containing freezing information.
        layerwise_lrs (list of float, optional): Learning rates per layer, progressively larger for deeper layers.

    Returns:
        param_groups (list): Parameter groups ready for optimizer setup.
    """

    layers = [name for name, _ in model.named_parameters()]

    # Prompt for freezing only if not set
    if cfg.get_freezed_layer_index() is None:
        print("Model Layers:")
        for i, layer in enumerate(layers):
            print(f"{i}: {layer}")
        
        from time import sleep
        sleep(0.5)

        while True:
            try:
                end_index = int(input(f"Enter the last layer index to freeze (-1 for no freezing, 0 to {len(layers)-1}): "))
                if end_index == -1 or (0 <= end_index < len(layers)):
                    cfg.set_freezed_layer_index(end_index)
                    break
                else:
                    print("Invalid index. Try again.")
            except ValueError:
                print("Invalid input. Enter a number.")

    end_index = cfg.get_freezed_layer_index()

    # Prepare parameter groups for optimizer
    param_groups = []

    if layerwise_lrs is None:
        base_lr = cfg.optimizer.get("base_lr", 1e-4)
        layerwise_lrs = [base_lr * (0.8 ** i) for i in reversed(range(len(layers)))]

    # Special case: No layers frozen
    if end_index == -1:
        for i, (name, param) in enumerate(model.named_parameters()):
            param.requires_grad = True
            param_groups.append({'params': param, 'lr': layerwise_lrs[i]})
    else:
        for i, (name, param) in enumerate(model.named_parameters()):
            if i <= end_index:
                param.requires_grad = False
            else:
                param.requires_grad = True
                param_groups.append({'params': param, 'lr': layerwise_lrs[i]})

    return param_groups


import numpy as np
from torch.utils.data import DataLoader
from monai.data.dataset import Dataset

def _worker_init_fn(worker_id):
        np.random.seed(42 + worker_id)
        random.seed(42 + worker_id)
        torch.manual_seed(42 + worker_id)

def make_loader(
    image_paths: np.ndarray,
    labels: np.ndarray,
    transforms,
    cfg,
    shuffle: bool = True
) -> DataLoader:
    """
    Build a MONAI DataLoader,
    Returns
    -------
    torch.utils.data.DataLoader
    """
    # 2) Build dict-list
    data_dicts = [{"image": p, "label": l}
                  for p, l in zip(image_paths, labels)]
    
    if len(data_dicts) == 0:
        raise ValueError("No data found in the provided paths and labels.")

    # 3) Dataset with transforms
    ds = Dataset(data=data_dicts, transform=transforms)

    # 4) DataLoader
    loader = DataLoader(
        ds,
        batch_size=cfg.data_loading["batch_size"],
        shuffle=shuffle,
        num_workers=cfg.data_loading["num_workers"],
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        generator=torch.Generator().manual_seed(42)
    )
    return loader
    
def make_unlabeled_loader(
    image_paths: np.ndarray,
    transforms,
    cfg,
    shuffle: bool = False
) -> DataLoader:
    """
    Build a MONAI DataLoader for unlabeled data,
    it creates fictional labels (1) for the unlabeled data.) for the sake of consistency and compatibility
    with previouse code. just remember that these labels are not real.
    -------
    return torch.utils.data.DataLoader
    """
    
    # 2) Build dict-list
    data_dicts = [{"image": p, "label":1} for p in image_paths]
    
    if len(data_dicts) == 0:
        raise ValueError("No data found in the provided paths.")

    # 3) Dataset with transforms
    ds = Dataset(data=data_dicts, transform=transforms)
    
    # 4) DataLoader
    loader = DataLoader(
        ds,
        batch_size=cfg.data_loading["batch_size"],
        shuffle=shuffle,
        num_workers=cfg.data_loading["num_workers"],
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        generator=torch.Generator().manual_seed(42)
    )
    
    return loader


def nested_cv_stratified_by_patient(df, cfg, labels_np, pat_labels, unique_pat_ids, train_transforms, val_transforms, pretrained_weights, class_names, model_factory=None, model_manager=None):
    """
    Perform nested cross-validation stratified by patient, with inner-loop hyperparameter tuning (learning rate)
    and outer-loop model evaluation. This function is designed for medical imaging tasks where data is grouped by patient,
    and ensures that images from the same patient do not appear in both train and test sets.

    Args:
        df (pd.DataFrame): DataFrame with columns ['image_path', 'label', 'patient_id'].
        cfg: Configuration object with all training, data, and optimizer settings.
        model: PyTorch model instance (used if model_manager is None).
        model_manager (optional): Object with a setup_model method for model instantiation.
        labels_np (np.ndarray): Array of all image-level labels.
        pat_labels (np.ndarray): Array of patient-level labels (for stratification).
        unique_pat_ids (np.ndarray): Array of unique patient IDs.
        train_transforms: MONAI or torchvision transforms for training data.
        val_transforms: MONAI or torchvision transforms for validation/test data.
        pretrained_weights: Path or identifier for pretrained weights (if any).
        class_names (list): List of class names for plotting and evaluation.

    Returns:
    
        - per_fold_metrics (dict): Dictionary of training/validation metrics for each fold.
        - fold_results (list): List of dictionaries with test set metrics for each outer fold.

    Workflow:
        - Outer StratifiedKFold splits by patient (ensuring no patient leakage).
        - Inner StratifiedKFold (on training patients) for hyperparameter tuning (learning rate via Optuna).
        - Optionally applies oversampling/undersampling for class balance.
        - Trains model with early stopping on a hold-out validation set.
        - Evaluates on the outer test set, saves learning curves and confusion matrices.
        - Aggregates and prints cross-validation results.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from utils.data_visualization_functions import plot_learning_curves, plot_confusion_matrix
    from utils.test_functions import evaluate_model
    import optuna, re ,os
    import pandas as pd
    
    def extract_patient_id(image_path):
        # Example: parse from the file name
        # In real code, you might have a different pattern
        # NOTE : This regex assumes the patient ID is a 4-digit number in the file name
        match = re.search(r'(\d{4})', image_path)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Could not extract patient ID from {image_path}")
        
    # Determine the number of classes automatically from the dataset labels
    unique_overall_labels = np.unique(labels_np) # Or df['label'].unique()
    num_classes = len(unique_overall_labels)
    print(f"Detected {num_classes} unique classes: {sorted(unique_overall_labels)}")

    per_fold_metrics = {
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
    
    # create folders for saving images to log
    cm_dir = "confusion_matrices"
    learning_dir = "learning_curves"
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(learning_dir, exist_ok=True)
    
    # ===============================
    # 1) Outer CV (by patient)
    # ===============================
    outer_skf = StratifiedKFold(
        n_splits=cfg.data_splitting["num_folds"],
        shuffle=True,
        random_state=cfg.data_splitting["random_seed"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outer_test_fold_results = []

    num_inner_folds_hyp = 3
    n_lr_trials = 4
    # minority_label = minority_label
    # print("unique labels", np.unique(labels_np))
    # print("unique labels overall:", unique_overall_labels) # Use the globally determined ones

    # -----------------------------------------------------------------------
    # REPLACE the loop over (train_index, test_index) with patient-level split
    # the splits select different patients ids
    # -----------------------------------------------------------------------
    for fold_idx, (train_pat_idx, test_pat_idx) in enumerate(outer_skf.split(unique_pat_ids, pat_labels)):
        print(f"\n===== OUTER FOLD {fold_idx+1} / {cfg.data_splitting['num_folds']} =====")
        
        # Initialize fold entry
        for key in per_fold_metrics:
            per_fold_metrics[key][fold_idx] = []

        # Get the actual patient IDs for train/test in this fold
        train_patients = unique_pat_ids[train_pat_idx]
        test_patients  = unique_pat_ids[test_pat_idx]

        # Filter df to get image-level train/test splits
        # filter the DataFrame to get the image paths and labels for the patients in the train and test sets
        # we use the patient IDs to filter the DataFrame
        train_mask = df["patient_id"].isin(train_patients)  # returns a boolean mask to use later with df.loc
        test_mask  = df["patient_id"].isin(test_patients)

        X_train_outer = df.loc[train_mask, "image_path"].values
        y_train_outer = df.loc[train_mask, "label"].values

        X_test_outer  = df.loc[test_mask, "image_path"].values
        y_test_outer  = df.loc[test_mask, "label"].values

        outer_train_size = len(X_train_outer)
        outer_test_size = len(X_test_outer)
        
        print(f" Outer Train samples: {outer_train_size} | Outer Test samples: {outer_test_size}")

        # =======================================================
        # STEP A: Inner CV with Optuna to tune the learning rate
        # =======================================================
        def objective(trial):
            candidate_lr = trial.suggest_float("lr", 1e-7, 1e-3, log=True)

            # --  a DataFrame from X_train_outer, y_train_outer (same as we did before to stratify by patient)
            df_outer_train = pd.DataFrame({"image_path": X_train_outer, "label": y_train_outer})
            df_outer_train["patient_id"] = df_outer_train["image_path"].apply(extract_patient_id)
            df_pat_inner = df_outer_train.groupby("patient_id", as_index=False)["label"].first()
            inner_pat_ids = df_pat_inner["patient_id"].values
            inner_pat_labels = df_pat_inner["label"].values

            inner_skf = StratifiedKFold(
                n_splits=num_inner_folds_hyp,
                shuffle=True,
                random_state=cfg.data_splitting["random_seed"]
            )

            inner_val_losses = []

            # -- For each inner fold, pick out the patients, then their images
            for inner_train_idx, inner_val_idx in inner_skf.split(inner_pat_ids, inner_pat_labels):
                these_train_pats = inner_pat_ids[inner_train_idx]
                these_val_pats   = inner_pat_ids[inner_val_idx]

                train_mask_inner = df_outer_train["patient_id"].isin(these_train_pats)
                val_mask_inner   = df_outer_train["patient_id"].isin(these_val_pats)

                X_train_inner = df_outer_train.loc[train_mask_inner, "image_path"].values
                y_train_inner = df_outer_train.loc[train_mask_inner, "label"].values
                X_val_inner   = df_outer_train.loc[val_mask_inner,   "image_path"].values
                y_val_inner   = df_outer_train.loc[val_mask_inner,   "label"].values

                if cfg.training["oversample"]:
                    X_train_inner_bal, y_train_inner_bal = oversample_minority(
                        X_train_inner, y_train_inner,
                        random_seed=cfg.data_splitting["random_seed"]
                    )
                elif cfg.training["undersample"]:
                    X_train_inner_bal, y_train_inner_bal = undersample_majority(
                        X_train_inner, y_train_inner,
                        random_seed=cfg.data_splitting["random_seed"]
                    )
                else:
                    X_train_inner_bal, y_train_inner_bal = X_train_inner, y_train_inner

                train_loader_inner = make_loader(
                    image_paths=np.array(X_train_inner_bal),
                    labels=np.array(y_train_inner_bal),
                    transforms = train_transforms,
                    cfg=cfg,
                    shuffle=True
                )
                
                val_loader_inner = make_loader(
                    image_paths=np.array(X_val_inner),
                    labels=np.array(y_val_inner),
                    transforms = val_transforms,
                    cfg=cfg,
                    shuffle=False
                )
                
                if model_manager is not None:
                    print(f"Using model manager: {model_manager}")
                    model_inner, device_inner = model_manager.setup_model(num_classes=num_classes, pretrained_weights= pretrained_weights)
                else:
                    print(f"Using passed model factory and not the model manager")
                    device_inner = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model_inner = model_factory(candidate_lr).to(device_inner)
                    
                #model_inner, device_inner = setup_model(cfg)
                if cfg.training["transfer_learning"] or cfg.training["fine_tuning"]:
                    freeze_layers_up_to(model_inner, cfg)

                # Weighted loss if needed
                if (cfg.training["weighted_loss"] and
                    not (cfg.training["oversample"] or cfg.training["undersample"])):
                    unique_labels_inner = np.unique(y_train_inner)
                    class_weights = compute_class_weight(
                        class_weight='balanced',
                        classes=unique_labels_inner,
                        y=y_train_inner
                    )
                    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device_inner)
                    loss_function_inner = nn.CrossEntropyLoss(weight=class_weights)
                else:
                    loss_function_inner = nn.CrossEntropyLoss()

                trainable_params = [p for p in model_inner.parameters() if p.requires_grad]  # filter the parameters of model_inner, taking only the ones that have requires_grad=True
                
                optimizer_inner = optim.Adam(
                    trainable_params, 
                    lr=candidate_lr,
                    weight_decay=float(cfg.optimizer["weight_decay"])
                )

                inner_num_epochs = 3
                print("number of inner epochs used for hyper-param tuning: ", inner_num_epochs)
                best_inner_loss = float("inf")

                for ep in range(inner_num_epochs):
                    train_loss, train_acc = train_epoch(
                        model_inner, train_loader_inner, optimizer_inner,
                        loss_function_inner, device_inner
                    )
                    
                    val_loss_inner, _, _, _, _, _, _, _= val_epoch(
                        model_inner, val_loader_inner, loss_function_inner, device_inner
                    )
                    
                    if val_loss_inner < best_inner_loss:
                        best_inner_loss = val_loss_inner

                inner_val_losses.append(best_inner_loss)

            return float(np.mean(inner_val_losses))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_lr_trials)

        best_lr = study.best_params["lr"]
        print(f"  [OUTER FOLD {fold_idx+1}] Best LR from inner CV = {best_lr:.4f}")

        # =======================================================
        # STEP B: Retrain on outer-train set, early stopping on hold-out
        # (Unchanged; below lines remain the same, just using X_train_outer, y_train_outer)
        # =======================================================
        X_train_for_es, X_val_for_es, y_train_for_es, y_val_for_es = train_test_split(
            X_train_outer, 
            y_train_outer,
            test_size=cfg.data_splitting["val_set_size"],
            stratify=y_train_outer,
            random_state=cfg.data_splitting["random_seed"]
        )

        internal_train_size = len(X_train_for_es)
        internal_val_size = len(X_val_for_es)
        print(f"Train samples for es (pre-oversampling): {internal_train_size}, "
            f"Val samples for es: {internal_val_size}")

        # [No change in oversampling, dataset creation, training loop, etc...]
        if cfg.training["oversample"]:
            X_train_for_es_bal, y_train_for_es_bal = oversample_minority(
                X_train_for_es, y_train_for_es,
                random_seed=cfg.data_splitting["random_seed"]
            )
        elif cfg.training["undersample"]:
            X_train_for_es_bal, y_train_for_es_bal = undersample_majority(
                X_train_for_es, y_train_for_es,
                random_seed=cfg.data_splitting["random_seed"]
            )
        else:
            X_train_for_es_bal, y_train_for_es_bal = X_train_for_es, y_train_for_es

        internal_train_size = len(X_train_for_es)
        internal_val_size  = len(X_val_for_es)
        print(f" final train set size (post over/undersampling if any): {len(X_train_for_es_bal)}")
        print(f" final val set size (post over/undersampling if any): {len(X_val_for_es)}")

        train_loader_es = make_loader(
            image_paths=np.array(X_train_for_es_bal),
            labels=np.array(y_train_for_es_bal),
            transforms=train_transforms,
            cfg=cfg,
            shuffle=True
        )
        
        val_loader_es = make_loader(
            image_paths=np.array(X_val_for_es),
            labels=np.array(y_val_for_es),
            transforms=val_transforms,
            cfg=cfg,
            shuffle=False
        )

        if model_manager is not None:
            print("Using model manager:")
            model, device = model_manager.setup_model(num_classes = num_classes, pretrained_weights=pretrained_weights)
        else:
            print(f"Using passed model and not the model manager")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model_factory(best_lr).to(device)
        
        if cfg.training["transfer_learning"] or cfg.training["fine_tuning"]:
            freeze_layers_up_to(model, cfg)
        
        print_model_summary(model)

        if (cfg.training["weighted_loss"] and
            not (cfg.training["oversample"] or cfg.training["undersample"])):
            unique_labels_outer = np.unique(y_train_outer)
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=unique_labels_outer,
                y=y_train_outer
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            loss_function = nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss_function = nn.CrossEntropyLoss()

        print(f"learning rate (from inner CV best) = {best_lr}")
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=best_lr,
            weight_decay=float(cfg.optimizer["weight_decay"])
        )
        
        using_cosine_scheduler = False  
        if cfg.get_model_name().lower() == "vit":
            # Use cosine annealing with warm restarts
            print("Using CosineAnnealingWarmRestarts scheduler")
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=7,      # e.g., reset cycle every 5 epochs
                T_mult=2, # e.g., double the cycle length every time
                eta_min=5e-6
            )
            using_cosine_scheduler = True
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5)

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(cfg.training["num_epochs"]):
            current_lr = optimizer.param_groups[0]['lr']

            if cfg.training["mixup_alpha"] > 0:
                train_loss, train_acc = train_epoch_mixUp(
                    model, train_loader_es, optimizer, loss_function, device,
                    mixup_alpha=cfg.training["mixup_alpha"]
                )
            else:
                train_loss, train_acc = train_epoch(
                    model, train_loader_es, optimizer, loss_function, device
                )

            val_loss, val_acc, val_prec, val_recall, val_f1, val_balanced_acc, val_roc_auc, val_mcc = val_epoch(
                model, val_loader_es, loss_function, device
            )
            if using_cosine_scheduler:
                scheduler.step()
            else:
                scheduler.step(val_loss)

            per_fold_metrics['train_loss'][fold_idx].append(train_loss)
            per_fold_metrics['val_loss'][fold_idx].append(val_loss)
            per_fold_metrics['train_accuracy'][fold_idx].append(train_acc)
            per_fold_metrics['val_accuracy'][fold_idx].append(val_acc)
            per_fold_metrics['val_f1'][fold_idx].append(val_f1)
            per_fold_metrics['val_balanced_accuracy'][fold_idx].append(val_balanced_acc)
            per_fold_metrics['val_precision'][fold_idx].append(val_prec)
            per_fold_metrics['val_recall'][fold_idx].append(val_recall)
            per_fold_metrics['val_auc'][fold_idx].append(val_roc_auc)
            per_fold_metrics['val_mcc'][fold_idx].append(val_mcc)


            print(f" Epoch {epoch+1}/{cfg.training['num_epochs']}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Val F1: {val_f1:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}, "
                f"lr: {current_lr:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'best_model_fold_{fold_idx}.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve == cfg.training["early_stopping_patience"]:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # =======================================================================
        #  End of training loop for this fold
        #  proceeding testing on the outer test set and saving the results + graphs
        # =========================================================================

        # create and save the loss/accuracy plots for this fold
        from utils.data_visualization_functions import plot_learning_curves
        fig_learning_curves = plot_learning_curves(
            per_fold_metrics['train_loss'][fold_idx],
            per_fold_metrics['val_loss'][fold_idx],
            per_fold_metrics['train_accuracy'][fold_idx],
            per_fold_metrics['val_accuracy'][fold_idx]
        )
        # 2) Save each confusion matrix PNG into the cm_dir
        learning_curves_filename = f"learning_curves_fold_{fold_idx+1}.png"
        learning_curves_path = os.path.join(learning_dir, learning_curves_filename)
        fig_learning_curves.savefig(learning_curves_path, dpi=100, bbox_inches='tight')
        plt.close(fig_learning_curves)  # close fig to free memory
        # Final evaluation on outer test
        model.load_state_dict(torch.load(f'best_model_fold_{fold_idx}.pth'))
        #PRINT THE TEST SET DISTRIBUTION

        class_counts = pd.Series(y_test_outer).value_counts()
        print(f"Test set class counts: {class_counts.to_dict()}")
        test_set_distribution = pd.Series(y_test_outer).value_counts(normalize=True)
        print(f"Test set distribution: {test_set_distribution.to_dict()}")
        
        test_loader = make_loader(
            image_paths=np.array(X_test_outer),
            labels=np.array(y_test_outer),
            transforms=val_transforms,
            cfg=cfg,
            shuffle=False
        )

        # Original call to val_epoch for testing
        # final_test_loss, _, _, _, _, _, _, _ = val_epoch(model, test_loader, loss_function, device)
        # Final evaluation on the *outer test set* of this fold
        (final_test_loss,
        final_test_acc,          # <- accuracy now
        final_test_prec,
        final_test_recall,
        final_test_f1,
        final_test_balanced_acc,
        test_auc,
        test_mcc) = val_epoch(model, test_loader, loss_function, device)

        
        print(f" [FOLD {fold_idx+1} FINAL] Test Loss: {final_test_loss:.4f} | "
            f"Test Acc: {final_test_acc:.4f} | Test AUC: {test_auc:.4f} | Test MCC: {test_mcc:.4f}"
            f"Test Balanced Acc: {final_test_balanced_acc:.4f} | Test Precision: {final_test_prec:.4f} | Test Recall: {final_test_recall:.4f} | Test F1: {final_test_f1:.4f}")
        
        
        eval_results = evaluate_model(
            model=model,
            dataloader=test_loader,
            class_names=class_names,
            return_misclassified=True
        )

        # Visualize confusion matrix
        confusion_matrix_fig = plot_confusion_matrix(
            eval_results['metrics']['confusion_matrix'],
            class_names=class_names,
            title=f'Test Set Confusion Matrix (Fold {fold_idx+1}'
        )

        # 2) Save each confusion matrix PNG into the cm_dir
        cm_filename = f"confusion_matrix_fold_{fold_idx+1}.png"
        cm_path = os.path.join(cm_dir, cm_filename)
        confusion_matrix_fig.savefig(cm_path, dpi=100, bbox_inches='tight')
        plt.close(confusion_matrix_fig)  # close fig to free memory

        outer_test_fold_results.append({
            "fold": fold_idx+1,
            "test_loss": final_test_loss,
            "test_acc": final_test_acc,
            "test_f1": final_test_f1,
            "test_balanced_acc": final_test_balanced_acc,
            "test_auc": test_auc,
            "test_precision": final_test_prec,
            "test_recall": final_test_recall,
            "test_mcc": test_mcc,
            "best_lr": best_lr
        })
        
    # =====================================================================
    # 10) After all the models trained on folds have been trained and tested
    # take the mean of the metrics over all folds
    # ====================================================================
    print("-------------------------------------------------")
    print("\nCross-validation results (outer folds):")
    for res in outer_test_fold_results:
        print(f"  Fold {res['fold']}: "
            f"Test Loss = {res['test_loss']:.4f}, "
            f"Test Acc = {res['test_acc']:.4f}, "
            f"Test F1 = {res['test_f1']:.4f}, "
            f"Test Balanced Acc = {res['test_balanced_acc']:.4f}, "
            f"Test MCC: {res.get('test_mcc', 'N/A'):.4f}, "
            f"(Best LR={res['best_lr']:.6f})"
            )
    
    return per_fold_metrics, outer_test_fold_results

def solve_cuda_oom():
    """Completely clear CUDA memory by:
    1. Running Python garbage collection
    2. Emptying PyTorch's CUDA cache
    3. Resetting PyTorch's CUDA memory stats
    
    Returns:
        dict: Memory stats before and after clearing
    """
    import torch
    import gc
    
    # Get initial memory stats
    initial_stats = {
        'allocated': torch.cuda.memory_allocated(),
        'reserved': torch.cuda.memory_reserved(),
        'max_allocated': torch.cuda.max_memory_allocated(),
        'max_reserved': torch.cuda.max_memory_reserved()
    }
    
    # Clear all Python references
    gc.collect()
    
    # Empty CUDA cache
    torch.cuda.empty_cache()
    
    # Reset memory stats to get accurate post-clear measurements
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    
    # Get final memory stats
    final_stats = {
        'allocated': torch.cuda.memory_allocated(),
        'reserved': torch.cuda.memory_reserved()
    }
    
    # Print summary
    print(f"Memory cleared - Before: {initial_stats['allocated']/1024**2:.2f}MB -> After: {final_stats['allocated']/1024**2:.2f}MB")
    print(f"Peak memory during session: {initial_stats['max_allocated']/1024**2:.2f}MB")
    
    return {'initial': initial_stats, 'final': final_stats}


class MLPClassifierHead(nn.Module):
    """
    A generic 3-layer MLP classifier head.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
    
class LinearProbeHead(nn.Module):
    """
    A simple linear classifier head for linear probing.
    Only a single fully-connected layer, no activation/dropout.
    """
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    @property
    def out_dim(self):
        return self.fc.out_features
    
    @property
    def in_dim(self):
        return self.fc.in_features

    def forward(self, x):
        """
        Args:
            x: feature tensor of shape (batch_size, input_dim)
        Returns:
            logits tensor of shape (batch_size, num_classes)
        """
        return self.fc(x)
    
def remove_projection_head(encoder: nn.Module) -> nn.Module:
    """Replace any known classifier attribute with nn.Identity()."""
    for attr in ["fc", "classifier", "class_layers", "out", "head"]:
        if hasattr(encoder, attr):
            setattr(encoder, attr, nn.Identity())
            print(f"Removed {attr} from encoder.")
            return encoder
        else:
            print(f"Encoder does not have attribute '{attr}' to remove.")
            return encoder
    # raise ValueError("Encoder does not have a known classifier head to remove.")
    
def _remove_linear_probe_head(backbone):
        # if isinstance(self.backbone.fc, LinearProbeHead):
        # print(self.backbone.fc.__class__.__name__)
        if hasattr(backbone, "fc"):
            if  backbone.fc.__class__.__name__ == "LinearProbeHead":
                print("removing linear probe head")
                backbone.fc = nn.Identity()
            
        elif hasattr(backbone, "classifier"):
            if backbone.classifier.__class__.__name__ == "LinearProbeHead":
                backbone.classifier = nn.Identity()
        else:
            raise RuntimeError(
                    "Could not remove linear probe head"
                )
        
               
class BaseSSLClassifier(nn.Module):
    """
    Wrap an SSL-pretrained encoder and "plug in" any classifier head.
    
    Args:
        encoder:          a pretrained nn.Module (e.g. ResNet50 without its fc)
        num_classes:      number of output classes
        freeze_backbone:  if True, encoder parameters are frozen
        backbone_output_dim:
                          if known, skip the dummy forward pass to infer it
        input_shape:      shape for dummy tensor when inferring dims it has to be the final shape after trasnformaions are applied
        head:             an already-built nn.Module to use as the classifier head
        head_factory:     a callable (out_dim, num_classes) -> nn.Module,
                          used only if `head` is None
    """
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        backbone_output_dim: Optional[int] = None,
        input_shape: Tuple[int,int,int,int] = (1, 3, 300, 300),
        head: Optional[nn.Module] = None,
    ):
        super().__init__()
        # Print a summary of the encoder for debugging and verification
        # print(f"Encoder summary: {encoder}")
        # Store the modified encoder as part of the model
        self.encoder = remove_projection_head(encoder)
        # 2) infer encoder out-dim by a dummy forward, if not provided
        # takes input shape as (batch_size, channels, height, width)
        # and do a forward pass to get the output dimension
        # NOTE: this is a bit of a hack, but it works for most encoders
        if backbone_output_dim is None:
            was_training = self.encoder.training
            self.encoder.eval()
            with torch.no_grad():
                device = next(self.encoder.parameters()).device
                # create a dummy input tensor with the specified shape
                dummy = torch.randn(*input_shape, device=device)
                out = self.encoder(dummy)
                if out.dim() > 2:
                    out = out.view(out.size(0), -1)
                elif out.dim() == 1:
                    out = out.unsqueeze(0)
                backbone_output_dim = out.shape[1]
                print(f"Encoder output dim: {backbone_output_dim}")
            self.encoder.train(was_training) # restore encoder training mode (the one before the dummy forward pass)

        # 3) optionally freeze the backbone
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # 4) choose the classifier head
        #    priority: head instance > head_factory() > simple default
        head = LinearProbeHead(backbone_output_dim, num_classes) if head is None else head # type: ignore
        if head is not None:
            print(f"Using provided classifier head: {head} of class {head.__class__.__name__}")
            self.classifier = head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SSL+classifier model.

        Args:
            images (torch.Tensor): 
                A batch of input images, shape (B_size, C, H, W).

        Returns:
            torch.Tensor:
                Logits for each class, shape (batch_size, num_classes).
        """
        # 1) Encode the raw images into high-level feature maps
        #    e.g. ResNet50 backbone will output shape (batch_size, F, h, w)
        features = self.encoder(images)

        # 2) If the encoder output is spatial (dim > 2), flatten it to a vector
        #    Now flattened_features has shape (batch_size, F * h * w)
        if features.dim() > 2: # in the case the output is (batch_size, num_channels, height, width)
            batch_size = features.size(0)  
            flattened_features = features.view(batch_size, -1) # flatten to (batch_size, F * h * w)
        else: # case the output is already flat (batch_size, feature_dim) skip flattening
            # Some encoders already produce a vector (e.g. ViT), so no flatten needed
            flattened_features = features

        # 3) Run the flattened feature vector through your classifier head
        #    Produces raw logits of shape (batch_size, num_classes)
        logits = self.classifier(flattened_features)

        return logits

# Example usage in a PyTorch Lightning module for supervised fine-tuning:
class SSLClassifierModule(pl.LightningModule):
    def __init__(self, encoder: nn.Module, num_classes: int = 2, freeze_encoder: bool = True, backbone_output_dim: int = None, lr: float = 1e-4, input_shape: Tuple[int,int,int,int] = (1, 3, 300, 300)):
        """
        Initialize the SSL Classifier Module.
        call BaseSSLClassifier to create the model and set up the loss function and optimizer.
        BaseSSLClassifier is take the pretrained encoder, figure out the output dim, removes the projection head, and add a new classifier head(LinearProbe usually).
        Args:
            encoder: Pretrained encoder model.
            num_classes: Number of output classes.
            freeze_encoder: Whether to freeze the encoder parameters.
            backbone_output_dim: Output dimension of the encoder.
            lr: Learning rate for the optimizer.
            input_shape: Shape of the input tensor.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"]) # save all hyperparameters except the encoder
        self.classifier_model = BaseSSLClassifier(encoder, num_classes, freeze_encoder, backbone_output_dim, input_shape=input_shape)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.classifier_model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)

from typing import Dict, Tuple, Optional

def get_best_fold_idx(outer_fold_test_results: list[Dict], metric="test_balanced_acc") -> int:
            """
            Get the index of the best fold based on a specified metric.

            Args:
                outer_fold_test_results (list of dict): List containing test results for 
                    each outer fold. Each element should be a dictionary with metrics 
                    as keys.
                metric (str): The metric name to use for selecting the best fold. 
                    Default is "test_balanced_acc".

            Returns:
                int: The index of the fold with the highest value for the specified metric.

            Example:
                best_idx = get_best_fold_idx(results, metric="test_f1")
            """

            print(outer_fold_test_results)
            best_fold_idx = np.argmax([r[metric] for r in outer_fold_test_results])
            # print(f"Best Balanced Accuracy Fold Index: {best_bac_fold_idx}")
            best_fold_result = outer_fold_test_results[best_fold_idx]
            print(f"Best {metric} Fold Result: {best_fold_result}")
            fold_idx = best_fold_result["fold"]
            return fold_idx