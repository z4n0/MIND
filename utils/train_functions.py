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
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from configs.ConfigLoader import ConfigLoader
from torch.cuda.amp import GradScaler, autocast
from torch.cuda.amp.autocast_mode import autocast
import warnings

# Suppress specific UserWarnings from PyTorch AMP
warnings.filterwarnings("ignore", message=".*`torch.cuda.amp.autocast(args...)` is deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*`torch.cuda.amp.GradScaler(args...)` is deprecated.*", category=UserWarning)


class MixupState:
    """Reusable GPU buffer to avoid per-batch allocations."""
    def __init__(self) -> None:
        self.buf: torch.Tensor | None = None

    def ensure_buf(self, x: torch.Tensor) -> torch.Tensor:
        if self.buf is None or self.buf.shape != x.shape or self.buf.dtype != x.dtype:
            self.buf = torch.empty_like(x, device=x.device)
        return self.buf


def mixup_inplace(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    device: torch.device,
    state: MixupState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    In-place MixUp to minimize peak memory:
      - buf <- x[idx]                        (1 extra full-size buffer)
      - x.mul_(lam).add_(buf, alpha=1-lam)   (overwrite x with the mix)
    Returns: (mixed_x, y_a, y_b, lam)
    """
    if alpha <= 0.0:
        return x, y, y, 1.0

    lam = float(torch.distributions.Beta(alpha, alpha).sample(()).item())
    idx = torch.randperm(x.size(0), device=device)

    buf = state.ensure_buf(x)
    buf.copy_(x[idx])                           # buf = shuffled(x)
    x.mul_(lam).add_(buf, alpha=1.0 - lam)      # x = lam*x + (1-lam)*buf  (in-place)
    return x, y, y[idx], lam


def train_epoch(
    model: nn.Module, 
    loader, 
    optimizer: torch.optim.Optimizer, 
    loss_function, 
    device: torch.device, 
    clip_value: float = 1.0, 
    print_batch_stats: bool = False,
    enable_amp: bool = True,   # come prima
    use_bf16: Optional[bool] = False,
):
    """
    Trains the model for one epoch (NO MixUp).
    Uses AMP if enable_amp=True and CUDA is available.
    Includes gradient clipping and NaN checks.
    Returns: (avg_loss, avg_accuracy)
    """
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    num_valid_batches = 0

    amp_active = enable_amp and (device.type == 'cuda')
    # scaler = GradScaler() if amp_active else None   #DEPRECATED
    # Use new API - GradScaler only needed for float16, not bfloat16
    scaler = torch.amp.GradScaler('cuda') if (amp_active and not use_bf16) else None
    # Determine dtype
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    # print(f"Training on device: {device}")
    for i, batch in enumerate(loader):
        images_batch = batch["image"].to(device)
        labels_batch = batch["label"].to(device).long()

        if i == 0 and print_batch_stats:
            print(f"\n--- Input Batch Stats (Batch {i}) ---")
            print(f"  Images shape: {images_batch.shape}, dtype: {images_batch.dtype}")
            print(f"  Images min: {images_batch.min()}, max: {images_batch.max()}, mean: {images_batch.mean()}")
            print(f"  Images has NaN: {torch.isnan(images_batch).any()}, has Inf: {torch.isinf(images_batch).any()}")
            print(f"  Labels shape: {labels_batch.shape}, dtype: {labels_batch.dtype}")
            print(f"  Labels unique: {torch.unique(labels_batch)}")
            print("-------------------------------------\n")

        optimizer.zero_grad()

        # AMP on CUDA only
        # if amp_active: #DEPRECATED
        #     with autocast():
        #         outputs = model(images_batch)
        #         loss = loss_function(outputs, labels_batch)
        if amp_active:
            with torch.autocast(device_type='cuda', dtype=dtype):
                outputs = model(images_batch)
                loss = loss_function(outputs, labels_batch)
        else:
            outputs = model(images_batch)
            loss = loss_function(outputs, labels_batch)

        # NaN/Inf guard
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss at training batch {i}. Skipping batch.")
            continue

        # Backward + step (with optional AMP scaler)
        if amp_active and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            scaler.step(optimizer)
            scaler.update()
        else: # BFLOAT16 OR NO AMP
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            optimizer.step()

        # Stats
        epoch_loss += loss.item()
        num_valid_batches += 1

        with torch.no_grad():
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels_batch).sum().item()
            total += labels_batch.size(0)

    avg_loss = epoch_loss / max(1, num_valid_batches)
    avg_accuracy = correct / max(1, total)
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
    # scaler = torch.cuda.amp.GradScaler() #DEPRECATED
    scaler = torch.amp.GradScaler('cuda')

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
        
        # with torch.cuda.amp.autocast(): #DEPRECATED
        #     outputs = model(images_batch)
        #     # Mixup loss
        #     loss = mixup_criterion(loss_function, outputs, labels_a, labels_b, lam)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images_batch)
            loss = mixup_criterion(loss_function, outputs, labels_a, labels_b, lam)
        # Backprop with AMP
        # Backprop with optional scaler
        if scaler is not None:  # float16
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # bfloat16 or float32
            loss.backward()
            optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
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

import numpy as np
from sklearn.utils import resample  # Questa è la funzione chiave per il ricampionamento

def oversample_minority(train_x, train_y, random_seed=42):
    """
    Esegue l'oversampling di ogni classe in modo che tutte le classi abbiano 
    lo stesso numero di campioni della classe maggioritaria.
    Questa funzione funziona sia per dataset binari che multi-classe.
    
    Args:
        train_x: Array dei path delle immagini
        train_y: Array delle etichette (label)
        random_seed: Seed per la riproducibilità
        
    Returns:
        balanced_x: Array delle feature bilanciato
        balanced_y: Array delle etichette bilanciato
    """
    
    train_x = np.array(train_x) # [image1_path, image2_path, image3_path, ..., imageN_path]
    train_y = np.array(train_y) # [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2, ...]
    
    # Analizza l'array delle etichette (train_y)
    # unique_classes: conterrà un array delle classi uniche (es. [0, 1] per PD/MSA)
    # class_counts: conterrà il numero di campioni per ciascuna classe (es. [300, 150])
    unique_classes, class_counts = np.unique(train_y, return_counts=True)
    
    # Trova il numero massimo di campioni tra tutte le classi.
    # Questo sarà il nostro "obiettivo" (target_count).
    # Esempio: se le counts sono [300, 150], target_count sarà 300.
    target_count = np.max(class_counts)
    
    # Inizializza due liste vuote che useremo per accumulare
    # i nuovi dati bilanciati per ciascuna classe.
    oversampled_x_list = []
    oversampled_y_list = []
    
    for cls in unique_classes:
        # Trova tutti gli indici (posizioni) nell'array train_y
        # che corrispondono alla classe corrente (cls).
        # Esempio: se cls=1, trova tutti gli indici dove train_y è 1.
        cls_indices = np.where(train_y == cls)[0]
        
        # --- Qui avviene la magia ---
        # Usa la funzione 'resample' di scikit-learn.
        cls_oversampled_x = resample(
            train_x[cls_indices],   # 1. Prendi i campioni (X) solo di QUESTA classe
            replace=True,           # 2. 'replace=True' è FONDAMENTALE. 
                                    #    Significa: "pesca un campione, usalo, e RIMETTILO NEL MUCCHIO".
                                    #    Questo permette di pescare lo stesso campione più volte.
                                    #    È così che avviene la duplicazione (oversampling).
            n_samples=target_count, # 3. Chiedi di "pescare" un numero di campioni pari a target_count.
                                    #    - Se la classe è minoritaria (150 campioni, target 300),
                                    #      pescherà 300 volte, duplicando campioni.
                                    #    - Se la classe è maggioritaria (300 campioni, target 300),
                                    #      pescherà 300 volte. (Bootstrap sampling)
            random_state=random_seed 
        )
        
        # Crea un array di etichette (Y) per i campioni appena creati.
        # Sarà un array riempito con il valore della classe corrente (es. [1, 1, 1, ..., 1])
        # e lungo esattamente 'target_count'.
        cls_oversampled_y = np.full(target_count, cls)
        
        # Aggiungi i dati e le etichette (ora bilanciati per questa classe)
        # alle nostre liste contenitore.
        oversampled_x_list.append(cls_oversampled_x)
        oversampled_y_list.append(cls_oversampled_y)
    
    # --- 3. Unione e Mescolamento ---
    # Alla fine del ciclo, le liste contengono gli array di ciascuna classe.
    # 'np.concatenate' li unisce tutti in un unico, grande array.
    # Ora abbiamo un dataset dove ogni classe ha 'target_count' campioni.
    balanced_x = np.concatenate(oversampled_x_list)
    balanced_y = np.concatenate(oversampled_y_list)
    
    np.random.seed(random_seed)
    
    # Crea un array di indici (da 0 a N-1) e mescolali casualmente.
    # Esempio: se N=4, crea [0, 1, 2, 3] e lo mescola in [2, 0, 3, 1].
    shuffle_indices = np.random.permutation(len(balanced_x))
    
    # Applica il mescolamento: riordina sia 'x' che 'y' usando
    # *esattamente* la stessa sequenza di indici casuali.
    # Questo mantiene la corrispondenza corretta tra campione (x) ed etichetta (y).
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
    training_cfg = cfg.get_training() or {}
    strategy = str(training_cfg.get("freeze_strategy", "index")).lower()

    if strategy in ("name", "auto"):
        applied, trainable_params, total_params = _freeze_layers_by_name(model, cfg, training_cfg)
        if applied:
            print(f"[freeze_layers_up_to] Name-based freezing applied ({trainable_params}/{total_params} trainable parameters).")
            _freeze_batchnorm_running_stats(model)
            # Preserve existing API: return stored index if available, else -1
            stored_index = cfg.get_freezed_layer_index()
            return stored_index if stored_index is not None else -1
        if strategy == "name":
            print("[freeze_layers_up_to] Name-based strategy requested but no keywords matched. Falling back to index-based strategy.")

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

    _freeze_batchnorm_running_stats(model)
    
    return end_index


def _freeze_batchnorm_running_stats(model: nn.Module) -> None:
    """
    Put BatchNorm modules whose affine parameters are frozen into eval mode so
    their running mean/variance are not updated during training.
    """
    bn_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )

    def _bn_eval_pre_hook(module, _inputs):
        if module.training:
            module.train(False)

    for module in model.modules():
        if isinstance(module, bn_types):
            # If any parameter still requires grad, keep updates enabled.
            if any(param.requires_grad for param in module.parameters()):
                continue
            module.train(False)
            if getattr(module, "_freeze_bn_hook", None) is None:
                handle = module.register_forward_pre_hook(_bn_eval_pre_hook)
                module._freeze_bn_hook = handle


def _freeze_layers_by_name(
    model: nn.Module,
    cfg: ConfigLoader,
    training_cfg: dict,
) -> tuple[bool, int, int]:
    """
    Freeze backbone parameters by matching substrings on parameter names.
    Returns (applied, trainable_count, total_params).
    """
    keywords = training_cfg.get("freeze_trainable_keywords")
    if keywords is None:
        model_name = cfg.get_model_name().lower()
        # Default keywords per architecture family
        if "densenet" in model_name or "efficientnet" in model_name:
            keywords = ("classifier", "class_layers", "head", "out", "fc")
        elif "resnet" in model_name or "seresnet" in model_name:
            keywords = ("fc", "classifier", "head", "class_layers", "out")
        elif "vit" in model_name or "swin" in model_name or "convnext" in model_name:
            keywords = ("head", "classifier", "mlp_head", "proj")
        else:
            return (False, 0, sum(1 for _ in model.parameters()))

    if isinstance(keywords, str):
        keywords = [keywords]
    keywords = tuple(kw.lower() for kw in keywords)

    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        lower_name = name.lower()
        should_train = any(kw in lower_name for kw in keywords)
        param.requires_grad = should_train
        if should_train:
            trainable_params += 1

    if trainable_params == 0:
        return (False, trainable_params, total_params)
    return (True, trainable_params, total_params)


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
