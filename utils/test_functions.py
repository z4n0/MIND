from sklearn.metrics import classification_report, precision_score, recall_score  # <-- Add this
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef, roc_auc_score
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_extraction_functions import extract_labels_meaning
# =============================================================================
# Reusable Evaluation Utilities
# =============================================================================

def calculate_classification_metrics(true_labels, predictions, confidences=None, class_names=None, probs=None):
    """
    Calculate comprehensive classification metrics for both binary and multi-class tasks.

    Inputs:
        true_labels (array-like): Ground truth labels.
        predictions (array-like): Model's predicted labels.
        confidences (array-like, optional): Confidence scores for predictions (values between 0 and 1).
        class_names (list of str, optional): Names of the classes for detailed reporting.

    Returns:
        metrics (dict): Dictionary containing overall metrics:
            - accuracy
            - precision
            - recall
            - F1-score
            - balanced accuracy
            - confusion matrix
        Additionally, if confidences are provided, the dictionary includes
            - mean confidence and a histogram of confidence values.
        Finally, if class_names is provided, the dictionary includes a detailed
        classification report.
    """
    # Determine which averaging to use for the scores
    num_classes = len(np.unique(true_labels))
    avg = 'binary' if num_classes == 2 else 'weighted'
    
    # Compute core metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, average=avg),
        'recall': recall_score(true_labels, predictions, average=avg),
        'f1': f1_score(true_labels, predictions, average=avg),
        'balanced_accuracy': balanced_accuracy_score(true_labels, predictions),
        'confusion_matrix': confusion_matrix(true_labels, predictions)
    }

    # MCC
    try:
        metrics['mcc'] = matthews_corrcoef(true_labels, predictions)
    except Exception:
        pass

    # AUC: prefer full probability vectors if available
    try:
        n_classes = len(np.unique(true_labels))
        if probs is not None:
            if n_classes == 2:
                metrics['auc'] = roc_auc_score(true_labels, probs[:, 1])
            else:
                metrics['auc'] = roc_auc_score(true_labels, probs, multi_class='ovr', average='macro')
        elif confidences is not None and n_classes == 2:
            metrics['auc'] = roc_auc_score(true_labels, confidences)
    except Exception:
        pass
    
    # Confidence-related metrics (optional)
    if confidences is not None:
        metrics.update({
            'mean_confidence': np.mean(confidences),
            'confidence_histogram': np.histogram(confidences, bins=10, range=(0, 1))[0]
        })
    
    # Detailed classification report (if class names are provided)
    if class_names is not None:
        metrics['classification_report'] = classification_report(
            true_labels,
            predictions,
            target_names=class_names,
            output_dict=True
        )
    
    return metrics

def evaluate_model(model, dataloader, class_names, device=None, return_misclassified=False):
    """
    Comprehensive model evaluation with metrics and optional misclassified samples.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches.
        class_names (list of str): List of class names for classification metrics.
        device (torch.device, optional): Device for computations. Defaults to GPU if available.
        return_misclassified (bool, optional): Include misclassified samples. Default False.

    Returns:
        dict: Contains predictions, true labels, confidences, metrics, and 
              optionally misclassified samples.
        
    Example output structure:
        {
            'predictions': np.array([0, 1, 1, 1, 0]),
            'true_labels': np.array([0, 1, 0, 1, 0]),
            'confidences': np.array([0.89, 0.92, 0.54, 0.78, 0.95]),
            'probs': np.array([[0.89, 0.11], [0.08, 0.92], ...]),
            'misclassified': [  # Only if return_misclassified=True
                {
                    'image': np.array(...),  # Shape: (C, H, W)
                    'true_label': 0,
                    'pred_label': 1,
                    'confidence': 0.54,
                    'batch_idx': 0
                }
            ],
            'metrics': {
                'accuracy': 0.60,
                'f1': 0.57,
                'balanced_accuracy': 0.67,
                'confusion_matrix': np.array([[2, 1], [1, 1]]),
                'mean_confidence': 0.816,
                'classification_report': {...}
            }
        }
    """
    # Determine computation device (GPU if available)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to target device

    # Initialize storage containers
    results = {
        'predictions': [],    # Model's predicted class indices
        'true_labels': [],    # Ground truth labels
        'confidences': [],    # Confidence scores for predictions
        'misclassified': [],  # Stores problematic cases if requested
        'probs': [],    # Model's predicted probabilities
    }

    # Set model to evaluation mode (disables dropout/batchnorm)
    model.eval()

    # Disable gradient calculation for faster inference
    with torch.no_grad():
        # Process data in batches to save memory
        for batch_idx, batch in enumerate(dataloader):
            # Move data to target device
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()  # Ensure labels are integers

            # Forward pass through model
            # print(f"model class name: {model.__class__.__name__}")
            model_name_lower = model.__class__.__name__.lower()
            is_vit = 'vit' in model_name_lower and not 'module' in model_name_lower
            if is_vit: #ViTFinetuneModule should be excluded from this check
                outputs, _ = model(images) #the discarded is hidden states
            else: # not vit
                  # Raw model outputs (logits)
                  outputs = model(images)  # Raw model outputs (logits)

            # Convert logits to probabilities using softmax
            probs = torch.softmax(outputs, dim=1)
            probs_np = probs.cpu().numpy()
            results['probs'].extend(probs_np)

            # Get confidence scores and predicted class indices
            confs, preds = torch.max(probs, dim=1)

            # Move data back to CPU and convert to numpy
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            confs_np = confs.cpu().numpy()

            # Store batch results
            results['predictions'].extend(preds_np)
            results['true_labels'].extend(labels_np)
            results['confidences'].extend(confs_np)

            # Optional: Capture misclassified samples for analysis
            if return_misclassified:
                # Create boolean mask of incorrect predictions
                mask = preds_np != labels_np

                # Collect details for misclassified samples
                misclassified = [{
                    'image': images[i].cpu().numpy(),  # Original image data
                    'true_label': labels_np[i],        # Actual class
                    'pred_label': preds_np[i],         # Model's prediction
                    'confidence': confs_np[i],         # How confident was the model
                    'batch_idx': batch_idx             # For debugging batches
                } for i in np.where(mask)[0]]  # np.where(mask)[0] gets indices of True values

                results['misclassified'].extend(misclassified)

    # Convert lists to numpy arrays for easier analysis
    for key in ['predictions', 'true_labels', 'confidences', 'probs']:
        results[key] = np.array(results[key])

    # Calculate performance metrics
    results['metrics'] = calculate_classification_metrics(
        results['true_labels'],
        results['predictions'],
        confidences=results['confidences'],
        class_names=class_names,
        probs=results.get('probs')
    )

    # Return appropriate data based on request
    return results if return_misclassified else {k:v for k,v in results.items() if k != 'misclassified'}