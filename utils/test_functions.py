from sklearn.metrics import classification_report, precision_score, recall_score  # <-- Add this
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_extraction_functions import extract_labels_meaning
# =============================================================================
# Reusable Evaluation Utilities
# =============================================================================

def calculate_classification_metrics(true_labels, predictions, confidences=None, class_names=None):
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

    Inputs:
    - model (torch.nn.Module): The PyTorch model to evaluate.
    - dataloader (torch.utils.data.DataLoader): DataLoader providing batches of images and labels.
    - class_names (list of str): List of class names for detailed classification metrics.
    - device (torch.device, optional): Device to perform computations on (CPU/GPU). Defaults to available GPU or CPU.
    - return_misclassified (bool, optional): Whether to include misclassified samples in the results. Default is False.

    Returns:
    - results (dict): Contains predictions, true labels, confidences, metrics, and optionally misclassified samples.
      If return_misclassified is False, misclassified samples are excluded from the output.

    mock_results = {
    'predictions': np.array([0, 1, 1, 1, 0]),
    'true_labels': np.array([0, 1, 0, 1, 0]),
    'confidences': np.array([0.89, 0.92, 0.54, 0.78, 0.95]),
    
    # List of misclassified samples (only if return_misclassified=True)
    'misclassified': [
        {
            'image': np.array(3, 224, 224),  # Mock RGB image
            'true_label': 0,
            'pred_label': 1,
            'confidence': 0.54,
            'batch_idx': 0
        },
        # Add more misclassified samples as needed
    ],
    
    # Dictionary containing all metrics
    'metrics': 
    {
        'accuracy': 0.60,
        'f1': 0.57,
        'balanced_accuracy': 0.67,
        'confusion_matrix': np.array([
            [2, 1],  # True Negatives, False Positives
            [1, 1]   # False Negatives, True Positives
        ]),
        'mean_confidence': 0.816,
        'confidence_histogram': np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 2]),  # 10 bins
        'classification_report': {
            'MSA': {  # Example for binary classification 
                'precision': 0.50,
                'recall': 0.67,
                'f1-score': 0.57,
                'support': 3
            },
            'control': {
                'precision': 0.50,
                'recall': 0.33,
                'f1-score': 0.40,
                'support': 2
            },
            'accuracy': 0.60,mock_results = {
    # Array of model predictions (e.g., for 5 samples)
    'predictions': np.array([0, 1, 1, 1, 0]),
    
    # Array of true labels
    'true_labels': np.array([0, 1, 0, 1, 0]),
    
    # Array of confidence scores
    'confidences': np.array([0.89, 0.92, 0.54, 0.78, 0.95]),
    
    # List of misclassified samples (only if return_misclassified=True)
    'misclassified': 
    [
        {
            'image': np.random.rand(3, 224, 224),  # Mock RGB image
            'true_label': 0,
            'pred_label': 1,
            'confidence': 0.54,
            'batch_idx': 0
        },
        # Add more misclassified samples as needed
    ],
    
    # Dictionary containing all metrics
    'metrics': {
        'accuracy': 0.60,
        'f1': 0.57,
        'balanced_accuracy': 0.67,
        'confusion_matrix': np.array([
            [2, 1],  # True Negatives, False Positives
            [1, 1]   # False Negatives, True Positives
        ]),
        'mean_confidence': 0.816,
        'confidence_histogram': np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 2]),  # 10 bins
        'classification_report': {
            'cat': {  # Example for binary classification (cats vs dogs)
                'precision': 0.50,
                'recall': 0.67,
                'f1-score': 0.57,
                'support': 3
            },
            'dog': {
                'precision': 0.50,
                'recall': 0.33,
                'f1-score': 0.40,
                'support': 2
            },
            'accuracy': 0.60,
            'macro avg': {
                'precision': 0.50,
                'recall': 0.50,
                'f1-score': 0.485,
                'support': 5
            },
            'weighted avg': {
                'precision': 0.50,
                'recall': 0.60,
                'f1-score': 0.504,
                'support': 5
                    }
                }
            }
            'macro avg': {
                'precision': 0.50,
                'recall': 0.50,
                'f1-score': 0.485,
                'support': 5
            },
            'weighted avg': {
                'precision': 0.50,
                'recall': 0.60,
                'f1-score': 0.504,
                'support': 5
            }
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
        'misclassified': []   # Stores problematic cases if requested
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
            print(f"model class name: {model.__class__.__name__}")
            model_name_lower = model.__class__.__name__.lower()
            if 'vit' in model_name_lower and not 'module' in model_name_lower: #ViTFinetuneModule should be excluded from this check
                outputs, hidden_states = model(images)
            else:
                  # Raw model outputs (logits)
                  outputs = model(images)  # Raw model outputs (logits)

            # Convert logits to probabilities using softmax
            probs = torch.softmax(outputs, dim=1)

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
    for key in ['predictions', 'true_labels', 'confidences']:
        results[key] = np.array(results[key])

    # Calculate performance metrics
    results['metrics'] = calculate_classification_metrics(
        results['true_labels'],
        results['predictions'],
        confidences=results['confidences'],
        class_names=class_names
    )

    # Return appropriate data based on request
    return results if return_misclassified else {k:v for k,v in results.items() if k != 'misclassified'}