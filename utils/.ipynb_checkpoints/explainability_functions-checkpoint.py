import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from .data_extraction_functions import extract_labels_meaning
from .data_visualization_fun import min_max_normalization

def threshold_heatmap(heatmap, threshold=0.5, normalize=True):
    """
    Apply thresholding to a heatmap to emphasize important regions.

    Args:
        heatmap (np.ndarray): The original heatmap (2D array).
        threshold (float): Values below this threshold will be set to 0. Default is 0.5.
        normalize (bool): Whether to normalize the heatmap after thresholding. Default is True.

    Returns:
        np.ndarray: The thresholded heatmap.
    """
    # Apply thresholding
    heatmap_thresholded = heatmap.copy()
    heatmap_thresholded[heatmap < threshold] = 0  # Set values below the threshold to 0

    # Normalize the heatmap if required
    if normalize:
        heatmap_min, heatmap_max = heatmap_thresholded.min(), heatmap_thresholded.max()
        if heatmap_max > heatmap_min:  # Avoid division by zero
            heatmap_thresholded = (heatmap_thresholded - heatmap_min) / (heatmap_max - heatmap_min)

    return heatmap_thresholded


def get_overlay_heatmap(image, heatmap, alpha=0.5, cmap="jet"):
    """
    Overlay a heatmap on an image with transparency for zero-valued pixels.

    Args:
        image (torch.Tensor): Original image tensor (C, H, W).
        heatmap (np.ndarray): Heatmap (H, W).
        alpha (float): Transparency of the heatmap overlay.
        cmap (str): Colormap for the heatmap.

    Returns:
        heatmap_rgba: Displays the overlay plot and returns the RGBA heatmap transparent for zero values
    """
    # Ensure the image is in [H, W, C] format and scaled to [0, 1]
    image_np = image.permute(1, 2, 0).cpu().numpy()
    if image_np.max() > 1:
        image_np = image_np / 255.0  # Scale pixel values to [0, 1]

    # Create an RGBA heatmap with transparency for zero-valued pixels
    cmap = plt.get_cmap(cmap)
    heatmap_rgba = cmap(heatmap)  # Convert heatmap to RGBA (values in [0, 1])

    # Set alpha channel based on heatmap values
    heatmap_rgba[..., 3] = np.where(heatmap > 0, alpha, 0)  # Transparent for zero values

    # Plot original image and overlay
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot original image
    ax1.imshow(image_np)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Plot overlay
    ax2.imshow(image_np)
    ax2.imshow(heatmap_rgba, interpolation='nearest')  # Overlay heatmap with transparency
    ax2.set_title('GradCAM Overlay')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    return heatmap_rgba


def visualize_gradcam_for_single_image(model, test_loader, gradcam_obj, min_max_normalization, threshold=0.5):
    """
    Gets a single image from the test_loader, generates and visualizes GradCAM for it.
    Includes both the original heatmap and the thresholded heatmap using the get_overlay_heatmap function.

    Args:
        model: The PyTorch model.
        test_loader: The DataLoader for the test dataset.
        gradcam_obj: Initialized GradCAM object (e.g., GradCAM or GradCAM++).
        min_max_normalization: Function to rescale image/array to [0, 1].
        threshold: Threshold value for heatmap thresholding (default is 0.5).
    """
    # Get the device from the model
    device = next(model.parameters()).device
    model.eval()

    # Get a batch from your test_loader
    example_batch = next(iter(test_loader))
    images = example_batch["image"].to(device)  # (B, C, H, W) if 2D
    labels = example_batch["label"].to(device)  # (B,)
    print(f"Images batch shape: {images.shape}")

    # Select the first image from the batch
    single_image = images[0].unsqueeze(0)       # => (1, C, H, W)
    single_label = labels[0].unsqueeze(0)       # => (1,)

    # Generate GradCAM heatmap using the provided object
    cam_result = gradcam_obj(x=single_image)    # => shape (1, 1, H, W)
    print("GradCAM output shape:", cam_result.shape)

    # Squeeze the heatmap to (H, W)
    cam_tensor = cam_result.squeeze(0).squeeze(0).cpu().numpy()  # => shape (H, W)
    print("GradCAM output shape after squeeze:", cam_tensor.shape)

    # Apply thresholding to the heatmap
    cam_tensor_thresholded = threshold_heatmap(cam_tensor, threshold=threshold, normalize=True)

    # Convert the single_image to a NumPy array
    input_image_np = single_image.squeeze(0).cpu().numpy()  # => (C, H, W)
    input_image_np = np.transpose(input_image_np, (1, 2, 0))  # => (H, W, C)

    # Normalize the input image to [0, 1]
    input_image_np = min_max_normalization(input_image_np)

    # Use the get_overlay_heatmap function to visualize the results
    print("Original Heatmap Overlay:")
    get_overlay_heatmap(torch.tensor(input_image_np).permute(2, 0, 1), cam_tensor, alpha=0.6, cmap="jet")

    print(f"Thresholded({threshold}) Heatmap Overlay:")
    get_overlay_heatmap(torch.tensor(input_image_np).permute(2, 0, 1), cam_tensor_thresholded, alpha=0.6, cmap="jet")


def generate_and_save_gradcam_batch(
    model,
    loader,
    gradcam_obj,
    output_dir,
    class1_name,
    class0_name,
    run_name=None,
    experiment_name=None
):
    """
    Generates and saves GradCAM visualizations for a batch of images.

    Args:
        model (torch.nn.Module): The trained model
        test_loader (DataLoader): DataLoader containing test images
        gradcam_obj: Initialized GradCAM object (e.g., GradCAM or GradCAM++)
        output_dir (str): Base directory for saving outputs
        class1_name (str): Name of class 1
        class0_name (str): Name of class 0
        min_max_normalization (callable): Function to normalize arrays to [0,1]
        run_name (str, optional): MLflow run name for output directory
        experiment_name (str, optional): MLflow experiment name for output directory
    """
    # Get device from model
    device = next(model.parameters()).device
    model.eval()

    # Create output directory
    # Get GradCAM variant name from the object's class
    gradcam_variant = gradcam_obj.__class__.__name__.lower()  # will be 'gradcam' or 'gradcampp'

    # Create output directory with GradCAM variant in the path
    cam_output_dir = os.path.join(output_dir, f"{gradcam_variant}_outputs", 
                                run_name if run_name else experiment_name)
    os.makedirs(cam_output_dir, exist_ok=True)

    print(f"Saving {gradcam_variant} visualizations to: {cam_output_dir}")

    for batch_idx, batch_data in enumerate(loader):
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        print(f"Processing batch {batch_idx+1}, shape: {images.shape}")

        # Compute GradCAM for entire batch using provided object
        #cam_batch = gradcam_obj(x=images)
        
        # Process each image in batch
        for i in range(images.shape[0]):
            # Extract a single image and add a batch dimension
            image_tensor = images[i].unsqueeze(0)  # shape becomes [1, C, H, W]
            # Optionally, specify the class index if needed (e.g., class_idx=0)
            single_cam_tensor = gradcam_obj(x=image_tensor)  # or gradcam_obj(x=image_tensor, class_idx=0)
            
            # Remove batch dimension for processing
            single_image = image_tensor[0].cpu().numpy()
            single_cam = single_cam_tensor[0].cpu().numpy()
                
            # Get image label
            image_label = extract_labels_meaning(
                [labels[i].item()], 
                class1_name, 
                class0_name
            )
            #print(f"Processing image {i+1} (Label: {image_label})")

            # Prepare images for visualization
            single_image = np.transpose(single_image, (1, 2, 0))
            single_cam = single_cam.squeeze()

            # Normalize images
            single_image_norm = min_max_normalization(single_image,channel_wise=True)
            cam_normalized = min_max_normalization(single_cam)

            # Create visualization
            fig, axarr = plt.subplots(1, 2, figsize=(10, 4))

            # Original image
            axarr[0].imshow(single_image_norm, 
                          cmap='gray' if single_image.shape[-1] == 1 else None)
            axarr[0].set_title(f"Original Image (Label: {image_label})")
            axarr[0].axis('off')

            # GradCAM overlay
            axarr[1].imshow(single_image_norm, 
                          cmap='gray' if single_image.shape[-1] == 1 else None)
            axarr[1].imshow(cam_normalized, cmap='jet', alpha=0.5)
            axarr[1].set_title("Grad-CAM Overlay")
            axarr[1].axis('off')

            plt.tight_layout()

            # Save visualization
            save_path = os.path.join(cam_output_dir, f"batch_{batch_idx}_img_{i}.png")
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved Grad-CAM overlay: {save_path}")
        
    return cam_output_dir
    

def process_and_save_batch_gradcam_and_Overlay(
    model, 
    test_loader, 
    gradcam_obj,
    base_dir,
    class1_name,
    class0_name, 
    min_max_rescale_for_display, 
    threshold=0.5,
    run_name=None,
    experiment_name=None
):
    """
    Processes a DataLoader, generates Grad-CAM visualizations for each image, and saves them.
    Includes both original and thresholded overlays.

    Args:
        model: The PyTorch model
        test_loader: The DataLoader for the dataset
        gradcam_obj: Initialized GradCAM object (GradCAM or GradCAM++)
        base_dir: The base directory to save output
        class1_name: Name of class 1 for labels
        class0_name: Name of class 0 for labels
        min_max_rescale_for_display: Function for min-max rescaling
        threshold: Threshold value for heatmap thresholding (default: 0.5)
        run_name: MLflow run name (optional)
        experiment_name: MLflow experiment name (optional)
    """
    # Get device from model
    device = next(model.parameters()).device
    model.eval()

    # Create output directory with GradCAM variant in the path
    gradcam_variant = gradcam_obj.__class__.__name__.lower()
    output_dir = os.path.join(base_dir, f"{gradcam_variant}_outputs", 
                             run_name if run_name else experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, batch_data in enumerate(test_loader):
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        for i in range(images.shape[0]):
            single_image = images[i].unsqueeze(0)
            single_label = labels[i].unsqueeze(0)

            # Get label meaning using built-in function
            labels_list = [0, 1]
            class_names = {0: class0_name, 1: class1_name}
            image_label = class_names[single_label.item()]

            # Generate GradCAM heatmap
            cam_result = gradcam_obj(x=single_image, class_idx=1)
            cam_tensor = cam_result.squeeze(0).squeeze(0).cpu().numpy()

            # Process image and apply thresholding
            input_image_np = single_image.squeeze(0).cpu().numpy()
            input_image_np = np.transpose(input_image_np, (1, 2, 0))
            input_image_np = min_max_rescale_for_display(input_image_np)
            
            # Use existing functions for normalization and thresholding
            cam_normalized = min_max_normalization(cam_tensor)
            cam_thresholded = threshold_heatmap(cam_tensor, 
                                              threshold=threshold, 
                                              normalize=True)
            cam_rgba_trasholded = get_overlay_heatmap(torch.tensor(input_image_np).permute(2, 0, 1), cam_thresholded, alpha=0.6, cmap="jet")

            # Create visualization
            fig, axarr = plt.subplots(1, 3, figsize=(15, 5))

            # Original Image
            axarr[0].imshow(input_image_np)
            axarr[0].set_title(f"Original Image (Label: {image_label})")
            axarr[0].axis('off')

            # Normalized Overlay
            axarr[1].imshow(input_image_np)
            axarr[1].imshow(cam_normalized, cmap="jet", alpha=0.5)
            axarr[1].set_title(f"{gradcam_variant.upper()} Overlay (Normalized)")
            axarr[1].axis('off')

            # Thresholded Overlay
            axarr[2].imshow(input_image_np)
            axarr[2].imshow(cam_rgba_trasholded, cmap="jet", alpha=0.5)
            axarr[2].set_title(f"{gradcam_variant.upper()} Overlay (Thresholded, t={threshold})")
            axarr[2].axis('off')

            plt.tight_layout()

            # Save visualization
            save_path = os.path.join(output_dir, "threshold", f"batch_{batch_idx}_img_{i}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved {gradcam_variant.upper()} overlay: {save_path}")
    
    return output_dir
