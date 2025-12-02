import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from .data_extraction_functions import extract_labels_meaning
from .data_visualization_functions import min_max_normalization

def format_label_text(label_idx, class_names):
    """
    Return a string representation like '1 (MSA)' if a mapping is available.
    """
    if label_idx is None:
        return "N/A"
    idx = int(label_idx)
    label_name = None
    if class_names is not None:
        try:
            if isinstance(class_names, dict):
                label_name = class_names.get(idx, class_names.get(str(idx)))
            else:
                label_name = class_names[idx]
        except (IndexError, KeyError, TypeError):
            label_name = None
    return f"{idx} ({label_name})" if label_name is not None else str(idx)

def _extract_logits(model_output):
    """
    Handle different model output formats and return a tensor of logits.
    """
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, (list, tuple)):
        return model_output[0]
    if isinstance(model_output, dict):
        for key in ("logits", "output", "out"):
            if key in model_output:
                return model_output[key]
        first_key = next(iter(model_output))
        return model_output[first_key]
    raise TypeError(f"Unsupported model output type: {type(model_output)}")

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
        image (torch.Tensor): Original image tensor (C, H, W). May have 3 or 4 channels.
        heatmap (np.ndarray): Heatmap (H, W).
        alpha (float): Transparency for the heatmap overlay.
        cmap (str): Colormap to use.
    
    Returns:
        heatmap_rgba (np.ndarray): The RGBA overlay.
    """
    # Bring image to [H, W, C] format.
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    # If values exceed 1, assume they are in [0,255] and scale down.
    if image_np.max() > 1:
        image_np = image_np / 255.0

    # Reorder channels based on the input format.
    # For 3-channel input: (G, B, R) => (R, G, B)
    # For 4-channel input: (G, B, Red, Gray) => (R, G, B)
    if image_np.shape[-1] == 3:
        image_np = image_np[..., [2, 0, 1]]
    elif image_np.shape[-1] == 4:
        image_np = image_np[..., [2, 0, 1]]
    
    # Create an RGBA heatmap.
    cmap_fn = plt.get_cmap(cmap)
    heatmap_rgba = cmap_fn(heatmap)
    # Set the alpha channel: only nonzero heatmap regions are overlaid.
    heatmap_rgba[..., 3] = np.where(heatmap > 0, alpha, 0)

    # Plot the original image and overlay for visual debugging.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(image_np)
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(image_np)
    ax2.imshow(heatmap_rgba, interpolation='nearest')
    ax2.set_title('GradCAM Overlay')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

    return heatmap_rgba

def generate_and_save_gradcam_batch(
    model,
    loader,
    gradcam_obj,
    output_dir,
    class_names,  # single parameter for class names (list or dict)
    run_name=None,
    experiment_name=None
):
    """
    Generates and saves GradCAM visualizations for a batch of images.
    
    For each image:
      - Converts the tensor to a NumPy array in (H, W, C) order.
      - Reorders channels: if 3-channel (from G, B, R) to (R, G, B); if 4-channel (from G, B, Red, Gray) to (R, G, B).
      - Applies min-max normalization.
      - Overlays the GradCAM heatmap.
    
    Args:
        model (torch.nn.Module): Trained model.
        loader (DataLoader): DataLoader containing images.
        gradcam_obj: An initialized GradCAM object.
        output_dir (str): Base directory for saving outputs.
        class_names: List or dictionary for mapping label indices to names.
        run_name (str, optional): Run name for output folder.
        experiment_name (str, optional): Experiment name if run_name is not provided.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    device = next(model.parameters()).device
    model.eval()

    gradcam_variant = gradcam_obj.__class__.__name__.lower()
    cam_output_dir = os.path.join(output_dir, f"{gradcam_variant}_outputs",
                                  run_name if run_name else experiment_name)
    os.makedirs(cam_output_dir, exist_ok=True)
    print(f"Saving {gradcam_variant} visualizations to: {cam_output_dir}")

    for batch_idx, batch_data in enumerate(loader):
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        print(f"Processing batch {batch_idx+1}, shape: {images.shape}")

        for i in range(images.shape[0]):
            image_tensor = images[i].unsqueeze(0)
            single_cam_tensor = gradcam_obj(x=image_tensor)

            with torch.no_grad():
                logits = _extract_logits(model(image_tensor))
            pred_idx = torch.argmax(logits, dim=1).item()
            label_val = labels[i].item()
            pred_color = "lime" if pred_idx == label_val else "red"
            true_label_text = format_label_text(label_val, class_names)
            pred_label_text = format_label_text(pred_idx, class_names)
            
            # Get the image and heatmap from tensors.
            single_image = image_tensor[0].cpu().numpy()   # shape: (C, H, W)
            single_cam = single_cam_tensor.squeeze().cpu().numpy()

            # Convert image to (H, W, C)
            single_image = np.transpose(single_image, (1, 2, 0))
            # Reorder channels:
            if single_image.shape[-1] == 3:
                # For 3-channel images: (G, B, R) -> (R, G, B)
                display_image = single_image[..., [2, 0, 1]]
            elif single_image.shape[-1] == 4:
                # For 4-channel images: (G, B, Red, Gray) -> (R, G, B)
                display_image = single_image[..., [2, 0, 1]]
            else:
                display_image = single_image

            # Normalize display image and heatmap.
            single_image_norm = min_max_normalization(display_image, channel_wise=True)
            cam_normalized = min_max_normalization(single_cam)

            # Create visualization with 2 panels: original and overlay.
            fig, axarr = plt.subplots(1, 2, figsize=(10, 4))
            axarr[0].imshow(single_image_norm, cmap='gray' if single_image_norm.shape[-1]==1 else None)
            axarr[0].set_title("Original Image")
            axarr[0].text(
                0.01,
                0.99,
                f"True: {true_label_text}",
                transform=axarr[0].transAxes,
                fontsize=11,
                color="white",
                va="top",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=3),
            )
            axarr[0].text(
                0.01,
                0.86,
                f"Pred: {pred_label_text}",
                transform=axarr[0].transAxes,
                fontsize=11,
                color=pred_color,
                va="top",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=3),
            )
            axarr[0].axis('off')

            axarr[1].imshow(single_image_norm, cmap='gray' if single_image_norm.shape[-1]==1 else None)
            axarr[1].imshow(cam_normalized, cmap='jet', alpha=0.5)
            axarr[1].set_title("Grad-CAM Overlay")
            axarr[1].axis('off')
            plt.tight_layout()

            save_path = os.path.join(cam_output_dir, f"batch_{batch_idx}_img_{i}.png")
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved Grad-CAM overlay: {save_path}")

    return cam_output_dir

def visualize_gradcam_for_single_image(model, test_loader, gradcam_obj, min_max_normalization, threshold=0.5, overlay_alpha_DAPI=0.2, random_seed=None):
    """
    Retrieves a random image from the test_loader, generates GradCAM heatmaps,
    and visualizes the results. Supports both 3‑channel and 4‑channel images.
    
    Args:
        model: The PyTorch model.
        test_loader: DataLoader for the test dataset.
        gradcam_obj: An initialized GradCAM object (e.g., GradCAM or GradCAM++).
        min_max_normalization: Function to normalize an array to [0, 1].
        threshold (float): Threshold for heatmap thresholding (default 0.5).
        overlay_alpha_DAPI (float): Transparency for overlaying the extra (gray) channel.
        random_seed (int, optional): Seed for random selection. Set for reproducibility.
    """
    import numpy as np
    import torch
    import random

    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # Get device from model
    device = next(model.parameters()).device
    model.eval()

    # Get the total number of batches
    num_batches = len(test_loader)
    
    # Choose a random batch
    batch_idx = random.randint(0, num_batches - 1)
    
    # Iterate to the selected batch
    for i, batch_data in enumerate(test_loader):
        if i == batch_idx:
            # We've reached our randomly chosen batch
            example_batch = batch_data
            break
    
    images = example_batch["image"].to(device)  # (B, C, H, W)
    labels = example_batch["label"].to(device)  # (B,)
    print(f"Images batch shape: {images.shape}")

    # Choose a random image within the batch
    img_idx = random.randint(0, images.shape[0] - 1)
    print(f"Selected random image {img_idx} from batch {batch_idx}")
    
    # Use the selected image
    single_image = images[img_idx].unsqueeze(0)  # shape: (1, C, H, W)
    single_label = labels[img_idx].unsqueeze(0)
    
    # ==================== ADD THIS DEBUGGING BLOCK ====================
    print("\n--- Model Output Debug ---")
    with torch.no_grad():
        # Get the raw scores (logits) from the model
        logits = model(single_image)
        
        # Get the probabilities after softmax
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

        print(f"Raw Logits: {logits.cpu().numpy()}")
        print(f"Probabilities: {probabilities.cpu().numpy()}")
        print(f"Predicted Class: {predicted_class.item()}")
    print("--------------------------\n")
    # =================================================================
    # Generate GradCAM heatmap.
    cam_result = gradcam_obj(x=single_image)  # expected shape: (1, 1, H, W)
    print("GradCAM output shape:", cam_result.shape)
    cam_tensor = cam_result.squeeze(0).squeeze(0).cpu().numpy()  # shape: (H, W)
    print("GradCAM output shape after squeeze:", cam_tensor.shape)
    
    # Threshold the heatmap.
    cam_tensor_thresholded = threshold_heatmap(cam_tensor, threshold=threshold, normalize=True)
    
    # Convert single_image to NumPy array (shape will be (C, H, W)).
    input_image_np = single_image.squeeze(0).cpu().numpy()  # (C, H, W)
    # Transpose to (H, W, C).
    input_image_np = np.transpose(input_image_np, (1, 2, 0))
    
    # Now, handle channel reordering based on number of channels.
    if input_image_np.shape[-1] == 3:
        # For 3-channel images, assume order (Green, Blue, Red) and reorder to (Red, Green, Blue).
        composite_rgb = input_image_np[..., [2, 0, 1]]
        # There is no extra overlay.
        display_image = composite_rgb
    elif input_image_np.shape[-1] == 4:
        # For 4-channel images, assume order (Green, Blue, Red, Gray).
        # Build base RGB from channels: Red from index 2, Green from index 0, Blue from index 1.
        composite_rgb = input_image_np[..., [2, 0, 1]]
        # The extra channel (gray) is taken from index 3.
        overlay_channel = input_image_np[..., 3]
        display_image = composite_rgb
    else:
        display_image = input_image_np

    # Normalize the base RGB composite.
    if display_image.ndim == 3:
        display_image = min_max_normalization(display_image, channel_wise=True)
    
    # For 4-channel case, normalize the extra (gray) channel separately.
    if input_image_np.shape[-1] == 4:
        overlay_norm = min_max_normalization(overlay_channel)
    else:
        overlay_norm = None

    # For the GradCAM overlay, we want to use our composite image as the background.
    # Convert display_image back to a tensor (channel-first).
    from torch import tensor
    display_for_overlay = tensor(display_image).permute(2, 0, 1)

    # Visualize the overlays using get_overlay_heatmap.
    print("Original Heatmap Overlay:")
    # get_overlay_heatmap will now reorder channels if needed; but here our display_for_overlay is already correct.
    get_overlay_heatmap(display_for_overlay, cam_tensor, alpha=0.6, cmap="jet")
    
    # Additionally, if we have an extra overlay (4-channel case), show it:
    if overlay_norm is not None:
        print(f"Overlay (gray channel) over composite with alpha={overlay_alpha_DAPI}:")
        # Create an overlay plot from the gray channel.
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(display_image)
        ax.imshow(overlay_norm, cmap='gray', alpha=overlay_alpha_DAPI)
        ax.set_title("4-channel composite with Gray overlay")
        ax.axis('off')
        plt.show()
    
    print(f"Thresholded ({threshold}) Heatmap Overlay:")
    get_overlay_heatmap(display_for_overlay, cam_tensor_thresholded, alpha=0.6, cmap="jet")

def process_and_save_batch_gradcam_and_Overlay(
    model, 
    test_loader, 
    gradcam_obj,
    base_dir,
    class_names,              # Same single parameter for class names (list or dict)
    min_max_rescale_for_display, 
    threshold=0.5,
    run_name=None,
    experiment_name=None,
    overlay_alpha_DAPI=0.2,   # The alpha for your extra (gray) channel
    input_channel_order="RGB" # New parameter to specify input channel semantics
):
    """
    Processes a DataLoader, generates GradCAM visualizations for each image, 
    and saves multi-panel figures showing:
      - Panel 1: The image as it appears in `show_misclassified_images`
                 (if 4-channel: we overlay the gray channel on top of the base RGB).
      - Panel 2: The same base image with the GradCAM heatmap overlaid.
      - Panel 3: The base image with a thresholded GradCAM heatmap.

    This ensures your images look exactly the same as in `show_misclassified_images` 
    for the base color channels and DAPI (gray) overlay.
    
    Args:
        model (torch.nn.Module): The model.
        test_loader (DataLoader): DataLoader for the test set.
        gradcam_obj: Initialized GradCAM object (GradCAM or GradCAM++), providing heatmaps.
        base_dir (str): Base directory to save outputs.
        class_names: List or dict for mapping label indices to names.
        min_max_rescale_for_display: Function that scales an image to [0,1].
        threshold (float): Threshold for the heatmap (default: 0.5).
        run_name (str, optional): Run name for output folders.
        experiment_name (str, optional): Experiment name if run_name is None.
        overlay_alpha_DAPI (float): Transparency for the extra (gray) channel overlay 
                                    (like in `show_misclassified_images`).
        input_channel_order (str): Expected channel order of the input images.
            - "RGB" (default): Assumes input is (Red, Green, Blue) or (Red, Green, Blue, Gray).
              This is correct if `from_GBR_to_RGB` was applied in the DataLoader.
            - "GBR": Assumes input is (Green, Blue, Red) or (Green, Blue, Gray, Red).
              This is correct for raw confocal data without permutation.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    device = next(model.parameters()).device
    model.eval()

    gradcam_variant = gradcam_obj.__class__.__name__.lower()
    output_dir = os.path.join(
        base_dir,
        f"{gradcam_variant}_outputs",
        run_name if run_name else experiment_name
    )
    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, batch_data in enumerate(test_loader):
        images = batch_data["image"].to(device)  # shape: (B, C, H, W)
        labels = batch_data["label"].to(device)
        
        print(f"Processing batch {batch_idx + 1}, shape: {images.shape}")

        for i in range(images.shape[0]):
            single_image = images[i].unsqueeze(0)  # shape: (1, C, H, W)
            single_label = labels[i].unsqueeze(0)
            label_val = single_label.item()

            with torch.no_grad():
                logits = _extract_logits(model(single_image))
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_color = "lime" if pred_idx == label_val else "red"
            true_label_text = format_label_text(label_val, class_names)
            pred_label_text = format_label_text(pred_idx, class_names)

            # Generate GradCAM for this image.
            cam_result = gradcam_obj(x=single_image, class_idx=label_val)
            # shape might be (1, 1, H, W)
            cam_tensor = cam_result.squeeze(0).squeeze(0).cpu().numpy()  # shape: (H, W)

            # Convert single_image to numpy: (C, H, W) → (H, W, C)
            input_image_np = single_image.squeeze(0).cpu().numpy()
            input_image_np = np.transpose(input_image_np, (1, 2, 0))  # now (H, W, C)

            # Logic to extract RGB composite and optional Overlay based on input order
            overlay = None
            
            if input_channel_order == "RGB":
                # Case 1: Input is already processed into RGB (or RGBGr)
                # Expected 3-ch: (Red, Green, Blue)
                # Expected 4-ch: (Red, Green, Blue, Gray)
                if input_image_np.shape[-1] == 3:
                    base_rgb = input_image_np
                    composite = base_rgb
                elif input_image_np.shape[-1] == 4:
                    base_rgb = input_image_np[..., :3] # First 3 are R, G, B
                    overlay = input_image_np[..., 3]   # 4th is Gray
                    composite = base_rgb
                else:
                    composite = input_image_np

            elif input_channel_order == "GBR":
                # Case 2: Input is raw confocal data
                # Expected 3-ch: (Green, Blue, Red)
                # Expected 4-ch: (Green, Blue, Gray, Red)
                if input_image_np.shape[-1] == 3:
                    # (Green, Blue, Red) → (Red, Green, Blue)
                    base_rgb = input_image_np[..., [2, 0, 1]]
                    composite = base_rgb
                elif input_image_np.shape[-1] == 4:
                    # (Green, Blue, Gray, Red)
                    # base_rgb from channels [3, 0, 1] => (Red, Green, Blue)
                    base_rgb = input_image_np[..., [3, 0, 1]]
                    # overlay is channel index 2 => Gray
                    overlay = input_image_np[..., 2]
                    composite = base_rgb
                else:
                    composite = input_image_np
            else:
                raise ValueError(f"Unknown input_channel_order '{input_channel_order}'. Use 'RGB' or 'GBR'.")

            # Normalize/scale for display, 
            # applying the same approach as you do in `show_misclassified_images`.
            # We'll do a separate step for composite and overlay if needed.
            composite = min_max_rescale_for_display(composite)  # or do your channel-wise function
            if overlay is not None:
                # Also scale the overlay channel to [0,1].
                # If you want channel-wise scaling, do it. Typically, this is grayscale => global min_max.
                min_val, max_val = overlay.min(), overlay.max()
                if max_val > min_val:
                    overlay = (overlay - min_val) / (max_val - min_val)
                else:
                    overlay = np.zeros_like(overlay, dtype=np.float32)

            # Apply thresholding to the GradCAM heatmap if desired.
            cam_thresholded = threshold_heatmap(cam_tensor, threshold=threshold, normalize=True)

            # We'll create 3 panels:
            # 1) Exactly as in show_misclassified_images (base RGB + optional gray overlay)
            # 2) base RGB (+ optional gray) with GradCAM overlay
            # 3) base RGB (+ optional gray) with thresholded GradCAM overlay
            fig, axarr = plt.subplots(1, 3, figsize=(20, 6))

            # Panel 1: "Original" composite with the gray overlay if 4 channels
            axarr[0].imshow(composite)
            if overlay is not None:
                axarr[0].imshow(overlay, cmap='gray', alpha=overlay_alpha_DAPI)
            axarr[0].set_title("Original Image", fontsize=14)
            axarr[0].text(
                0.01,
                0.99,
                f"True: {true_label_text}",
                transform=axarr[0].transAxes,
                fontsize=12,
                color="white",
                va="top",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=3),
            )
            axarr[0].text(
                0.01,
                0.86,
                f"Pred: {pred_label_text}",
                transform=axarr[0].transAxes,
                fontsize=12,
                color=pred_color,
                va="top",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=3),
            )
            axarr[0].axis('off')

            # Panel 2: same composite, plus GradCAM overlay
            # assume GradCAM is in [0,1], so we can alpha-blend it
            cam_normalized = min_max_normalization(cam_tensor)
            axarr[1].imshow(composite)
            if overlay is not None:
                axarr[1].imshow(overlay, cmap='gray', alpha=overlay_alpha_DAPI)
            axarr[1].imshow(cam_normalized, cmap='jet', alpha=0.5)
            axarr[1].set_title("GradCAM Overlay", fontsize=14)
            axarr[1].axis('off')

            # Panel 3: thresholded GradCAM overlay
            # We'll color the thresholded areas in jet, alpha=0.6, 0 if below threshold
            axarr[2].imshow(composite)
            if overlay is not None:
                axarr[2].imshow(overlay, cmap='gray', alpha=overlay_alpha_DAPI)
            # We'll convert thresholded heatmap to RGBA via get_overlay_heatmap, or just do:
            # thresholded => 1 where > threshold => 0 otherwise
            axarr[2].imshow(cam_thresholded, cmap='jet', alpha=0.6)
            axarr[2].set_title(f"Thresholded GradCAM (t={threshold})", fontsize=14)
            axarr[2].axis('off')

            plt.tight_layout()

            # Save the final figure
            save_path = os.path.join(output_dir, f"batch_{batch_idx}_img_{i}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved {gradcam_variant.upper()} overlay with DAPI-like channel to: {save_path}")

    return output_dir
