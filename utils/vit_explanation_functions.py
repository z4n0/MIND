import os
import re
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List

# ---------------------------------
# 2) Attention Rollout
# ---------------------------------
def compute_attention_rollout_matrix(all_layers_attention: List[torch.Tensor], head_fusion: str = 'mean'):
    """
    Computes Attention Rollout to visualize information flow through the Transformer layers.
    
    This method tracks how attention propagates from the input to the final layer,
    explicitly accounting for residual (skip) connections.

    Args:
        all_layers_attention (List[torch.Tensor]): List of attention tensors for each layer.
                                                   Each tensor has shape [Batch, Num_Heads, Tokens, Tokens].
        head_fusion (str): Strategy to aggregate attention heads ('mean', 'max'). Default: 'mean'.

    Returns:
        The aggregated attention map (Rollout) with shape [Batch, Tokens, Tokens].
    """
    rollout_matrix = None

    # Iterate through layers from the first (closest to input) to the last
    for layer_idx, layer_attn in enumerate(all_layers_attention):
        
        # 1. Fusion of Attention Heads
        # usually average the heads to get a holistic view of the layer's attention.
        # Input: [B, Heads, N, N] -> Output: [B, N, N]
        if head_fusion == 'mean':
            attn_fused = layer_attn.mean(dim=1)
        elif head_fusion == 'max':
            attn_fused = layer_attn.max(dim=1)[0]
        else:
             attn_fused = layer_attn.mean(dim=1) 

        batch_size, num_tokens, _ = attn_fused.shape

        # 2. Handling Residual Connections
        # In a Transformer, the output is defined as y = Attention(x) + x.
        # To model the information flow that "skips" the attention mechanism (the '+ x' part),
        # we add an Identity matrix. This represents implicit self-attention.
        identity_matrix = torch.eye(num_tokens, device=attn_fused.device).unsqueeze(0).expand(batch_size, -1, -1)
        attn_fused_with_residual = attn_fused + identity_matrix

        # 3. Normalization
        # After adding the identity matrix, values no longer sum to 1.
        # We re-normalize to maintain a valid probability distribution.
        row_sums = attn_fused_with_residual.sum(dim=-1, keepdim=True)
        attn_normalized = attn_fused_with_residual / (row_sums + 1e-7) # 1e-7 for numerical stability

        # 4. Matrix Multiplication (The "Rollout")
        # We multiply the current layer's attention matrix with the accumulated rollout matrix.
        # Mathematically: Rollout_L = Attention_L * Rollout_(L-1)
        if rollout_matrix is None:
            rollout_matrix = attn_normalized
        else:
            rollout_matrix = torch.matmul(attn_normalized, rollout_matrix)

    return rollout_matrix

def normalize_image(img: np.ndarray, per_channel: bool = True) -> np.ndarray:
    """
    Scales `img` to [0..1]. 
    If `per_channel=True` and shape=[H,W,C], normalizes each channel independently.
    Otherwise, min-max across the entire array.
    """
    if img.ndim == 3 and per_channel:
        H, W, C = img.shape
        out = np.zeros_like(img, dtype=np.float32)
        for c in range(C):
            chan = img[..., c]
            mn, mx = chan.min(), chan.max()
            if (mx - mn) < 1e-7:
                out[..., c] = 0.0
            else:
                out[..., c] = (chan - mn) / (mx - mn)
        return out
    else:
        mn, mx = img.min(), img.max()
        if (mx - mn) < 1e-7:
            return np.zeros_like(img, dtype=np.float32)
        return (img - mn) / (mx - mn)

# ---------------------------------
# 3) Overlay Visualization
# ---------------------------------
def overlay_rgb_and_gray(base_rgb, gray_channel, alpha=0.4):
    """
    Overlays a single gray channel onto a base RGB image with alpha transparency.
    
    Args:
        base_rgb (np.ndarray): [H, W, 3] in [0..1].
        gray_channel (np.ndarray): [H, W] in [0..1].
        alpha (float): overlay transparency in [0..1].
    """
    plt.imshow(base_rgb)  # Show base
    plt.imshow(gray_channel, cmap='gray', alpha=alpha)  # Gray overlay


def visualize_image(ax, img_4ch_or_3ch, overlay_alpha=0.4, normalize=True):
    """
    Displays either a 3-channel or 4-channel NumPy image (channels-first or channels-last).
    For 4-channel images with shape [H,W,4] or [4,H,W], we interpret them as:
        G(0), B(1), Gray(2), R(3) 
    and reorder to: 
        base RGB = (R,G,B), overlay Gray = channel 2.

    For 3-channel images, we just reorder [2, 1, 0] -> (R,G,B).
    If shape is [C,H,W], we transpose to [H,W,C].
    Finally, we min-max normalize if `normalize=True`.
    """
    # Ensure np array
    if isinstance(img_4ch_or_3ch, torch.Tensor):
        img_4ch_or_3ch = img_4ch_or_3ch.cpu().numpy()

    arr = img_4ch_or_3ch
    # Move channels to last dimension if needed
    if arr.ndim == 3 and arr.shape[0] in (3,4):
        # channels-first -> channels-last
        arr = np.transpose(arr, (1,2,0))

    H, W, *rest = arr.shape
    if len(rest) == 0:
        # Single-channel grayscale
        base_rgb = arr
        if normalize:
            base_rgb = normalize_image(base_rgb, per_channel=False)
        ax.imshow(base_rgb, cmap='gray')
        ax.set_title("Single-Channel Grayscale", fontsize=8)
        ax.axis('off')
        return

    # If 3 or 4 channels in last dimension
    num_ch = arr.shape[-1]
    if num_ch == 3:
        # reorder from (G,B,R) -> (R,G,B), if that's your convention
        # But if your data is ALREADY (R,G,B), then skip the reorder
        # For demonstration, let's do a reorder from (G,B,R) => (R,G,B)
        arr_rgb = arr[..., [2, 0, 1]]
        if normalize:
            arr_rgb = normalize_image(arr_rgb, per_channel=True)
        ax.imshow(arr_rgb)
        ax.set_title("3-Channel as RGB", fontsize=8)
        ax.axis('off')

    elif num_ch == 4:
        # interpret: G(0), B(1), Gray(2), R(3) -> base=(R,G,B), overlay=Gray(2)
        base_rgb = arr[..., [3, 0, 1]]  # shape [H,W,3]
        overlay = arr[..., 2]          # shape [H,W]

        if normalize:
            base_rgb = normalize_image(base_rgb, per_channel=True)
            overlay = normalize_image(overlay, per_channel=False)

        ax.imshow(base_rgb)  # base
        ax.imshow(overlay, cmap='gray', alpha=overlay_alpha)  # overlay
        ax.set_title("4-Channel: RGB + Gray Overlay", fontsize=8)
        ax.axis('off')

    else:
        ax.set_title(f"Unexpected channels={num_ch}", fontsize=8)
        ax.axis('off')


def save_attention_overlays_side_by_side(
    data_loader,
    model,
    output_directory,
    device,
    heatmap_alpha=0.5
):
    """
    Generates side-by-side comparisons: 
    1. The original microscopy image (reconstructed to RGB).
    2. The same image with the Vision Transformer's attention map overlaid.

    This function handles the specific channel ordering of confocal microscopes:
    - 3 Channels: Assumes (Green, Blue, Red) -> Converts to (Red, Green, Blue).
    - 4 Channels: Assumes (Green, Blue, Gray-Structure, Red) -> Uses RGB + Gray.

    Args:
        data_loader (DataLoader): Yields batches with 'image' and 'label'.
        model (torch.nn.Module): The trained ViT model. Must store attention in 'blk.attn.att_mat'.
        output_directory (str): Where to save the resulting PNGs.
        device (torch.device): 'cuda' or 'cpu'.
        heatmap_alpha (float): Transparency of the attention heatmap (0.0 to 1.0).
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    print(f"Starting visualization. Saving to: {output_directory}")

    with torch.no_grad():
        for batch_index, batch_data in enumerate(data_loader):
            # Move data to GPU/CPU
            # Shape: [Batch_Size, Channels, Height, Width]
            input_tensor = batch_data["image"].to(device)
            labels_tensor = batch_data["label"].to(device)

            # --- 1. Forward Pass ---
            # We need the forward pass to populate the attention matrices inside the model blocks
            model_outputs, _ = model(input_tensor)
            _, predicted_labels = torch.max(model_outputs, dim=1)

            # --- 2. Extract Attention & Compute Rollout ---
            # Retrieve raw attention matrices from each Transformer block
            # Note: This requires your ViT blocks to save 'att_mat' during forward()
            layer_attentions = [block.attn.att_mat for block in model.blocks]
            
            # Aggregate attention across layers using the Rollout method
            # Result Shape: [Batch_Size, Total_Tokens, Total_Tokens]
            attention_rollout = compute_attention_rollout_matrix(layer_attentions)

            # --- 3. Prepare Data for Visualization (CPU conversion) ---
            batch_images_np = input_tensor.cpu().numpy()
            predicted_np = predicted_labels.cpu().numpy()
            rollout_np = attention_rollout.cpu().numpy()

            # Iterate through each image in the current batch
            for item_idx in range(batch_images_np.shape[0]):
                
                # Extract single image: [Channels, Height, Width]
                image_chw = batch_images_np[item_idx]
                
                # Transpose to [Height, Width, Channels] for Matplotlib
                if image_chw.ndim == 3 and image_chw.shape[0] in (3, 4):
                    image_hwc = np.transpose(image_chw, (1, 2, 0))
                else:
                    image_hwc = image_chw # Fallback for grayscale or errors

                # --- 4. Channel Management (Microscopy Specifics) ---
                display_image_rgb = None
                structure_channel_gray = None
                plot_title = "Original"

                # Case A: 3-Channel Images
                # saves as (Green, Blue, Red). We want RGB for display.
                if image_hwc.ndim == 3 and image_hwc.shape[-1] == 3:
                    # Permute: Idx 2 (Red) -> 0, Idx 0 (Green) -> 1, Idx 1 (Blue) -> 2
                    display_image_rgb = image_hwc[..., [2, 0, 1]].astype(np.float32)
                    display_image_rgb = normalize_image(display_image_rgb, per_channel=True)

                # Case B: 4-Channel Images (with structural/brightfield channel)
                # Assumed Input: Green(0), Blue(1), Gray/Structure(2), Red(3)
                elif image_hwc.ndim == 3 and image_hwc.shape[-1] == 4:
                    # RGB part: Red(3), Green(0), Blue(1)
                    display_image_rgb = image_hwc[..., [3, 0, 1]].astype(np.float32)
                    # Structural part: Gray(2)
                    structure_channel_gray = image_hwc[..., 2].astype(np.float32)
                    
                    display_image_rgb = normalize_image(display_image_rgb, per_channel=True)
                    structure_channel_gray = normalize_image(structure_channel_gray, per_channel=False)
                    plot_title = "Original (RGB + Morphology)"

                # Case C: Fallback (Grayscale or unknown)
                else:
                    display_image_rgb = normalize_image(image_hwc, per_channel=False)
                    plot_title = "Original (Grayscale)"

                # Create copies to ensure we don't modify the original arrays during plotting
                img_for_overlay = display_image_rgb.copy()
                gray_for_overlay = structure_channel_gray.copy() if structure_channel_gray is not None else None

                # --- 5. Process Attention Map ---
                # Get rollout for this specific image: [Total_Tokens, Total_Tokens]
                single_img_rollout = rollout_np[item_idx]
                
                # We focus on the attention flowing FROM the CLS token (index 0) TO all image patches (indices 1:)
                # This tells us which parts of the image the model used to classify.
                cls_token_attention = single_img_rollout[0, 1:] 
                
                # Calculate grid size (e.g., sqrt(196) = 14)
                grid_side_len = int(math.sqrt(cls_token_attention.shape[0]))
                
                # Reshape flattened patches back to 2D grid: [14, 14]
                attention_grid = cls_token_attention.reshape(grid_side_len, grid_side_len)

                # Upsample the coarse grid to match original image size (e.g., 224x224)
                img_height, img_width = image_hwc.shape[0], image_hwc.shape[1]
                
                # Convert to tensor for interpolation: [Batch=1, Channel=1, H, W]
                attn_tensor = torch.from_numpy(attention_grid).unsqueeze(0).unsqueeze(0).float()
                
                # Bilinear interpolation creates a smooth heatmap
                upsampled_attn = F.interpolate(
                    attn_tensor, 
                    size=(img_height, img_width), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Back to numpy for plotting
                heatmap_np = upsampled_attn.squeeze().cpu().numpy()
                heatmap_np = normalize_image(heatmap_np, per_channel=False)

                # --- 6. Plotting ---
                fig, (ax_original, ax_overlay) = plt.subplots(1, 2, figsize=(12, 6))

                # Left Panel: Original Image
                ax_original.set_title(plot_title, fontsize=10)
                ax_original.axis("off")
                ax_original.imshow(display_image_rgb)
                if gray_for_overlay is not None:
                    # Overlay structural details in grayscale (useful for skin boundaries)
                    ax_original.imshow(gray_for_overlay, cmap='gray', alpha=0.3)

                # Right Panel: Image + Attention Heatmap
                ax_overlay.set_title("Prediction Heatmap", fontsize=10)
                ax_overlay.axis("off")
                # 1. Base image
                ax_overlay.imshow(img_for_overlay)
                # 2. Structural gray (optional)
                if gray_for_overlay is not None:
                    ax_overlay.imshow(gray_for_overlay, cmap='gray', alpha=0.3)
                # 3. Attention Heatmap (Jet colormap is standard for intensity)
                ax_overlay.imshow(heatmap_np, cmap='jet', alpha=heatmap_alpha)

                # Overall Title
                pred_class_idx = int(predicted_np[item_idx])
                fig.suptitle(
                    f"Batch {batch_index}, Sample {item_idx} | Predicted Class: {pred_class_idx}",
                    fontsize=11
                )

                # --- 7. Saving ---
                file_name = f"overlay_b{batch_index}_s{item_idx}_pred{pred_class_idx}.png"
                full_path = os.path.join(output_directory, file_name)
                
                plt.savefig(full_path, bbox_inches="tight", dpi=100)
                plt.close(fig) # Close figure to free memory