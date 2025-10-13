import os
import re
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------------
# 1) Normalization Helpers

# ---------------------------------
# 2) Attention Rollout
# ---------------------------------
def compute_rollout(attentions):
    """
    Computes the aggregated (rolled out) attention across layers.

    Args:
        attentions (List[torch.Tensor]): each shape [B, num_heads, tokens, tokens].
    Returns:
        torch.Tensor: shape [B, tokens, tokens].
    """
    rollout = None
    for attn in attentions:
        # Average over heads => shape [B, tokens, tokens]
        attn_heads_fused = attn.mean(dim=1)
        B, N, _ = attn_heads_fused.shape

        # Add identity matrix to account for residual
        I = torch.eye(N, device=attn_heads_fused.device).unsqueeze(0).expand(B, -1, -1)
        attn_heads_fused = attn_heads_fused + I

        # Row-normalize
        sums = attn_heads_fused.sum(dim=-1, keepdim=True)
        attn_heads_fused = attn_heads_fused / (sums + 1e-7)

        # Multiply across layers
        if rollout is None:
            rollout = attn_heads_fused
        else:
            rollout = torch.matmul(attn_heads_fused, rollout)
    return rollout  # shape [B, tokens, tokens]

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


# ---------------------------------
# 4) Main Function: Save Attention Overlays
# ---------------------------------
def save_attention_overlays_side_by_side(
    test_loader,
    model,
    output_path,
    device,
    overlay_alpha=0.5
):
    """
    Generates side-by-side figures: left = original image, right = image + attention overlay.
    Handles both 3ch and 4ch input images:
      - 3ch assumed order (G,B,R) -> reorder to (R,G,B).
      - 4ch assumed order (G,B,Gray(2),R(3)) -> base (R,G,B) plus channel2 (Gray).
        Then we overlay the final attention map in 'jet' colormap on top.

    Args:
        test_loader (DataLoader): yields dict with 'image' and 'label'
        model (torch.nn.Module): trained ViT with blocks storing attn in 'att_mat'
        output_path (str): folder for output
        device (torch.device): 'cuda' or 'cpu'
        overlay_alpha (float): alpha for the attention heatmap
    """
    os.makedirs(output_path, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images_batch = batch["image"].to(device)  # [B, C, H, W]
            labels_batch = batch["label"].to(device)

            # Forward pass: populate attention
            outputs, _ = model(images_batch)
            _, predicted = torch.max(outputs, dim=1)

            # Gather attentions
            attentions = [blk.attn.att_mat for blk in model.blocks]
            rollout = compute_rollout(attentions)  # [B, tokens, tokens]

            images_np = images_batch.cpu().numpy()  # [B, C, H, W]
            predicted_np = predicted.cpu().numpy()
            rollout_np = rollout.cpu().numpy()       # [B, tokens, tokens]

            for i in range(images_np.shape[0]):
                # Convert [C,H,W] -> [H,W,C] if needed
                img_cxhxw = images_np[i]  # shape [C,H,W]
                if img_cxhxw.ndim == 3 and img_cxhxw.shape[0] in (3,4):
                    img_hwc = np.transpose(img_cxhxw, (1,2,0))  # [H,W,C]
                else:
                    # single channel or something unexpected
                    img_hwc = img_cxhxw

                # ---------- Prepare Base Image (Left Panel) ----------
                base_rgb = None
                overlay_gray = None
                title_left = "Original"

                if img_hwc.ndim == 3 and img_hwc.shape[-1] == 3:
                    # reorder from (G,B,R) => (R,G,B) if that matches your convention
                    base_rgb = img_hwc[..., [2,0,1]].astype(np.float32)
                    base_rgb = normalize_image(base_rgb, per_channel=True)

                elif img_hwc.ndim == 3 and img_hwc.shape[-1] == 4:
                    # interpret channels: G(0),B(1),Gray(2),R(3)
                    base_rgb = img_hwc[..., [3,0,1]].astype(np.float32)
                    overlay_gray = img_hwc[..., 2].astype(np.float32)
                    base_rgb = normalize_image(base_rgb, per_channel=True)
                    overlay_gray = normalize_image(overlay_gray, per_channel=False)
                    title_left = "Original (RGB + Gray)"

                else:
                    # fallback single channel
                    base_rgb = normalize_image(img_hwc, per_channel=False)
                    title_left = "Original (Grayscale)"

                # ---------- Prepare the Right Panel with Overlay ----------
                # We copy the same base image for overlay
                base_for_overlay = base_rgb.copy() if isinstance(base_rgb, np.ndarray) else base_rgb
                gray_for_overlay = overlay_gray.copy() if isinstance(overlay_gray, np.ndarray) else None

                # Compute upsampled attention
                rollout_sample = rollout_np[i]    # [tokens, tokens]
                cls_attn = rollout_sample[0, 1:]  # skip CLS token in col 0
                grid_size = int(math.sqrt(cls_attn.shape[0]))
                attn_map = cls_attn.reshape(grid_size, grid_size)

                H, W = img_hwc.shape[0], img_hwc.shape[1]
                attn_map_t = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0).float()  # [1,1,g,g]
                attn_map_up = F.interpolate(attn_map_t, size=(H, W), mode='bilinear', align_corners=False)
                attn_map_up_np = attn_map_up.squeeze().cpu().numpy()
                # normalize for visualization
                attn_map_up_np = normalize_image(attn_map_up_np, per_channel=False)

                # ---------- Plotting ----------
                fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12,6))

                # Left panel: just the original
                ax_left.set_title(title_left, fontsize=10)
                ax_left.axis("off")
                ax_left.imshow(base_rgb)  # base image
                if gray_for_overlay is not None:
                    ax_left.imshow(gray_for_overlay, cmap='gray', alpha=0.3)

                # Right panel: original + attention
                ax_right.set_title("With Attention Overlay", fontsize=10)
                ax_right.axis("off")
                ax_right.imshow(base_for_overlay)
                if gray_for_overlay is not None:
                    ax_right.imshow(gray_for_overlay, cmap='gray', alpha=0.3)
                ax_right.imshow(attn_map_up_np, cmap='jet', alpha=overlay_alpha)

                # Title
                pred_label = int(predicted_np[i])
                fig.suptitle(
                    f"Batch {batch_idx}, Sample {i} | Predicted: {pred_label}",
                    fontsize=11
                )

                # Save
                outname = f"overlay_side_by_side_batch{batch_idx}_sample{i}.png"
                outpath = os.path.join(output_path, outname)
                plt.savefig(outpath, bbox_inches="tight", dpi=100)
                plt.close(fig)