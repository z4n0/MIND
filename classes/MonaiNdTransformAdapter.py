import torch
import torch.nn as nn
from monai.transforms.compose import Compose

class MonaiNdTransformAdapter(nn.Module):
    """Adapts MONAI non-dictionary transforms to work with tensor-based interfaces like BYOL code.
    it expects a MONAI Compose transform that applies to each image tensor NOT DICT in a batch.
    it takes (B, C, H, W) tensor as input and applies the MONAI transforms to each image in the batch.
    returns a tensor of the same shape (B, C, H, W) after applying the transforms.
    This is useful for integrating MONAI transforms into PyTorch-based training loops or pipelines.
    Args:
        monai_compose_transform (Compose): A MONAI non dict Compose transform that applies to each image tensor.
    returns:
        torch.Tensor: A tensor of the same shape (B, C, H, W) after applying the transforms.
    """
    def __init__(self, monai_compose_transform: Compose):
        super().__init__()
        self.transform = monai_compose_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be a batch of images (B, C, H, W)
        # MONAI array transforms expect (C, H, W)
        # Apply transforms to each image in the batch
        if x.dim() != 4:
            raise ValueError("Input tensor must be of shape (B, C, H, W)")
        if x.shape[0] == 0 or x.shape[0] > 64:
            raise ValueError("Batch size must be between 1 and 64, inclusive. SOMETHING IS OFF WITH THE BATCH SHAPE")
        augmented_batch = [self.transform(img_tensor) for img_tensor in x]
        augmented_batch = [torch.as_tensor(img) if not isinstance(img, torch.Tensor) else img for img in augmented_batch]
        return torch.stack(augmented_batch, dim=0)