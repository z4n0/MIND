import torch
import torch.nn as nn
from monai.transforms.compose import Compose

class MonaiDictTransformAdapter(nn.Module):
    """
    A PyTorch module that wraps MONAI dictionary-based Compose transforms for batch processing.
    This class allows you to apply MONAI's dictionary-style transforms (which expect inputs as dictionaries,
    e.g., {"image": tensor}) to each image in a batch of tensors. It is useful for integrating MONAI
    augmentations into PyTorch training pipelines.
    Args:
        monai_dict_compose_transform (Compose): A MONAI Compose object containing dictionary-based transforms.
        image_key (str, optional): The key used in the input/output dictionaries for the image tensor. Defaults to "image".
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Applies the MONAI dictionary-based transform to each image in the input batch.
            Args:
                x (torch.Tensor): A batch of image tensors with shape (B, C, H, W).
            Returns:
                torch.Tensor: A batch of augmented image tensors with the same shape as input.
    """
    def __init__(self, monai_dict_compose_transform: Compose, image_key: str = "image"):
        super().__init__()
        self.transform = monai_dict_compose_transform
        self.image_key = image_key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be a batch of image tensors (B, C, H, W)
        # MONAI dictionary transforms expect a dict like {"image": tensor} for each item
        
        augmented_batch = []
        for img_tensor in x:  # Iterate over images in the batch
            # Wrap the single image tensor in a dictionary
            input_dict = {self.image_key: img_tensor}
            # Apply the MONAI dictionary-based transform
            output_dict = self.transform(input_dict)
            # Extract the augmented image tensor from the output dictionary
            augmented_batch.append(output_dict[self.image_key])
            
        # Stack the individually augmented image tensors back into a batch
        return torch.stack(augmented_batch, dim=0)