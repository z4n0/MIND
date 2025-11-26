from monai.transforms import MapTransform
from typing import Sequence, Union
import torch


class ChannelAblationD(MapTransform):
    """
    Zero-out specific channels in a multi-channel confocal microscopy image.

    This transform is used to study the importance of individual fluorescence channels 
    (G, B, Gr, R) in classifying PD vs. MSA. Instead of removing channels, it zeros them 
    out to keep the model architecture unchanged.

    Args:
        keys (Sequence[str]): Keys of dictionary items to transform (e.g., ["image"]).
        channels_to_ablate (Union[int, Sequence[int]]): 
            Index or list of indices of channels to zero out (0-indexed).
            Channel mapping after RGB conversion:
                - 0 = Red 
                - 1 = Green 
                - 2 = Blue 
                - 3 = Gray
        allow_missing_keys (bool): If True, skip missing keys without error. Default: False.

    Raises:
        TypeError: If the image is not a torch.Tensor.
        ValueError: If the tensor is not 3D (C, H, W) or channel indices are invalid.

    Notes:
        - Apply this transform after channel reordering but before normalization.
        - Ablation is applied during both training and validation to test channel importance.
    """

    def __init__(
        self,
        keys: Sequence[str],
        channels_to_ablate: Union[int, Sequence[int]],
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Initialize the transform and validate channel indices.

        Args:
            keys: Dictionary keys to apply ablation to (e.g., ["image"]).
            channels_to_ablate: Index or list of indices of channels to zero out.
            allow_missing_keys: If True, skip missing keys without error.
        """
        super().__init__(keys, allow_missing_keys)

        # Ensure channels_to_ablate is a list
        if isinstance(channels_to_ablate, int):
            if channels_to_ablate < 0:
                raise ValueError(f"Channel index must be non-negative, got {channels_to_ablate}")
            self.channels_to_ablate = [channels_to_ablate]
        else:
            channels_list = list(channels_to_ablate)
            if not channels_list:
                raise ValueError("At least one channel must be specified for ablation.")
            if any(ch < 0 for ch in channels_list):
                raise ValueError(f"All channel indices must be non-negative, got {channels_list}")
            self.channels_to_ablate = sorted(channels_list)

    def __call__(self, data: dict) -> dict:
        """
        Zero out the specified channels in the image tensor.

        Args:
            data: Dictionary containing the image tensor(s) and metadata.

        Returns:
            Dictionary with the specified channels zeroed out.

        Raises:
            TypeError: If the image is not a torch.Tensor.
            ValueError: If the tensor shape is not (C, H, W) or channel indices are invalid.
        """
        d = dict(data)

        for key in self.key_iterator(d):
            img = d[key]

            # Ensure the image is a torch.Tensor
            if not isinstance(img, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for key '{key}', got {type(img).__name__}.")

            # Ensure the image has 3 dimensions (C, H, W)
            if img.ndim != 3:
                raise ValueError(f"Expected 3D tensor (C, H, W) for key '{key}', got shape {img.shape}.")

            num_channels = img.shape[0]

            # Check if channel indices are valid
            invalid_channels = [ch for ch in self.channels_to_ablate if ch >= num_channels]
            if invalid_channels:
                raise ValueError(
                    f"Invalid channel indices {invalid_channels} for {num_channels}-channel image. "
                    f"Valid range: [0, {num_channels - 1}]."
                )

            # Zero out the specified channels
            img_ablated = img.clone()
            for channel_idx in self.channels_to_ablate:
                img_ablated[channel_idx, :, :] = 0.0

            d[key] = img_ablated

        return d