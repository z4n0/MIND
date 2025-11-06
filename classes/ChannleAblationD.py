from monai.transforms import MapTransform
from typing import Sequence, Union
import torch


class ChannelAblationD(MapTransform):
    """
    Zero-out (ablate) specific channels in a multi-channel confocal microscopy image.
    
    This transform is designed for systematic ablation studies to quantify the
    individual contribution of fluorescence channels (G, B, Gr, R) to PD vs. MSA
    classification performance, following methodology established in multi-modal
    medical imaging literature [Sudre et al., MedIA 2017; Kamnitsas et al., 
    MedIA 2017].
    
    Scientific Rationale:
        Zeroing channels (rather than removing them) preserves model architecture
        invariance across ablation conditions, isolating the information
        contribution of each biomarker while maintaining constant network capacity.
        This mimics real-world imaging failures (photobleaching, failed antibody
        staining) better than physical channel removal.
    
    Args:
        keys (Sequence[str]): Keys of dictionary items to transform (e.g., ["image"]).
        channels_to_ablate (Union[int, Sequence[int]]): 
            Single channel index or list of channel indices to zero out (0-indexed).
            For G-B-Gr-R → R-G-B-Gr conversion (after `from_GBR_to_RGB`):
                - 0 = Red (cytoplasm autofluorescence)
                - 1 = Green (insulin, β-cell marker)
                - 2 = Blue (DAPI, nuclear stain)
                - 3 = Far-red (granuphilin, α-synuclein co-localization)
        allow_missing_keys (bool): If True, silently skip missing keys. Default: False.
        
    Raises:
        TypeError: If image is not a torch.Tensor.
        ValueError: If tensor is not 3D (C,H,W) or channel indices are out of bounds.
        
    Examples:
        >>> # Single-channel ablation (granuphilin only)
        >>> transform = ChannelAblationD(
        ...     keys=["image"],
        ...     channels_to_ablate=3  # Integer input
        ... )
        >>> data = {"image": torch.randn(4, 256, 256)}
        >>> output = transform(data)
        >>> assert output["image"][3, :, :].sum() == 0  # Channel 3 zeroed
        
        >>> # Multi-channel ablation (green + blue)
        >>> transform = ChannelAblationD(
        ...     keys=["image"],
        ...     channels_to_ablate=[1, 2]  # List input
        ... )
        >>> output = transform(data)
        >>> assert output["image"][1:3, :, :].sum() == 0  # Channels 1,2 zeroed
        
    Transform Pipeline Placement:
        1. LoadImaged (custom TIFF reader)
        2. Resized (e.g., to 224×224)
        3. ScaleIntensityd (to [0,1] range)
        4. LambdaD: from_GBR_to_RGB (G-B-Gr-R → R-G-B-Gr channel reordering)
        5. **ChannelAblationD** ← Insert here (before augmentation)
        6. RandFlipd, RandRotate90d (spatial augmentations)
        7. RandGaussianNoised, RandHistogramShiftd (intensity augmentations)
        8. NormalizeIntensityd (fold-specific mean/std)
        9. EnsureTyped
        
    Note:
        - Must be applied **after** channel reordering but **before** normalization
          to ensure ablated channels contribute zero to mean/std calculations,
          preserving biological plausibility of remaining channels.
        - Ablation applies to **both** training and validation sets (training-time
          ablation) to test channel necessity, not inference-time dependence.
        - For architectural ablation (physically removing channels), use
          `ChannelRemovalD` with corresponding model input layer adaptation.
          
    References:
        - Sudre et al., "Generalised Dice overlap as a deep learning loss function
          for highly unbalanced segmentations", Medical Image Analysis 2017.
        - Kamnitsas et al., "Efficient multi-scale 3D CNN with fully connected CRF
          for accurate brain lesion segmentation", Medical Image Analysis 2017.
    """
    
    def __init__(
        self,
        keys: Sequence[str],
        channels_to_ablate: Union[int, Sequence[int]],
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Initialize channel ablation transform with index validation.
        
        Args:
            keys: Dictionary keys to apply ablation to (typically ["image"]).
            channels_to_ablate: Single int or list of ints specifying channels
                               to zero out (0-indexed after RGB conversion).
            allow_missing_keys: Whether to skip missing keys silently.
        """
        super().__init__(keys, allow_missing_keys)
        
        # Normalize input to list for uniform processing
        if isinstance(channels_to_ablate, int):
            if channels_to_ablate < 0:
                raise ValueError(
                    f"Channel index must be non-negative, got {channels_to_ablate}"
                )
            self.channels_to_ablate = [channels_to_ablate]
        else:
            # Validate list inputs
            channels_list = list(channels_to_ablate)
            if not channels_list:
                raise ValueError(
                    "Must specify at least one channel to ablate. "
                    "For no ablation, omit this transform."
                )
            if any(ch < 0 for ch in channels_list):
                raise ValueError(
                    f"All channel indices must be non-negative, got {channels_list}"
                )
            self.channels_to_ablate = sorted(channels_list)
        
    def __call__(self, data: dict) -> dict:
        """
        Apply channel ablation by zeroing specified channels.
        
        This operation is performed in-place on a cloned tensor to avoid
        modifying the original data dictionary, ensuring thread safety for
        multi-worker data loading.
        
        Args:
            data: Dictionary containing image tensor(s) and metadata.
            
        Returns:
            Dictionary with ablated image tensor(s) (zeroed channels).
            
        Raises:
            TypeError: If image under specified key is not torch.Tensor.
            ValueError: If tensor shape is not (C,H,W) or channel indices
                       exceed the number of available channels.
        """
        d = dict(data)
        
        for key in self.key_iterator(d):
            img = d[key]
            
            # Type validation
            if not isinstance(img, torch.Tensor):
                raise TypeError(
                    f"Expected torch.Tensor for key '{key}', "
                    f"got {type(img).__name__}. "
                    f"Ensure EnsureTyped is not applied before this transform."
                )
            
            # Shape validation
            if img.ndim != 3:
                raise ValueError(
                    f"Expected 3D tensor with shape (C,H,W) for key '{key}', "
                    f"got {img.ndim}D tensor with shape {img.shape}. "
                    f"For batch processing, apply this transform before batching."
                )
            
            num_channels = img.shape[0]
            
            # Channel index bounds checking
            invalid_channels = [
                ch for ch in self.channels_to_ablate if ch >= num_channels
            ]
            if invalid_channels:
                raise ValueError(
                    f"Channel indices {invalid_channels} out of bounds for "
                    f"{num_channels}-channel image (key: '{key}'). "
                    f"Valid range: [0, {num_channels-1}]. "
                    f"Current mapping: 0=R, 1=G, 2=B, 3=Gr (after RGB conversion)."
                )
            
            # Perform ablation via zeroing (clone to avoid in-place modification)
            img_ablated = img.clone()
            for channel_idx in self.channels_to_ablate:
                img_ablated[channel_idx, :, :] = 0.0
            
            d[key] = img_ablated
            
        return d
    
    def __repr__(self) -> str:
        """
        Return a string representation for debugging and logging.
        
        Returns:
            Human-readable string describing the transform configuration.
        """
        channels_str = (
            f"[{self.channels_to_ablate[0]}]" 
            if len(self.channels_to_ablate) == 1 
            else str(self.channels_to_ablate)
        )
        return (
            f"{self.__class__.__name__}("
            f"keys={self.keys}, "
            f"channels_to_ablate={channels_str}, "
            f"allow_missing_keys={self.allow_missing_keys})"
        )