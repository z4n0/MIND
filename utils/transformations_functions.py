# import numpy as np
from monai.transforms import (
    Compose,
    # LoadImaged,
    # EnsureChannelFirstd,
    Resized,
    ScaleIntensityd,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    # RandRotated,
    RandAdjustContrastd,
    RandBiasFieldd,  # Simulates a smooth, spatially varying intensity bias field (common in microscopy due to uneven illumination). Highly recommended.
    EnsureTyped,
    RandSpatialCropd,
    LambdaD,
    OneOf,
    RandHistogramShiftd,
    # DataStatsd,  # Computes statistics (min, max, mean std) of the image intensities.
    # AddCoordinateChannelsd, 
    # RandCoarseDropoutd,
    # CenterSpatialCropd,
    # RandCropd,
    # Rand2DElasticd,
    # RandGridPatchd,
    # ScaleIntensityRangePercentilesd,
    # ClipIntensityPercentilesd,  # Clip intensities at specified low and high percentiles to remove extreme outlier pixels.
    # HistogramNormalized  # Standardizes image histograms, can be useful if illumination/staining varies significantly.
)

from monai.transforms.transform import MapTransform, Transform  # ← Add Transform here
from classes.ChannleAblationD import ChannelAblationD
import random
import torch
from typing import List, Tuple
from monai.transforms import OneOf
# from monai.transforms.intensity.dictionary import (
#     RandHistogramShiftd, RandAdjustContrastd, RandGaussianNoised,
#     RandBiasFieldd, RandCoarseDropoutd
# )
from monai.transforms.intensity.dictionary import RandCoarseDropoutd
from monai.transforms.utility.dictionary import LambdaD, Identityd
from configs.ConfigLoader import ConfigLoader
from monai.transforms.compose import Compose
from monai.transforms.spatial.dictionary import Resized, RandFlipd, RandRotate90d, Rand2DElasticd
# from monai.transforms.intensity.dictionary import ScaleIntensityd, NormalizeIntensityd, RandGaussianNoised, RandHistogramShiftd, RandAdjustContrastd
from monai.transforms.utility.dictionary import LambdaD,EnsureTyped
# from timm import is_model_pretrained
from classes.PrintShapeTransform import PrintShapeTransform
from monai.data.dataset import Dataset
from monai.data.dataloader import DataLoader
from torchvision.models import DenseNet121_Weights, DenseNet169_Weights, DenseNet201_Weights, ResNet50_Weights, ResNet18_Weights, ViT_B_16_Weights
from classes.CustomTiffFileReader import CustomTiffFileReader
from typing import List, Tuple
from monai.transforms.transform import MapTransform
import random 
# from monai.utils.misc import set_determinism
from utils.reproducibility_functions import set_global_seed
set_global_seed(42)

IMAGENET_WEIGHTS = {
    "densenet121": DenseNet121_Weights.DEFAULT,
    "densenet169": DenseNet169_Weights.DEFAULT,
    "densenet201": DenseNet201_Weights.DEFAULT,
    "resnet50": ResNet50_Weights.DEFAULT ,
    "resnet18": ResNet18_Weights.DEFAULT,
    "vit": ViT_B_16_Weights.DEFAULT, 
}

IMAGENET_NORMALIZATION_PARAMS_RGB = {
    "subtrahend": [0.485, 0.456, 0.406],
    "divisor": [0.229, 0.224, 0.225],
}

def from_GBR_to_RGB(image: torch.Tensor) -> torch.Tensor:
    """
    Convert an image or batch of images from G-B-R channel order to R-G-B.
    if the image is 4 channels, it will pass through unchanged, no permutation is done.
    
    Args:
        image (torch.Tensor): 
            - 3D tensor of shape (C, H, W), or
            - 4D tensor of shape (B, C, H, W).
    Returns:
        torch.Tensor: same shape as input, but with channels permuted from G-B-R to R-G-B.
    
    Raises:
        TypeError: if input is not a torch.Tensor.
        ValueError: if tensor is not 3D or 4D
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image)}")

    # print(f"Image shape: {image.shape}") (C,H,W)
    ndim = image.ndim
    single_image = (ndim == 3)
    if single_image: # Single image: (C, H, W)
        c, h, w = image.shape
        if c == 3:
            # Permute 3-channel GBR to RGB
            return image[[2, 0, 1], :, :]
        elif c == 4:
            # Input order is assumed G, B, Gr, R (GBGrR)
            # Convert to R, G, B, Gr as expected downstream
            return image[[3, 0, 1, 2], :, :]
        else:
            raise ValueError(f"Expected 3 or 4 channels in dim 0, got {c}")
    else:
        raise ValueError(f"Unsupported tensor shape {image.shape}; "
                         "expected 3D (C,H,W)")
        

def get_preNormalization_transforms_list(cfg, is_supported_by_torchvision=False)->List[MapTransform]:
    """
        Get the list of transforms to be applied before normalization.
        Only Loading, Resizing, Scaling transformations have to be here
        ie all transformations that can also be applied to the validation set
        
        if is_supported_by_torchvision is True, then the resizing and scaling is handled by torchvision
    """
    is_model_pretrained = cfg.training["transfer_learning"]
    base_transforms_list = [
        CustomTiffFileReader(keys=["image"]), # loads the image from the path as a numpy array (C,H,W) in GBR format
        EnsureTyped(keys=["image"], data_type="tensor", dtype=torch.float32), # Ensure image is a tensor
        EnsureTyped(keys=["label"], data_type="tensor", dtype=torch.int64), # Ensure label is a tensor
        LambdaD(keys="image", func=from_GBR_to_RGB), # Ensure the tensor is in RGB format (since pre-trained models expect RGB)
        # robust per-image rescale to [0,1] using percentiles (handles exposure/bleaching)
        #     ScaleIntensityRangePercentilesd(
        #         keys=["image"], lower=1.0, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        #     ),
        # # fold-wise (compute on train split only!) per-channel z-score
        # NormalizeIntensityd(keys=["image"], channel_wise=True),
        # base_transforms_list.append(LambdaD(keys="image", func=lambda x: x / 255.0))
        # ClipIntensityPercentilesd(keys="image", lower=0.5, upper=99.5),
        # ScaleIntensityd(keys="image", minv=0.0, maxv=1.0, channel_wise=True)
        # LambdaD(keys="image", func=lambda x: x / 255.0),
          # scales intensities to [0,1] before computing stats since most color transforms expect [0,1] range
        ]
    
    if not is_supported_by_torchvision or not is_model_pretrained: 
        #if using a custom model or a pretrained model not supported by torchvision
        # Resize the image to the desired spatial size
        base_transforms_list.append(
            Resized(keys="image", spatial_size=cfg.get_spatial_size(),
                    mode='bilinear', size_mode='all') # bilinear interpolation during resizing ie for each pixel is taken the average of the 4 nearest pixels
        )
        # ScaleIntensityRangePercentilesd(
        #         keys=["image"], lower=1.0, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        #     ),
        #     NormalizeIntensityd(keys=["image"], channel_wise=True),
        base_transforms_list.append(ScaleIntensityd(keys="image"))  # scales intensities to [0,1] before computing stats since most color transforms expect [0,1] range
        # base_transforms_list.append(LambdaD(keys="image", func=lambda x: x / 255.0)) 
    else: # If using a pretrained model the resizingm, scaling and cropping is handled by torchvision
        print(f"Using pretrained model: {is_model_pretrained} and supported by torchvision: {bool(is_supported_by_torchvision)} hence the resizing and scaling is handled by torchvision.Weights.Transforms")

    return base_transforms_list

def get_preNormalization_transforms_list_pretrained_tv(cfg:ConfigLoader)->List[MapTransform]:
    """
        Get the list of transforms to be applied before normalization.
        Only Loading, Resizing, Scaling transformations have to be here
        ie all transformations that can also be applied to the validation set
        
        if is_supported_by_torchvision is True, then the resizing and scaling is handled by torchvision
    """
    
    base_transforms_list = [
        CustomTiffFileReader(keys=["image"]), # loads the image from the path as a numpy array (C,H,W) in GBR format
        EnsureTyped(keys=["image"], data_type="tensor", dtype=torch.float32), # Ensure image is a tensor
        EnsureTyped(keys=["label"], data_type="tensor", dtype=torch.int64), # Ensure label is a tensor
        LambdaD(keys="image", func=from_GBR_to_RGB), # Ensure the tensor is in RGB format (since pre-trained models expect RGB)
        Resized(keys="image", spatial_size=cfg.get_spatial_size(),
                        mode='bilinear', size_mode='all'), # bilinear interpolation during resizing ie for each pixel is taken the average of the 4 nearest pixels
        # IMPORTANT for pretrained models: scale by a fixed factor (1/255)
        # to match torchvision semantics instead of per-image min/max scaling.
        # ClipIntensityPercentilesd(keys="image", lower=0.5, upper=99.5),
        # ScaleIntensityd(keys="image", minv=0.0, maxv=1.0, channel_wise=True),
        ScaleIntensityd(keys="image"),  # perform min max scaling scales intensities to [0,1] before computing stats since most color transforms expect [0,1] range
        # LambdaD(keys="image", func=lambda x: x / 255.0),
    ]

    return base_transforms_list


def _get_spatial_augmentations(cfg):
    """
    Get the list of spatial augmentation transforms.
    """
    H = cfg.data_augmentation["resize_spatial_size"][0]
    W = cfg.data_augmentation["resize_spatial_size"][1]
    crop_size = (int(0.95*H), int(0.95*W))
    spatial_transforms = [
            RandFlipd(
                keys="image", 
                prob=cfg.data_augmentation["rand_flip_prob"], 
                spatial_axis=cfg.data_augmentation["rand_flip_spatial_axes"][0]
            ),
            RandFlipd(
                keys="image", 
                prob=cfg.data_augmentation["rand_flip_prob"], 
                spatial_axis=cfg.data_augmentation["rand_flip_spatial_axes"][1]
            ),
            RandRotate90d(
                keys="image", 
                prob=cfg.data_augmentation["rand_rotate90_prob"], 
                max_k=cfg.data_augmentation["rand_rotate90_max_k"]
            ),
            # RandSpatialCropd(
            #     keys="image",
            #     roi_size=crop_size,
            #     random_center=True,
            # ),
            # Rand2DElasticd(
            #     keys=["image"],
            #     spacing=cfg.data_augmentation.get("rand_2d_elastic_spacing", (30, 30)), # Spacing for the grid control points
            #     magnitude_range=cfg.data_augmentation.get("rand_2d_elastic_magnitude", (0.2, 1.5)), # Magnitude of the elastic deformation 
            #     prob=0.20,
            #     rotate_range=cfg.data_augmentation.get("rand_2d_elastic_rotate_range", (np.pi / 14)), # rotation range between -pi/12 and pi/12
            #     scale_range=cfg.data_augmentation.get("rand_2d_elastic_scale_range", (0.93, 1.1)), #zoom range between 93% and 110%
            #     mode='bilinear',
            #     padding_mode='border',
            # ),
            # RandCoarseDropoutd(
            #     keys=["image"],
            #     holes=1,
            #     spatial_size=(int(0.1*H), int(0.1*W)),  # bind H,W from cfg or use relative
            #     max_holes=3,
            #     fill_value=0.0,
            #     prob=0.10
            # ),
        ]
    return spatial_transforms

# ---------- helpers ----------

def _poisson_gaussian_lambda(
    keys: Tuple[str, ...] = ("image",),
    gain_range: Tuple[float, float] = (30.0, 80.0),
    sigma_range: Tuple[float, float] = (0.005, 0.015)
) -> LambdaD:
    """
    Apply Poisson–Gaussian noise model to simulate realistic confocal microscopy noise.
    
    Scientific Justification:
    Confocal microscopy exhibits mixed noise characteristics:
    - **Poisson component**: photon shot noise (signal-dependent)
    - **Gaussian component**: readout/thermal noise (signal-independent)
    
    Model: y = Poisson(gain × x) / gain + N(0, σ²)
    where x ∈ [0,1] is the scaled input intensity.
    
    Args:
        keys (Tuple[str, ...]): Dictionary keys to apply the transform to.
            Default: ("image",)
        gain_range (Tuple[float, float]): Range [g_min, g_max] for Poisson gain factor.
            Higher gain → lower shot noise (typical confocal: 30–80).
            Default: (30.0, 80.0)
        sigma_range (Tuple[float, float]): Range [σ_min, σ_max] for Gaussian noise std.
            Typical values for 8-bit scaled images: 0.005–0.015.
            Default: (0.005, 0.015)
    
    Returns:
        LambdaD: MONAI dictionary transform that applies the noise model.
    
    Example:
        >>> transform = _poisson_gaussian_lambda(
        ...     keys=("image",), gain_range=(40.0, 70.0), sigma_range=(0.01, 0.02)
        ... )
        >>> noisy_dict = transform({"image": clean_tensor})
    """
    def _fn(x: torch.Tensor) -> torch.Tensor:
        """
        Inner function that applies the noise model.
        
        Args:
            x (torch.Tensor): Input tensor with values in [0, 1].
                Shape: (C, H, W) or (C, D, H, W) for 3D images.
        
        Returns:
            torch.Tensor: Noisy tensor, clamped to [0, 1].
        """
        # Ensure input is in valid range [0, 1]
        x = x.float().clamp(0, 1)
        # Sample random gain and sigma for this augmentation instance
        g = random.uniform(*gain_range)
        s = random.uniform(*sigma_range)
        # Apply Poisson noise: scale up, sample, scale back
        # torch.poisson(λ) samples from Poisson(λ), so we use λ = g*x
        y = torch.poisson(x * g) / g
        # Add Gaussian readout noise
        y = y + s * torch.randn_like(y)
        # Clamp to valid intensity range
        return y.clamp(0, 1)
    
    return LambdaD(keys=keys, func=_fn)


def _weak_bleedthrough_lambda(
    keys: Tuple[str, ...] = ("image",),
    max_frac: float = 0.05
):
    """
    Simulate weak spectral bleed-through (channel cross-talk) in multi-channel imaging.
    
    Scientific Justification:
    In fluorescence/confocal microscopy, spectral overlap between emission filters
    causes weak signal leakage between channels (typically <5% per channel pair).
    This is modeled via a linear mixing matrix M:
    
        y = M @ x
    
    where M[i,j] represents the fraction of channel j's signal appearing in channel i.
    We construct M as:
        - M[i,i] = 1.0 (diagonal, main signal preserved)
        - M[i,j] = random([0, max_frac]) for i ≠ j (off-diagonal, cross-talk)
    
    This augmentation improves model robustness to acquisition variability and is
    standard practice in multi-channel biomedical imaging
    
    Args:
        keys (Tuple[str, ...]): Dictionary keys to apply the transform to.
            Default: ("image",)
        max_frac (float): Maximum bleed-through fraction for off-diagonal terms.
            Typical values: 0.03–0.07 for confocal (3–7% cross-talk).
            Default: 0.05
    
    Returns:
        LambdaD: MONAI dictionary transform that applies spectral mixing.
        
    Notes:
        - Input images must have ≥3 channels; 1-2 channel images are returned unchanged.
    
    Example:
        >>> transform = _weak_bleedthrough_lambda(keys=("image",), max_frac=0.07)
        >>> mixed_dict = transform({"image": rgb_tensor})  # Shape: (C, H, W)
    """
    def _fn(x: torch.Tensor) -> torch.Tensor:
        """
        Inner function that applies spectral mixing.
        
        Args:
            x (torch.Tensor): Input tensor with shape (C, H, W) or (C, D, H, W).
                Must have C ≥ 3 for mixing to be applied.
        
        Returns:
            torch.Tensor: Mixed tensor with same shape as input.
        """
        x = x.float()
        
        # Only apply to images with 3+ channels
        if x.ndim != 3 or x.shape[0] < 3:
            raise ValueError("Input tensor must have at least 3 channels for bleed-through simulation.")
        
        # Number of channels to mix (up to 4: RGB + grayscale)
        C = min(4, x.shape[0])
        
        # Construct mixing matrix: diagonal=1.0, off-diagonal=random[0, max_frac]
        M = torch.eye(C, dtype=x.dtype, device=x.device)
        for i in range(C):
            for j in range(C):
                if i != j:
                    M[i, j] = random.uniform(0.0, max_frac)
        
        # Apply mixing: reshape to (C, N) for matrix multiplication
        # M @ x[:C] with shapes (C, C) @ (C, H*W) = (C, H*W)
        mixed = (M @ x[:C].reshape(C, -1)).reshape_as(x[:C])
        
        # Clamp mixed channels to [0, 1] and concatenate with unchanged channels
        x = torch.cat([mixed.clamp(0, 1), x[C:]], dim=0)
        
        return x
    
    return LambdaD(keys=keys, func=_fn)


def _monitor_range(x: torch.Tensor) -> torch.Tensor:
    """Optional: log % of out-of-range pixels before the final clamp."""
    below = (x < 0).float().mean().item() * 100.0
    above = (x > 1).float().mean().item() * 100.0
    if below > 0.1 or above > 0.1:
        print(f"[warn] out-of-range: <0={below:.2f}%, >1={above:.2f}%")
    return x


def _get_intensity_augmentations(cfg:ConfigLoader) -> List[Transform]:
    """
    Confocal-friendly photometric augs using OneOf(+Identityd).
    Returns a list of monai Transforms to be appended to your train pipeline.
    A single clamp to [0,1] is placed at the end of the block.
    
    Args:
        cfg: Configuration object with data augmentation containing the preset value
        it always uses the preset from cfg
        .

    """
    H, W = cfg.get_spatial_size()
    preset = cfg.get_intensity_augmentation_preset()

    if preset is None or preset is False or str(preset).lower() in {
        "none", "off", "no", "0"
    }:
        return []

    if preset == "light":
        intensity_one: Transform = OneOf(
            transforms=[
                Identityd(keys=["image"]),
                RandHistogramShiftd(keys=["image"], prob=1.0,
                                    num_control_points=(3, 5)),
                RandAdjustContrastd(keys=["image"], prob=1.0,
                                    gamma=(0.95, 1.05)),
                _poisson_gaussian_lambda(keys=("image",),
                                         gain_range=(30.0, 80.0),
                                         sigma_range=(0.005, 0.015)),
                RandBiasFieldd(keys=["image"], prob=1.0,
                               degree=2, coeff_range=(0.01, 0.03)),
                _weak_bleedthrough_lambda(keys=("image",), max_frac=0.05),
            ],
            weights=[0.40, 0.20, 0.18, 0.12, 0.06, 0.04],
        )
        extras: List[Transform] = [
            RandCoarseDropoutd(
                keys=["image"], prob=0.05,
                holes=1, max_holes=2,
                spatial_size=(int(0.08 * H), int(0.08 * W)),
                fill_value=0.0,
            )
        ]

    elif preset == "medium":
        intensity_one = OneOf(
            transforms=[
                Identityd(keys=["image"]),
                RandHistogramShiftd(keys=["image"], prob=1.0,
                                    num_control_points=(4, 6)),
                RandAdjustContrastd(keys=["image"], prob=1.0,
                                    gamma=(0.90, 1.10)),
                _poisson_gaussian_lambda(keys=("image",),
                                         gain_range=(20.0, 70.0),
                                         sigma_range=(0.008, 0.02)),
                RandBiasFieldd(keys=["image"], prob=1.0,
                               degree=2, coeff_range=(0.015, 0.04)),
                _weak_bleedthrough_lambda(keys=("image",), max_frac=0.07),
                RandGaussianNoised(keys=["image"], prob=1.0,
                                   mean=0.0, std=0.015),
            ],
            weights=[0.30, 0.20, 0.18, 0.14, 0.08, 0.06, 0.04],
        )
        extras = [
            RandCoarseDropoutd(
                keys=["image"], prob=0.07,
                holes=1, max_holes=2,
                spatial_size=(int(0.08 * H), int(0.08 * W)),
                fill_value=0.0,
            )
        ]

    elif preset == "heavy":
        intensity_one = OneOf(
            transforms=[
                Identityd(keys=["image"]),
                RandHistogramShiftd(keys=["image"], prob=1.0,
                                    num_control_points=(4, 7)),
                RandAdjustContrastd(keys=["image"], prob=1.0,
                                    gamma=(0.85, 1.15)),
                _poisson_gaussian_lambda(keys=("image",),
                                         gain_range=(10.0, 60.0),
                                         sigma_range=(0.01, 0.03)),
                RandBiasFieldd(keys=["image"], prob=1.0,
                               degree=3, coeff_range=(0.02, 0.06)),
                _weak_bleedthrough_lambda(keys=("image",), max_frac=0.10),
                RandGaussianNoised(keys=["image"], prob=1.0,
                                   mean=0.0, std=0.02),
            ],
            weights=[0.25, 0.20, 0.18, 0.15, 0.10, 0.07, 0.05],
        )
        extras = [
            RandCoarseDropoutd(
                keys=["image"], prob=0.10,
                holes=1, max_holes=3,
                spatial_size=(int(0.10 * H), int(0.10 * W)),
                fill_value=0.0,
            )
        ]
    else:
        raise ValueError(f"Unknown preset: {preset}")

    # Final block: OneOf → extras → clamp
    intensity_block: List[Transform] = [
        intensity_one,
        *extras,
        # LambdaD(keys=["image"], func=_monitor_range),
        LambdaD(keys=["image"], func=lambda x: x.clamp(0.0, 1.0)),
    ]
    return intensity_block


# Compute global normalization parameters from training set
# global_mean, global_std = compute_dataset_mean_std(train_images_paths, cfg)
def compute_dataset_mean_std(
    image_paths: list[str], 
    cfg, 
    is_supported_by_torchvision: bool = False
) -> dict:
    """
    Computes per-channel mean and standard deviation using VECTORIZED Welford's
    algorithm for numerical stability and performance.
    
    If cfg.get_use_ablation() is True, this function applies ChannelAblationD
    BEFORE computing statistics, ensuring normalization parameters reflect the
    actual ablated distribution
    
    Args:
        image_paths: List of file paths to training images
        cfg: ConfigLoader instance with ablation/model configuration
        is_supported_by_torchvision: Whether to use torchvision preprocessing
        
    Returns:
        dict: {"mean": [μ₀, μ₁, ...], "std": [σ₀, σ₁, ...]} per-channel stats
              For ablated channels, mean ≈ 0.0 and std ≈ 1e-6
              
    """
    from classes.ChannleAblationD import ChannelAblationD
    
    # Build minimal preprocessing pipeline (load → resize → scale)
    minimal_transforms_list = get_preNormalization_transforms_list(
        cfg, 
        is_supported_by_torchvision=is_supported_by_torchvision
    )
    
    # CRITICAL: Apply ablation BEFORE computing statistics if enabled
    if cfg.get_use_ablation():
        channels_to_ablate = cfg.get_channels_to_ablate()
        print(f"   [stats] Applying ablation to channels {channels_to_ablate} "
              f"before computing mean/std")
        minimal_transforms_list.append(
            ChannelAblationD(keys=["image"], channels_to_ablate=channels_to_ablate)
        )
    
    minimal_transforms_for_stats = Compose(minimal_transforms_list)
    dataset = Dataset(
        data=[{"image": img_path, "label": 0} for img_path in image_paths],
        transform=minimal_transforms_for_stats
    )
    
    # Configure data loading
    batch_size = cfg.data_loading.get("batch_size", 16)
    num_workers = cfg.data_loading.get("num_workers", 4)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Optimize CPU→GPU transfer
    )

    # Initialize Welford accumulators
    n_total = 0  # Total pixel count across all batches
    running_mean = None  # Per-channel running mean
    M2 = None  # Per-channel sum of squared deviations (Welford's M2)
    
    print(f"   [stats] Computing statistics over {len(image_paths)} images "
          f"(batch_size={batch_size})...")
    
    for batch_idx, batch in enumerate(loader):
        imgs = batch["image"]  # [B, C, H, W]
        
        # Ensure float normalization (range [0,1]) (mostly redundant since already done in the transformations passed to
        # the Dataset, but kept here for safety since it's applied only if the max become greater than 1)
        if imgs.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
            imgs = imgs.float() / 255.0 if imgs.max() > 2 else imgs.float()
        
        # Move to GPU if available (massive speedup for large batches)
        device = imgs.device
        
        # Reshape to [C, N] where N = B*H*W (all spatial pixels per channel)
        B, C, H, W = imgs.shape
        imgs_reshaped = imgs.permute(1, 0, 2, 3).reshape(C, -1)  # [C, B*H*W]
        n_batch = imgs_reshaped.shape[1]
        
        # Initialize accumulators on first batch
        if running_mean is None:
            running_mean = torch.zeros(C, dtype=torch.float64, device=device)
            M2 = torch.zeros(C, dtype=torch.float64, device=device)
        
        # Compute batch statistics
        batch_mean = imgs_reshaped.double().mean(dim=1)  # [C]
        batch_var = imgs_reshaped.double().var(dim=1, unbiased=False)  # [C]
        
        # Update global statistics using parallel Welford formula
        delta = batch_mean - running_mean
        n_total_new = n_total + n_batch
        
        # Update mean: μ_new = (n*μ_old + n_batch*μ_batch) / (n + n_batch)
        running_mean += delta * (n_batch / n_total_new)
        
        # Update M2 (sum of squared deviations)
        # M2_new = M2_old + M2_batch + δ² * n * n_batch / (n + n_batch)
        M2 += batch_var * n_batch + delta**2 * (n_total * n_batch / n_total_new)
        
        n_total = n_total_new
        
        # Progress logging every 5 batches
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
            processed_images = min((batch_idx + 1) * batch_size, len(image_paths))
            print(f"   [stats] Processed {batch_idx + 1}/{len(loader)} batches "
                  f"({processed_images}/{len(image_paths)} images, "
                  f"{n_total:,} pixels)")
    
    # === FINALIZE STATISTICS ===
    if n_total == 0 or running_mean is None or M2 is None:
        print("  Warning: No data processed. Returning default stats.")
        num_channels = cfg.get_model_input_channels()
        return {"mean": [0.5] * num_channels, "std": [0.5] * num_channels}
    
    # Compute final std using Bessel's correction (n-1 for sample std)
    final_mean = running_mean.cpu().tolist()
    final_std = torch.sqrt(M2 / (n_total - 1)).cpu().tolist()
    
    # Prevent division by zero for ablated channels (std ≈ 0)
    final_std = [max(s, 1e-6) for s in final_std]
    
    # === VALIDATION LOGGING ===
    if cfg.get_use_ablation():
        channels_to_ablate = cfg.get_channels_to_ablate()
        print(f"   [stats]  Computed statistics with ablation (n={n_total:,} pixels):")
        for ch_idx in range(len(final_mean)):
            status = "ABLATED" if ch_idx in channels_to_ablate else "active"
            print(f"      Channel {ch_idx} ({status}): "
                  f"mean={final_mean[ch_idx]:.6f}, std={final_std[ch_idx]:.6f}")
    else:
        print(f"   [stats]  Computed statistics (n={n_total:,} pixels):")
        for ch_idx in range(len(final_mean)):
            print(f"      Channel {ch_idx}: "
                  f"mean={final_mean[ch_idx]:.6f}, std={final_std[ch_idx]:.6f}")
    
    return {"mean": final_mean, "std": final_std}

def is_supported_by_torchvision(model_name: str) -> bool:
    """
    Check if the model is supported by torchvision ie if it is in the IMAGENET_WEIGHTS dictionary
    """
    return IMAGENET_WEIGHTS.get(model_name.lower()) is not None  # ← .lower() handles ALL cases

def get_transforms(cfg: ConfigLoader, fold_specific_stats: dict[str, float] | None = None):
    
    is_pretrained = cfg.get_transfer_learning() 
    supported_by_torchvision = is_supported_by_torchvision(cfg.get_model_name()) 
    print(f"Model {cfg.get_model_name()} supported by torchvision: {supported_by_torchvision} and pretrained: {is_pretrained}")
    
    # select the normalization params if you are using a pretrained model
    print(f"Using pretrained model: {is_pretrained}")
    
    color_transforms = cfg.get_intensity_augmentation_preset() not in ["none", "off", "no", "0"]
    
    # Initialize transform lists
    train_transforms_list = []
    val_transforms_list = []
    
    print(f"Using pretrained model: {is_pretrained} and supported by torchvision: {supported_by_torchvision} with color transforms: {color_transforms}")
    # train_transforms_list, val_transforms_list = get_custom_transforms_lists(cfg, color_transforms, fold_specific_stats)
    if not supported_by_torchvision or not is_pretrained:  
        print("the model is not supported by torchvision or is not pretrained")
        train_transforms_list, val_transforms_list = get_custom_transforms_lists(cfg, fold_specific_stats)
    elif supported_by_torchvision and is_pretrained: # If you are using a pretrained model(even if it's micronet), use the torchvision transforms
        print("the model is supported by torchvision and is pretrained")
        train_transforms_list, val_transforms_list = get_pretrained_model_transforms_list_without_crop(cfg)
    else: # Fallback or unsupported pretrained model
        raise ValueError(f"Model {cfg.get_model_name()} is marked pretrained but not supported by torchvision or has no specific transform pipeline defined. Falling back to basic transforms without specific normalization.")
    
    #set the random state for the transforms
    train_transforms = Compose(train_transforms_list).set_random_state(42)
    val_transforms = Compose(val_transforms_list).set_random_state(42)
    test_transforms = val_transforms
    
    # preview_input_shape = (3, *cfg.data_augmentation["resize_spatial_size"])
    # print(f"--- Previewing train_transforms output with input shape: {preview_input_shape} ---")
    # if fold_specific_stats:
    #     preview_transform_output(train_transforms, preview_input_shape, fold_specific_stats.get("mean"), fold_specific_stats.get("std"))
    # else:       
    #     preview_transform_output(train_transforms, preview_input_shape)
    # print("--- End of train_transforms preview ---")
    
    return train_transforms, val_transforms, test_transforms

def get_custom_transforms_lists(cfg: ConfigLoader, fold_specific_stats: dict, crop: bool = False):
    """
    Generate training and validation transform lists for custom models.
    
    Scientific rationale:
    - Ablation must be applied BEFORE normalization to both train and val
    - This ensures mean/std are computed on the actual ablated distribution
    - Consistent with ablation study best practices (Zeiler & Fergus, ECCV 2014)
    """
    print(f"Model {cfg.get_model_name()} not supported using custom transforms")
    supported_by_torchvision = is_supported_by_torchvision(cfg.get_model_name())
    
    # Pipeline di base che carica e ridimensiona
    train_transforms_list = get_preNormalization_transforms_list(cfg, supported_by_torchvision)
    # Aggiungi il cropping casuale per forzare il modello a ignorare i bordi.
    # Prima calcoliamo la dimensione del crop, ad esempio il 90% della dimensione dell'immagine.
    crop = bool(cfg.get_use_crop())

    if crop:
        original_size = cfg.get_crop_size()  # Es: (512, 512)
        crop_percentage = cfg.get_crop_percentage()
        
        # Ensure original_size and crop_percentage are valid
        if original_size is None or crop_percentage is None:
            raise ValueError("original_size and crop_percentage must not be None")
        if not isinstance(original_size, (tuple, list)) or len(original_size) != 2:
            raise ValueError(f"original_size must be a tuple/list of length 2, got: {original_size}")
        if not (0 < crop_percentage <= 1):
            raise ValueError(f"crop_percentage must be between 0 and 1, got: {crop_percentage}")

        crop_size = (int(original_size[0] * crop_percentage), int(original_size[1] * crop_percentage))

        print(f"Applying random crop with size: {crop_size}")
        train_transforms_list.append(
            RandSpatialCropd(keys=["image"], roi_size=crop_size, random_size=False)
        )
        
        
    # Ora applica le altre aumentazioni spaziali sull'immagine croppata
    # print("No spatial augmentations are applied currently")
    spatial_transforms = _get_spatial_augmentations(cfg)
    train_transforms_list.extend(spatial_transforms)

    # Aggiungi le aumentazioni di intensità saranno decise da come è il preset
    intensity_transforms = _get_intensity_augmentations(cfg)
    train_transforms_list.extend(intensity_transforms)
    
    # Optional ablation of channels - MUST be applied BEFORE normalization
    if cfg.get_use_ablation():
        channels_to_ablate = cfg.get_channels_to_ablate()
        print(f"  Applying channel ablation to channels: {channels_to_ablate}")
        train_transforms_list.append(
            ChannelAblationD(keys=["image"], channels_to_ablate=channels_to_ablate)
        )

    # apply normalization
    if fold_specific_stats:
        print(f"Applying fold-specific normalization with mean: {fold_specific_stats['mean']}, std: {fold_specific_stats['std']}")
        train_transforms_list.append(
            NormalizeIntensityd(
                keys=["image"],
                subtrahend=torch.tensor(fold_specific_stats["mean"]),
                divisor=torch.tensor(fold_specific_stats["std"]),
                channel_wise=True
            )
        )
    else: #do normal channel wise normalization with mean 0.5 and std 0.5
        print("NO FOLD SPECIFIC STATS provided, using default x channel normalization")
        train_transforms_list.append(
            NormalizeIntensityd(
                keys=["image"],
                channel_wise=True,
            )
        )
    
    # --- VALIDATION TRANSFORMS ---
    val_transforms_list = get_preNormalization_transforms_list(cfg, supported_by_torchvision)
    
    if cfg.get_use_ablation():
        channels_to_ablate = cfg.get_channels_to_ablate()
        print(f"  Applying channel ablation to validation set (channels: {channels_to_ablate})")
        val_transforms_list.append(
            ChannelAblationD(keys=["image"], channels_to_ablate=channels_to_ablate)
        )
    
    # Then apply normalization (same as train)
    if fold_specific_stats and fold_specific_stats.get("mean") is not None and fold_specific_stats.get("std") is not None:
        val_transforms_list.append(
            NormalizeIntensityd(
                keys=["image"],
                subtrahend=torch.tensor(fold_specific_stats["mean"]),
                divisor=torch.tensor(fold_specific_stats["std"]),
                channel_wise=True
            )
        )
    else:
        print("NO FOLD SPECIFIC STATS provided, using default x channel normalization")
        val_transforms_list.append(
            NormalizeIntensityd(
                keys=["image"],
                channel_wise=True,
            )
        )
        
    return train_transforms_list, val_transforms_list


def get_pretrained_model_transforms_list_without_crop(cfg:ConfigLoader)->Tuple[List[MapTransform], List[MapTransform]]:
    """
    emulates torchvision pretrained model transforms without center crop.
    Handles both 3-channel and 4-channel inputs.
    
    For 3-channel: applies standard ImageNet normalization
    For 4-channel: skips ImageNet normalization (uses scaling only)
                   The adapted first conv layer will handle the 4th channel
    """

    num_channels = cfg.get_model_input_channels()  # Get number of channels from config
    imagenet_weights = IMAGENET_WEIGHTS.get(cfg.get_model_name().lower(), None)
    
    # use_color_transforms = cfg.get_intensity_augmentation_preset() not in ["none", "off", "no", "0"]

    # Force pre-normalization to include resize and scale (no center crop)
    train_transforms_list = get_preNormalization_transforms_list_pretrained_tv(cfg)
    train_transforms_list.extend(_get_spatial_augmentations(cfg))
    # if use_color_transforms:
    train_transforms_list.extend(_get_intensity_augmentations(cfg))
        
    if cfg.get_use_ablation():
            channels_to_ablate = cfg.get_channels_to_ablate()
            print(f" Applying channel ablation to channels (pretrained path): {channels_to_ablate}")
            train_transforms_list.append(
                ChannelAblationD(keys=["image"], channels_to_ablate=channels_to_ablate)
            )
    
    # Apply normalization based on number of channels
    if num_channels == 3 and imagenet_weights is not None:
        # Standard 3-channel ImageNet normalization
        tv_t = imagenet_weights.transforms()
        mean = torch.tensor(tv_t.mean)  # [0.485, 0.456, 0.406]
        std  = torch.tensor(tv_t.std)   # [0.229, 0.224, 0.225]
        
        # Mathematical operation:
        # x_norm[c] = (x[c] - mean[c]) / std[c]  for c in {R, G, B}
        train_transforms_list.append(
            NormalizeIntensityd(
                keys=["image"], 
                subtrahend=mean,  # mean vector
                divisor=std,      # std vector
                channel_wise=True # apply per-channel
            )
        )
        print(f" Using ImageNet normalization for 3-channel input")
    elif num_channels == 4:
        # For 4-channel: skip ImageNet normalization
        # The channel adaptation layer will learn to handle the 4th channel
        # ScaleIntensityd was already applied in new_get_preNormalization_transforms_list
        print(f" Using 4-channel input: Skipping ImageNet normalization (using [0,1] scaling only) skipping normalization")
        # print(f" The adapted first conv layer will handle channel-specific normalization")
    else:
        # Fallback for other channel counts
        print(f" Using {num_channels} channels without ImageNet normalization")
        
    # -------------------VAL TRANSFORMS ----------------------------------------------------
    # Validation: resize-only + same normalization as training
    val_transforms_list = get_preNormalization_transforms_list_pretrained_tv(cfg)
    
    # For 4-channel or other cases, no additional normalization (scaling already done in pre-normalization)
    if cfg.get_use_ablation():
        channels_to_ablate = cfg.get_channels_to_ablate()
        # Apply the same ablation to validation for distribution consistency
        val_transforms_list.append(
            ChannelAblationD(keys=["image"], channels_to_ablate=channels_to_ablate)
        )
        
    # apply imagenet Normalization 
    if num_channels == 3 and imagenet_weights is not None:
        tv_t = imagenet_weights.transforms()
        mean = torch.tensor(tv_t.mean)  # [0.485, 0.456, 0.406]
        std  = torch.tensor(tv_t.std)   # [0.229, 0.224, 0.225]
        val_transforms_list.append(
            NormalizeIntensityd(
                keys=["image"], 
                subtrahend=mean,  # mean vector
                divisor=std,      # std vector
                channel_wise=True # apply per-channel
            )
        )
    
    return train_transforms_list, val_transforms_list


def get_convert_to_tensor_transform_list():
    return [
        EnsureTyped(keys=["image"], data_type="tensor", dtype=torch.float32),
        EnsureTyped(keys=["label"], data_type="tensor", dtype=torch.int64),
    ]
