import numpy as np
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
    RandHistogramShiftd,
    DataStatsd,  # Computes statistics (min, max, mean std) of the image intensities.
    AddCoordinateChannelsd,  # Adds channels representing spatial coordinates; can sometimes help CNNs learn spatial relationships.
    RandCoarseDropoutd,
    CenterSpatialCropd,
    # RandCropd,
    # Rand2DElasticd,
    # RandGridPatchd,
    ClipIntensityPercentilesd,  # Clip intensities at specified low and high percentiles to remove extreme outlier pixels.
    HistogramNormalized  # Standardizes image histograms, can be useful if illumination/staining varies significantly.
)

import random
import torch
from typing import List, Tuple
from monai.transforms import OneOf
from monai.transforms.intensity.dictionary import (
    RandHistogramShiftd, RandAdjustContrastd, RandGaussianNoised,
    RandBiasFieldd, RandCoarseDropoutd
)
from monai.transforms.utility.dictionary import LambdaD, Identityd
from configs.ConfigLoader import ConfigLoader
from monai.transforms.compose import Compose
from monai.transforms.spatial.dictionary import Resized, RandFlipd, RandRotate90d, Rand2DElasticd
from monai.transforms.intensity.dictionary import ScaleIntensityd, NormalizeIntensityd, RandGaussianNoised, RandHistogramShiftd, RandAdjustContrastd
from monai.transforms.utility.dictionary import LambdaD,EnsureTyped
# from timm import is_model_pretrained
from classes.PrintShapeTransform import PrintShapeTransform
from monai.data.dataset import Dataset
from monai.data.dataloader import DataLoader
from torchvision.models import DenseNet121_Weights, DenseNet169_Weights, DenseNet201_Weights, ResNet50_Weights, ResNet18_Weights
from classes.CustomTiffFileReader import CustomTiffFileReader
from typing import List, Tuple
from monai.transforms.transform import MapTransform
import random 
from monai.utils.misc import set_determinism
from utils.reproducibility_functions import set_global_seed
set_global_seed(42)

IMAGENET_WEIGHTS = {
    "densenet121": DenseNet121_Weights.DEFAULT,
    "densenet169": DenseNet169_Weights.DEFAULT,
    "densenet201": DenseNet201_Weights.DEFAULT,
    "resnet50": ResNet50_Weights.DEFAULT ,
    "resnet18": ResNet18_Weights.DEFAULT,
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
            Channels must be exactly 3.
    Returns:
        torch.Tensor: same shape as input, but with channels permuted from G-B-R to R-G-B.
    
    Raises:
        TypeError: if input is not a torch.Tensor.
        ValueError: if tensor is not 3D or 4D, or if C != 3.
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
            # Pass 4-channel image through
            return image
        else:
            raise ValueError(f"Expected 3 or 4 channels in dim 0, got {c}")

    batch_image = (ndim == 4)
    if batch_image: # Batch: (B, C, H, W)
        b, c, h, w = image.shape
        if b>64:
            raise ValueError(f"Batch size {b} is too large, expected <= 64 look at batch shape {image.shape}")
        if c == 3:
            # Permute 3-channel GBR to RGB for each image in batch
            return image[:, [2, 0, 1], :, :]
        elif c == 4:
            # Pass 4-channel batch through
            return image
        else:
            raise ValueError(f"Expected 3 or 4 channels in dim 1, got {c}")

    else:
        raise ValueError(f"Unsupported tensor shape {image.shape}; "
                         "expected 3D (C,H,W) or 4D (B,C,H,W)")
        
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
    ]
    
    if not is_supported_by_torchvision or not is_model_pretrained: #if using a custom model or a pretrained model not supported by torchvision
        # Resize the image to the desired spatial size
        base_transforms_list.append(
            Resized(keys="image", spatial_size=cfg.data_augmentation["resize_spatial_size"],
                    mode='bilinear', size_mode='all') # bilinear interpolation during resizing ie for each pixel is taken the average of the 4 nearest pixels
        )
        base_transforms_list.append(ScaleIntensityd(keys="image"))  # scales intensities to [0,1] before computing stats since most color transforms expect [0,1] range

    else: # If using a pretrained model the resizingm, scaling and cropping is handled by torchvision
        print(f"Using pretrained model: {is_model_pretrained} and supported by torchvision: {bool(is_supported_by_torchvision)} hence the resizing and scaling is handled by torchvision.Weights.Transforms")

    return base_transforms_list

def new_get_preNormalization_transforms_list(cfg:ConfigLoader)->List[MapTransform]:
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
        ScaleIntensityd(keys="image"),  # scales intensities to [0,1] before computing stats since most color transforms expect [0,1] range
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
    keys=("image",), gain_range=(30.0, 80.0), sigma_range=(0.005, 0.015)
):
    """
    Poisson–Gaussian noise: y = Poisson(gain * x)/gain + N(0, sigma), x in [0,1].
    """
    def _fn(x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(0, 1)
        g = random.uniform(*gain_range)
        s = random.uniform(*sigma_range)
        y = torch.poisson(x * g) / g
        y = y + s * torch.randn_like(y)
        return y.clamp(0, 1)
    return LambdaD(keys=keys, func=_fn)


def _weak_bleedthrough_lambda(keys=("image",), max_frac=0.05):
    """
    Very small channel mixing on RGB to mimic spectral cross-talk.
    """
    def _fn(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.ndim != 3 or x.shape[0] < 3:
            return x
        C = 3
        M = torch.eye(C, dtype=x.dtype, device=x.device)
        for i in range(C):
            for j in range(C):
                if i != j:
                    M[i, j] = random.uniform(0.0, max_frac)
        mixed = (M @ x[:C].reshape(C, -1)).reshape_as(x[:C])
        x = torch.cat([mixed.clamp(0, 1), x[C:]], dim=0)
        return x
    return LambdaD(keys=keys, func=_fn)


# ---------- main block ----------

def _get_intensity_augmentations(cfg, preset: str = "medium") -> List:
    """
    Confocal-friendly photometric augs using OneOf(+Identityd) so that
    *at most one* op is applied per image with a controllable overall prob.
    Presets: "light" (recommended), "medium" (if underfitting), "heavy" (ablate).
    """
    # image size for gentle dropout (optional)
    H, W = cfg.data_augmentation["resize_spatial_size"]
    preset = cfg.get_intensity_augmentation_preset()

    # Allow disabling intensity augs via preset
    if preset is None or preset is False or str(preset).lower() in ("none", "off", "no", "0"):
        return []

    if preset == "light":
        # Overall apply probability ≈ 0.6 via Identityd weight
        intensity_one = OneOf(
            transforms=[
                Identityd(keys=["image"]),  # "do nothing" branch
                RandHistogramShiftd(keys=["image"], prob=1.0, num_control_points=(3, 5)),
                RandAdjustContrastd(keys=["image"],  prob=1.0, gamma=(0.95, 1.05)),
                _poisson_gaussian_lambda(keys=("image",), gain_range=(30.0, 80.0),
                                         sigma_range=(0.005, 0.015)),
                RandBiasFieldd(keys=["image"], prob=1.0, degree=2, coeff_range=(0.01, 0.03)),
                _weak_bleedthrough_lambda(keys=("image",), max_frac=0.05),
            ],
            weights=[0.40, 0.20, 0.18, 0.12, 0.06, 0.04],   # sums arbitrary; acts as probabilities
        )
        extras = [
            RandCoarseDropoutd(  # rare, small occlusion; optional
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
                RandHistogramShiftd(keys=["image"], prob=1.0, num_control_points=(4, 6)),
                RandAdjustContrastd(keys=["image"],  prob=1.0, gamma=(0.90, 1.10)),
                _poisson_gaussian_lambda(keys=("image",), gain_range=(20.0, 70.0),
                                         sigma_range=(0.008, 0.02)),
                RandBiasFieldd(keys=["image"], prob=1.0, degree=2, coeff_range=(0.015, 0.04)),
                _weak_bleedthrough_lambda(keys=("image",), max_frac=0.07),
                RandGaussianNoised(keys="image", prob=1.0, mean=0.0, std=0.015),
            ],
            weights=[0.30, 0.20, 0.18, 0.14, 0.08, 0.06, 0.04],
        )
        extras = [
            RandCoarseDropoutd(keys=["image"], prob=0.07,
                               holes=1, max_holes=2,
                               spatial_size=(int(0.08 * H), int(0.08 * W)))
        ]

    elif preset == "heavy":  # for ablation/diagnostics only
        intensity_one = OneOf(
            transforms=[
                Identityd(keys=["image"]),
                RandHistogramShiftd(keys=["image"], prob=1.0, num_control_points=(4, 7)),
                RandAdjustContrastd(keys=["image"],  prob=1.0, gamma=(0.85, 1.15)),
                _poisson_gaussian_lambda(keys=("image",), gain_range=(10.0, 60.0),
                                         sigma_range=(0.01, 0.03)),
                RandBiasFieldd(keys=["image"], prob=1.0, degree=3, coeff_range=(0.02, 0.06)),
                _weak_bleedthrough_lambda(keys=("image",), max_frac=0.10),
                RandGaussianNoised(keys="image", prob=1.0, mean=0.0, std=0.02),
            ],
            weights=[0.25, 0.20, 0.18, 0.15, 0.10, 0.07, 0.05],
        )
        extras = [
            RandCoarseDropoutd(keys=["image"], prob=0.10,
                               holes=1, max_holes=3,
                               spatial_size=(int(0.1 * H), int(0.1 * W)))
        ]
    else:
        raise ValueError(f"Unknown preset: {preset}")

    return [intensity_one] + extras


# def _get_intensity_augmentations(cfg):
#     """
#     Get the list of intensity augmentation transforms.
#     """
#     intensity_transforms = [
#             RandHistogramShiftd(keys=["image"], prob=0.2, num_control_points=(3,5)), # num_control_points=(4, 7)
#             RandAdjustContrastd(keys=["image"], prob=0.25, gamma=(0.9, 1.1)),
#             RandGaussianNoised(
#                     keys="image", 
#                     prob=cfg.data_augmentation["rand_gaussian_noise_prob"],
#                     mean=cfg.data_augmentation["rand_gaussian_noise_mean"],
#                     std=cfg.data_augmentation["rand_gaussian_noise_std"]
#                 ),
            
#         ]
#     return intensity_transforms 

# Compute global normalization parameters from your training set
# global_mean, global_std = compute_dataset_mean_std(train_images_paths, cfg)
def compute_dataset_mean_std(image_paths: list[str], cfg, is_supported_by_torchvision=False):
    """
    Computes the mean and standard deviation for a given set of image paths.
    Applies minimal transforms to bring images to a state comparable to
    what they would be just before normalization in the main pipeline.
    
    #NOTE: this minimal transforms should be the same as the ones used in the main pipeline.
    #the images to be in the exact same state as they would be just before the NormalizeIntensityd transform 
    # is applied in your main training/validation pipeline. (disregarding transformations which do not change pixel intensities)
    """
    minimal_transforms_for_stats = Compose(get_preNormalization_transforms_list(cfg,is_supported_by_torchvision=is_supported_by_torchvision))
    dataset = Dataset(data=[{"image": img_path, "label": 0} for img_path in image_paths], transform=minimal_transforms_for_stats)
    
    # Use batch_size from cfg if available, else a default.
    # Handle potential missing num_workers in cfg for this specific loader.
    batch_size = cfg.get_batch_size() if cfg.data_loading.get("batch_size", None) is not None else 16
    num_workers = cfg.get_num_workers() if cfg.data_loading.get("num_workers", None) is not None else 4

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    all_means, all_stds = [], []
    for batch_idx, batch in enumerate(loader):
        imgs = batch["image"]  # [B, C, H, W]
        # Compute per-channel mean and std over spatial dims, then average over batch
        # Ensure imgs is float for mean/std calculation
        if imgs.dtype == torch.uint8 or imgs.dtype == torch.int16 or imgs.dtype == torch.int32 or imgs.dtype == torch.int64:
            imgs = imgs.float() / 255.0 if imgs.max() > 1 else imgs.float()

        current_means = imgs.mean(dim=[0, 2, 3]) # Mean over C, H, W
        current_stds = imgs.std(dim=[0, 2, 3])   # Std over C, H, W
        
        # Handle cases where a batch might have 0 std (e.g., all black images)
        # although ScaleIntensity should prevent this if images are not all identical.
        current_stds[current_stds == 0] = 1e-6 # Replace 0 std with a tiny value to avoid NaN/inf

        all_means.append(current_means)
        all_stds.append(current_stds)

        # Optional: print progress for large datasets
        # if batch_idx % 50 == 0:
        #     print(f"  Stats computation: processed batch {batch_idx+1}/{len(loader)}")

    if not all_means or not all_stds:
        # Fallback if dataset was empty or all images failed to load, though unlikely.
        print("Warning: No data to compute mean/std. Returning default [0.5]*3, [0.5]*3")
        num_channels = 3 # Assuming 3 channels (RGB)
        return {"mean": [0.5] * num_channels, "std": [0.5] * num_channels}

    # Aggregate means and stds:
    # For mean: average of batch means
    # For std: use formula for pooled std or simply average of batch stds as an approximation.
    # Averaging stds is not strictly correct but often used.
    # A more correct pooled variance: S_p^2 = sum((n_i-1)S_i^2 + n_i * (mean_i - global_mean)^2) / (N-k)
    # For simplicity here, we'll average the means and stds.
    final_mean = torch.stack(all_means).mean(dim=0).tolist()
    final_std = torch.stack(all_stds).mean(dim=0).tolist()

    # print(f"Computed Mean: {final_mean}, Computed Std: {final_std} for the current dataset split.")
    return {"mean": final_mean, "std": final_std}

def is_supported_by_torchvision(model_name:str)->bool:
    """
    Check if the model is supported by torchvision ie if it is in the IMAGENET_WEIGHTS dictionary
    """
    return IMAGENET_WEIGHTS.get(model_name.lower()) is not None

def get_transforms(cfg: ConfigLoader, fold_specific_stats: dict[str, float] | None = None) -> Tuple[Compose, Compose, Compose]:
    """
    Create and return the training, validation, and test transforms.
    
    Args:
        cfg: A configuration object/dictionary with data augmentation parameters.
        fold_specific_stats: A dictionary with keys 'mean' and 'std' used for fold-specific normalization.
        
    Returns:
        tuple (train_transforms, val_transforms, test_transforms)
        each of these is a Compose object containing the necessary transformations list.
    """
    # select the normalization params if you are using a pretrained model
    is_pretrained = cfg.get_transfer_learning()
    print(f"Using pretrained model: {is_pretrained}")
    
    color_transforms = cfg.get_intensity_augmentation_preset() not in ["none", "off", "no", "0"]
    
    # Initialize transform lists
    train_transforms_list = []
    val_transforms_list = []
    supported_by_torchvision = is_supported_by_torchvision(cfg.get_model_name())
    
    print(f"Using pretrained model: {is_pretrained} and supported by torchvision: {supported_by_torchvision} with color transforms: {color_transforms}")
    # train_transforms_list, val_transforms_list = get_custom_transforms_lists(cfg, color_transforms, fold_specific_stats)
    if not supported_by_torchvision or not is_pretrained: 
        print("the model is not supported by torchvision or is not pretrained")
        # ie. if you are not using a pretrained model or using a pretrained model that is not supported by torchvision you can use custom transforms
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
    Generate training and validation transform lists for custom models. ie not pretrained models
    """
    print(f"Model {cfg.get_model_name()} not supported using custom transforms")
    supported_by_torchvision = is_supported_by_torchvision(cfg.get_model_name())
    
    # Pipeline di base che carica e ridimensiona
    train_transforms_list = get_preNormalization_transforms_list(cfg, supported_by_torchvision)

    # --- INSERISCI QUESTA PARTE ---
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
    spatial_transforms = _get_spatial_augmentations(cfg)
    train_transforms_list.extend(spatial_transforms)

    # Aggiungi le aumentazioni di intensità saranno decise da come è il preset
    intensity_transforms = _get_intensity_augmentations(cfg)
    train_transforms_list.extend(intensity_transforms)
        
    # Applica la normalizzazione
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
    
    # Per la validazione, usiamo un crop centrale invece che casuale per avere risultati consistenti
    val_transforms_list = get_preNormalization_transforms_list(cfg, supported_by_torchvision)
    if crop:
        val_transforms_list.append(
            CenterSpatialCropd(keys="image", roi_size=crop_size),   
        )
    
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

# def get_pretrained_model_transforms_list(cfg, color_transforms:bool)->Tuple[List[MapTransform], List[MapTransform]]:
#     """
#     Get the transforms for pretrained models.
#     """
#     supported_by_torchvision = is_supported_by_torchvision(cfg.get_model_name())
#     imagenet_weights = IMAGENET_WEIGHTS.get(cfg.get_model_name().lower(), None) # Check if the model is supported by torchvision
#     torchvision_norm_transforms = imagenet_weights.transforms() # resize 256x256, center crop, convert to tensor, normalize using imagenet mean and std
#         # these are the default transforms for imagenet weights
#         # def forward(self, img: Tensor) -> Tensor:
#             # img = F.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
#             # img = F.center_crop(img, self.crop_size)
#             # if not isinstance(img, Tensor):
#             #     img = F.pil_to_tensor(img)
#             # img = F.convert_image_dtype(img, torch.float) #Converts the image to floating point and SCALES to [0.0, 1.0].
#             # img = F.normalize(img, mean=self.mean, std=self.std)
#             # return img
            
#     print(f"Using Imagenet/micronet pretrained model--> using torchvision transforms")
#     print(imagenet_weights.transforms().describe())
#     # Training transforms for pretrained models
#     train_transforms_list = get_preNormalization_transforms_list(cfg, supported_by_torchvision)
#     train_transforms_list.extend(_get_spatial_augmentations(cfg))
    
#     if color_transforms:
#         train_transforms_list.extend(_get_intensity_augmentations(cfg))
    
#     # Apply torchvision transforms (includes resize, crop, ToTensor (if applicable), and ImageNet normalization)
#     # NOTE: resize,crop,Tensor conversion,scaling are applied before the normalization hence this reapplication should not have any effect
#     # even if reapplied
#     train_transforms_list.append(LambdaD(keys="image", func=torchvision_norm_transforms))
#     # train_transforms_list.append(PrintShapeTransform(keys=["image"]))

#     # Validation transforms for pretrained models
#     val_transforms_list = get_preNormalization_transforms_list(cfg, supported_by_torchvision)
#     val_transforms_list.append(LambdaD(keys="image", func=torchvision_norm_transforms))
    
#     return train_transforms_list, val_transforms_list


def get_pretrained_model_transforms_list_without_crop(cfg:ConfigLoader)->Tuple[List[MapTransform], List[MapTransform]]:
    """
    Get the transforms for pretrained models without crop.
    Handles both 3-channel and 4-channel inputs.
    
    For 3-channel: applies standard ImageNet normalization
    For 4-channel: skips ImageNet normalization (uses scaling only)
                   The adapted first conv layer will handle the 4th channel
    """
    from monai.transforms import Resized, ScaleIntensityd, NormalizeIntensityd

    num_channels = cfg.get_model_input_channels()  # Get number of channels from config
    imagenet_weights = IMAGENET_WEIGHTS.get(cfg.get_model_name().lower(), None)
    
    use_color_transforms = cfg.get_intensity_augmentation_preset() not in ["none", "off", "no", "0"]

    # Force pre-normalization to include resize and scale (no center crop)
    train_transforms_list = new_get_preNormalization_transforms_list(cfg)
    train_transforms_list.extend(_get_spatial_augmentations(cfg))
    if use_color_transforms:
        train_transforms_list.extend(_get_intensity_augmentations(cfg))
    
    # Apply normalization based on number of channels
    if num_channels == 3 and imagenet_weights is not None:
        # Standard 3-channel ImageNet normalization
        tv_t = imagenet_weights.transforms()
        mean = torch.tensor(tv_t.mean)
        std  = torch.tensor(tv_t.std)
        train_transforms_list.append(
            NormalizeIntensityd(keys=["image"], subtrahend=mean, divisor=std, channel_wise=True)
        )
        print(f"✓ Using ImageNet normalization for 3-channel input")
    elif num_channels == 4:
        # For 4-channel: skip ImageNet normalization
        # The channel adaptation layer will learn to handle the 4th channel
        # ScaleIntensityd was already applied in new_get_preNormalization_transforms_list
        print(f"✓ Using 4-channel input: Skipping ImageNet normalization (using [0,1] scaling only)")
        print(f"  → The adapted first conv layer will handle channel-specific normalization")
    else:
        # Fallback for other channel counts
        print(f"⚠️  Using {num_channels} channels without ImageNet normalization")
    
    # -----------------------------------------------------------------------
    # Validation: resize-only + same normalization as training
    val_transforms_list = new_get_preNormalization_transforms_list(cfg)
    
    if num_channels == 3 and imagenet_weights is not None:
        tv_t = imagenet_weights.transforms()
        mean = torch.tensor(tv_t.mean)
        std  = torch.tensor(tv_t.std)
        val_transforms_list.append(
            NormalizeIntensityd(keys=["image"], subtrahend=mean, divisor=std, channel_wise=True)
        )
    # For 4-channel or other cases, no additional normalization (scaling already done in pre-normalization)
    
    return train_transforms_list, val_transforms_list

def get_convert_to_tensor_transform_list():
    return [
        EnsureTyped(keys=["image"], data_type="tensor", dtype=torch.float32),
        EnsureTyped(keys=["label"], data_type="tensor", dtype=torch.int64),
    ]

def check_normalization(img, mean_val, std_val, tolerance=0.12, threshold=0.990):
    """
    Check that most of the image data falls within [mean - 3*std, mean + 3*std].

    Args:
        img (np.ndarray): The image tensor.
        mean_val (float): The mean value of the image.
        std_val (float): The standard deviation of the image.
        tolerance (float): Allowed tolerance for the bounds.
        threshold (float): Minimum fraction of values that must be within the range.

    Raises:
        ValueError: If the fraction of values within the range is less than the threshold.
    """
    lower_bound = mean_val - 3 * std_val - tolerance
    upper_bound = mean_val + 3 * std_val + tolerance

    within_range = np.logical_and(img >= lower_bound, img <= upper_bound)
    fraction_within = np.mean(within_range)

    print(f"Fraction of values within [{lower_bound:.2f}, {upper_bound:.2f}]: {fraction_within:.4f}")
    
    if fraction_within < threshold:
        raise ValueError(
            f"Only {fraction_within:.2%} of values are within the expected range. "
            "Normalization may have failed."
        )
    print("EVERYTHING IS FINE")
        
def preview_transform_output(
    pipeline: Compose,
    input_shape: Tuple[int, int, int],
    channels_mean=None,
    channels_std=None,
) -> Tuple[int, ...]:
    """
    Apply a MONAI Compose pipeline *after* any file‐loading transforms
    to a dummy image and print/return its output shape.

    This function will automatically strip out any CustomTiffFileReader
    (or other file‐based reader transforms) from the start of the pipeline
    so you can feed in a tensor directly.

    Args:
        pipeline:     your full Compose object (may start with file‐readers).
        input_shape:  (C, H, W) of the tensor *after* your resize step.
        mean:         Mean values for each channel. Can be a single value or a list of values.
        std:          Standard deviation values for each channel. Can be a single value or a list of values.

    Returns:
        The shape of the transformed image tensor, e.g. (C_out, H_out, W_out…).
    """
    # 1) Filter out any file‐reader transforms
    filtered_transforms = [
        t for t in pipeline.transforms
        if not isinstance(t, CustomTiffFileReader)
    ]
    # 2) Build a new Compose just from the remaining steps
    sub_pipeline = Compose(filtered_transforms)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 3) Create a dummy sample
    print(f"Creating a dummy sample with shape: {input_shape}")
    c, h, w = input_shape

    # Create dummy image with channel-wise normal distribution
    dummy_image = torch.zeros((c, h, w), dtype=torch.float32, device=device)
    if channels_mean is None:
        channels_mean = [0.0] * c
    if channels_std is None:
        channels_std = [1.0] * c

    for channel in range(c):
        dummy_image[channel] = torch.normal(
            mean=channels_mean[channel],
            std=channels_std[channel],
            size=(h, w),
            dtype=torch.float32,
            device=device
        )
    print("values with a RANDOM tensor input")
    dummy = {
        "image": dummy_image,
        "label": torch.tensor(0, dtype=torch.int64, device=device),
    }
    
    print("before transforms")
    print(f"Image dtype: {dummy_image.dtype}")
    print(f"Image shape: {dummy_image.shape}")
    print(f"Image mean: {dummy_image.mean()}")
    print(f"Image std: {dummy_image.std()}")
    
    # 4) Run it
    out = sub_pipeline(dummy)

    # 5) Inspect
    img = out["image"]
    max_val = img.max()
    min_val = img.min()
    mean_val = img.mean()
    std_val = img.std()
    print("after transforms")
    print(f"Image dtype: {img.dtype}")
    print(f"Image min: {min_val}")
    print(f"Image max: {max_val}")
    print(f"Image mean: {mean_val}")
    print(f"Image std: {std_val}")
    check_normalization(img, mean_val, std_val)
    shape = tuple(img.shape)
    print(f"[preview] after transforms, image tensor shape is: {shape}")
    return shape
