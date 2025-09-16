# byol_transforms.py
from typing import List
import numpy as np
import torch
from monai.transforms import (
    Compose, RandSpatialCropd, RandFlipd, RandRotate90d, RandAffined,
    Rand2DElasticd, RandGaussianNoised, RandGaussianSmoothd,
    RandAdjustContrastd, RandHistogramShiftd,
    RandCoarseDropoutd
)
from monai.transforms.intensity.array import NormalizeIntensity

# ------------------------- BUILDERS ------------------------------------------
# ---------------------- NON-DICT BUILDERS ------------------------------------
# Each element now expects a single torch.Tensor / np.ndarray (C,H,W).

from monai.transforms import (
    RandSpatialCrop, RandFlip, RandRotate90, Rand2DElastic,
    RandGaussianSmooth, RandGaussianNoise, RandAdjustContrast,
    RandHistogramShift, RandAffined, RandCoarseDropout
)
import numpy as np

def get_byol_view1_aug(cfg):
    """Global crop (≈55-100 % area) + moderate deformations, non-dict version."""
    return [
        # — spatial —
        RandSpatialCrop(
            roi_size=cfg.data_augmentation["resize_spatial_size"],
            random_center=True,               # keeps 55–100 % of area
            max_roi_size=None,
        ),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandRotate90(prob=0.5, max_k=2),      # ±180°
        Rand2DElastic(
            spacing=(75, 75), magnitude_range=(2, 4),
            prob=0.20, rotate_range=np.pi / 36,
            scale_range=(0.97, 1.03),
            padding_mode="border", mode="bilinear"
        ),
        # — photometric —
        RandGaussianSmooth(prob=0.5,
                           sigma_x=(0.2, 1.2), sigma_y=(0.2, 1.2)),
        RandGaussianNoise(prob=0.4, mean=0.0, std=0.015),
        RandAdjustContrast(prob=0.40, gamma=(0.8, 1.2)),
        RandHistogramShift(prob=0.25),
        # RandChannelDropout has no non-dict analogue; omit or implement custom
        # — occlusion —
        RandCoarseDropout(
            holes=1, spatial_size=(32, 32),
            prob=0.15
        ),
    ]


def get_byol_view2_aug(cfg):
    """Local crop (≈35-60 % area) + stronger deformations, non-dict version."""
    return [
        RandSpatialCrop(
            roi_size=cfg.data_augmentation["resize_spatial_size"],
            random_center=True,               # 35-60 % area (randomly sampled)
            # random_size_deviation handled internally by RandSpatialCrop
        ),
        RandAdjustContrast(prob=0.50, gamma=(0.85, 1.15)),  # pre-geometry
        # RandAffined(
        #     rotate_range=(0, 0, 0), translate_range=(0.05, 0.05),
        #     scale_range=(0.90, 1.10), shear_range=(0.05, 0.05),
        #     prob=0.50, mode="bilinear", padding_mode="border"
        # ),
        Rand2DElastic(
            spacing=(75, 75), magnitude_range=(2, 4),
            prob=0.35, rotate_range=np.pi / 36,
            scale_range=(0.95, 1.05),
            padding_mode="border", mode="bilinear"
        ),
        RandGaussianSmooth(prob=0.5,
                           sigma_x=(0.3, 1.0), sigma_y=(0.3, 1.0)),
        RandGaussianNoise(prob=0.60, mean=0.0, std=0.020),
        RandCoarseDropout(
            holes=1, spatial_size=(24, 24),
            prob=0.20
        ),
    ]


# from utils.transformations_functions import _get_preNormalization_transforms_list
from utils.transformations_functions import from_GBR_to_RGB
from classes.CustomTiffFileReader import CustomTiffFileReader
from monai.transforms import (
    DataStats, EnsureChannelFirst, EnsureType, Lambda,Resize, ScaleIntensity, NormalizeIntensity
)
from monai.transforms.utility.array import DataStats

def _get_preNormalization_nonDict_transforms_list(
    cfg,
    is_supported_by_torchvision: bool = False
) -> List:
    """
    Get the list of array-based transforms to apply before normalization.
    Only loading, resizing, scaling transformations go here
    (i.e. all transforms you’d also want on the validation set).

    If is_supported_by_torchvision is True, resizing/scaling is
    assumed to be handled by torchvision’s pretrained‐model pipeline.
    """
    is_model_pretrained = cfg.training["transfer_learning"]
    transforms: List = [
        # load a TIFF into a (H, W, C) numpy array
        # CustomTiffFileReader(keys=["image"]),
        #COMPUTE STATS
        DataStats(prefix="After_LoadImage", data_shape=True, value_range=True),
        # ensure channel-first C×H×W
        # EnsureChannelFirst(),
        # convert to torch.FloatTensor
        # EnsureType(dtype=torch.float32),
        # swap from GBR → RGB
        # Lambda(lambda img: from_GBR_to_RGB(img)),
    ]

    if not is_supported_by_torchvision or not is_model_pretrained:
        # do our own resize + intensity scale
        transforms.append(
            Resize(
                spatial_size=cfg.data_augmentation["resize_spatial_size"],
                mode="bilinear",
                size_mode="all",
            )
        )
        transforms.append(ScaleIntensity())
    else:
        print(
            f"Using pretrained model: {is_model_pretrained} "
            f"and torchvision support: {bool(is_supported_by_torchvision)}, "
            "so resize/scale will be done by torchvision.Weights.Transforms"
        )

    return transforms

def get_byol_transforms_nd(cfg, normalize_intensity=True):
    """
    Returns two MONAI Compose objects: view1_transform, view2_transform.
    They already contain your pre-normalisation pipeline.
    """
    base = _get_preNormalization_nonDict_transforms_list(
        cfg, is_supported_by_torchvision=False
    )
    # remove the loading and tensor transformation
    # print(base)
    # base = base[:]
    
    norm = [
        NormalizeIntensity(nonzero=True, channel_wise=True)
    ]
    
    if normalize_intensity:
        transformations_view1 = base + get_byol_view1_aug(cfg) + norm
        transformations_view2 = base + get_byol_view2_aug(cfg) + norm
    else:
        transformations_view1 = base + get_byol_view1_aug(cfg)
        transformations_view2 = base + get_byol_view2_aug(cfg)

    view1 = Compose(transformations_view1).set_random_state(42)
    view2 = Compose(transformations_view2).set_random_state(43)

    return view1, view2
    
from typing import List
import numpy as np
import torch
from monai.transforms import (
    Compose, RandSpatialCropd, RandFlipd, RandRotate90d, RandAffined,
    Rand2DElasticd, RandGaussianNoised, RandGaussianSmoothd,
    RandAdjustContrastd, RandHistogramShiftd, RandCoarseDropoutd,
    Resized, ScaleIntensityd, EnsureTyped, LambdaD, NormalizeIntensityd, DataStats,
    RandGridDistortiond, RandZoomd, RandStdShiftIntensityd,
    RandBiasFieldd
)

def get_byol_view1_aug_dict(cfg):
    """Global crop (≈55-100 % area) + moderate deformations, dict version."""
    return [
        # — spatial —
        RandSpatialCropd(
            keys=["image"],
            roi_size=cfg.data_augmentation["resize_spatial_size"],
            random_center=True,               # keeps 55–100 % of area
            max_roi_size=None,
        ),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image"], prob=0.5, max_k=2),      # ±180°
        RandZoomd(
            keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.2, 
            keep_size=True, padding_mode="constant", mode="bilinear"
        ),
        RandGridDistortiond(
            keys=["image"], num_cells=5, distort_limit=(-0.03, 0.03), prob=0.15, # Reduced distort_limit
            mode="bilinear", padding_mode="border" # or "zeros"
        ),
        # Rand2DElasticd(
        #     keys=["image"],
        #     spacing=(75, 75), magnitude_range=(2, 4),
        #     prob=0.20, rotate_range=np.pi / 36,
        #     scale_range=(0.97, 1.03),
        #     padding_mode="border", mode="bilinear"
        # ),
        # — photometric —
        RandGaussianSmoothd(
            keys=["image"], 
            prob=0.5,
            sigma_x=(0.2, 1.2), sigma_y=(0.2, 1.2)
        ),
        RandGaussianNoised(keys=["image"], prob=0.4, mean=0.0, std=0.015),
        # RandPoissonNoised(keys=["image"], prob=0.3),
        RandAdjustContrastd(keys=["image"], prob=0.40, gamma=(0.8, 1.2)),
        RandStdShiftIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandBiasFieldd(keys=["image"], degree=3, coeff_range=(0.0, 0.05), prob=0.2), # Reduced coeff_range
        RandHistogramShiftd(keys=["image"], prob=0.25),
        # Now we can use RandChannelDropoutd with dictionary transforms
        # RandChannelDropoutd(keys=["image"], prob=0.20),
        # — occlusion —
        RandCoarseDropoutd(
            keys=["image"],
            holes=1, 
            spatial_size=(32, 32),
            fill_value=0.0,  # Fill value for the dropout holes
            prob=0.15
        ),
    ]


def get_byol_view2_aug_dict(cfg):
    """Local crop (≈35-60 % area) + stronger deformations."""
    return [
        # — spatial —
        RandSpatialCropd(
            keys=["image"],
            roi_size=cfg.data_augmentation["resize_spatial_size"],
            random_center=True,               # 35-60 % area (randomly sampled)
        ),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0), # Often good to have basic flips in both views
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image"], prob=0.5, max_k=3), # Allow full 360 for more variety
        RandZoomd(
            keys=["image"], min_zoom=0.85, max_zoom=1.15, prob=0.25, # Slightly wider zoom range
            keep_size=True, padding_mode="constant", mode="bilinear"
        ),
        RandGridDistortiond(
            keys=["image"], num_cells=5, distort_limit=(-0.05, 0.05), prob=0.20, # Slightly stronger
            mode="bilinear", padding_mode="border" # or "zeros"
        ),
        # Uncomment if you want to use this transform
        # RandAffined(
        #     keys=["image"],
        #     rotate_range=(0, 0, 0), translate_range=(0.05, 0.05),
        #     scale_range=(0.90, 1.10), shear_range=(0.05, 0.05),
        #     prob=0.50, mode="bilinear", padding_mode="border"
        # ),
        # Rand2DElasticd(
        #     keys=["image"],
        #     spacing=(75, 75), magnitude_range=(2, 4),
        #     prob=0.35, rotate_range=np.pi / 36,
        #     scale_range=(0.95, 1.05),
        #     padding_mode="border", mode="bilinear"
        # ),
        # — photometric —
        RandAdjustContrastd(keys=["image"], prob=0.50, gamma=(0.85, 1.15)),  # pre-geometry
        RandGaussianSmoothd(
            keys=["image"], 
            prob=0.5,
            sigma_x=(0.3, 1.0), sigma_y=(0.3, 1.0)
        ),
        RandGaussianNoised(keys=["image"], prob=0.60, mean=0.0, std=0.020),
        # RandPoissonNoised(keys=["image"], prob=0.4), # Slightly higher prob for view 2
        RandStdShiftIntensityd(keys=["image"], factors=0.15, prob=0.35), # Slightly stronger shift
        RandBiasFieldd(keys=["image"], degree=3, coeff_range=(0.0, 0.08), prob=0.25), # Slightly stronger bias
        RandHistogramShiftd(keys=["image"], prob=0.30), # Slightly higher prob
        # RandChannelDropoutd(keys=["image"], prob=0.30),
        # — occlusion —
        RandCoarseDropoutd(
            keys=["image"],
            holes=1, 
            spatial_size=(24, 24), # Could be slightly smaller or more holes for local view
            fill_value=0.0,  # Fill value for the dropout holes
            prob=0.20
        ),
    ]


from utils.transformations_functions import from_GBR_to_RGB
from classes.CustomTiffFileReader import CustomTiffFileReader
from monai.transforms.utility.array import DataStats
from monai.transforms.utility.dictionary import DataStatsd

# def _get_preNormalization_dict_transforms_list(
#     cfg,
#     is_supported_by_torchvision: bool = False
# ) -> List:
#     """
#     Get the list of dictionary-based transforms to apply before normalization.
#     """
#     is_model_pretrained = cfg.training["transfer_learning"]
#     transforms: List = [
#         # load a TIFF into a (H, W, C) numpy array
#         # CustomTiffFileReader(keys=["image"]),
#         # COMPUTE STATS
#         DataStatsd(keys=["image"], prefix="After_LoadImage", data_shape=True, value_range=True),
#         # convert to torch.FloatTensor
#         EnsureTyped(keys=["image"], dtype=torch.float32),
#         # swap from GBR → RGB
#         # LambdaD(keys=["image"], func=from_GBR_to_RGB),
#     ]

#     if not is_supported_by_torchvision or not is_model_pretrained:
#         # do our own resize + intensity scale
#         transforms.append(
#             Resized(
#                 keys=["image"],
#                 spatial_size=cfg.data_augmentation["resize_spatial_size"],
#                 mode="bilinear",
#                 size_mode="all",
#             )
#         )
#         transforms.append(ScaleIntensityd(keys=["image"]))
#     else:
#         print(
#             f"Using pretrained model: {is_model_pretrained} "
#             f"and torchvision support: {bool(is_supported_by_torchvision)}, "
#             "so resize/scale will be done by torchvision.Weights.Transforms"
#         )

#     return transforms

from utils.transformations_functions import get_preNormalization_transforms_list

def get_byol_transforms_dict(cfg, normalize_intensity=True):
    """
    Returns two MONAI Compose objects with dictionary transforms:
    view1_transform, view2_transform.
    """
    # base = _get_preNormalization_dict_transforms_list(
    #     cfg, is_supported_by_torchvision=False
    # )
    
    base = get_preNormalization_transforms_list(
        cfg, is_supported_by_torchvision=False
    )
    
    base = base[3:] # removing the loading and tensor transformation
    
    norm = [
        # DataStatsd(keys=["image"], prefix="After_LoadImage", data_shape=True, value_range=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        DataStatsd(keys=["image"], prefix="After_Normalization", data_shape=True, value_range=True)
    ]
    
    if normalize_intensity:
        transformations_view1 = base + get_byol_view1_aug_dict(cfg) + norm
        transformations_view2 = base + get_byol_view2_aug_dict(cfg) + norm
    else:
        transformations_view1 = base + get_byol_view1_aug_dict(cfg)
        transformations_view2 = base + get_byol_view2_aug_dict(cfg)

    view1 = Compose(transformations_view1).set_random_state(42)
    view2 = Compose(transformations_view2).set_random_state(43)

    return view1, view2
