import torch
from classes.CustomTiffFileReader import CustomTiffFileReader

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Resized,
    CenterSpatialCropd,
    ScaleIntensityd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandHistogramShiftd,
    RandAdjustContrastd,
    NormalizeIntensityd,
)

from typing import Tuple, Union


#TODO: add a function to convert the image to RGB if it is not already
def get_transforms(cfg, color_transforms: bool = True):
    """
    MONAI-based pipeline for training, validation, and test transforms on
    1024x1024x3 UInt8 TIFF images. Returns train, val, test Compose objects.

    Steps:
      1) Load TIFF -> Tensor(float32)
      2) Ensure channel first
      3) Resize + center crop to 224×224
      4) Scale intensities to [0,1]
      5) Apply spatial and optional color augmentations (train only)
      6) Normalize with ImageNet mean/std
    """
    # ImageNet normalization parameters
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # --- TRAIN TRANSFORMS ---
    train_list = [
        # a) Read and convert to tensor
        CustomTiffFileReader(keys=["image"]),
        # EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        
        EnsureTyped(keys=["image", "label"], data_type="tensor", dtype=torch.float32),

        # b) Spatial resizing and cropping
        Resized(
            keys="image",
            spatial_size=cfg.data_augmentation["resize_spatial_size"],
            mode='bilinear'
        ),
        CenterSpatialCropd(keys="image", roi_size=(224, 224)),

        # c) Scale intensities to [0,1]
        ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),

        # d) Spatial augmentations
        RandFlipd(
            keys="image",
            prob=cfg.data_augmentation["rand_flip_prob"],
            spatial_axis=0
        ),
        RandRotate90d(
            keys="image",
            prob=cfg.data_augmentation["rand_rotate90_prob"],
            max_k=cfg.data_augmentation["rand_rotate90_max_k"]
        ),
        RandGaussianNoised(
            keys="image",
            prob=cfg.data_augmentation["rand_gaussian_noise_prob"],
            mean=cfg.data_augmentation["rand_gaussian_noise_mean"],
            std=cfg.data_augmentation["rand_gaussian_noise_std"]
        ),
    ]

    # e) Optional color augmentations
    if color_transforms:
        train_list += [
            RandHistogramShiftd(keys=["image"], prob=0.3),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 1.2)),
        ]

    # f) Final normalization
    train_list.append(
        NormalizeIntensityd(
            keys=["image"],
            subtrahend=mean,
            divisor=std,
            channel_wise=True
        )
    )
    train_transforms = Compose(train_list)

    # --- VALIDATION / TEST TRANSFORMS (no augmentations) ---
    val_list = [
        CustomTiffFileReader(keys=["image"]),
        # EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureTyped(keys=["image", "label"], data_type="tensor", dtype=torch.float32),
        Resized(
            keys="image",
            spatial_size=cfg.data_augmentation["resize_spatial_size"],
            mode='bilinear'
        ),
        CenterSpatialCropd(keys="image", roi_size=(224, 224)),
        ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
        NormalizeIntensityd(
            keys=["image"],
            subtrahend=mean,
            divisor=std,
            channel_wise=True
        )
    ]
    val_transforms = Compose(val_list)
    test_transforms = val_transforms
    test_transforms = val_transforms
    in_shape = (3, *cfg.data_augmentation["resize_spatial_size"])
    preview_transform_output(train_transforms, in_shape)
    return train_transforms, val_transforms, test_transforms


def preview_transform_output(
    pipeline: Compose,
    input_shape: Tuple[int, int, int],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[int, ...]:
    """
    Apply a MONAI Compose pipeline after file-reading transforms to a dummy image
    and print/return its output shape and stats.
    """
    # 1) Filter out any file-reader transforms
    filtered = [t for t in pipeline.transforms if not isinstance(t, CustomTiffFileReader)]
    sub_pipeline = Compose(filtered)

    # 2) Dummy sample
    c, h, w = input_shape
    dummy = {
        "image": torch.rand((c, h, w), dtype=torch.float32, device=device),
        "label": torch.tensor(0, dtype=torch.int64, device=device),
    }
    out = sub_pipeline(dummy)
    img = out["image"]
    #print stats of the image after the transformations
    max_val = img.max()
    min_val = img.min()
    mean_val = img.mean()
    std_val = img.std()
    print(f"Image dtype: {img.dtype}")
    print(f"Image device: {img.device}")
    print(f"Image min: {min_val}")
    print(f"Image max: {max_val}")
    print(f"Image mean: {mean_val}")
    print(f"Image std: {std_val}")
    if min_val < (std_val*(-3))-0.1 or max_val > (3*std_val)+0.1:
        print("after normalization and intensity scaling, values should be in range [0,+- 3std] since std should be close to 1 it should be between [-3,3]")
        raise ValueError("Image values are not in the range [0,+- 3std] if normalized properly they should")
    shape = tuple(img.shape)
    print(f"[preview] after transforms, image tensor shape is: {shape}")
    return shape
