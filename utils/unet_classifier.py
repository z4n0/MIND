import torch
from torch import nn
from typing import Optional

try:
    from monai.networks.nets import BasicUNet
except Exception as e:  # pragma: no cover
    BasicUNet = None  # type: ignore


class UNetClassifier(nn.Module):
    """
    Simple U-Net-based classifier using MONAI's BasicUNet as a feature extractor.

    Pipeline:
      x -> BasicUNet(in_channels, out_channels=feat_channels)
        -> AdaptiveAvgPool2d(1) -> Flatten -> Dropout(optional) -> Linear(feat_channels, num_classes)

    Notes:
    - This is meant for fast experimentation: no segmentation loss, just GAP over
      the U-Net output to get an image-level prediction.
    - Works with arbitrary input spatial sizes (GAP handles it).
    - Expects 2D inputs of shape (N, C, H, W).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feat_channels: int = 64,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if BasicUNet is None:
            raise ImportError("MONAI is required for UNetClassifier (pip install monai)")

        # Use BasicUNet as a backbone that outputs `feat_channels` maps
        self.backbone = BasicUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feat_channels,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p and dropout_p > 0 else nn.Identity()
        self.head = nn.Linear(feat_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone outputs (N, feat_channels, H, W)
        feats = self.backbone(x)
        # Global average pooling -> (N, feat_channels, 1, 1)
        z = self.pool(feats)
        z = torch.flatten(z, 1)  # (N, feat_channels)
        z = self.dropout(z)
        logits = self.head(z)  # (N, num_classes)
        return logits

