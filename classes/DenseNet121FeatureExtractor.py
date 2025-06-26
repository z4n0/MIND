# ---------------------
# 1. Define a DenseNet121 Feature Extractor
# ---------------------
import torch.nn as nn
from monai.networks.nets import DenseNet121

class DenseNet121FeatureExtractor(nn.Module):
    #constructor
    def __init__(self, in_channels=3, out_channels=2):
        """
        Use MONAI's DenseNet121 and remove the final classifier so we can extract features.
        Args:
            in_channels (int): Number of input channels (3 for RGB).
            out_channels (int): Dummy number for classification (won't be used).
        """
        super(DenseNet121FeatureExtractor, self).__init__()
        self.model = DenseNet121(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    
    def forward(self, x):
        # Get convolutional features: shape (B, C, H, W)
        features = self.model.features(x)
        #print(f"Features shape: {features.shape}")
        # Apply global average pooling to reduce spatial dimensions to 1x1
        pooled = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        #print(f"Pooled shape: {pooled.shape}")
        # Flatten to (B, C)
        feature_vector = pooled.view(pooled.size(0), -1)
        #print(f"Feature vector shape: {feature_vector.shape}")
        return feature_vector