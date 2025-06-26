import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
import torch.nn.functional as F
from monai.networks.blocks import SABlock

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(SimpleCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 1/2 spatial resolution
        )
        # Second block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 1/4 spatial resolution
        )
        # Third block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 1/8 spatial resolution
        )
        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x is expected to be of shape (B, C, H, W) with C=3 for RGB
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x now has shape (B, 128, H', W') with H' = H/8 (approximately)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # flatten x to (B, 128) from (B, 128, 1, 1)
        x = self.fc(x)
        return x


class SimpleMONAICNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(SimpleMONAICNN, self).__init__()
        # First convolutional block: Conv-BN-ReLU with 32 output channels.
        self.conv1 = Convolution(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            strides=1,
            padding=1,    # preserve spatial size
            act="RELU",
            norm="BATCH",
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 1/2 spatial resolution
        
        # Second block: 32 -> 64 channels.
        self.conv2 = Convolution(
            spatial_dims=2,
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            strides=1,
            padding=1,
            act="RELU",
            norm="BATCH",
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 1/4 spatial resolution
        
        # Third block: 64 -> 128 channels.
        self.conv3 = Convolution(
            spatial_dims=2,
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            strides=1,
            padding=1,
            act="RELU",
            norm="BATCH",
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)  # 1/8 spatial resolution
        
        # Global pooling and classifier.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x should have shape (B, 3, H, W)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.global_pool(x)  # shape (B, 128, 1, 1)
        x = x.view(x.size(0), -1)  # flatten to (B, 128)
        x = self.fc(x)
        return x

# --- CBAM Implementation ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, kernel_size=1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_cat)
        return self.sigmoid(x_out)

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

# --- Simple CNN with CBAM ---
class SimpleCNNWithCBAM(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(SimpleCNNWithCBAM, self).__init__()
        # First block: Conv -> BN -> ReLU -> Pool -> CBAM
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # reduce spatial dims by 2
        self.cbam1 = CBAM(32)
        
        # Second block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.cbam2 = CBAM(64)
        
        # Third block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.cbam3 = CBAM(128)
        
        # Global average pooling and final classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Input shape: (B, 3, H, W)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.cbam1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.cbam2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.cbam3(x)
        
        # x now is of shape (B, 128, H', W') with H' = H/8
        x = self.global_pool(x)  # shape: (B, 128, 1, 1)
        x = x.view(x.size(0), -1)  # flatten to (B, 128)
        x = self.fc(x)
        return x
    
class SimpleCNNWithSABlock(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, img_size=224, hidden_size=64, num_heads=4):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            num_classes (int): Number of output classes.
            img_size (int): Input image size (assumed square).
            hidden_size (int): Number of channels after the first convolution.
            num_heads (int): Number of attention heads for the SABlock.
        """
        super(SimpleCNNWithSABlock, self).__init__()
        # Initial convolution block: produces feature maps of shape (B, hidden_size, H, W)
        self.conv1 = Convolution(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=3,
            strides=1,
            padding=1,
            act="RELU",
            norm="BATCH"
        )
        # Downsample using max pooling to reduce spatial size.
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # output: (B, hidden_size, img_size/2, img_size/2)
        
        # Compute sequence length from the pooled feature map.
        pooled_size = img_size // 2  # assuming square images
        sequence_length = pooled_size * pooled_size
        
        # SABlock expects input shape (B, sequence_length, hidden_size).
        self.sa = SABlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=0.0,
            qkv_bias=True,
            save_attn=False,
            dim_head=None,
            hidden_input_size=hidden_size,
            causal=False,
            sequence_length=sequence_length,
            rel_pos_embedding="decomposed",  # or None, depending on your needs
            input_size=(pooled_size, pooled_size),
            attention_dtype=None,
            include_fc=True,
            use_combined_linear=True,
            use_flash_attention=False
        )
        
        # Global average pooling and a final fully connected layer.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: (B, 3, img_size, img_size)
        x = self.conv1(x)       # (B, hidden_size, img_size, img_size)
        x = self.pool1(x)       # (B, hidden_size, img_size/2, img_size/2)
        
        # Reshape features to a sequence for the SABlock.
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # (B, H*W, hidden_size)
        
        # Apply self-attention.
        x = self.sa(x)  # (B, H*W, hidden_size)
        
        # Reshape back to spatial feature maps.
        x = x.transpose(1, 2).view(B, C, H, W)  # (B, hidden_size, H, W)
        
        # Global pooling and classification.
        x = self.global_pool(x)  # (B, hidden_size, 1, 1)
        x = x.view(B, -1)        # (B, hidden_size)
        x = self.fc(x)           # (B, num_classes)
        return x
