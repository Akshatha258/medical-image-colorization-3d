"""
3D Reconstruction Network for Medical Images
Converts 2D colorized slices into 3D volumetric representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """3D Convolution Block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ResidualBlock3D(nn.Module):
    """3D Residual Block for better gradient flow"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv3DBlock(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class Reconstruction3DNet(nn.Module):
    """
    3D Reconstruction Network
    
    Takes a stack of 2D colorized images and reconstructs a 3D volume
    
    Args:
        in_channels: Number of input channels (3 for RGB/LAB)
        out_channels: Number of output channels (3 for RGB/LAB)
        base_features: Base number of features
        num_slices: Number of 2D slices to stack
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_features=32, num_slices=16):
        super().__init__()
        self.num_slices = num_slices
        
        # Initial 3D convolution
        self.init_conv = Conv3DBlock(in_channels, base_features, kernel_size=3, padding=1)
        
        # Encoder
        self.enc1 = nn.Sequential(
            Conv3DBlock(base_features, base_features * 2, stride=2),
            ResidualBlock3D(base_features * 2)
        )
        
        self.enc2 = nn.Sequential(
            Conv3DBlock(base_features * 2, base_features * 4, stride=2),
            ResidualBlock3D(base_features * 4)
        )
        
        self.enc3 = nn.Sequential(
            Conv3DBlock(base_features * 4, base_features * 8, stride=2),
            ResidualBlock3D(base_features * 8)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock3D(base_features * 8),
            ResidualBlock3D(base_features * 8)
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(base_features * 8, base_features * 4, 
                             kernel_size=2, stride=2),
            nn.BatchNorm3d(base_features * 4),
            nn.ReLU(inplace=True),
            ResidualBlock3D(base_features * 4)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(base_features * 4, base_features * 2, 
                             kernel_size=2, stride=2),
            nn.BatchNorm3d(base_features * 2),
            nn.ReLU(inplace=True),
            ResidualBlock3D(base_features * 2)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(base_features * 2, base_features, 
                             kernel_size=2, stride=2),
            nn.BatchNorm3d(base_features),
            nn.ReLU(inplace=True),
            ResidualBlock3D(base_features)
        )
        
        # Output layer
        self.out_conv = nn.Conv3d(base_features, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)
               where D is the depth (number of slices)
        """
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decoder
        d3 = self.dec3(b)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        
        # Output
        out = self.out_conv(d1)
        return torch.tanh(out)


class DepthEstimationNet(nn.Module):
    """
    Depth Estimation Network
    Estimates depth information from 2D images to aid 3D reconstruction
    """
    
    def __init__(self, in_channels=3, base_features=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            Conv3DBlock(in_channels, base_features, kernel_size=(1, 7, 7), padding=(0, 3, 3)),
            Conv3DBlock(base_features, base_features * 2, stride=(1, 2, 2)),
            ResidualBlock3D(base_features * 2),
            Conv3DBlock(base_features * 2, base_features * 4, stride=(1, 2, 2)),
            ResidualBlock3D(base_features * 4),
            Conv3DBlock(base_features * 4, base_features * 8, stride=(1, 2, 2)),
            ResidualBlock3D(base_features * 8)
        )
        
        self.depth_head = nn.Sequential(
            nn.Conv3d(base_features * 8, base_features * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_features * 4, 1, kernel_size=1),
            nn.Sigmoid()  # Depth in range [0, 1]
        )
    
    def forward(self, x):
        features = self.encoder(x)
        depth = self.depth_head(features)
        return depth


class HybridReconstruction3D(nn.Module):
    """
    Hybrid 3D Reconstruction combining depth estimation and volumetric reconstruction
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_features=32, num_slices=16):
        super().__init__()
        
        self.depth_net = DepthEstimationNet(in_channels, base_features)
        self.reconstruction_net = Reconstruction3DNet(
            in_channels + 1,  # +1 for depth channel
            out_channels, 
            base_features, 
            num_slices
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, D, H, W)
        Returns:
            reconstructed: 3D volume (B, C, D, H, W)
            depth: Depth map (B, 1, D, H, W)
        """
        # Estimate depth
        depth = self.depth_net(x)
        
        # Upsample depth to match input resolution
        depth_upsampled = F.interpolate(
            depth, 
            size=x.shape[2:], 
            mode='trilinear', 
            align_corners=True
        )
        
        # Concatenate depth with input
        x_with_depth = torch.cat([x, depth_upsampled], dim=1)
        
        # Reconstruct 3D volume
        reconstructed = self.reconstruction_net(x_with_depth)
        
        return reconstructed, depth


if __name__ == "__main__":
    # Test the models
    batch_size = 2
    channels = 3
    depth = 16
    height = 128
    width = 128
    
    # Test Reconstruction3DNet
    model = Reconstruction3DNet(in_channels=3, out_channels=3, num_slices=depth)
    x = torch.randn(batch_size, channels, depth, height, width)
    output = model(x)
    print(f"Reconstruction3DNet:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test HybridReconstruction3D
    hybrid_model = HybridReconstruction3D(in_channels=3, out_channels=3, num_slices=depth)
    reconstructed, depth_map = hybrid_model(x)
    print(f"\nHybridReconstruction3D:")
    print(f"  Input shape: {x.shape}")
    print(f"  Reconstructed shape: {reconstructed.shape}")
    print(f"  Depth map shape: {depth_map.shape}")
    print(f"  Parameters: {sum(p.numel() for p in hybrid_model.parameters()):,}")
