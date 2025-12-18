"""
Complete Colorization Model
Combines U-Net colorization with 3D reconstruction
"""

import torch
import torch.nn as nn
from .unet import UNet, AttentionUNet
from .reconstruction_3d import HybridReconstruction3D


class MedicalImageColorization3D(nn.Module):
    """
    End-to-end model for 2D to 3D medical image colorization
    
    Pipeline:
    1. Takes grayscale 2D slices (L channel in LAB space)
    2. Predicts AB channels using U-Net
    3. Reconstructs 3D volume from colorized slices
    
    Args:
        use_attention: Use attention U-Net instead of standard U-Net
        base_features_3d: Base features for 3D reconstruction network
        num_slices: Number of slices for 3D reconstruction
    """
    
    def __init__(self, use_attention=True, base_features_3d=32, num_slices=16):
        super().__init__()
        
        # 2D Colorization network (predicts AB channels from L channel)
        if use_attention:
            self.colorization_net = AttentionUNet(n_channels=1, n_classes=2)
        else:
            self.colorization_net = UNet(n_channels=1, n_classes=2)
        
        # 3D Reconstruction network
        self.reconstruction_net = HybridReconstruction3D(
            in_channels=3,  # LAB channels
            out_channels=3,
            base_features=base_features_3d,
            num_slices=num_slices
        )
        
        self.num_slices = num_slices
    
    def forward(self, x, return_intermediate=False):
        """
        Args:
            x: Input grayscale images (B, 1, D, H, W) or (B, 1, H, W)
            return_intermediate: Return intermediate colorization results
        
        Returns:
            If return_intermediate:
                colorized_2d: 2D colorized images (B, 3, D, H, W)
                reconstructed_3d: 3D reconstructed volume (B, 3, D, H, W)
                depth: Depth estimation (B, 1, D, H, W)
            Else:
                reconstructed_3d: 3D reconstructed volume (B, 3, D, H, W)
        """
        # Handle both 4D (B, C, H, W) and 5D (B, C, D, H, W) inputs
        if x.dim() == 4:
            # Single slice: (B, 1, H, W) -> add depth dimension
            x = x.unsqueeze(2)  # (B, 1, 1, H, W)
        
        batch_size, _, depth, height, width = x.shape
        
        # Colorize each 2D slice
        colorized_slices = []
        for i in range(depth):
            slice_2d = x[:, :, i, :, :]  # (B, 1, H, W)
            ab_channels = self.colorization_net(slice_2d)  # (B, 2, H, W)
            
            # Combine L channel (input) with predicted AB channels
            lab_image = torch.cat([slice_2d, ab_channels], dim=1)  # (B, 3, H, W)
            colorized_slices.append(lab_image)
        
        # Stack colorized slices: (B, 3, D, H, W)
        colorized_2d = torch.stack(colorized_slices, dim=2)
        
        # 3D Reconstruction
        reconstructed_3d, depth_map = self.reconstruction_net(colorized_2d)
        
        if return_intermediate:
            return colorized_2d, reconstructed_3d, depth_map
        else:
            return reconstructed_3d
    
    def colorize_only(self, x):
        """
        Only perform 2D colorization without 3D reconstruction
        
        Args:
            x: Input grayscale image (B, 1, H, W)
        
        Returns:
            colorized: Colorized image in LAB space (B, 3, H, W)
        """
        ab_channels = self.colorization_net(x)
        colorized = torch.cat([x, ab_channels], dim=1)
        return colorized
    
    def reconstruct_only(self, colorized_slices):
        """
        Only perform 3D reconstruction from pre-colorized slices
        
        Args:
            colorized_slices: Colorized images (B, 3, D, H, W)
        
        Returns:
            reconstructed: 3D volume (B, 3, D, H, W)
            depth: Depth map (B, 1, D, H, W)
        """
        return self.reconstruction_net(colorized_slices)


class LightweightColorization3D(nn.Module):
    """
    Lightweight version for faster inference and lower memory usage
    """
    
    def __init__(self, num_slices=8):
        super().__init__()
        
        # Smaller U-Net
        self.colorization_net = UNet(n_channels=1, n_classes=2, bilinear=True)
        
        # Smaller 3D network
        self.reconstruction_net = HybridReconstruction3D(
            in_channels=3,
            out_channels=3,
            base_features=16,  # Reduced from 32
            num_slices=num_slices
        )
        
        self.num_slices = num_slices
    
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(2)
        
        batch_size, _, depth, height, width = x.shape
        
        # Colorize slices
        colorized_slices = []
        for i in range(depth):
            slice_2d = x[:, :, i, :, :]
            ab_channels = self.colorization_net(slice_2d)
            lab_image = torch.cat([slice_2d, ab_channels], dim=1)
            colorized_slices.append(lab_image)
        
        colorized_2d = torch.stack(colorized_slices, dim=2)
        
        # 3D Reconstruction
        reconstructed_3d, _ = self.reconstruction_net(colorized_2d)
        
        return reconstructed_3d


def get_model(model_type='full', **kwargs):
    """
    Factory function to get the appropriate model
    
    Args:
        model_type: 'full', 'lightweight', or 'attention'
        **kwargs: Additional arguments for model initialization
    
    Returns:
        model: Initialized model
    """
    if model_type == 'full':
        return MedicalImageColorization3D(use_attention=False, **kwargs)
    elif model_type == 'attention':
        return MedicalImageColorization3D(use_attention=True, **kwargs)
    elif model_type == 'lightweight':
        return LightweightColorization3D(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the complete model
    print("Testing MedicalImageColorization3D...")
    model = MedicalImageColorization3D(use_attention=True, num_slices=16)
    
    # Test with single slice
    x_single = torch.randn(2, 1, 256, 256)
    output_single = model(x_single)
    print(f"Single slice input: {x_single.shape}")
    print(f"Output: {output_single.shape}")
    
    # Test with multiple slices
    x_multi = torch.randn(2, 1, 16, 256, 256)
    colorized, reconstructed, depth = model(x_multi, return_intermediate=True)
    print(f"\nMultiple slices input: {x_multi.shape}")
    print(f"Colorized 2D: {colorized.shape}")
    print(f"Reconstructed 3D: {reconstructed.shape}")
    print(f"Depth map: {depth.shape}")
    
    # Test colorize only
    colorized_only = model.colorize_only(x_single)
    print(f"\nColorize only output: {colorized_only.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
