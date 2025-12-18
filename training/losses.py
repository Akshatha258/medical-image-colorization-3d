"""
Loss functions for medical image colorization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import lpips


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features
    """
    
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = layers
        
        # Extract specific layers
        self.blocks = nn.ModuleList()
        layer_map = {
            'relu1_2': 4,
            'relu2_2': 9,
            'relu3_3': 16,
            'relu4_3': 23
        }
        
        prev_idx = 0
        for layer_name in layers:
            idx = layer_map[layer_name]
            block = nn.Sequential(*[vgg[i] for i in range(prev_idx, idx + 1)])
            self.blocks.append(block)
            prev_idx = idx + 1
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)
        
        Returns:
            loss: Perceptual loss
        """
        # Ensure 3 channels
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        loss = 0.0
        x_pred = pred
        x_target = target
        
        for block in self.blocks:
            x_pred = block(x_pred)
            x_target = block(x_target)
            loss += F.l1_loss(x_pred, x_target)
        
        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss
    """
    
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)
        
        Returns:
            loss: 1 - SSIM (lower is better)
        """
        channel = pred.size(1)
        
        if channel != self.channel:
            self.window = self.create_window(self.window_size, channel)
            self.channel = channel
        
        self.window = self.window.to(pred.device)
        
        ssim_value = self.ssim(pred, target, self.window, self.window_size, channel, self.size_average)
        return 1 - ssim_value


class CombinedLoss(nn.Module):
    """
    Combined loss for medical image colorization
    """
    
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, ssim_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.ssim_loss = SSIMLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # L1 loss
        l1 = self.l1_loss(pred, target)
        
        # Perceptual loss
        perceptual = self.perceptual_loss(pred, target)
        
        # SSIM loss
        ssim = self.ssim_loss(pred, target)
        
        # Combined loss
        total_loss = (
            self.l1_weight * l1 +
            self.perceptual_weight * perceptual +
            self.ssim_weight * ssim
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1.item(),
            'perceptual': perceptual.item(),
            'ssim': ssim.item()
        }
        
        return total_loss, loss_dict


class Reconstruction3DLoss(nn.Module):
    """
    Loss for 3D reconstruction
    """
    
    def __init__(self, l1_weight=1.0, smoothness_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.smoothness_weight = smoothness_weight
        self.l1_loss = nn.L1Loss()
    
    def smoothness_loss(self, volume):
        """
        Encourage smooth transitions between slices
        
        Args:
            volume: 3D volume (B, C, D, H, W)
        
        Returns:
            smoothness: Smoothness loss
        """
        # Compute differences between adjacent slices
        diff = volume[:, :, 1:, :, :] - volume[:, :, :-1, :, :]
        smoothness = torch.mean(torch.abs(diff))
        return smoothness
    
    def forward(self, pred_volume, target_volume):
        """
        Args:
            pred_volume: Predicted 3D volume (B, C, D, H, W)
            target_volume: Target 3D volume (B, C, D, H, W)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # L1 loss
        l1 = self.l1_loss(pred_volume, target_volume)
        
        # Smoothness loss
        smoothness = self.smoothness_loss(pred_volume)
        
        # Combined loss
        total_loss = self.l1_weight * l1 + self.smoothness_weight * smoothness
        
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1.item(),
            'smoothness': smoothness.item()
        }
        
        return total_loss, loss_dict


class TotalLoss(nn.Module):
    """
    Total loss combining 2D colorization and 3D reconstruction
    """
    
    def __init__(self, colorization_weight=1.0, reconstruction_weight=0.3):
        super().__init__()
        self.colorization_weight = colorization_weight
        self.reconstruction_weight = reconstruction_weight
        
        self.colorization_loss = CombinedLoss()
        self.reconstruction_loss = Reconstruction3DLoss()
    
    def forward(self, pred_2d, target_2d, pred_3d, target_3d):
        """
        Args:
            pred_2d: Predicted 2D colorized images
            target_2d: Target 2D colorized images
            pred_3d: Predicted 3D volume
            target_3d: Target 3D volume
        
        Returns:
            total_loss: Total combined loss
            loss_dict: Dictionary of all losses
        """
        # 2D colorization loss
        color_loss, color_dict = self.colorization_loss(pred_2d, target_2d)
        
        # 3D reconstruction loss
        recon_loss, recon_dict = self.reconstruction_loss(pred_3d, target_3d)
        
        # Combined loss
        total_loss = (
            self.colorization_weight * color_loss +
            self.reconstruction_weight * recon_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'colorization': color_loss.item(),
            'reconstruction': recon_loss.item(),
            **{f'color_{k}': v for k, v in color_dict.items()},
            **{f'recon_{k}': v for k, v in recon_dict.items()}
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test losses
    print("Testing loss functions...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test CombinedLoss
    pred = torch.rand(2, 3, 256, 256).to(device)
    target = torch.rand(2, 3, 256, 256).to(device)
    
    loss_fn = CombinedLoss().to(device)
    loss, loss_dict = loss_fn(pred, target)
    
    print("\nCombinedLoss:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # Test Reconstruction3DLoss
    pred_3d = torch.rand(2, 3, 16, 128, 128).to(device)
    target_3d = torch.rand(2, 3, 16, 128, 128).to(device)
    
    recon_loss_fn = Reconstruction3DLoss().to(device)
    recon_loss, recon_dict = recon_loss_fn(pred_3d, target_3d)
    
    print("\nReconstruction3DLoss:")
    for key, value in recon_dict.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nLoss functions test complete!")
