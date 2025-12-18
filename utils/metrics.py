"""
Evaluation metrics for medical image colorization
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips


class MetricsCalculator:
    """Calculate various metrics for image quality assessment"""
    
    def __init__(self, device='cuda'):
        self.device = device
        # Initialize LPIPS model for perceptual similarity
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()
    
    def calculate_psnr(self, pred, target, data_range=1.0):
        """
        Calculate Peak Signal-to-Noise Ratio
        
        Args:
            pred: Predicted image (B, C, H, W) or (H, W)
            target: Target image (B, C, H, W) or (H, W)
            data_range: Data range (default 1.0 for normalized images)
        
        Returns:
            psnr_value: PSNR in dB
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Handle batch dimension
        if pred.ndim == 4:
            psnr_values = []
            for i in range(pred.shape[0]):
                # Convert to (H, W, C) for skimage
                pred_img = np.transpose(pred[i], (1, 2, 0))
                target_img = np.transpose(target[i], (1, 2, 0))
                psnr_val = psnr(target_img, pred_img, data_range=data_range)
                psnr_values.append(psnr_val)
            return np.mean(psnr_values)
        else:
            return psnr(target, pred, data_range=data_range)
    
    def calculate_ssim(self, pred, target, data_range=1.0):
        """
        Calculate Structural Similarity Index
        
        Args:
            pred: Predicted image
            target: Target image
            data_range: Data range
        
        Returns:
            ssim_value: SSIM score
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Handle batch dimension
        if pred.ndim == 4:
            ssim_values = []
            for i in range(pred.shape[0]):
                pred_img = np.transpose(pred[i], (1, 2, 0))
                target_img = np.transpose(target[i], (1, 2, 0))
                ssim_val = ssim(target_img, pred_img, 
                               data_range=data_range, 
                               channel_axis=2)
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)
        else:
            return ssim(target, pred, data_range=data_range)
    
    def calculate_lpips(self, pred, target):
        """
        Calculate Learned Perceptual Image Patch Similarity
        
        Args:
            pred: Predicted image (B, C, H, W), range [-1, 1]
            target: Target image (B, C, H, W), range [-1, 1]
        
        Returns:
            lpips_value: LPIPS distance (lower is better)
        """
        if not isinstance(pred, torch.Tensor):
            pred = torch.from_numpy(pred).float()
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).float()
        
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        # Ensure 3 channels for LPIPS
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            lpips_value = self.lpips_model(pred, target).mean().item()
        
        return lpips_value
    
    def calculate_mae(self, pred, target):
        """
        Calculate Mean Absolute Error
        
        Args:
            pred: Predicted image
            target: Target image
        
        Returns:
            mae_value: MAE
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        return np.mean(np.abs(pred - target))
    
    def calculate_mse(self, pred, target):
        """
        Calculate Mean Squared Error
        
        Args:
            pred: Predicted image
            target: Target image
        
        Returns:
            mse_value: MSE
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        return np.mean((pred - target) ** 2)
    
    def calculate_all_metrics(self, pred, target, data_range=1.0):
        """
        Calculate all metrics at once
        
        Args:
            pred: Predicted image
            target: Target image
            data_range: Data range
        
        Returns:
            metrics: Dictionary of all metrics
        """
        metrics = {
            'psnr': self.calculate_psnr(pred, target, data_range),
            'ssim': self.calculate_ssim(pred, target, data_range),
            'mae': self.calculate_mae(pred, target),
            'mse': self.calculate_mse(pred, target),
        }
        
        # LPIPS requires specific format
        try:
            # Convert to [-1, 1] range for LPIPS
            pred_lpips = pred * 2 - 1 if pred.max() <= 1.0 else pred
            target_lpips = target * 2 - 1 if target.max() <= 1.0 else target
            metrics['lpips'] = self.calculate_lpips(pred_lpips, target_lpips)
        except Exception as e:
            print(f"LPIPS calculation failed: {e}")
            metrics['lpips'] = None
        
        return metrics


def calculate_3d_iou(pred_volume, target_volume, threshold=0.5):
    """
    Calculate 3D Intersection over Union
    
    Args:
        pred_volume: Predicted 3D volume
        target_volume: Target 3D volume
        threshold: Threshold for binarization
    
    Returns:
        iou: IoU score
    """
    if isinstance(pred_volume, torch.Tensor):
        pred_volume = pred_volume.detach().cpu().numpy()
    if isinstance(target_volume, torch.Tensor):
        target_volume = target_volume.detach().cpu().numpy()
    
    # Binarize
    pred_binary = (pred_volume > threshold).astype(np.float32)
    target_binary = (target_volume > threshold).astype(np.float32)
    
    # Calculate IoU
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary) - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou


def calculate_dice_coefficient(pred_volume, target_volume, threshold=0.5):
    """
    Calculate Dice Coefficient (F1 score for segmentation)
    
    Args:
        pred_volume: Predicted volume
        target_volume: Target volume
        threshold: Threshold for binarization
    
    Returns:
        dice: Dice coefficient
    """
    if isinstance(pred_volume, torch.Tensor):
        pred_volume = pred_volume.detach().cpu().numpy()
    if isinstance(target_volume, torch.Tensor):
        target_volume = target_volume.detach().cpu().numpy()
    
    # Binarize
    pred_binary = (pred_volume > threshold).astype(np.float32)
    target_binary = (target_volume > threshold).astype(np.float32)
    
    # Calculate Dice
    intersection = np.sum(pred_binary * target_binary)
    dice = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(target_binary) + 1e-8)
    
    return dice


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calculator = MetricsCalculator(device=device)
    
    # Create dummy images
    pred = torch.rand(2, 3, 256, 256)
    target = pred + torch.randn(2, 3, 256, 256) * 0.1  # Add some noise
    
    # Calculate metrics
    metrics = calculator.calculate_all_metrics(pred, target)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key.upper()}: {value:.4f}")
    
    # Test 3D metrics
    pred_volume = torch.rand(1, 1, 16, 128, 128)
    target_volume = pred_volume + torch.randn(1, 1, 16, 128, 128) * 0.1
    
    iou = calculate_3d_iou(pred_volume, target_volume)
    dice = calculate_dice_coefficient(pred_volume, target_volume)
    
    print(f"\n3D Metrics:")
    print(f"  IoU: {iou:.4f}")
    print(f"  Dice: {dice:.4f}")
