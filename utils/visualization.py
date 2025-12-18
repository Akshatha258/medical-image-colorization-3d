"""
Visualization utilities for medical images
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from pathlib import Path


def visualize_2d_colorization(grayscale, colorized, save_path=None):
    """
    Visualize 2D colorization results
    
    Args:
        grayscale: Grayscale input image (H, W) or (1, H, W)
        colorized: Colorized output image (3, H, W) or (H, W, 3)
        save_path: Path to save the figure
    """
    if isinstance(grayscale, torch.Tensor):
        grayscale = grayscale.detach().cpu().numpy()
    if isinstance(colorized, torch.Tensor):
        colorized = colorized.detach().cpu().numpy()
    
    # Handle channel dimensions
    if grayscale.ndim == 3 and grayscale.shape[0] == 1:
        grayscale = grayscale[0]
    
    if colorized.ndim == 3 and colorized.shape[0] == 3:
        colorized = np.transpose(colorized, (1, 2, 0))
    
    # Normalize to [0, 1]
    grayscale = (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min() + 1e-8)
    colorized = (colorized - colorized.min()) / (colorized.max() - colorized.min() + 1e-8)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(grayscale, cmap='gray')
    axes[0].set_title('Input (Grayscale)', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(colorized)
    axes[1].set_title('Output (Colorized)', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_3d_slices(volume, num_slices=9, save_path=None, title='3D Volume Slices'):
    """
    Visualize slices from a 3D volume
    
    Args:
        volume: 3D volume (D, H, W) or (C, D, H, W)
        num_slices: Number of slices to display
        save_path: Path to save the figure
        title: Figure title
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    
    # Handle channel dimension
    if volume.ndim == 4:
        volume = volume[0]  # Take first channel
    
    depth = volume.shape[0]
    indices = np.linspace(0, depth - 1, num_slices, dtype=int)
    
    # Create grid
    rows = int(np.ceil(np.sqrt(num_slices)))
    cols = int(np.ceil(num_slices / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_slices > 1 else [axes]
    
    for i, idx in enumerate(indices):
        slice_2d = volume[idx]
        
        # Normalize
        slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
        
        axes[i].imshow(slice_2d, cmap='gray')
        axes[i].set_title(f'Slice {idx}/{depth-1}', fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_slices, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_comparison(input_img, pred_img, target_img=None, save_path=None):
    """
    Visualize input, prediction, and optionally target
    
    Args:
        input_img: Input image
        pred_img: Predicted image
        target_img: Target image (optional)
        save_path: Path to save the figure
    """
    if isinstance(input_img, torch.Tensor):
        input_img = input_img.detach().cpu().numpy()
    if isinstance(pred_img, torch.Tensor):
        pred_img = pred_img.detach().cpu().numpy()
    if target_img is not None and isinstance(target_img, torch.Tensor):
        target_img = target_img.detach().cpu().numpy()
    
    # Handle channel dimensions
    def prepare_image(img):
        if img.ndim == 3:
            if img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            if img.shape[2] == 1:
                img = img[:, :, 0]
        return (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    input_img = prepare_image(input_img)
    pred_img = prepare_image(pred_img)
    
    # Create figure
    num_cols = 3 if target_img is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(num_cols * 5, 5))
    
    axes[0].imshow(input_img, cmap='gray' if input_img.ndim == 2 else None)
    axes[0].set_title('Input', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(pred_img, cmap='gray' if pred_img.ndim == 2 else None)
    axes[1].set_title('Prediction', fontsize=14)
    axes[1].axis('off')
    
    if target_img is not None:
        target_img = prepare_image(target_img)
        axes[2].imshow(target_img, cmap='gray' if target_img.ndim == 2 else None)
        axes[2].set_title('Ground Truth', fontsize=14)
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(train_losses, val_losses, metrics_history=None, save_path=None):
    """
    Plot training and validation curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        metrics_history: Dictionary of metric histories
        save_path: Path to save the figure
    """
    num_plots = 2 if metrics_history is None else 2 + len(metrics_history)
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 5, 4))
    
    if num_plots == 2:
        axes = [axes[0], axes[1]]
    
    # Plot losses
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot learning rate if available
    axes[1].plot(train_losses, linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Train Loss', fontsize=12)
    axes[1].set_title('Training Loss (Log Scale)', fontsize=14)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Plot metrics
    if metrics_history:
        for idx, (metric_name, values) in enumerate(metrics_history.items()):
            axes[idx + 2].plot(values, linewidth=2)
            axes[idx + 2].set_xlabel('Epoch', fontsize=12)
            axes[idx + 2].set_ylabel(metric_name.upper(), fontsize=12)
            axes[idx + 2].set_title(f'{metric_name.upper()} over Epochs', fontsize=14)
            axes[idx + 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_gif_from_slices(volume, output_path, duration=100):
    """
    Create an animated GIF from 3D volume slices
    
    Args:
        volume: 3D volume (D, H, W)
        output_path: Path to save GIF
        duration: Duration per frame in milliseconds
    """
    from PIL import Image
    
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    
    if volume.ndim == 4:
        volume = volume[0]
    
    # Normalize
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    volume = (volume * 255).astype(np.uint8)
    
    # Create frames
    frames = []
    for i in range(volume.shape[0]):
        frame = Image.fromarray(volume[i])
        frames.append(frame)
    
    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    
    print(f"Saved GIF to {output_path}")


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization functions...")
    
    # Create output directory
    output_dir = Path("results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test 2D colorization
    grayscale = np.random.rand(256, 256)
    colorized = np.random.rand(256, 256, 3)
    visualize_2d_colorization(
        grayscale, 
        colorized, 
        save_path=output_dir / "2d_colorization.png"
    )
    
    # Test 3D slices
    volume = np.random.rand(16, 128, 128)
    visualize_3d_slices(
        volume, 
        num_slices=9, 
        save_path=output_dir / "3d_slices.png"
    )
    
    # Test comparison
    input_img = np.random.rand(1, 256, 256)
    pred_img = np.random.rand(3, 256, 256)
    target_img = np.random.rand(3, 256, 256)
    visualize_comparison(
        input_img, 
        pred_img, 
        target_img, 
        save_path=output_dir / "comparison.png"
    )
    
    # Test training curves
    train_losses = [1.0 - 0.05 * i + np.random.rand() * 0.1 for i in range(50)]
    val_losses = [1.0 - 0.04 * i + np.random.rand() * 0.15 for i in range(50)]
    plot_training_curves(
        train_losses, 
        val_losses, 
        save_path=output_dir / "training_curves.png"
    )
    
    print("Visualization tests complete!")
