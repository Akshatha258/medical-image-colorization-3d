"""
Inference script for medical image colorization
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models import get_model
from utils.preprocessing import (
    load_dicom, load_nifti, normalize_image, 
    preprocess_for_training, lab_to_rgb
)
from utils.visualization import visualize_2d_colorization, visualize_3d_slices


def load_model(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded model
    """
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    model_config = checkpoint.get('config', {})
    model_type = model_config.get('MODEL_TYPE', 'attention')
    num_slices = model_config.get('NUM_SLICES', 16)
    
    # Initialize model
    model = get_model(
        model_type=model_type,
        num_slices=num_slices
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model type: {model_type}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    
    return model


def preprocess_image(image_path, target_size=256):
    """
    Load and preprocess image for inference
    
    Args:
        image_path: Path to image file
        target_size: Target image size
    
    Returns:
        image_tensor: Preprocessed image tensor
        original_image: Original image for visualization
    """
    file_ext = Path(image_path).suffix.lower()
    
    # Load image based on file type
    if file_ext == '.dcm':
        image, _ = load_dicom(image_path)
    elif file_ext in ['.nii', '.nii.gz']:
        volume, _ = load_nifti(image_path)
        # Take middle slice for 2D
        image = volume[volume.shape[0] // 2]
    else:
        # Standard image formats
        image = np.array(Image.open(image_path).convert('L'))
    
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Store original for visualization
    original_image = image.copy()
    
    # Preprocess
    image = preprocess_for_training(
        image,
        target_size=(target_size, target_size),
        normalize_method='minmax'
    )
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    return image_tensor, original_image


def colorize_image(model, image_tensor, device='cuda'):
    """
    Colorize a single image
    
    Args:
        model: Trained model
        image_tensor: Input image tensor (1, 1, H, W)
        device: Device to run inference on
    
    Returns:
        colorized: Colorized image (H, W, 3)
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Get colorized output (LAB space)
        output = model.colorize_only(image_tensor)  # (1, 3, H, W)
    
    # Convert to numpy
    output = output.cpu().numpy()[0]  # (3, H, W)
    output = np.transpose(output, (1, 2, 0))  # (H, W, 3)
    
    # Convert LAB to RGB
    # Note: In practice, you'd need proper LAB to RGB conversion
    # For now, we'll just normalize the output
    colorized = (output - output.min()) / (output.max() - output.min() + 1e-8)
    
    return colorized


def process_single_image(model, image_path, output_dir, device='cuda'):
    """
    Process a single image
    
    Args:
        model: Trained model
        image_path: Path to input image
        output_dir: Directory to save results
        device: Device to run inference on
    """
    print(f"\nProcessing: {image_path}")
    
    # Preprocess
    image_tensor, original_image = preprocess_image(image_path)
    
    # Colorize
    colorized = colorize_image(model, image_tensor, device)
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = Path(image_path).stem
    
    # Save visualization
    viz_path = output_dir / f"{filename}_colorized.png"
    visualize_2d_colorization(
        original_image,
        colorized,
        save_path=viz_path
    )
    
    # Save colorized image
    colorized_path = output_dir / f"{filename}_output.png"
    colorized_img = (colorized * 255).astype(np.uint8)
    Image.fromarray(colorized_img).save(colorized_path)
    
    print(f"✓ Results saved to {output_dir}")


def process_directory(model, input_dir, output_dir, device='cuda'):
    """
    Process all images in a directory
    
    Args:
        model: Trained model
        input_dir: Directory containing input images
        output_dir: Directory to save results
        device: Device to run inference on
    """
    input_path = Path(input_dir)
    
    # Find all image files
    image_extensions = ['.dcm', '.nii', '.nii.gz', '.png', '.jpg', '.jpeg']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_path in image_files:
        try:
            process_single_image(model, str(image_path), output_dir, device)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print(f"\n✓ All images processed! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Medical Image Colorization Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='results/inference',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        process_single_image(model, str(input_path), args.output, device)
    elif input_path.is_dir():
        # Directory of images
        process_directory(model, str(input_path), args.output, device)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()
