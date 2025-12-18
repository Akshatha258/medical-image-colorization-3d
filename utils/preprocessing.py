"""
Data preprocessing utilities for medical images
"""

import os
import numpy as np
import pydicom
from PIL import Image
import SimpleITK as sitk
from skimage import color, exposure
import torch
from pathlib import Path


def load_dicom(file_path):
    """
    Load DICOM file and return pixel array
    
    Args:
        file_path: Path to DICOM file
    
    Returns:
        image: Numpy array of image data
        metadata: DICOM metadata
    """
    try:
        dicom = pydicom.dcmread(file_path)
        image = dicom.pixel_array.astype(np.float32)
        
        # Apply rescale slope and intercept if available
        if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
            image = image * dicom.RescaleSlope + dicom.RescaleIntercept
        
        return image, dicom
    except Exception as e:
        print(f"Error loading DICOM {file_path}: {e}")
        return None, None


def apply_windowing(image, window_center, window_width):
    """
    Apply windowing to CT images (HU units)
    
    Args:
        image: Input image in HU units
        window_center: Window center (level)
        window_width: Window width
    
    Returns:
        windowed_image: Windowed image normalized to [0, 1]
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    
    windowed = np.clip(image, img_min, img_max)
    windowed = (windowed - img_min) / (img_max - img_min)
    
    return windowed


def normalize_image(image, method='minmax'):
    """
    Normalize image to [0, 1] range
    
    Args:
        image: Input image
        method: 'minmax', 'zscore', or 'percentile'
    
    Returns:
        normalized: Normalized image
    """
    if method == 'minmax':
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            normalized = (image - img_min) / (img_max - img_min)
        else:
            normalized = image
    
    elif method == 'zscore':
        mean, std = image.mean(), image.std()
        if std > 0:
            normalized = (image - mean) / std
            # Clip to reasonable range
            normalized = np.clip(normalized, -3, 3)
            normalized = (normalized + 3) / 6  # Scale to [0, 1]
        else:
            normalized = image
    
    elif method == 'percentile':
        p2, p98 = np.percentile(image, (2, 98))
        normalized = np.clip(image, p2, p98)
        normalized = (normalized - p2) / (p98 - p2)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.astype(np.float32)


def rgb_to_lab(rgb_image):
    """
    Convert RGB image to LAB color space
    
    Args:
        rgb_image: RGB image in range [0, 1]
    
    Returns:
        lab_image: LAB image (L: [0, 100], A: [-128, 127], B: [-128, 127])
    """
    lab = color.rgb2lab(rgb_image)
    return lab


def lab_to_rgb(lab_image):
    """
    Convert LAB image to RGB color space
    
    Args:
        lab_image: LAB image
    
    Returns:
        rgb_image: RGB image in range [0, 1]
    """
    rgb = color.lab2rgb(lab_image)
    return np.clip(rgb, 0, 1)


def preprocess_for_training(image, target_size=(256, 256), normalize_method='minmax'):
    """
    Preprocess image for training
    
    Args:
        image: Input grayscale image
        target_size: Target size (H, W)
        normalize_method: Normalization method
    
    Returns:
        processed: Preprocessed image ready for training
    """
    # Normalize
    image = normalize_image(image, method=normalize_method)
    
    # Resize
    if image.shape != target_size:
        image = np.array(Image.fromarray(image).resize(target_size, Image.BILINEAR))
    
    # Ensure float32
    image = image.astype(np.float32)
    
    return image


def load_nifti(file_path):
    """
    Load NIfTI file (common for MRI data)
    
    Args:
        file_path: Path to .nii or .nii.gz file
    
    Returns:
        image: 3D numpy array
        metadata: SimpleITK image object with metadata
    """
    try:
        sitk_image = sitk.ReadImage(file_path)
        image = sitk.GetArrayFromImage(sitk_image)
        return image, sitk_image
    except Exception as e:
        print(f"Error loading NIfTI {file_path}: {e}")
        return None, None


def save_nifti(image, output_path, reference_image=None):
    """
    Save numpy array as NIfTI file
    
    Args:
        image: 3D numpy array
        output_path: Output file path
        reference_image: Reference SimpleITK image for metadata
    """
    sitk_image = sitk.GetImageFromArray(image)
    
    if reference_image is not None:
        sitk_image.CopyInformation(reference_image)
    
    sitk.WriteImage(sitk_image, output_path)


def create_slices_from_volume(volume, num_slices=16, axis=0):
    """
    Extract evenly spaced slices from 3D volume
    
    Args:
        volume: 3D numpy array (D, H, W)
        num_slices: Number of slices to extract
        axis: Axis along which to slice (0, 1, or 2)
    
    Returns:
        slices: List of 2D slices
    """
    depth = volume.shape[axis]
    
    if num_slices >= depth:
        indices = range(depth)
    else:
        indices = np.linspace(0, depth - 1, num_slices, dtype=int)
    
    slices = []
    for idx in indices:
        if axis == 0:
            slice_2d = volume[idx, :, :]
        elif axis == 1:
            slice_2d = volume[:, idx, :]
        else:
            slice_2d = volume[:, :, idx]
        
        slices.append(slice_2d)
    
    return slices


def augment_image(image, augmentation_type='flip'):
    """
    Apply data augmentation
    
    Args:
        image: Input image
        augmentation_type: Type of augmentation
    
    Returns:
        augmented: Augmented image
    """
    if augmentation_type == 'flip_horizontal':
        return np.fliplr(image)
    elif augmentation_type == 'flip_vertical':
        return np.flipud(image)
    elif augmentation_type == 'rotate_90':
        return np.rot90(image)
    elif augmentation_type == 'rotate_180':
        return np.rot90(image, 2)
    elif augmentation_type == 'rotate_270':
        return np.rot90(image, 3)
    else:
        return image


def batch_preprocess(input_dir, output_dir, file_extension='.dcm', 
                     target_size=(256, 256), normalize_method='minmax'):
    """
    Batch preprocess all images in a directory
    
    Args:
        input_dir: Input directory containing medical images
        output_dir: Output directory for preprocessed images
        file_extension: File extension to process
        target_size: Target image size
        normalize_method: Normalization method
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = list(input_path.glob(f'*{file_extension}'))
    print(f"Found {len(files)} files to process")
    
    for i, file_path in enumerate(files):
        try:
            # Load image
            if file_extension == '.dcm':
                image, _ = load_dicom(file_path)
            elif file_extension in ['.nii', '.nii.gz']:
                volume, _ = load_nifti(file_path)
                # Take middle slice for 2D processing
                image = volume[volume.shape[0] // 2]
            else:
                image = np.array(Image.open(file_path).convert('L'))
            
            if image is None:
                continue
            
            # Preprocess
            processed = preprocess_for_training(
                image, 
                target_size=target_size,
                normalize_method=normalize_method
            )
            
            # Save as numpy array
            output_file = output_path / f"{file_path.stem}.npy"
            np.save(output_file, processed)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(files)} files")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Preprocessing complete! Saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess medical images')
    parser.add_argument('--input', type=str, required=True, help='Input directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--extension', type=str, default='.dcm', help='File extension')
    parser.add_argument('--size', type=int, default=256, help='Target image size')
    parser.add_argument('--normalize', type=str, default='minmax', 
                       choices=['minmax', 'zscore', 'percentile'])
    
    args = parser.parse_args()
    
    batch_preprocess(
        args.input,
        args.output,
        file_extension=args.extension,
        target_size=(args.size, args.size),
        normalize_method=args.normalize
    )
