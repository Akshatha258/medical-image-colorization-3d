"""
Dataset and DataLoader utilities for medical image colorization
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .preprocessing import load_dicom, load_nifti, normalize_image, rgb_to_lab


class MedicalImageDataset(Dataset):
    """
    Dataset for medical image colorization
    
    For training, this can load paired grayscale-color images or
    use synthetic colorization for unpaired data
    """
    
    def __init__(self, data_dir, mode='train', transform=None, 
                 num_slices=16, image_size=256):
        """
        Args:
            data_dir: Directory containing preprocessed images
            mode: 'train', 'val', or 'test'
            transform: Albumentations transform
            num_slices: Number of slices for 3D reconstruction
            image_size: Image size (assumes square images)
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        self.num_slices = num_slices
        self.image_size = image_size
        
        # Load file paths
        self.image_files = sorted(list(self.data_dir.glob('*.npy')))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")
        
        print(f"Loaded {len(self.image_files)} images for {mode} mode")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = np.load(image_path).astype(np.float32)
        
        # Ensure 2D
        if image.ndim == 3:
            image = image[image.shape[0] // 2]  # Take middle slice
        
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        
        # For training, we need both input (L channel) and target (AB channels)
        # Since we have grayscale images, we'll use the same image as both
        # In practice, you'd have paired color images or use a different strategy
        
        return {
            'input': image,  # Grayscale (L channel)
            'target': image.repeat(2, 1, 1),  # Placeholder for AB channels
            'filename': image_path.name
        }


class MedicalImage3DDataset(Dataset):
    """
    Dataset for 3D medical image volumes
    """
    
    def __init__(self, data_dir, mode='train', num_slices=16, image_size=256):
        """
        Args:
            data_dir: Directory containing 3D volumes
            mode: 'train', 'val', or 'test'
            num_slices: Number of slices to extract
            image_size: Target image size
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.num_slices = num_slices
        self.image_size = image_size
        
        # Load volume file paths
        self.volume_files = sorted(list(self.data_dir.glob('*.nii*')))
        
        if len(self.volume_files) == 0:
            raise ValueError(f"No NIfTI files found in {data_dir}")
        
        print(f"Loaded {len(self.volume_files)} volumes for {mode} mode")
    
    def __len__(self):
        return len(self.volume_files)
    
    def __getitem__(self, idx):
        # Load volume
        volume_path = self.volume_files[idx]
        volume, _ = load_nifti(str(volume_path))
        
        if volume is None:
            # Return dummy data if loading fails
            return {
                'input': torch.zeros(1, self.num_slices, self.image_size, self.image_size),
                'filename': volume_path.name
            }
        
        # Normalize
        volume = normalize_image(volume, method='minmax')
        
        # Extract slices
        depth = volume.shape[0]
        if self.num_slices >= depth:
            indices = range(depth)
            # Pad if necessary
            if self.num_slices > depth:
                padding = self.num_slices - depth
                volume = np.pad(volume, ((0, padding), (0, 0), (0, 0)), mode='edge')
        else:
            indices = np.linspace(0, depth - 1, self.num_slices, dtype=int)
        
        slices = []
        for i in range(self.num_slices):
            if i < len(indices):
                slice_2d = volume[indices[i]]
            else:
                slice_2d = volume[-1]  # Use last slice for padding
            
            # Resize if needed
            if slice_2d.shape != (self.image_size, self.image_size):
                from PIL import Image
                slice_2d = np.array(
                    Image.fromarray(slice_2d).resize(
                        (self.image_size, self.image_size), 
                        Image.BILINEAR
                    )
                )
            
            slices.append(slice_2d)
        
        # Stack slices: (num_slices, H, W)
        volume_tensor = torch.from_numpy(np.stack(slices, axis=0)).float()
        
        # Add channel dimension: (1, num_slices, H, W)
        volume_tensor = volume_tensor.unsqueeze(0)
        
        return {
            'input': volume_tensor,
            'filename': volume_path.name
        }


def get_transforms(mode='train', image_size=256):
    """
    Get augmentation transforms
    
    Args:
        mode: 'train' or 'val'/'test'
        image_size: Target image size
    
    Returns:
        transform: Albumentations transform
    """
    if mode == 'train':
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
        ])
    
    return transform


def get_dataloader(data_dir, mode='train', batch_size=8, num_workers=4,
                   num_slices=16, image_size=256, dataset_type='2d'):
    """
    Get DataLoader for training/validation/testing
    
    Args:
        data_dir: Data directory
        mode: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of worker processes
        num_slices: Number of slices for 3D
        image_size: Image size
        dataset_type: '2d' or '3d'
    
    Returns:
        dataloader: PyTorch DataLoader
    """
    transform = get_transforms(mode=mode, image_size=image_size)
    
    if dataset_type == '2d':
        dataset = MedicalImageDataset(
            data_dir=data_dir,
            mode=mode,
            transform=transform,
            num_slices=num_slices,
            image_size=image_size
        )
    else:
        dataset = MedicalImage3DDataset(
            data_dir=data_dir,
            mode=mode,
            num_slices=num_slices,
            image_size=image_size
        )
    
    shuffle = (mode == 'train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train')
    )
    
    return dataloader


if __name__ == "__main__":
    # Test the dataset
    print("Testing MedicalImageDataset...")
    
    # Create dummy data
    dummy_dir = Path("data/processed")
    dummy_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(5):
        dummy_image = np.random.rand(256, 256).astype(np.float32)
        np.save(dummy_dir / f"image_{i}.npy", dummy_image)
    
    # Test dataloader
    dataloader = get_dataloader(
        data_dir=dummy_dir,
        mode='train',
        batch_size=2,
        num_workers=0,
        dataset_type='2d'
    )
    
    for batch in dataloader:
        print(f"Input shape: {batch['input'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"Filenames: {batch['filename']}")
        break
    
    print("\nDataset test complete!")
