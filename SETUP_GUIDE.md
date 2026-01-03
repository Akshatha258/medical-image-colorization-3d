# 🚀 Complete Setup and Running Guide

This guide will walk you through setting up and running the Medical Image Colorization project from scratch.

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works too)
- 8GB+ RAM (16GB+ recommended)
- Git

## 🔧 Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Akshatha258/medical-image-colorization-3d.git

# Navigate to the project directory
cd medical-image-colorization-3d
```

## 🐍 Step 2: Create Virtual Environment

### On Linux/Mac:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### On Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

## 📦 Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (choose based on your system)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

**Note**: If you encounter issues with specific packages, install them individually:
```bash
pip install pydicom SimpleITK nibabel opencv-python matplotlib scikit-image albumentations lpips tqdm pyyaml
```

## 📁 Step 4: Prepare Your Data

### Option A: Use Sample Data (for testing)

Create dummy data to test the pipeline:

```bash
# Create data directories
mkdir -p data/raw data/processed/train data/processed/val data/processed/test

# Run this Python script to create dummy data
python << EOF
import numpy as np
from pathlib import Path

# Create dummy training data
train_dir = Path("data/processed/train")
val_dir = Path("data/processed/val")

for i in range(20):
    dummy_image = np.random.rand(256, 256).astype(np.float32)
    np.save(train_dir / f"image_{i}.npy", dummy_image)

for i in range(5):
    dummy_image = np.random.rand(256, 256).astype(np.float32)
    np.save(val_dir / f"image_{i}.npy", dummy_image)

print("✓ Dummy data created!")
EOF
```

### Option B: Use Real Medical Images

1. **Download medical imaging datasets:**
   - NIH Chest X-ray: https://www.kaggle.com/datasets/nih-chest-xrays/data
   - LIDC-IDRI: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
   - Or use your own DICOM/NIfTI files

2. **Place raw data:**
```bash
# Copy your DICOM or NIfTI files to:
cp /path/to/your/images/* data/raw/
```

3. **Preprocess the data:**
```bash
# For DICOM files
python utils/preprocessing.py --input data/raw --output data/processed/train --extension .dcm --size 256

# For NIfTI files
python utils/preprocessing.py --input data/raw --output data/processed/train --extension .nii --size 256

# For standard images (PNG, JPG)
python utils/preprocessing.py --input data/raw --output data/processed/train --extension .png --size 256
```

4. **Split data into train/val/test:**
```bash
python << EOF
import shutil
from pathlib import Path
import random

processed_dir = Path("data/processed/train")
val_dir = Path("data/processed/val")
test_dir = Path("data/processed/test")

val_dir.mkdir(exist_ok=True)
test_dir.mkdir(exist_ok=True)

files = list(processed_dir.glob("*.npy"))
random.shuffle(files)

# 70% train, 15% val, 15% test
val_split = int(len(files) * 0.7)
test_split = int(len(files) * 0.85)

for f in files[val_split:test_split]:
    shutil.move(str(f), str(val_dir / f.name))

for f in files[test_split:]:
    shutil.move(str(f), str(test_dir / f.name))

print(f"✓ Data split complete!")
print(f"  Train: {len(list(processed_dir.glob('*.npy')))} files")
print(f"  Val: {len(list(val_dir.glob('*.npy')))} files")
print(f"  Test: {len(list(test_dir.glob('*.npy')))} files")
EOF
```

## 🎯 Step 5: Configure Training

Edit `training/config.py` to customize your training:

```python
# Key parameters to adjust:
BATCH_SIZE = 4          # Reduce if out of memory
NUM_EPOCHS = 100        # Number of training epochs
LEARNING_RATE = 1e-4    # Learning rate
IMAGE_SIZE = 256        # Image size (256 or 512)
MODEL_TYPE = 'attention' # 'full', 'attention', or 'lightweight'
```

## 🏋️ Step 6: Train the Model

### Quick Start (with dummy data):
```bash
python training/train.py
```

### Full Training:
```bash
# Train with default settings
python training/train.py

# Monitor training in real-time
# The script will show progress bars and metrics
```

### Training Output:
```
Training Configuration
==================================================
Model Type: attention
Batch Size: 4
Learning Rate: 0.0001
Num Epochs: 100
...
==================================================

Epoch 1/100
----------------------------------------------------------------------
Training: 100%|████████████| 5/5 [00:10<00:00,  2.00s/it]
Train Loss: 0.4523 | L1: 0.3421 | Perceptual: 0.0892 | SSIM: 0.0210
Validation: 100%|████████████| 2/2 [00:03<00:00,  1.50s/it]
Val Loss: 0.4123 | PSNR: 24.56 | SSIM: 0.7823
✓ New best model saved! (Loss: 0.4123)
```

### Resume Training:
```bash
# Edit config.py and set:
# RESUME_CHECKPOINT = "checkpoints/checkpoint_epoch_50.pth"

python training/train.py
```

## 🔮 Step 7: Run Inference

### On a Single Image:
```bash
python inference/predict.py \
    --checkpoint checkpoints/best_model.pth \
    --input data/test/sample_image.npy \
    --output results/inference \
    --device cuda
```

### On a Directory:
```bash
python inference/predict.py \
    --checkpoint checkpoints/best_model.pth \
    --input data/test/ \
    --output results/inference \
    --device cuda
```

### Using CPU (if no GPU):
```bash
python inference/predict.py \
    --checkpoint checkpoints/best_model.pth \
    --input data/test/sample_image.npy \
    --output results/inference \
    --device cpu
```

## 📊 Step 8: View Results

After training and inference, check these directories:

```bash
# Training curves and metrics
results/training_curves.png

# Sample validation outputs
results/samples/val_sample.png

# Inference results
results/inference/
├── image_001_colorized.png
├── image_001_output.png
├── image_002_colorized.png
└── ...

# Model checkpoints
checkpoints/
├── best_model.pth
├── final_model.pth
└── checkpoint_epoch_*.pth
```

## 🐛 Troubleshooting

### Issue 1: CUDA Out of Memory
```bash
# Solution: Reduce batch size in training/config.py
BATCH_SIZE = 2  # or even 1
```

### Issue 2: No module named 'models'
```bash
# Solution: Make sure you're in the project root directory
cd medical-image-colorization-3d
python training/train.py
```

### Issue 3: PyTorch not detecting GPU
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 4: Missing dependencies
```bash
# Install missing packages individually
pip install package_name

# Or reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

### Issue 5: No data found
```bash
# Make sure data is in correct format and location
ls data/processed/train/  # Should show .npy files
ls data/processed/val/    # Should show .npy files
```

## 📈 Monitoring Training

### Option 1: Terminal Output
Training progress is shown in real-time with progress bars and metrics.

### Option 2: TensorBoard (Optional)
```bash
# Install tensorboard
pip install tensorboard

# Run tensorboard
tensorboard --logdir logs/

# Open browser to http://localhost:6006
```

### Option 3: Weights & Biases (Optional)
```bash
# Install wandb
pip install wandb

# Login
wandb login

# Enable in config.py
USE_WANDB = True
WANDB_PROJECT = "medical-image-colorization"

# Run training
python training/train.py
```

## 🎓 Next Steps

1. **Experiment with hyperparameters** in `training/config.py`
2. **Try different model architectures** (full, attention, lightweight)
3. **Use real medical imaging datasets**
4. **Fine-tune on specific imaging modalities** (CT, MRI, X-ray)
5. **Implement 3D reconstruction** for volumetric data

## 📚 Additional Resources

- **Documentation**: Check individual Python files for detailed docstrings
- **Examples**: See `notebooks/` directory (coming soon)
- **Issues**: Report bugs on GitHub Issues
- **Datasets**: 
  - NIH Chest X-ray: https://www.kaggle.com/datasets/nih-chest-xrays/data
  - TCIA: https://www.cancerimagingarchive.net/

## 💡 Quick Commands Reference

```bash
# Setup
git clone https://github.com/Akshatha258/medical-image-colorization-3d.git
cd medical-image-colorization-3d
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Preprocess data
python utils/preprocessing.py --input data/raw --output data/processed/train

# Train
python training/train.py

# Inference
python inference/predict.py --checkpoint checkpoints/best_model.pth --input data/test/

# View results
ls results/
```

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] Data prepared and in correct directories
- [ ] Config file reviewed and customized
- [ ] Training runs without errors
- [ ] Checkpoints are being saved
- [ ] Inference produces output images
- [ ] Results look reasonable

## 🎉 Success!

If you've completed all steps, you should now have:
- ✅ A trained medical image colorization model
- ✅ Saved checkpoints for future use
- ✅ Colorized output images
- ✅ Training curves and metrics

Happy training! 🚀
