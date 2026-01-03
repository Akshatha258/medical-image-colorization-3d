# ⚡ Quick Start Guide

Get up and running in 5 minutes!

## 🚀 Fast Setup

```bash
# 1. Clone and enter directory
git clone https://github.com/Akshatha258/medical-image-colorization-3d.git
cd medical-image-colorization-3d

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 4. Create dummy data for testing
python -c "
import numpy as np
from pathlib import Path

for split in ['train', 'val']:
    dir_path = Path(f'data/processed/{split}')
    dir_path.mkdir(parents=True, exist_ok=True)
    num_samples = 20 if split == 'train' else 5
    for i in range(num_samples):
        np.save(dir_path / f'image_{i}.npy', np.random.rand(256, 256).astype('float32'))
print('✓ Dummy data created!')
"

# 5. Train the model
python training/train.py

# 6. Run inference
python inference/predict.py \
    --checkpoint checkpoints/best_model.pth \
    --input data/processed/val/ \
    --output results/inference
```

## 📊 Expected Output

After training, you'll see:
```
Epoch 1/100
----------------------------------------------------------------------
Training: 100%|████████████| 5/5 [00:10<00:00]
Train Loss: 0.4523 | L1: 0.3421 | Perceptual: 0.0892
Val Loss: 0.4123 | PSNR: 24.56 | SSIM: 0.7823
✓ New best model saved!
```

Results will be in:
- `checkpoints/` - Trained models
- `results/` - Visualizations and outputs

## 🎯 Next Steps

1. **Use real data**: Replace dummy data with actual medical images
2. **Adjust config**: Edit `training/config.py` for your needs
3. **Monitor training**: Check `results/training_curves.png`
4. **Run inference**: Test on new images

## 💡 Common Commands

```bash
# Train with custom config
python training/train.py

# Resume training
# Edit config.py: RESUME_CHECKPOINT = "checkpoints/checkpoint_epoch_50.pth"
python training/train.py

# Inference on single image
python inference/predict.py --checkpoint checkpoints/best_model.pth --input image.png --output results/

# Preprocess real data
python utils/preprocessing.py --input data/raw --output data/processed/train --extension .dcm
```

## 🆘 Quick Troubleshooting

**Out of memory?**
```python
# In training/config.py, reduce:
BATCH_SIZE = 2
```

**No GPU?**
```bash
# Use CPU
python inference/predict.py --checkpoint checkpoints/best_model.pth --input image.png --device cpu
```

**Import errors?**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

For detailed instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)
