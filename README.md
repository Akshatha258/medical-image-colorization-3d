# 2D to 3D Medical Image Colorization

A deep learning project for converting grayscale 2D medical images into colorized 3D representations using U-Net and 3D CNN architecture.

## 🎯 Project Overview

This project implements a hybrid deep learning approach to:
- Colorize grayscale medical images (X-rays, CT scans, MRI)
- Generate 3D volumetric representations from 2D slices
- Provide accurate medical image visualization

## 🏗️ Architecture

- **Stage 1**: U-Net for 2D colorization (LAB color space)
- **Stage 2**: 3D CNN for depth estimation and volumetric reconstruction
- **Loss Functions**: Perceptual loss, L1 reconstruction, SSIM

## 📁 Project Structure

```
medical-image-colorization-3d/
├── data/
│   ├── raw/              # Original DICOM/images
│   ├── processed/        # Preprocessed data
│   └── augmented/        # Augmented dataset
├── models/
│   ├── unet.py          # U-Net architecture
│   ├── reconstruction_3d.py  # 3D reconstruction network
│   └── colorization_model.py # Complete model
├── utils/
│   ├── preprocessing.py  # Data preprocessing
│   ├── visualization.py  # Visualization tools
│   ├── metrics.py       # Evaluation metrics
│   └── data_loader.py   # Dataset loaders
├── training/
│   ├── train.py         # Training script
│   ├── validate.py      # Validation script
│   └── config.py        # Configuration
├── inference/
│   └── predict.py       # Inference script
├── notebooks/
│   └── experiments.ipynb # Jupyter experiments
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Akshatha258/medical-image-colorization-3d.git
cd medical-image-colorization-3d

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Place your medical images in data/raw/
python utils/preprocessing.py --input data/raw --output data/processed
```

### Training

```bash
# Train the model
python training/train.py --config training/config.py --epochs 100
```

### Inference

```bash
# Generate colorized 3D images
python inference/predict.py --input path/to/image.dcm --output results/
```

## 📊 Datasets

Recommended datasets:
- NIH Chest X-ray Dataset
- LIDC-IDRI (Lung CT)
- BraTS (Brain MRI)
- Medical Segmentation Decathlon

## 🛠️ Technical Stack

- **Framework**: PyTorch
- **Medical Imaging**: PyDICOM, SimpleITK, NiBabel
- **Visualization**: Matplotlib, VTK, Plotly
- **Training**: PyTorch Lightning, Weights & Biases

## 📈 Evaluation Metrics

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- 3D IoU (Intersection over Union)

## 🔬 Model Details

### U-Net Colorization Network
- Encoder: 5 downsampling blocks
- Decoder: 5 upsampling blocks with skip connections
- Output: LAB color space (L channel from input, predict AB)

### 3D Reconstruction Network
- Input: Colorized 2D slices
- 3D Convolutions with residual connections
- Output: Volumetric 3D representation

## 📝 License

MIT License

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaboration, please open an issue.

## 🙏 Acknowledgments

- U-Net architecture based on Ronneberger et al.
- Medical imaging preprocessing inspired by MONAI framework
