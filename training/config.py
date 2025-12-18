"""
Configuration file for training
"""

import torch
from pathlib import Path


class Config:
    """Training configuration"""
    
    # Paths
    DATA_DIR = Path("data/processed")
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEST_DIR = DATA_DIR / "test"
    CHECKPOINT_DIR = Path("checkpoints")
    LOG_DIR = Path("logs")
    RESULTS_DIR = Path("results")
    
    # Model
    MODEL_TYPE = 'attention'  # 'full', 'attention', or 'lightweight'
    USE_ATTENTION = True
    BASE_FEATURES_3D = 32
    NUM_SLICES = 16
    
    # Training
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Optimizer
    OPTIMIZER = 'adam'  # 'adam', 'adamw', or 'sgd'
    BETAS = (0.9, 0.999)
    
    # Scheduler
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'cosine'  # 'cosine', 'step', or 'plateau'
    SCHEDULER_PATIENCE = 10  # For ReduceLROnPlateau
    SCHEDULER_FACTOR = 0.5
    T_MAX = 50  # For CosineAnnealingLR
    
    # Loss weights
    LOSS_L1_WEIGHT = 1.0
    LOSS_PERCEPTUAL_WEIGHT = 0.1
    LOSS_SSIM_WEIGHT = 0.5
    LOSS_3D_WEIGHT = 0.3
    
    # Data
    IMAGE_SIZE = 256
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Augmentation
    USE_AUGMENTATION = True
    HORIZONTAL_FLIP_PROB = 0.5
    VERTICAL_FLIP_PROB = 0.5
    ROTATION_PROB = 0.5
    
    # Training settings
    GRADIENT_CLIP = 1.0
    MIXED_PRECISION = True  # Use automatic mixed precision
    ACCUMULATION_STEPS = 1  # Gradient accumulation
    
    # Validation
    VAL_FREQUENCY = 1  # Validate every N epochs
    SAVE_FREQUENCY = 5  # Save checkpoint every N epochs
    
    # Logging
    LOG_FREQUENCY = 10  # Log every N batches
    USE_WANDB = False  # Use Weights & Biases
    WANDB_PROJECT = "medical-image-colorization"
    WANDB_ENTITY = None
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Reproducibility
    SEED = 42
    
    # Early stopping
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 15
    
    # Resume training
    RESUME_CHECKPOINT = None  # Path to checkpoint to resume from
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        cls.VAL_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Print configuration"""
        print("\n" + "="*50)
        print("Training Configuration")
        print("="*50)
        print(f"Model Type: {cls.MODEL_TYPE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Num Epochs: {cls.NUM_EPOCHS}")
        print(f"Image Size: {cls.IMAGE_SIZE}")
        print(f"Num Slices: {cls.NUM_SLICES}")
        print(f"Device: {cls.DEVICE}")
        print(f"Mixed Precision: {cls.MIXED_PRECISION}")
        print(f"Use Augmentation: {cls.USE_AUGMENTATION}")
        print(f"Use Scheduler: {cls.USE_SCHEDULER}")
        print(f"Use Early Stopping: {cls.USE_EARLY_STOPPING}")
        print("="*50 + "\n")


# Create a default config instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    config.create_dirs()
    config.print_config()
    print("Configuration test complete!")
