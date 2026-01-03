"""
Simple training script for medical image colorization
Fixed version without use_attention parameter conflict
"""

import os
import sys
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from pathlib import Path
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models import get_model
from utils.data_loader import get_dataloader
from utils.metrics import MetricsCalculator, AverageMeter
from utils.visualization import visualize_comparison, plot_training_curves
from training.config import Config
from training.losses import CombinedLoss


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, config):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc='Training')
    
    for batch_idx, batch in enumerate(pbar):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        if config.MIXED_PRECISION:
            with autocast():
                outputs = model.colorize_only(inputs)
                loss, loss_dict = criterion(outputs, targets)
        else:
            outputs = model.colorize_only(inputs)
            loss, loss_dict = criterion(outputs, targets)
        
        # Backward pass
        if config.MIXED_PRECISION:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                optimizer.step()
                optimizer.zero_grad()
        
        losses.update(loss_dict['total'], inputs.size(0))
        pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    return {'loss': losses.avg}


def validate(model, dataloader, criterion, metrics_calculator, device, config):
    """Validate the model"""
    model.eval()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc='Validation')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model.colorize_only(inputs)
            loss, loss_dict = criterion(outputs, targets)
            
            psnr = metrics_calculator.calculate_psnr(outputs, targets)
            
            losses.update(loss_dict['total'], inputs.size(0))
            psnr_meter.update(psnr, inputs.size(0))
            
            pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'psnr': f'{psnr_meter.avg:.2f}'})
            
            # Save sample
            if batch_idx == 0:
                save_path = config.RESULTS_DIR / 'samples' / f'val_sample.png'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                visualize_comparison(
                    inputs[0].cpu(),
                    outputs[0].cpu(),
                    targets[0].cpu(),
                    save_path=save_path
                )
    
    return {'loss': losses.avg, 'psnr': psnr_meter.avg}


def save_checkpoint(model, optimizer, epoch, best_loss, config, filename='checkpoint.pth'):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }
    save_path = config.CHECKPOINT_DIR / filename
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def main():
    """Main training function"""
    
    # Configuration
    config = Config()
    config.create_dirs()
    config.print_config()
    
    # Set seed
    set_seed(config.SEED)
    
    # Device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Model - FIXED: Removed use_attention parameter
    print("\nInitializing model...")
    model = get_model(
        model_type=config.MODEL_TYPE,
        base_features_3d=config.BASE_FEATURES_3D,
        num_slices=config.NUM_SLICES
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Data loaders
    print("\nLoading data...")
    train_loader = get_dataloader(
        data_dir=config.TRAIN_DIR,
        mode='train',
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        image_size=config.IMAGE_SIZE,
        dataset_type='2d'
    )
    
    val_loader = get_dataloader(
        data_dir=config.VAL_DIR,
        mode='val',
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        image_size=config.IMAGE_SIZE,
        dataset_type='2d'
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Loss function
    criterion = CombinedLoss(
        l1_weight=config.LOSS_L1_WEIGHT,
        perceptual_weight=config.LOSS_PERCEPTUAL_WEIGHT,
        ssim_weight=config.LOSS_SSIM_WEIGHT
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = None
    if config.USE_SCHEDULER:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_MAX)
    
    # Mixed precision scaler
    scaler = GradScaler() if config.MIXED_PRECISION else None
    
    # Metrics calculator
    metrics_calculator = MetricsCalculator(device=device)
    
    # Training history
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    print("\nStarting training...")
    print("=" * 70)
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 70)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config
        )
        train_losses.append(train_metrics['loss'])
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        
        # Validate
        if (epoch + 1) % config.VAL_FREQUENCY == 0:
            val_metrics = validate(
                model, val_loader, criterion, metrics_calculator, device, config
            )
            val_losses.append(val_metrics['loss'])
            print(f"Val Loss: {val_metrics['loss']:.4f} | PSNR: {val_metrics['psnr']:.2f}")
            
            # Save best model
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                save_checkpoint(model, optimizer, epoch, best_loss, config, 'best_model.pth')
                print(f"✓ New best model saved! (Loss: {best_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if config.USE_EARLY_STOPPING and patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_FREQUENCY == 0:
            save_checkpoint(model, optimizer, epoch, best_loss, config, f'checkpoint_epoch_{epoch + 1}.pth')
    
    # Save final model
    save_checkpoint(model, optimizer, config.NUM_EPOCHS - 1, best_loss, config, 'final_model.pth')
    
    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(train_losses, val_losses, save_path=config.RESULTS_DIR / 'training_curves.png')
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
