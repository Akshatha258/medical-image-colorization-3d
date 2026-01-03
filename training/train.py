"""
Main training script for medical image colorization
"""

import os
import sys
import torch
import torch.nn as nn
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, config):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    l1_losses = AverageMeter()
    perceptual_losses = AverageMeter()
    ssim_losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc='Training')
    
    for batch_idx, batch in enumerate(pbar):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass with mixed precision
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
        
        # Update metrics
        losses.update(loss_dict['total'], inputs.size(0))
        l1_losses.update(loss_dict['l1'], inputs.size(0))
        perceptual_losses.update(loss_dict['perceptual'], inputs.size(0))
        ssim_losses.update(loss_dict['ssim'], inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'l1': f'{l1_losses.avg:.4f}',
            'perc': f'{perceptual_losses.avg:.4f}',
            'ssim': f'{ssim_losses.avg:.4f}'
        })
    
    return {
        'loss': losses.avg,
        'l1': l1_losses.avg,
        'perceptual': perceptual_losses.avg,
        'ssim': ssim_losses.avg
    }


def validate(model, dataloader, criterion, metrics_calculator, device, config):
    """Validate the model"""
    model.eval()
    
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc='Validation')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model.colorize_only(inputs)
            loss, loss_dict = criterion(outputs, targets)
            
            # Calculate metrics
            psnr = metrics_calculator.calculate_psnr(outputs, targets)
            ssim = metrics_calculator.calculate_ssim(outputs, targets)
            
            # Update meters
            losses.update(loss_dict['total'], inputs.size(0))
            psnr_meter.update(psnr, inputs.size(0))
            ssim_meter.update(ssim, inputs.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'psnr': f'{psnr_meter.avg:.2f}',
                'ssim': f'{ssim_meter.avg:.4f}'
            })
            
            # Save sample visualizations
            if batch_idx == 0:
                save_path = config.RESULTS_DIR / 'samples' / f'val_sample.png'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                visualize_comparison(
                    inputs[0].cpu(),
                    outputs[0].cpu(),
                    targets[0].cpu(),
                    save_path=save_path
                )
    
    return {
        'loss': losses.avg,
        'psnr': psnr_meter.avg,
        'ssim': ssim_meter.avg
    }


def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, config, filename='checkpoint.pth'):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_loss': best_loss,
        'config': {
            'MODEL_TYPE': config.MODEL_TYPE,
            'IMAGE_SIZE': config.IMAGE_SIZE,
            'NUM_SLICES': config.NUM_SLICES,
        }
    }
    
    save_path = config.CHECKPOINT_DIR / filename
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, scheduler, config):
    """Load training checkpoint"""
    checkpoint_path = config.RESUME_CHECKPOINT
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch + 1}")
    
    return epoch + 1, best_loss


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
    
    # Model
    print("\nInitializing model...")
    model = get_model(
        model_type=config.MODEL_TYPE,
        use_attention=config.USE_ATTENTION,
        base_features_3d=config.BASE_FEATURES_3D,
        num_slices=config.NUM_SLICES
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
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
    if config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            betas=config.BETAS,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            betas=config.BETAS,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=config.WEIGHT_DECAY
        )
    
    # Scheduler
    scheduler = None
    if config.USE_SCHEDULER:
        if config.SCHEDULER_TYPE == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.T_MAX
            )
        elif config.SCHEDULER_TYPE == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        elif config.SCHEDULER_TYPE == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.SCHEDULER_FACTOR,
                patience=config.SCHEDULER_PATIENCE
            )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.MIXED_PRECISION else None
    
    # Metrics calculator
    metrics_calculator = MetricsCalculator(device=device)
    
    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    if config.RESUME_CHECKPOINT:
        start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, config)
    
    # Training history
    train_losses = []
    val_losses = []
    val_psnr_history = []
    val_ssim_history = []
    
    # Early stopping
    patience_counter = 0
    
    # Training loop
    print("\nStarting training...")
    print("=" * 70)
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 70)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config
        )
        train_losses.append(train_metrics['loss'])
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"L1: {train_metrics['l1']:.4f} | "
              f"Perceptual: {train_metrics['perceptual']:.4f} | "
              f"SSIM: {train_metrics['ssim']:.4f}")
        
        # Validate
        if (epoch + 1) % config.VAL_FREQUENCY == 0:
            val_metrics = validate(
                model, val_loader, criterion, metrics_calculator, device, config
            )
            val_losses.append(val_metrics['loss'])
            val_psnr_history.append(val_metrics['psnr'])
            val_ssim_history.append(val_metrics['ssim'])
            
            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"PSNR: {val_metrics['psnr']:.2f} | "
                  f"SSIM: {val_metrics['ssim']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_loss, config,
                    filename='best_model.pth'
                )
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
            if config.SCHEDULER_TYPE == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_FREQUENCY == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_loss, config,
                filename=f'checkpoint_epoch_{epoch + 1}.pth'
            )
        
        # Print learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, config.NUM_EPOCHS - 1, best_loss, config,
        filename='final_model.pth'
    )
    
    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(
        train_losses,
        val_losses,
        metrics_history={'psnr': val_psnr_history, 'ssim': val_ssim_history},
        save_path=config.RESULTS_DIR / 'training_curves.png'
    )
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Models saved to: {config.CHECKPOINT_DIR}")
    print(f"Results saved to: {config.RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
