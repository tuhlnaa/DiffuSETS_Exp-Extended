import os
import time
import torch
import wandb

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from typing import Dict, Any


def _prepare_conditions(label: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """Prepare conditional inputs for the model.
    
    Args:
        label: Dictionary containing 'gender', 'age', and 'hr' keys
        device: Target device
        
    Returns:
        Dictionary of condition tensors, each of shape (B, 1, 1)
    """
    # Gender encoding: 'M' -> 1, otherwise -> 0
    gender = torch.tensor(
        [1 if g == 'M' else 0 for g in label['gender']], 
        dtype=torch.float32,
        device=device
    ).unsqueeze(1).unsqueeze(2)  # (B, 1, 1)

    # Age and heart rate
    age = label['age'].unsqueeze(1).unsqueeze(2).to(device)
    hr = label['hr'].to(torch.float32).unsqueeze(1).unsqueeze(2).to(device)

    return {
        'gender': gender,
        'age': age,
        'heart_rate': hr,  # Consistent naming with underscore
    }


def _prepare_text_embeddings(text_embed: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert text embeddings to tensor format.
    
    Args:
        text_embed: Text embeddings array of shape (Batch, Dimension)
        device: Target device
        
    Returns:
        Tensor of shape (D, 1, B)
    """
    text_embed = torch.from_numpy(text_embed).float()
    text_embed = text_embed.t().unsqueeze(1)  # (D, 1, B)
    return text_embed.to(device)


def train_epoch(
    dataloader: DataLoader,
    unet: nn.Module,
    diffusion_scheduler: Any,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    use_conditions: bool = False,
    num_repetitions: int = 1,
) -> float:
    """Train for one epoch.
    
    Args:
        dataloader: Training data loader
        unet: U-Net model
        diffusion_scheduler: Diffusion noise scheduler
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        device: Training device
        use_conditions: Whether to use conditional inputs
        num_repetitions: Number of times to repeat the epoch
        
    Returns:
        Average loss over the epoch
    """
    unet.train()
    total_loss = 0.0
    num_batches = 0

    for _ in range(num_repetitions):
        for data, label in dataloader:
            # Move data to device
            latent = data.to(device)
            batch_size = latent.shape[0]

            # Prepare text embeddings
            text_embed = _prepare_text_embeddings(
                np.array(label['text_embed']), device
            )

            # Sample random timesteps (with replacement - samples can repeat)
            max_timesteps = diffusion_scheduler.config.num_train_timesteps
            timesteps = torch.randint(1, max_timesteps - 1, (batch_size,)).to(device)

            # Alternative: sample without replacement (all timesteps unique within batch)
            # timesteps = torch.randperm(max_timesteps - 2)[:batch_size] + 1
            # timesteps = timesteps.to(device)

            # Add noise to latents
            noise = torch.randn_like(latent).to(device)
            noisy_latent = diffusion_scheduler.add_noise(latent, noise, timesteps).to(device)

            # Prepare conditions if needed
            if use_conditions:
                conditions = _prepare_conditions(label, device)
                # Forward pass
                noise_pred = unet(noisy_latent, timesteps, text_embed, conditions)
            else:
                noise_pred = unet(noisy_latent, timesteps, text_embed)

            # Compute loss (mean MSE per sample)
            loss = F.mse_loss(noise_pred, noise, reduction='sum') / batch_size
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train_model(
    config: Dict[str, Any],
    save_dir: str,
    dataloader: DataLoader,
    diffusion_scheduler: Any,
    unet: nn.Module,
    hyperparams: Dict[str, Any],
    logger: Any,
) -> None:
    """Train the diffusion model.
    
    Args:
        config: Configuration dictionary with 'device' and 'condition' keys
        save_dir: Directory to save model checkpoints
        dataloader: Training data loader
        diffusion_scheduler: Diffusion noise scheduler
        unet: U-Net model to train
        hyperparams: Hyperparameters dict with 'lr', 'epochs', 'checkpoint_freq'
        logger: Logger instance
    """
    # Setup device and model
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    unet = unet.to(device)

    # Setup optimizer and scheduler
    total_steps = hyperparams['epochs'] * len(dataloader)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=hyperparams['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=0.1 * hyperparams['lr']
    )

    # Training loop
    best_loss = float('inf')
    checkpoint_freq = hyperparams.get('checkpoint_freq', 50)
    start_time = time.time()
    
    for epoch in range(1, hyperparams['epochs'] + 1):
        epoch_start = time.time()
        
        # Train one epoch
        avg_loss = train_epoch(
            dataloader=dataloader,
            unet=unet,
            diffusion_scheduler=diffusion_scheduler,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_conditions=config['condition'],
            num_repetitions=1,
        )
        
        # Logging
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f'Epoch: {epoch}/{hyperparams["epochs"]}, '
            f'Loss: {avg_loss:.4f}, LR: {current_lr:.6f}'
        )
        wandb.log({"train_loss": avg_loss, "learning_rate": current_lr}, step=epoch)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, 'unet_best.pth')
            torch.save(unet.state_dict(), best_path)
            logger.info(f'Epoch {epoch}: New best model saved (loss: {avg_loss:.4f})')
        
        # Periodic checkpoints
        if epoch % checkpoint_freq == 0:
            checkpoint_path = os.path.join(save_dir, f'unet_epoch_{epoch}.pth')
            torch.save(unet.state_dict(), checkpoint_path)
            logger.info(f'Checkpoint saved: unet_epoch_{epoch}.pth')
        
        # Time tracking
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        logger.info(f"Epoch time: {epoch_time:.2f}s, Total time: {total_time:.2f}s")
