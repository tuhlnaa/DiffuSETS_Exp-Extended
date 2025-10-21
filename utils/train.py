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
    loss_list = []
    unet.train()
    for _ in range(num_repetitions):
        for data, label in dataloader:
            # Prepare text embeddings
            text_embed = _prepare_text_embeddings(
                np.array(label['text_embed']), device
            )

            # Move latent to device
            latent = data.to(device)

            # t = torch.randperm(diffused_model.config.num_train_timesteps-2)[:latent.shape[0]] + 1 
            # compatible with larger batch size
            timesteps = torch.randint(1, diffusion_scheduler.config.num_train_timesteps - 1, (latent.shape[0],))

            noise = torch.randn(latent.shape, device=latent.device)
            noisy_latent = diffusion_scheduler.add_noise(latent, noise, timesteps)

            noisy_latent = noisy_latent.to(device)
            timesteps = timesteps.to(device)
            noise = noise.to(device)

            if use_conditions:
                gender = []
                age = label['age']
                hr = label['hr']
                conditions = {}
            
                for ch in label['gender']:
                    if ch == 'M':
                        gender.append(1)
                    else:
                        gender.append(0)
                gender = np.array(gender)
                gender = np.repeat(gender[:, np.newaxis], 1, axis=1)
                gender = np.repeat(gender[:, :, np.newaxis], 1, axis=2)
                gender = torch.Tensor(gender)
                gender = gender.to(device)
                conditions.update({'gender': gender})

                age = np.array(age)
                age = np.repeat(age[:, np.newaxis], 1, axis=1)
                age = np.repeat(age[:, :, np.newaxis], 1, axis=2)
                age = torch.Tensor(age)
                age = age.to(device)
                conditions.update({'age': age})

                hr = np.array(hr)
                hr = np.repeat(hr[:, np.newaxis], 1, axis=1)
                hr = np.repeat(hr[:, :, np.newaxis], 1, axis=2)
                hr = torch.Tensor(hr)
                hr = hr.to(device)
                conditions.update({'heart rate': hr})

                for key in conditions:
                    conditions[key] = conditions[key].to(device)
                noise_estim = unet(noisy_latent, timesteps, text_embed, conditions)
            else: 
                noise_estim = unet(noisy_latent, timesteps, text_embed)

            # Batchwise MSE loss 
            loss = F.mse_loss(noise_estim, noise, reduction='sum').div(noise.size(0))
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    return sum(loss_list) / len(loss_list)


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
