import argparse
import json
import logging
import torch

from diffusers import DDPMScheduler
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, Any

# Import custom modules
from dataset.mimic_iv_ecg_dataset import DictDataset
from unet.unet_conditional import ECGConditional
from unet.unet_nocondition import ECGNoCondition
from utils.train import train_model
from utils.train_novae import train_model_novae
from vae.vae_model import VAEDecoder


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DiffuSETS Training')
    parser.add_argument('config', type=str, help='Path to training configuration file')
    return parser.parse_args()


def setup_logger(log_dir: Path, exp_name: str) -> logging.Logger:
    """Configure and return a logger with file and console handlers."""
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    log_file = log_dir / 'train.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_next_experiment_dir(checkpoints_root: Path, exp_type: str) -> Path:
    """Find the next available experiment directory number."""
    if not checkpoints_root.exists():
        checkpoints_root.mkdir(parents=True, exist_ok=True)
    
    max_idx = 0
    prefix = f"{exp_type}_"
    
    for item in checkpoints_root.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            try:
                idx = int(item.name.split('_')[-1])
                max_idx = max(max_idx, idx)
            except (ValueError, IndexError):
                continue
    
    next_dir = checkpoints_root / f"{exp_type}_{max_idx + 1}"
    next_dir.mkdir(parents=True, exist_ok=True)
    
    return next_dir


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return configuration from JSON file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_unet(hyperparams: Dict[str, Any], meta: Dict[str, Any]) -> torch.nn.Module:
    """Build and return the appropriate UNet model."""
    use_vae_latent = meta['vae_latent']
    n_channels = 4 if use_vae_latent else 12
    
    num_train_steps = hyperparams['num_train_steps']
    kernel_size = hyperparams['unet_kernel_size']
    num_levels = hyperparams['unet_num_level']
    
    if meta['condition']:
        return ECGConditional(
            num_train_steps,
            kernel_size=kernel_size,
            num_levels=num_levels,
            n_channels=n_channels
        )
    else:
        return ECGNoCondition(
            num_train_steps,
            kernel_size=kernel_size,
            num_levels=num_levels,
            n_channels=n_channels
        )


def build_scheduler(hyperparams: Dict[str, Any]) -> DDPMScheduler:
    """Build and return the diffusion scheduler."""
    return DDPMScheduler(
        num_train_timesteps=hyperparams['num_train_steps'],
        beta_start=hyperparams['beta_start'],
        beta_end=hyperparams['beta_end']
    )


def main() -> None:
    """Main training entry point."""
    args = parse_args()
    config = load_config(args.config)
    
    # Extract configuration sections
    meta = config['meta']
    dependencies = config['dependencies']
    hyperparams = config['hyper_para']
    
    # Setup experiment directory
    checkpoints_root = Path(dependencies['checkpoints_dir'])
    save_weights_path = get_next_experiment_dir(checkpoints_root, meta['exp_type'])
    
    # Setup logging
    logger = setup_logger(save_weights_path, save_weights_path.name)
    logger.info(f"Experiment metadata: {meta}")
    logger.info(f"Hyperparameters: {hyperparams}")
    
    # Load dataset
    dataset_path = Path(dependencies['dataset_path'])
    train_dataset = DictDataset(dataset_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=True,
        num_workers=hyperparams.get('num_workers', 0)
    )
    
    # Build models
    unet = build_unet(hyperparams, meta)
    scheduler = build_scheduler(hyperparams)
    
    # Train based on configuration
    if meta['vae_latent']:
        train_model(
            meta=meta,
            save_weights_path=save_weights_path,
            dataloader=train_dataloader,
            diffused_model=scheduler,
            unet=unet,
            hyperparams=hyperparams,
            logger=logger
        )
    else:
        # Load VAE decoder
        decoder = VAEDecoder()
        vae_checkpoint_path = Path(dependencies['vae_path'])
        checkpoint = torch.load(vae_checkpoint_path, map_location='cpu')
        decoder.load_state_dict(checkpoint['decoder'])
        
        train_model_novae(
            meta=meta,
            save_weights_path=save_weights_path,
            dataloader=train_dataloader,
            diffused_model=scheduler,
            unet=unet,
            decoder=decoder,
            hyperparams=hyperparams,
            logger=logger
        )


if __name__ == '__main__':
    main()