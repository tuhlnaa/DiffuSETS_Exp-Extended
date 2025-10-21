import argparse
import json
import logging
import torch
import wandb

from diffusers import DDPMScheduler
from dotenv import load_dotenv
from pathlib import Path
from rich.logging import RichHandler
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Any

# Import custom modules
# from dataset.mimic_iv_ecg_dataset import DictDataset
from dataset.mimic_iv_ecg_datasetV2 import MIMIC_IV_ECG_VAE_Dataset
from unet.unet_conditional import ECGConditional
from unet.unet_nocondition import ECGNoCondition
from utils.config import ConfigurationManager, RichPrinter, init_seeds
from utils.train import train_model
from utils.train_novae import train_model_novae
from vae.vae_model import VAEDecoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


def get_next_experiment_dir(output_root: str, exp_name: str, exp_type: str) -> Path:
    """Find the next available experiment directory number."""
    output_root = Path(output_root)
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)
    
    max_idx = 0
    prefix = f"{exp_type}_"
    
    for item in output_root.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            try:
                idx = int(item.name.split('_')[-1])
                max_idx = max(max_idx, idx)
            except (ValueError, IndexError):
                continue
    
    next_dir = output_root / f"{exp_name}_{exp_type}_{max_idx + 1}"
    next_dir.mkdir(parents=True, exist_ok=True)
    
    return next_dir


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


def create_argument_parser(args=None) -> argparse.ArgumentParser:
    """Create and configure command line argument parser."""
    parser = argparse.ArgumentParser(
        description='DiffuSETS Training - Generate ECG signals using diffusion models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('config', type=str, help='Path to the training configuration JSON file')

    parsed_args = parser.parse_args()

    # Load and validate configuration
    config = ConfigurationManager.load_config(parsed_args.config)

    # Print configuration
    RichPrinter.print_dict(config, "Configuration")

    return config


def main() -> None:
    """Main training entry point."""
    load_dotenv()
    config = create_argument_parser()
    init_seeds(seed=42)

    wandb.init(project="DiffuSETS", group="Training", entity=None)
    wandb.run.name = "Experiment 1 V2"

    # Extract configuration sections
    meta = config['meta']
    dependencies = config['dependencies']
    hyperparams = config['hyper_para']
    
    # Setup experiment directory
    save_weights_path = get_next_experiment_dir(dependencies['output_dir'], meta['exp_name'], meta['exp_type'])
    
    # Setup logging
    logger.info(f"Experiment metadata: {meta}")
    logger.info(f"Hyperparameters: {hyperparams}")
    
    # Load dataset
    dataset_path = Path(dependencies['dataset_path'])
    train_dataset = MIMIC_IV_ECG_VAE_Dataset(dataset_path, subset_proportion=0.005)
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