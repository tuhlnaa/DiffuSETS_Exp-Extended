"""
DiffuSETS Inference Module

A clean, modular implementation for running ECG generation inference
using diffusion models with optional VAE latent space encoding.
"""

import argparse
import os
import torch

from typing import Dict, Any, Optional, Tuple
from diffusers import DDPMScheduler
from dotenv import load_dotenv

# Import custom modules
from utils.config import ConfigurationManager, RichDictPrinter, init_seeds
from utils.inference import batch_generate_ECG, batch_generate_ECG_novae


class ModelLoader:
    """Handles loading and initialization of models."""
    
    @staticmethod
    def load_unet(config: Dict[str, Any]) -> torch.nn.Module:
        """Load and initialize the UNet model based on configuration."""
        hyper_params = config['hyper_para']
        meta_config = config['meta']
        
        # Determine input channels based on VAE usage
        n_channels = 4 if meta_config['vae_latent'] else 12
        
        # Import and initialize appropriate UNet variant
        if meta_config['condition']:
            from unet.unet_conditional import ECGconditional
            unet = ECGconditional(
                hyper_params['num_train_steps'],
                kernel_size=hyper_params['unet_kernel_size'],
                num_levels=hyper_params['unet_num_level'],
                n_channels=n_channels
            )
        else:
            from unet.unet_nocondition import ECGnocondition
            unet = ECGnocondition(
                hyper_params['num_train_steps'],
                kernel_size=hyper_params['unet_kernel_size'],
                num_levels=hyper_params['unet_num_level'],
                n_channels=n_channels
            )
        
        # Load pre-trained weights
        unet_path = config['inference_setting']['unet_path']
        if not os.path.exists(unet_path):
            raise FileNotFoundError(f"UNet model file not found: {unet_path}")
        
        unet.load_state_dict(torch.load(unet_path, map_location='cpu'))
        
        return unet
    

    @staticmethod
    def load_vae_decoder(vae_path: str) -> torch.nn.Module:
        """Load VAE decoder if required."""
        from vae.vae_model import VAE_Decoder
        
        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE model file not found: {vae_path}")
        
        decoder = VAE_Decoder()
        checkpoint = torch.load(vae_path, map_location='cpu')
        decoder.load_state_dict(checkpoint['decoder'])
        return decoder


class DiffusionInference:
    """Main class for running diffusion-based ECG inference."""
   
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.settings = config['inference_setting']
        self.hyper_params = config['hyper_para']
        self.meta_config = config['meta']
       
        # Initialize models
        self.unet = ModelLoader.load_unet(config)
        self.scheduler, self.vae_decoder = self._create_scheduler_and_load_vae()


    def _create_scheduler_and_load_vae(self) -> Tuple[DDPMScheduler, Optional[torch.nn.Module]]:
        """Create and configure the diffusion scheduler, and load VAE decoder if needed."""
        # Create scheduler
        scheduler = DDPMScheduler(
            num_train_timesteps=self.hyper_params['num_train_steps'],
            beta_start=self.hyper_params['beta_start'],
            beta_end=self.hyper_params['beta_end']
        )
        scheduler.set_timesteps(self.settings['inference_timestep'])
       
        # Load VAE decoder if VAE latent space is being used
        vae_decoder = None
        if self.meta_config['vae_latent']:
            vae_path = self.config['dependencies']['vae_path']
            vae_decoder = ModelLoader.load_vae_decoder(vae_path)
       
        return scheduler, vae_decoder


    def run_inference(self) -> None:
        """Execute the inference pipeline with VAE or direct inference."""
        if self.meta_config['vae_latent']:
            # Run inference with VAE latent space
            batch_generate_ECG(
                settings=self.settings,
                unet=self.unet,
                diffused_model=self.scheduler,
                decoder=self.vae_decoder,
                condition=self.meta_config['condition']
            )
        else:
            # Run inference without VAE encoding
            batch_generate_ECG_novae(
                settings=self.settings,
                unet=self.unet,
                diffused_model=self.scheduler,
                condition=self.meta_config['condition']
            )


def create_argument_parser(args=None) -> argparse.ArgumentParser:
    """Create and configure command line argument parser."""
    parser = argparse.ArgumentParser(
        description='DiffuSETS Inference - Generate ECG signals using diffusion models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('config', type=str, help='Path to the training configuration JSON file')

    parsed_args = parser.parse_args()

    # Load and validate configuration
    config = ConfigurationManager.load_config(parsed_args.config)

    # Print configuration
    RichDictPrinter.print_dict(config, "Configuration")

    return config


def main() -> None:
    """Main entry point for the inference script."""
    # Load environment variables
    load_dotenv()
    config = create_argument_parser()
    
    # Initialize and run inference
    init_seeds()
    inference_engine = DiffusionInference(config)
    inference_engine.run_inference()
    

if __name__ == "__main__":
    main()
