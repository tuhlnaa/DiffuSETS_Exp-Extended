"""
DiffuSETS Inference Module

A clean, modular implementation for running ECG generation inference
using diffusion models with optional VAE latent space encoding.
"""

import argparse
import json
import os
import torch

from pathlib import Path
from typing import Dict, Any, Optional
from diffusers import DDPMScheduler
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration is invalid or incomplete."""
    pass


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


class ConfigurationManager:
    """Handles configuration loading and validation."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load and validate configuration from JSON file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with config_file.open('r') as f:
            config = json.load(f)
        
        ConfigurationManager._validate_config(config)
        ConfigurationManager._augment_config(config)

        return config
    

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """Validate that required configuration keys are present."""
        required_keys = [
            'inference_setting', 'hyper_para', 'meta'
        ]
        
        for key in required_keys:
            if key not in config:
                raise ConfigurationError(f"Missing required configuration key: {key}")
        
        # Validate meta configuration
        meta_required = ['condition', 'vae_latent', 'device']
        for key in meta_required:
            if key not in config['meta']:
                raise ConfigurationError(f"Missing required meta configuration key: {key}")
    

    @staticmethod
    def _augment_config(config: Dict[str, Any]) -> None:
        """Add environment variables and derived settings to configuration."""
        # Add OpenAI API key from environment
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY environment variable not set")
        
        config['inference_setting']['OPENAI_API_KEY'] = openai_api_key

        # Add device to inference settings for convenience
        config['inference_setting']['device'] = config['meta']['device']


class DiffusionInference:
    """Main class for running diffusion-based ECG inference."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.settings = config['inference_setting']
        self.hyper_params = config['hyper_para']
        self.meta_config = config['meta']
        
        # Initialize models
        self.unet = ModelLoader.load_unet(config)
        self.scheduler = self._create_scheduler()
        self.vae_decoder = self._load_vae_decoder() if self.meta_config['vae_latent'] else None
    

    def _create_scheduler(self) -> DDPMScheduler:
        """Create and configure the diffusion scheduler."""
        scheduler = DDPMScheduler(
            num_train_timesteps=self.hyper_params['num_train_steps'],
            beta_start=self.hyper_params['beta_start'],
            beta_end=self.hyper_params['beta_end']
        )
        scheduler.set_timesteps(self.settings['inference_timestep'])
        return scheduler
    

    def _load_vae_decoder(self) -> Optional[torch.nn.Module]:
        """Load VAE decoder if VAE latent space is being used."""
        if 'dependencies' not in self.config or 'vae_path' not in self.config['dependencies']:
            raise ConfigurationError("VAE path not specified in dependencies")
        
        vae_path = self.config['dependencies']['vae_path']
        return ModelLoader.load_vae_decoder(vae_path)
    

    def run_inference(self) -> None:
        """Execute the inference pipeline."""
        if self.meta_config['vae_latent']:
            self._run_vae_inference()
        else:
            self._run_direct_inference()
    

    def _run_vae_inference(self) -> None:
        """Run inference with VAE latent space."""
        from utils.inference import batch_generate_ECG
        
        batch_generate_ECG(
            settings=self.settings,
            unet=self.unet,
            diffused_model=self.scheduler,
            decoder=self.vae_decoder,
            condition=self.meta_config['condition']
        )
    

    def _run_direct_inference(self) -> None:
        """Run inference without VAE encoding."""
        from utils.inference_novae import batch_generate_ECG_novae
        
        batch_generate_ECG_novae(
            settings=self.settings,
            unet=self.unet,
            diffused_model=self.scheduler,
            condition=self.meta_config['condition']
        )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure command line argument parser."""
    parser = argparse.ArgumentParser(
        description='DiffuSETS Inference - Generate ECG signals using diffusion models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('config', type=str, help='Path to the training configuration JSON file'
                        )
    return parser


def main() -> None:
    """Main entry point for the inference script."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load and validate configuration
        config = ConfigurationManager.load_config(args.config)
        
        # Initialize and run inference
        inference_engine = DiffusionInference(config)
        inference_engine.run_inference()
        
    except (ConfigurationError, FileNotFoundError) as e:
        print(f"Configuration Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error during inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())