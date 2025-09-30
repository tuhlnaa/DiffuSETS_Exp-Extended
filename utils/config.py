"""
Configuration management for PyTorch training using OmegaConf.
Based on: https://github.com/huggingface/pytorch-image-models/blob/main/train.py
"""
import json
import os
import torch
import random
import numpy as np

from torch.backends import cudnn
from typing import Any, Dict, Union
from pathlib import Path
from rich.table import Table
from rich import box
from rich.pretty import Pretty
from rich.console import Console


def init_seeds(seed: int = 0, cuda_deterministic: bool = True) -> None:
    """Initialize random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        cuda_deterministic: If True, use deterministic CUDA operations (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if cuda_deterministic:
            cudnn.deterministic = True
            cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            cudnn.benchmark = True

            
class ConfigurationError(Exception):
    """Raised when configuration is invalid or incomplete."""
    pass


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
        
        ConfigurationManager._augment_config(config)

        return config

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
        
         
class RichDictPrinter:
    @staticmethod
    def print_dict(data: Union[Dict, str], title: str = "Dictionary") -> None:
        """Print dictionary details in a structured table.
        
        Args:
            data: Dictionary object or JSON string to display
            title: Title for the table display
        """
        console = Console()
        table = Table(title=title, box=box.ROUNDED)
        table.add_column("Parameter", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        # Handle JSON string input
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                console.print(f"[red]Error parsing JSON string: {e}[/red]")
                return
        
        # Ensure we have a dictionary
        if not isinstance(data, dict):
            console.print(f"[red]Error: Expected dictionary or JSON string, got {type(data)}[/red]")
            return
        
        # Add rows recursively for nested dictionaries
        def add_dict_to_table(d: Dict, prefix: str = "") -> None:
            for key, value in d.items():
                param_name = f"{prefix}{key}"
                if isinstance(value, dict):
                    add_dict_to_table(value, f"{param_name}.")
                else:
                    pretty_value = Pretty(value, indent_guides=False)
                    table.add_row(param_name, pretty_value)
        
        add_dict_to_table(data)
        console.print(table)
        console.print()  # Add spacing after table
