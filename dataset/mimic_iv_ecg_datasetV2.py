"""
MIMIC-IV-ECG VAE Dataset class compatible with DictDataset interface.
Loads data from individual .npz files instead of a single .pt file.

Usage:
    dataset = MIMIC_IV_ECG_VAE_Dataset(data_dir='./mimic_vae_npz')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
"""

import torch
import logging
import numpy as np

from pathlib import Path
from rich.logging import RichHandler
from torch.utils.data import Dataset
from typing import Tuple, Dict, Any
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


class MIMIC_IV_ECG_VAE_Dataset(Dataset):
    """
    Dataset class for loading MIMIC-IV-ECG VAE latent representations from .npz files.
   
    Compatible with DictDataset interface - returns (data, label) tuples where:
    - data: torch.Tensor of shape (num_leads, latent_dim)
    - label: dict containing metadata (text, subject_id, hr, age, gender, text_embed)
   
    Args:
        data_dir: Directory containing all .npz files
        subset_proportion: Proportion of dataset to use (0.0 to 1.0). Default is 1.0 (use all data).
    """
   
    def __init__(self, data_dir: str, subset_proportion: float = 1.0, logger: Any = logger):
        self.data_dir = Path(data_dir)
        all_file_paths = sorted(self.data_dir.glob("*.npz"))

        # Validate
        if len(all_file_paths) == 0:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")
        
        if not 0.0 < subset_proportion <= 1.0:
            raise ValueError(f"subset_proportion must be between 0.0 and 1.0, got {subset_proportion}")
        
        # Select subset of files if needed
        if subset_proportion < 1.0:
            num_files = int(len(all_file_paths) * subset_proportion)
            num_files = max(1, num_files)
            
            indices = np.random.choice(len(all_file_paths), size=num_files, replace=False)
            self.file_paths = [all_file_paths[i] for i in sorted(indices)]
        else:
            self.file_paths = all_file_paths
        
        logger.info(f"Using {len(self.file_paths)} out of {len(all_file_paths)} files ({subset_proportion*100:.1f}%)")

   
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.file_paths)
   

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Load and return a single sample.
       
        Args:
            idx: Sample index
       
        Returns:
            Tuple of (data, label):
            - data: torch.Tensor of shape (num_leads, latent_dim), e.g., (4, 128)
            - label: dict with keys:
                - 'text': str, ECG diagnosis text
                - 'subject_id': int, patient ID
                - 'hr': float, heart rate
                - 'age': int, patient age
                - 'gender': str, patient gender ('M' or 'F')
                - 'text_embed': list of float, text embedding (length 1536)
        """
        # Load .npz file
        npz_data = np.load(self.file_paths[idx], allow_pickle=True)
       
        # Extract data tensor
        data = torch.from_numpy(npz_data['data']).float()

        # Extract label fields
        label = {
            'text': str(npz_data['label_text'].item()),
            'subject_id': int(npz_data['label_subject_id'].item()),
            'hr': float(npz_data['label_hr'].item()),
            'age': int(npz_data['label_age'].item()),
            'gender': str(npz_data['label_gender'].item()),
            'text_embed': npz_data['label_text_embed'].tolist(),
        }
       
        return data, label
    

def create_dataloader(
    data_dir: str,
    batch_size: int,
    subset_proportion: float = 1.0,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for MIMIC-IV-ECG VAE dataset.
    
    Args:
        data_dir: Directory containing .npz files
        batch_size: Batch size for DataLoader
        subset_proportion: Proportion of dataset to use (0.0 to 1.0)
        shuffle: Whether to shuffle data

    Returns:
        Configured DataLoader instance
    """
    dataset = MIMIC_IV_ECG_VAE_Dataset(
        data_dir=data_dir,
        subset_proportion=subset_proportion,
    )

    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty! No .npz files found in {data_dir}")
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
        num_workers=0,
    )
    print(f"Dataset samples: {len(dataset)}, DataLoader batches: {len(data_loader)}")

    return data_loader
