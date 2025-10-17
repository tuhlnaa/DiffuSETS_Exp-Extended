"""
Script for loading and testing NPZ-based dataset.

This script loads the individual NPZ files created by split_mimic_vae_to_npz.py
and provides utilities for testing the dataset loading.

Usage:
    python usage_dataset.py ./output/mimic_vae_npz --batch_size 8
    python usage_dataset.py ./output/mimic_vae_npz --batch_size 8 --num_batches 3
"""

import argparse
import sys
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


class NPZDataset(Dataset):
    """Dataset class for loading individual NPZ files."""
    
    def __init__(self, npz_dir: str):
        """
        Initialize NPZ dataset.
        
        Args:
            npz_dir: Directory containing NPZ files (sample_*.npz)
        """
        self.npz_dir = Path(npz_dir)
        
        if not self.npz_dir.exists():
            raise FileNotFoundError(f"Directory not found: {npz_dir}")
        
        # Find all NPZ files
        self.npz_files = sorted(self.npz_dir.glob('sample_*.npz'))
        
        if len(self.npz_files) == 0:
            raise ValueError(f"No NPZ files found in {npz_dir}")
        
        console.print(f"[cyan]Loading dataset from:[/cyan] {npz_dir}")
        console.print(f"[green]✓ Found {len(self.npz_files)} samples[/green]")
    
    def __len__(self):
        return len(self.npz_files)
    
    def __getitem__(self, idx):
        """
        Load a single sample from NPZ file.
        
        Returns:
            data: The main data tensor
            label: Dictionary containing all label fields
        """
        npz_path = self.npz_files[idx]
        
        # Load NPZ file
        with np.load(npz_path, allow_pickle=True) as npz_data:
            # Extract main data
            data = torch.from_numpy(npz_data['data'])
            
            # Extract all label fields (those with 'label_' prefix)
            label = {}
            for key in npz_data.keys():
                if key.startswith('label_'):
                    # Remove 'label_' prefix
                    label_key = key[6:]  # len('label_') = 6
                    label[label_key] = torch.from_numpy(npz_data[key])
            
            # Optionally extract sample_id if needed
            # sample_id = npz_data['sample_id'][0]
        
        return data, label


def create_dataloader(
    npz_dir: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for NPZ dataset.
    
    Args:
        npz_dir: Directory containing NPZ files
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        DataLoader instance
    """
    dataset = NPZDataset(npz_dir)
    
    # Custom collate function to handle label dictionaries
    def collate_fn(batch):
        """Collate function for batching data with label dictionaries."""
        data_list = []
        label_dicts = []
        
        for data, label in batch:
            data_list.append(data)
            label_dicts.append(label)
        
        # Stack data tensors
        batch_data = torch.stack(data_list)
        
        # Combine label dictionaries
        batch_labels = {}
        if len(label_dicts) > 0:
            # Get all keys from first label dict
            label_keys = label_dicts[0].keys()
            
            for key in label_keys:
                # Stack all values for this key
                values = [label[key] for label in label_dicts]
                batch_labels[key] = torch.stack(values)
        
        return batch_data, batch_labels
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    return dataloader


def print_batch_info(
    batch_idx: int,
    data: torch.Tensor,
    labels: dict,
) -> None:
    """Print information about a single batch using rich tables."""
    table = Table(title=f"Batch {batch_idx + 1}", box=box.ROUNDED)
    
    table.add_column("Tensor", style="cyan", no_wrap=True)
    table.add_column("Shape", style="bright_white")
    table.add_column("Dtype", style="green")
    table.add_column("Device", style="yellow")
    table.add_column("Range", style="bright_cyan")
    
    # Data tensor
    table.add_row(
        "Data",
        str(data.shape),
        str(data.dtype),
        str(data.device),
        f"[{data.min():.4f}, {data.max():.4f}]",
    )
    
    # Label tensors
    for label_key, label_tensor in labels.items():
        table.add_row(
            f"Label: {label_key}",
            str(label_tensor.shape),
            str(label_tensor.dtype),
            str(label_tensor.device),
            f"[{label_tensor.min():.4f}, {label_tensor.max():.4f}]",
        )
    
    console.print(table)
    console.print()


def test_dataloader(
    dataloader: DataLoader,
    num_batches: int = 2,
) -> None:
    """Test dataloader by iterating through batches."""
    
    # Dataset Statistics
    stats_table = Table(title="Dataset Statistics", box=box.DOUBLE)
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="green", justify="right")
    
    stats_table.add_row("Total Samples", str(len(dataloader.dataset)))
    stats_table.add_row("Total Batches", str(len(dataloader)))
    stats_table.add_row("Batch Size", str(dataloader.batch_size))
    
    console.print(stats_table)
    console.print()
    
    # Iterate through batches
    for batch_idx, (data, labels) in enumerate(dataloader):
        print_batch_info(batch_idx, data, labels)
        
        if batch_idx >= num_batches - 1:
            break


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test NPZ dataset loading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("npz_dir", type=str, help="Directory containing NPZ files (sample_*.npz)")

    # Optional arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--num_batches", type=int, default=2, help="Number of batches to display for testing")
    parser.add_argument("--shuffle",action="store_true",help="Shuffle the data")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading")
    
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()
    
    # Print configuration
    config_table = Table(title="Configuration", box=box.DOUBLE)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("NPZ Directory", args.npz_dir)
    config_table.add_row("Batch Size", str(args.batch_size))
    config_table.add_row("Num Batches", str(args.num_batches))
    config_table.add_row("Shuffle", str(args.shuffle))
    config_table.add_row("Num Workers", str(args.num_workers))
    
    console.print(config_table)
    console.print()
    
    # Create dataloader
    dataloader = create_dataloader(
        npz_dir=args.npz_dir,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )
    
    # Test dataloader
    test_dataloader(dataloader, args.num_batches)


if __name__ == "__main__":
    main()