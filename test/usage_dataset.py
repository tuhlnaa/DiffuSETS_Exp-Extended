"""
Test script for MIMIC-IV-ECG VAE dataset loading and validation.

Usage:
    python test/usage_dataset.py ./data/mimic_vae_train
    python test/usage_dataset.py ./data/mimic_vae_train 8 --subset_proportion 0.1
    python test/usage_dataset.py ./data/mimic_vae_train 8 --num_batches 3
"""

import argparse
import sys
import torch

from pathlib import Path
from rich import box
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import RichPrinter, init_seeds
from dataset.mimic_iv_ecg_datasetV2 import create_dataloader

console = Console()


def test_dataloaders(
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    num_batches: int = 2,
) -> None:
    """Test dataloaders by iterating through batches."""
    
    # Dataset Statistics
    stats_table = Table(title="Dataset Statistics", box=box.DOUBLE)
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Training", style="green", justify="right")
    if val_loader is not None:
        stats_table.add_column("Validation", style="yellow", justify="right")
    
    stats_table.add_row("Samples", str(len(train_loader.dataset)), 
                        str(len(val_loader.dataset)) if val_loader else "N/A")
    stats_table.add_row("Batches", str(len(train_loader)), 
                        str(len(val_loader)) if val_loader else "N/A")
    
    console.print(stats_table)
    console.print()
    
    # Test training loader
    console.print("[bold green]Training DataLoader Samples:[/bold green]")
    for batch_idx, (data, labels) in enumerate(train_loader):
        print_batch_info(batch_idx, data, labels)
        if batch_idx >= num_batches - 1:
            break
    
    # Test validation loader if provided
    if val_loader is not None:
        console.print("[bold yellow]Validation DataLoader Samples:[/bold yellow]")
        for batch_idx, (data, labels) in enumerate(val_loader):
            print_batch_info(batch_idx, data, labels)
            if batch_idx >= num_batches - 1:
                break


def collate_fn(batch):
    """Custom collate function for batching VAE dataset samples.
    
    Args:
        batch: List of (data, label) tuples
    
    Returns:
        Tuple of (batched_data, batched_labels)
    """
    data_list = []
    label_dict = {
        'text': [],
        'subject_id': [],
        'hr': [],
        'age': [],
        'gender': [],
        'text_embed': [],
    }
    
    for data, label in batch:
        data_list.append(data)
        for key in label_dict.keys():
            label_dict[key].append(label[key])
    
    # Stack data tensors
    batched_data = torch.stack(data_list, dim=0)
    
    return batched_data, label_dict


def print_batch_info(
    batch_idx: int,
    data: torch.Tensor,
    labels: dict,
) -> None:
    """Print information about a single batch using rich tables."""
    
    # Main data table
    table = Table(title=f"Batch {batch_idx + 1} - Data Tensors", box=box.ROUNDED)
    
    table.add_column("Tensor", style="cyan", no_wrap=True)
    table.add_column("Shape", style="bright_white")
    table.add_column("Dtype", style="green")
    table.add_column("Device", style="yellow")
    table.add_column("Range", style="bright_cyan")
    
    table.add_row(
        "VAE Latents",
        str(data.shape),
        str(data.dtype),
        str(data.device),
        f"[{data.min():.4f}, {data.max():.4f}]",
    )
    
    console.print(table)
    
    # Label metadata table
    label_table = Table(title=f"Batch {batch_idx + 1} - Label Metadata", box=box.ROUNDED)
    label_table.add_column("Field", style="cyan", no_wrap=True)
    label_table.add_column("Type", style="green")
    label_table.add_column("Sample Values", style="bright_white")
    
    # Text field
    texts = labels['text']
    sample_text = texts[0][:80] + "..." if len(texts[0]) > 80 else texts[0]
    label_table.add_row("text", "str", f"'{sample_text}'")
    
    # Subject ID
    subject_ids = labels['subject_id']
    label_table.add_row("subject_id", "int", f"{subject_ids[0]}, {subject_ids[1] if len(subject_ids) > 1 else '...'}, ...")
    
    # Heart rate
    hrs = labels['hr']
    hr_sample = f"{hrs[0]:.1f}, {hrs[1]:.1f}, ..." if len(hrs) > 1 else f"{hrs[0]:.1f}"
    label_table.add_row("hr", "float", hr_sample)
    
    # Age
    ages = labels['age']
    age_sample = f"{ages[0]}, {ages[1]}, ..." if len(ages) > 1 else f"{ages[0]}"
    label_table.add_row("age", "int", age_sample)
    
    # Gender
    genders = labels['gender']
    gender_sample = f"'{genders[0]}', '{genders[1]}', ..." if len(genders) > 1 else f"'{genders[0]}'"
    label_table.add_row("gender", "str", gender_sample)
    
    # Text embedding
    text_embeds = labels['text_embed']
    embed_shape = f"({len(text_embeds)}, {len(text_embeds[0])})"
    label_table.add_row("text_embed", "list[float]", f"shape={embed_shape}")
    
    console.print(label_table)
    console.print()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test MIMIC-IV-ECG VAE dataset loading and preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument("train_dir", type=str, help="Directory containing training .npz files")

    # Optional arguments
    required.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--val_dir", type=str, default=None, help="Directory containing validation .npz files")
    parser.add_argument("--subset_proportion", type=float, default=1.0, help="Proportion of dataset to use (0.0 to 1.0)")
    parser.add_argument("--num_batches", type=int, default=1, help="Number of batches to display for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    RichPrinter.print_config(args, "Test Configuration")
    
    return args


def main() -> None:
    """Main execution function."""
    args = parse_args()
    init_seeds(args.seed)
    
    # Create training dataloader with custom collate function
    train_loader = create_dataloader(
        data_dir=args.train_dir,
        batch_size=args.batch_size,
        subset_proportion=args.subset_proportion,
        shuffle=True,
    )
    train_loader.collate_fn = collate_fn
    
    # Create validation dataloader if specified
    val_loader = None
    if args.val_dir is not None:
        val_loader = create_dataloader(
            data_dir=args.val_dir,
            batch_size=args.batch_size,
            subset_proportion=args.subset_proportion,
            shuffle=False,
        )
        val_loader.collate_fn = collate_fn
    
    # Test dataloaders
    test_dataloaders(train_loader, val_loader, args.num_batches)


if __name__ == "__main__":
    main()

"""
Using 794372 out of 794372 files (100.0%)
Dataset samples: 794372, DataLoader batches: 99296

     Dataset Statistics
╔═════════╦══════════╦═════╗
║ Metric  ║ Training ║     ║
╠═════════╬══════════╬═════╣
║ Samples ║   794372 ║ N/A ║
║ Batches ║    99296 ║ N/A ║
╚═════════╩══════════╩═════╝

Training DataLoader Samples:
                                Batch 1 - Data Tensors
╭─────────────┬─────────────────────────┬───────────────┬────────┬───────────────────╮
│ Tensor      │ Shape                   │ Dtype         │ Device │ Range             │
├─────────────┼─────────────────────────┼───────────────┼────────┼───────────────────┤
│ VAE Latents │ torch.Size([8, 4, 128]) │ torch.float32 │ cpu    │ [-0.7821, 0.6119] │
╰─────────────┴─────────────────────────┴───────────────┴────────┴───────────────────╯

                                           Batch 1 - Label Metadata
╭────────────┬───────┬───────────────────────────────────────────────────────────────────────────────────────╮
│ Field      │ Type  │ Sample Values                                                                         │
├────────────┼───────┼───────────────────────────────────────────────────────────────────────────────────────┤
│ text       │ str   │ 'Atrial fibrillation.|Left axis deviation|Incomplete LBBB|Anteroseptal infarct - ...' │
│ subject_id │ int   │ 10986205, 15554944, ...                                                               │
│ hr         │ float │ 87.9, 90.0, ...                                                                       │
│ age        │ int   │ 84, 88, ...                                                                           │
│ gender     │ str   │ 'F', 'F', ...                                                                         │
│ text_embed │ list  │ shape=(8, 1536)                                                                       │
╰────────────┴───────┴───────────────────────────────────────────────────────────────────────────────────────╯

"""