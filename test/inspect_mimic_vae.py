"""
Script to inspect and understand the structure of mimic_vae_0_new.pt file.

Usage:
python test/inspect_mimic_vae.py ./mimic_vae_0_new.pt
python test/inspect_mimic_vae.py ./mimic_vae_0_new.pt --batch_size 8 --num_batches 2
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
from rich.tree import Tree

console = Console()

class DictDataset(Dataset):
    """Dataset class for loading dictionary-based PyTorch data."""
    
    def __init__(self, path: str):
        console.print(f"[cyan]Loading dataset from:[/cyan] {path}")
        self.data_dict = torch.load(path, map_location='cpu', weights_only=False)
        self.keys = list(self.data_dict.keys())
        console.print(f"[green]✓ Loaded {len(self.keys)} samples[/green]")

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        latent_dict = self.data_dict[key]
        return latent_dict['data'], latent_dict['label']


def inspect_file_structure(path: str) -> dict:
    """Inspect the structure of the .pt file without loading everything into memory."""
    console.print("\n[bold cyan]═══ File Structure Inspection ═══[/bold cyan]\n")
    
    data_dict = torch.load(path, map_location='cpu', weights_only=False)
    
    # Basic statistics
    num_samples = len(data_dict)
    keys = list(data_dict.keys())
    
    info = {
        'num_samples': num_samples,
        'keys': keys, # e.g. 0 ..., 794370, 794371
        'sample_key': keys[0] if keys else None
    }
    
    # Inspect first sample structure
    if info['sample_key']:
        sample = data_dict[info['sample_key']]
        info['sample_structure'] = sample
        
        # Create tree view of structure
        tree = Tree("[bold]Sample Structure[/bold]")
        
        tree.add(f"[cyan]Top-level keys:[/cyan] {list(sample.keys())}")
        
        # Inspect 'data'
        if 'data' in sample:
            data_branch = tree.add("[yellow]'data' field[/yellow]")
            data = sample['data']
            if isinstance(data, torch.Tensor):
                data_branch.add(f"Type: Tensor")
                data_branch.add(f"Shape: {data.shape}")
                data_branch.add(f"Dtype: {data.dtype}")
                data_branch.add(f"Range: [{data.min():.2f}, {data.max():.2f}]")
            else:
                data_branch.add(f"Type: {type(data)}")
        
        # Inspect 'label'
        if 'label' in sample:
            label_branch = tree.add("[green]'label' field[/green]")
            label = sample['label']
            if isinstance(label, dict):
                label_branch.add(f"Type: dict with keys: {list(label.keys())}")
                for key, value in label.items():
                    if isinstance(value, torch.Tensor):
                        label_branch.add(f"  {key}: Tensor {value.shape}, dtype={value.dtype}")
                    elif isinstance(value, np.ndarray):
                        label_branch.add(f"  {key}: ndarray {value.shape}, dtype={value.dtype}")
                    elif isinstance(value, list):
                        label_branch.add(f"  {key}: list (len={len(value)})")
                        if len(value) > 0:
                            label_branch.add(f"    Sample element: {value[0]} (type={type(value[0])})")
                    else:
                        label_branch.add(f"  {key}: {type(value).__name__} = {value}")
            else:
                label_branch.add(f"Type: {type(label)}")
        
        console.print(tree)
        console.print()
    
    return info


def print_batch_info(
    batch_idx: int,
    data: torch.Tensor,
    label: dict,
) -> None:
    """Print detailed information about a batch."""
    
    # Data tensor info
    data_table = Table(title=f"Batch {batch_idx + 1} - Data Tensor", box=box.ROUNDED)
    data_table.add_column("Property", style="cyan", no_wrap=True)
    data_table.add_column("Value", style="bright_white")
    
    data_table.add_row("Shape", str(data.shape))
    data_table.add_row("Dtype", str(data.dtype))
    data_table.add_row("Device", str(data.device))
    data_table.add_row("Range", f"[{data.min():.2f}, {data.max():.2f}]")
    data_table.add_row("Mean", f"{data.mean():.2f}")
    data_table.add_row("Std", f"{data.std():.2f}")
    
    console.print(data_table)
    
    # Label info
    label_table = Table(title=f"Batch {batch_idx + 1} - Label Fields", box=box.ROUNDED)
    label_table.add_column("Field", style="cyan", no_wrap=True)
    label_table.add_column("Type", style="green")
    label_table.add_column("Shape/Length", style="yellow")
    label_table.add_column("Sample Value", style="bright_white")
    
    for key, value in label.items():
        if isinstance(value, torch.Tensor):
            sample_val = f"Range: [{value.min():.2f}, {value.max():.2f}]"
            label_table.add_row(key, "Tensor", str(value.shape), sample_val)
        elif isinstance(value, np.ndarray):
            sample_val = f"Range: [{value.min():.2f}, {value.max():.2f}]"
            label_table.add_row(key, "ndarray", str(value.shape), sample_val)
        elif isinstance(value, list):
            sample_val = str(value[0]) if len(value) > 0 else "empty"
            label_table.add_row(key, "list", f"len={len(value)}", sample_val)
        else:
            label_table.add_row(key, type(value).__name__, "-", str(value))
    
    console.print(label_table)
    console.print()


def test_dataloader(
    data_path: str,
    batch_size: int = 4,
    num_batches: int = 2,
) -> None:
    """Test the dataloader with the MIMIC VAE dataset."""
    
    console.print("\n[bold cyan]═══ DataLoader Testing ═══[/bold cyan]\n")
    
    # Create dataset and dataloader
    dataset = DictDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    
    # Dataset statistics
    stats_table = Table(title="Dataset Statistics", box=box.DOUBLE)
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="green", justify="right")
    
    stats_table.add_row("Total Samples", str(len(dataset)))
    stats_table.add_row("Batch Size", str(batch_size))
    stats_table.add_row("Number of Batches", str(len(dataloader)))
    
    console.print(stats_table)
    console.print()
    
    # Iterate through batches
    for batch_idx, (data, label) in enumerate(dataloader):
        print_batch_info(batch_idx, data, label)
        
        # Show example of text_embed processing (from the original code)
        if 'text_embed' in label and batch_idx == 0:
            console.print("[bold magenta]Example: text_embed processing[/bold magenta]")
            text_embed = label['text_embed']
            console.print(f"Original text_embed type: {type(text_embed)}")
            console.print(f"Original text_embed shape/length: {text_embed.shape if hasattr(text_embed, 'shape') else len(text_embed)}")
            
            # Apply the transformation from the original code
            text_embed_array = np.array(text_embed)
            console.print(f"After np.array: {text_embed_array.shape}")
            text_embed_transposed = text_embed_array.transpose(1, 0)
            console.print(f"After transpose: {text_embed_transposed.shape}")
            text_embed_expanded = np.repeat(text_embed_transposed[:, np.newaxis, :], 1, axis=1)
            console.print(f"After expand: {text_embed_expanded.shape}")
            console.print()
        
        if batch_idx >= num_batches - 1:
            break


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect and test MIMIC VAE dataset file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("data_path", type=str, help="Path to mimic_vae_0_new.pt file",)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader testing",)
    parser.add_argument("--num_batches", type=int, default=2, help="Number of batches to display",)
    parser.add_argument("--skip_loading", action="store_true", help="Skip dataloader testing, only show structure",)

    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()
    
    # Check if file exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        console.print(f"[bold red]Error: File not found: {args.data_path}[/bold red]")
        sys.exit(1)
    
    # Inspect structure
    info = inspect_file_structure(args.data_path)

    # Test dataloader if not skipped
    if not args.skip_loading:
        test_dataloader(
            args.data_path,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
        )
    
    console.print("[bold green]✓ Inspection complete![/bold green]")


if __name__ == "__main__":
    main()

"""
═══ File Structure Inspection ═══


═══ DataLoader Testing ═══

Loading dataset from: ./checkpoints/mimic_vae_0_new.pt
✓ Loaded 794372 samples
      Dataset Statistics
╔═══════════════════╦════════╗
║ Metric            ║  Value ║
╠═══════════════════╬════════╣
║ Total Samples     ║ 794372 ║
║ Batch Size        ║      8 ║
║ Number of Batches ║  99297 ║
╚═══════════════════╩════════╝

        Batch 1 - Data Tensor
╭──────────┬─────────────────────────╮
│ Property │ Value                   │
├──────────┼─────────────────────────┤
│ Shape    │ torch.Size([8, 4, 128]) │
│ Dtype    │ torch.float32           │
│ Device   │ cpu                     │
│ Range    │ [-0.60, 0.91]           │
│ Mean     │ 0.0093                  │
│ Std      │ 0.1710                  │
╰──────────┴─────────────────────────╯
                                                   Batch 1 - Label Fields
╭────────────┬────────┬─────────────────┬──────────────────────────────────────────────────────────────────────────────────╮
│ Field      │ Type   │ Shape/Length    │ Sample Value                                                                     │
├────────────┼────────┼─────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
│ text       │ list   │ len=8           │ Sinus rhythm|Possible right atrial abnormality|Borderline ECG                    │
│ subject_id │ Tensor │ torch.Size([8]) │ Range: [10000032.00, 10000560.00]                                                │
│ hr         │ Tensor │ torch.Size([8]) │ Range: [64.70, 100.15]                                                           │
│ age        │ Tensor │ torch.Size([8]) │ Range: [34.00, 53.00]                                                            │
│ gender     │ list   │ len=8           │ F                                                                                │
│ text_embed │ list   │ len=1536        │ tensor([-0.0105, -0.0105, -0.0130,  0.0027,  0.0014,  0.0014,  0.0027,  0.0013], │
│            │        │                 │        dtype=torch.float64)                                                      │
╰────────────┴────────┴─────────────────┴──────────────────────────────────────────────────────────────────────────────────╯

Example: text_embed processing
Original text_embed type: <class 'list'>
Original text_embed shape/length: 1536
After np.array: (1536, 8)
After transpose: (8, 1536)
After expand: (8, 1, 1536)
"""