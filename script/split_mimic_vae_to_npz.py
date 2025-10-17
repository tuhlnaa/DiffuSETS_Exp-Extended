"""
Script to split mimic_vae_0_new.pt into individual NPZ files.

This script processes the large PyTorch file in a memory-efficient way,
creating one NPZ file per sample.

Usage:
    python script/split_mimic_vae_to_npz.py ./mimic_vae_0_new.pt ./output/Dataset_Preprocessing/MIMIC-IV-ECG/mimic_vae_npz
"""
import argparse
import psutil
import sys
import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm


def convert_to_numpy(obj):
    """Convert PyTorch tensors and other objects to NumPy arrays with optimized dtypes."""
    if isinstance(obj, torch.Tensor):
        arr = obj.cpu().numpy()
    elif isinstance(obj, np.ndarray):
        arr = obj
    elif isinstance(obj, np.generic):
        # Handle NumPy scalar types (np.int64, np.float64, etc.)
        arr = np.asarray(obj)
    elif isinstance(obj, (int, float)):
        # Handle Python scalar types
        arr = np.asarray(obj)
    elif isinstance(obj, list):
        # For lists, try to convert to numpy array if possible
        try:
            arr = np.array(obj)
        except:
            # If conversion fails, return as is (will be pickled by npz)
            return obj
    else:
        return obj
   
    # Convert float64 to float32
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
   
    # Convert int64 to int32 (with overflow check)
    elif arr.dtype == np.int64:
        # Check if values fit in int32 range
        if arr.size > 0:
            min_val, max_val = np.min(arr), np.max(arr)
            if min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                return arr.astype(np.int32)
            else:
                print(f"Warning: int64 values outside int32 range (min={min_val}, max={max_val}), keeping int64")
                return arr
        return arr.astype(np.int32)
   
    return arr


def save_sample_to_npz(sample_id, sample_data, output_dir):
    """
    Save a single sample to an NPZ file.
    
    Args:
        sample_id: The sample identifier (key from original dict)
        sample_data: Dictionary containing 'data' and 'label' fields
        output_dir: Path to output directory
    """
    # Prepare data dictionary for NPZ
    npz_dict = {}
    
    # Add the main data tensor
    if 'data' in sample_data:
        npz_dict['data'] = convert_to_numpy(sample_data['data'])

    # Add label fields
    if 'label' in sample_data:
        label = sample_data['label']
        for key, value in label.items():
            # Use 'label_' prefix to avoid conflicts
            npz_key = f'label_{key}'
            npz_dict[npz_key] = convert_to_numpy(value)
    
    # Add sample ID
    npz_dict['sample_id'] = convert_to_numpy(np.array([sample_id]))

    # Save to NPZ file
    output_path = output_dir / f'sample_{sample_id:08d}.npz'
    np.savez(output_path, **npz_dict)


def split_pt_to_npz(input_path: str, output_dir: str):
    """
    Split the large PT file into individual NPZ files.
    
    Args:
        input_path: Path to the input .pt file
        output_dir: Directory to save NPZ files
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {input_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load the entire dictionary
    # Unfortunately, we need to load it all at once since it's a single dict
    print("Loading PyTorch file (this may take several minutes)...")
    data_dict = torch.load(input_path, map_location='cpu', weights_only=False)
    
    num_samples = len(data_dict)
    print(f"Loaded {num_samples} samples")
    print()
    
    # Get sorted keys for consistent ordering
    keys = sorted(data_dict.keys())
    
    # Process samples with progress bar
    print("Splitting into individual NPZ files...")
    for i, key in enumerate(tqdm(keys, desc="Processing samples")):
        sample_data = data_dict[key]
        save_sample_to_npz(key, sample_data, output_dir)
    
    print()
    print(f"✓ Successfully created {num_samples} NPZ files in {output_dir}")
    
    # Print summary
    sample_files = list(output_dir.glob('sample_*.npz'))
    total_size_mb = sum(f.stat().st_size for f in sample_files) / (1024 * 1024 * 1024)
    print(f"Total output size: {total_size_mb:.2f} GB")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split MIMIC VAE PT file into individual NPZ files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("input_path", type=str, help="Path to mimic_vae_0_new.pt file")
    parser.add_argument("output_dir", type=str, help="Directory to save NPZ files")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Check if input file exists
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: File not found: {args.input_path}")
        sys.exit(1)

    # Check available memory
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    if available_memory_gb < 60:
        raise MemoryError(f"Insufficient memory: {available_memory_gb:.2f}GB available, 60GB required")

    # Perform the split
    split_pt_to_npz(args.input_path, args.output_dir)


if __name__ == "__main__":
    main()