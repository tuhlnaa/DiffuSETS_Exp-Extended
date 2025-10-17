"""
Script to convert mimic_vae_0_new.pt to Apache Parquet format.

This script processes the large PyTorch file and converts it to Parquet,
which provides efficient columnar storage and compression.

Usage:
    python script/convert_mimic_vae_to_parquet.py ./mimic_vae_0_new.pt ./output/mimic_vae_dataset.parquet
    python script/convert_mimic_vae_to_parquet.py ./checkpoints/mimic_vae_0_new.pt ./output/mimic_vae_dataset.parquet
"""
import argparse
import psutil
import sys
import torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from tqdm import tqdm


def convert_to_numpy(obj):
    """Convert PyTorch tensors and other objects to NumPy arrays."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, list):
        try:
            return np.array(obj)
        except:
            return obj
    else:
        return obj


def prepare_sample_for_parquet(sample_id, sample_data):
    """
    Prepare a single sample for Parquet format.
    
    Args:
        sample_id: The sample identifier (key from original dict)
        sample_data: Dictionary containing 'data' and 'label' fields
        
    Returns:
        Dictionary with flattened structure suitable for Parquet
    """
    row = {'sample_id': sample_id}
    
    # Add the main data tensor (flatten if multi-dimensional)
    if 'data' in sample_data:
        data_array = convert_to_numpy(sample_data['data'])
        # Store as list for Parquet (Parquet supports list types)
        row['data'] = data_array.flatten().tolist()
        row['data_shape'] = data_array.shape
    
    # # Add label fields
    # if 'label' in sample_data:
    #     label = sample_data['label']
    #     for key, value in label.items():
    #         numpy_value = convert_to_numpy(value)
    #         # Handle different data types appropriately
    #         if numpy_value.ndim == 0:
    #             # Scalar value
    #             row[f'label_{key}'] = numpy_value.item()
    #         else:
    #             # Array value
    #             row[f'label_{key}'] = numpy_value.flatten().tolist()
    #             row[f'label_{key}_shape'] = numpy_value.shape

    # Add label fields
    if 'label' in sample_data:
        label = sample_data['label']
        for key, value in label.items():
            numpy_value = convert_to_numpy(value)
            # Handle different data types appropriately
            if isinstance(numpy_value, np.ndarray):
                if numpy_value.ndim == 0:
                    # Scalar array
                    row[f'label_{key}'] = numpy_value.item()
                else:
                    # Multi-dimensional array
                    row[f'label_{key}'] = numpy_value.flatten().tolist()
                    row[f'label_{key}_shape'] = numpy_value.shape
            else:
                # Primitive types (int, float, str, etc.)
                row[f'label_{key}'] = numpy_value
    
    return row


def convert_pt_to_parquet(input_path: str, output_path: str, batch_size: int = 10000):
    """
    Convert the large PT file to Parquet format.
    
    Args:
        input_path: Path to the input .pt file
        output_path: Path to output Parquet file
        batch_size: Number of samples to write at once (for memory efficiency)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {input_path}")
    print(f"Output file: {output_path}")
    print()
    
    # Load the entire dictionary
    print("Loading PyTorch file (this may take several minutes)...")
    data_dict = torch.load(input_path, map_location='cpu', weights_only=False)
    
    num_samples = len(data_dict)
    print(f"Loaded {num_samples} samples")
    print()
    
    # Get sorted keys for consistent ordering
    keys = sorted(data_dict.keys())
    
    # Process samples in batches
    print("Converting to Parquet format...")
    
    # Initialize Parquet writer (will be created on first batch)
    writer = None
    schema = None
    
    batch_data = []
    
    for i, key in enumerate(tqdm(keys, desc="Processing samples")):
        sample_data = data_dict[key]
        row = prepare_sample_for_parquet(key, sample_data)
        batch_data.append(row)
        
        # # Write batch when it reaches batch_size or at the end
        # if len(batch_data) >= batch_size or i == len(keys) - 1:
        #     # Convert batch to PyArrow table
        #     table = pa.Table.from_pylist(batch_data)
            
        #     if writer is None:
        #         # Create writer with schema from first batch
        #         schema = table.schema
        #         writer = pq.ParquetWriter(
        #             output_path,
        #             schema,
        #             compression='snappy',  # Good balance of speed and compression
        #             use_dictionary=True,   # Enable dictionary encoding
        #         )
            
        #     # Write the batch
        #     writer.write_table(table)
        #     batch_data = []

        # Convert batch to PyArrow table
        table = pa.Table.from_pylist(batch_data)
        
        if writer is None:
            # Create writer with schema from first batch
            schema = table.schema
            writer = pq.ParquetWriter(
                output_path,
                schema,
                compression='snappy',  # Good balance of speed and compression
                use_dictionary=True,   # Enable dictionary encoding
            )
        
        # Write the batch
        writer.write_table(table)
        batch_data = []

    # Close the writer
    if writer is not None:
        writer.close()
    
    print()
    print(f"✓ Successfully created Parquet file: {output_path}")
    
    # Print summary
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Output file size: {file_size_mb:.2f} MB")
        print(f"Compression ratio: {file_size_mb / num_samples:.4f} MB per sample")
        
        # Show how to read the file
        print()
        print("To read the Parquet file:")
        print(f"  import pyarrow.parquet as pq")
        print(f"  table = pq.read_table('{output_path}')")
        print(f"  df = table.to_pandas()  # Convert to pandas DataFrame")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert MIMIC VAE PT file to Apache Parquet format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("input_path", type=str, help="Path to mimic_vae_0_new.pt file")
    parser.add_argument("output_path", type=str, help="Path for output Parquet file")
    parser.add_argument("--batch_size", type=int, default=10000, help="Number of samples to write at once")
    
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
    
    # Perform the conversion
    convert_pt_to_parquet(args.input_path, args.output_path, args.batch_size)


if __name__ == "__main__":
    main()