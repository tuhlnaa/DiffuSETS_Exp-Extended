"""
# Convert files in default directory
python ./script/convert_embeddings_to_npy.py

# Specify both input and output directories
python ./script/convert_embeddings_to_npy.py --input_dir ./embeddings --output_dir ./embeddings_npy
"""

import torch
import numpy as np
from pathlib import Path


def convert_txt_to_npy(input_dir: str, output_dir: str = None) -> None:
    """
    Convert all .txt files containing tensor data to .npy files.
    
    Args:
        input_dir: Directory containing .txt files
        output_dir: Directory to save .npy files (defaults to input_dir)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all .txt files in the directory
    txt_files = list(input_path.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Found {len(txt_files)} .txt files to convert")
    
    for txt_file in txt_files:
        try:
            # Read the embedding from txt file
            with open(txt_file, 'r') as f:
                embedding_data = eval(f.readline())
            
            # Convert to tensor then to numpy
            tensor_embedding = torch.tensor(embedding_data)
            numpy_embedding = tensor_embedding.numpy()
            
            # Create output filename
            npy_filename = txt_file.stem + '.npy'
            npy_path = output_path / npy_filename
            
            # Save as .npy file
            np.save(npy_path, numpy_embedding)
            
            print(f"Converted: {txt_file.name} -> {npy_filename} "
                  f"(shape: {numpy_embedding.shape}, dtype: {numpy_embedding.dtype})")
            
        except Exception as e:
            print(f"Error converting {txt_file.name}: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert .txt embedding files to .npy format"
    )
    parser.add_argument("--input_dir", type=str, default="./non_mimic_embeddings", help="Directory containing .txt files")
    parser.add_argument("--output_dir",type=str,default=None,help="Directory to save .npy files (defaults to input_dir)")
    
    args = parser.parse_args()
    convert_txt_to_npy(args.input_dir, args.output_dir)