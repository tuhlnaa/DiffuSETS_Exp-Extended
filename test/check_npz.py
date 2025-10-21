import numpy as np
from pathlib import Path
from tqdm import tqdm


def check_npz_files(data_dir: str, output_file: str = "failed_files.txt"):
    """
    Check all .npz files in a directory and identify files that fail to load.
    
    Args:
        data_dir: Directory containing .npz files
        output_file: Path to save the list of failed files
    
    Returns:
        Tuple of (failed_files, total_files, failed_count)
    """
    data_dir = Path(data_dir)
    all_files = sorted(data_dir.glob("*.npz"))
    
    if len(all_files) == 0:
        print(f"No .npz files found in {data_dir}")
        return [], 0, 0
    
    failed_files = []
    
    print(f"Checking {len(all_files)} files...")
    
    for file_path in tqdm(all_files):
        try:
            # Try to load the file
            npz_data = np.load(file_path, allow_pickle=True)
            
            # Verify expected keys exist
            required_keys = ['data', 'label_text', 'label_subject_id', 
                           'label_hr', 'label_age', 'label_gender', 'label_text_embed']
            
            missing_keys = [key for key in required_keys if key not in npz_data.files]
            
            if missing_keys:
                failed_files.append({
                    'path': str(file_path),
                    'error': f'Missing keys: {missing_keys}',
                    'size_bytes': file_path.stat().st_size
                })
            
            npz_data.close()
            
        except EOFError as e:
            failed_files.append({
                'path': str(file_path),
                'error': f'EOFError: {str(e)}',
                'size_bytes': file_path.stat().st_size
            })
        except Exception as e:
            failed_files.append({
                'path': str(file_path),
                'error': f'{type(e).__name__}: {str(e)}',
                'size_bytes': file_path.stat().st_size
            })
    
    # Write results to file
    with open(output_file, 'w') as f:
        f.write(f"NPZ File Validation Report\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Total files checked: {len(all_files)}\n")
        f.write(f"Failed files: {len(failed_files)}\n")
        f.write(f"Success rate: {(len(all_files) - len(failed_files)) / len(all_files) * 100:.2f}%\n\n")
        
        if failed_files:
            f.write(f"Failed Files:\n")
            f.write(f"{'-'*80}\n")
            for i, failed in enumerate(failed_files, 1):
                f.write(f"\n{i}. {failed['path']}\n")
                f.write(f"   Error: {failed['error']}\n")
                f.write(f"   File size: {failed['size_bytes']} bytes\n")
        else:
            f.write("All files loaded successfully!\n")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Validation Complete!")
    print(f"{'='*80}")
    print(f"Total files: {len(all_files)}")
    print(f"Failed files: {len(failed_files)}")
    print(f"Success rate: {(len(all_files) - len(failed_files)) / len(all_files) * 100:.2f}%")
    print(f"\nResults saved to: {output_file}")
    
    return failed_files, len(all_files), len(failed_files)


if __name__ == "__main__":
    # Replace with your actual data directory
    data_directory = "/path/to/your/npz/files"
    
    failed, total, failed_count = check_npz_files(
        data_dir=data_directory,
        output_file="failed_npz_files_report.txt"
    )
    
    # Optionally: print paths of failed files for easy deletion/investigation
    if failed:
        print("\nFailed file paths:")
        for f in failed:
            print(f"  {f['path']}")