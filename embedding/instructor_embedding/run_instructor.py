"""
Embedding Generation Script

This script automates the process of generating embeddings for specified datasets
using the get_embedding.py script. It sets up the necessary environment variables
and runs the embedding generation for each combination of dataset and scale.

Usage:
    python run_embedding.py

The script uses subprocess to run get_embedding.py with specific arguments and
environment variables for each dataset and scale combination.
"""

import subprocess
import os

def run_embedding(dataset: str, scale: str) -> None:
    """
    Run the get_embedding.py script for a specific dataset and scale.

    This function constructs the command to run get_embedding.py with appropriate
    arguments and environment variables.

    Args:
        dataset (str): The name of the dataset to process.
        scale (str): The scale of the dataset (e.g., 'small', 'medium', 'large').

    Raises:
        subprocess.CalledProcessError: If the get_embedding.py script exits with a non-zero status.
    """
    command = [
        "python", "get_embedding.py",
        "--model_name", "hkunlp/instructor-large",
        "--scale", scale,
        "--task_name", dataset,
        "--data_path", f"../../datasets/{dataset}/{scale}.jsonl",
        "--result_file", f"../../datasets/{dataset}/{scale}_embeds.hdf5",
        "--measure"
    ]
    
    # Set up environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    
    try:
        subprocess.run(command, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running get_embedding.py for dataset {dataset} and scale {scale}: {e}")
        raise

def main() -> None:
    """
    Main function to run the embedding generation process.

    This function defines the datasets and scales to process, then iterates over
    them to run the embedding generation for each combination.
    """
    datasets = ["banking77"]
    scales = ["small"]
    
    for dataset in datasets:
        for scale in scales:
            print(f"Processing dataset: {dataset}, scale: {scale}")
            run_embedding(dataset, scale)
    
    print("Embedding generation completed for all datasets and scales.")

if __name__ == "__main__":
    main()