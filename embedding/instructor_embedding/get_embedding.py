"""
Clustering Evaluation Script

This script performs clustering evaluation on text data using the INSTRUCTOR model.
It can either load pre-computed embeddings or generate new ones, then evaluate
clustering performance if specified.

Usage:
    python script_name.py [arguments]

Arguments:
    --model_name: Name of the INSTRUCTOR model to use
    --task_name: Name of the task
    --data_path: Path to the input data file
    --cache_dir: Directory for caching model files
    --result_file: Path to save/load embeddings
    --prompt: Prompt for the model (default: model_name)
    --batch_size: Batch size for processing (default: -1)
    --checkpoint: Path to model checkpoint
    --scale: Scale of the task (default: "small")
    --measure: Flag to measure clustering performance
    --overwrite: Flag to overwrite existing embeddings
"""

import os
import sys
import json
import h5py
import torch
import logging
import argparse
import numpy as np
from instructor import INSTRUCTOR
from clustering_utils.evaluator import ClusteringEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_path):
    """
    Load and parse the input data file.

    Args:
        data_path (str): Path to the input JSON file.

    Returns:
        list: A list of dictionaries, each containing a data point.

    Raises:
        FileNotFoundError: If the specified file is not found.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    try:
        with open(data_path, 'r') as f:
            return [json.loads(l) for l in f]
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {data_path}")
        raise

def process_data(data, measure):
    """
    Process the loaded data into texts and labels.

    Args:
        data (list): List of dictionaries containing the data points.
        measure (bool): Flag indicating whether to extract labels for measurement.

    Returns:
        tuple: A tuple containing:
            - list: Texts extracted from the data.
            - list or None: Labels if measure is True, otherwise None.

    Raises:
        ValueError: If labels are required for measurement but not provided in the data.
    """
    texts, labels = [], []
    for datum in data:
        texts.append(datum['input'])
        if measure:
            if 'label' in datum:
                labels.append(datum['label'])
            else:
                raise ValueError("Label not provided for measurement!")
    return texts, labels if measure else (texts, None)

def load_or_compute_embeddings(args, texts, labels):
    """
    Load existing embeddings or compute new ones.

    Args:
        args (argparse.Namespace): Command-line arguments.
        texts (list): List of input texts.
        labels (list or None): List of labels if available.

    Returns:
        tuple: A tuple containing:
            - dict or None: Clustering measures if computed.
            - numpy.ndarray: Embeddings for the input texts.

    Raises:
        FileNotFoundError: If the specified embedding file or checkpoint is not found.
        torch.serialization.pickle.UnpicklingError: If there's an error loading the model checkpoint.
    """
    if os.path.exists(args.result_file) and not args.overwrite:
        logger.info("Loading existing embeddings...")
        with h5py.File(args.result_file, 'r') as f:
            embeds = np.asarray(f['embeds'])
        evaluator = ClusteringEvaluator(sentences=texts, labels=labels, args=args)
        measures = evaluator.eval_only(embeds)
    else:
        logger.info("Computing new embeddings...")
        model = INSTRUCTOR(args.model_name, cache_folder=args.cache_dir)
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            try:
                state_dict = torch.load(os.path.join(args.checkpoint, 'pytorch_model.bin'))
                model.load_state_dict(state_dict)
            except FileNotFoundError:
                logger.error(f"Checkpoint file not found: {args.checkpoint}")
                raise
            except torch.serialization.pickle.UnpicklingError:
                logger.error(f"Error loading checkpoint: {args.checkpoint}")
                raise

        args.prompt = args.prompt or args.model_name
        if args.prompt not in ['hkunlp/instructor-xl', 'hkunlp/instructor-base']:
            args.prompt = 'hkunlp/instructor-large'

        evaluator = ClusteringEvaluator(sentences=texts, labels=labels, args=args)
        measures, embeds = evaluator(model)

        with h5py.File(args.result_file, 'w') as f:
            f.create_dataset("embeds", data=embeds)

    return measures, embeds

def save_measures(measures, result_file):
    """
    Save clustering measures to a JSON file.

    Args:
        measures (dict): Dictionary containing clustering measures.
        result_file (str): Path to the result file.

    Raises:
        IOError: If there's an error writing to the output file.
    """
    output_file = result_file.replace(".hdf5", "_measures.json")
    try:
        with open(output_file, 'w') as f:
            json.dump(measures, f)
        logger.info(f"Measures saved to {output_file}")
    except IOError:
        logger.error(f"Error writing measures to file: {output_file}")
        raise

def main(args):
    """
    Main function to run the clustering evaluation script.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Raises:
        Exception: Any unexpected exceptions that occur during execution.
    """
    try:
        data = load_data(args.data_path)
        texts, labels = process_data(data, args.measure)
        
        measures, embeds = load_or_compute_embeddings(args, texts, labels)
        
        if measures is not None and args.measure:
            save_measures(measures, args.result_file)
            logger.info(f"Clustering measures: {measures}")
        
        logger.info("Processing completed.")
    except Exception as e:
        logger.exception(f"An error occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering Evaluation Script")
    parser.add_argument('--model_name', default=None, type=str, help="Name of the INSTRUCTOR model")
    parser.add_argument('--task_name', default=None, type=str, help="Name of the task")
    parser.add_argument('--data_path', default=None, type=str, help="Path to the input data file")
    parser.add_argument('--cache_dir', default=None, type=str, help="Directory for caching model files")
    parser.add_argument('--result_file', default=None, type=str, help="Path to save/load embeddings")
    parser.add_argument('--prompt', default=None, type=str, help="Prompt for the model")
    parser.add_argument('--batch_size', default=-1, type=int, help="Batch size for processing")
    parser.add_argument("--checkpoint", default=None, type=str, help="Path to model checkpoint")
    parser.add_argument("--scale", default="small", type=str, help="Scale of the task")
    parser.add_argument("--measure", action="store_true", help="Measure clustering performance")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings")
    
    args = parser.parse_args()
    main(args)