import os
import torch
import json
import h5py
import argparse
import numpy as np
import tqdm
from functools import partial
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerFast, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput
from typing import List, Dict

from e5_utils import logger, pool, move_to_cuda
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

def hungray_alignment(y_true, y_pred):
    """
    Aligns predicted cluster labels with true labels using the Hungarian Algorithm.
    
    Parameters:
    y_true (np.ndarray): Ground truth labels.
    y_pred (np.ndarray): Predicted cluster labels.

    Returns:
    ind (np.ndarray): Array of aligned indices.
    w (np.ndarray): Weight matrix representing alignment cost.
    """
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    """
    Calculates clustering accuracy (ACC) based on alignment of predicted and true labels.

    Parameters:
    y_true (np.ndarray): Ground truth labels.
    y_pred (np.ndarray): Predicted cluster labels.

    Returns:
    acc (float): Clustering accuracy score.
    """
    ind, w = hungray_alignment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

def clustering_score(y_true, y_pred):
    """
    Computes multiple clustering evaluation metrics: Accuracy, Adjusted Rand Index, and Normalized Mutual Information.
    
    Parameters:
    y_true (np.ndarray): Ground truth labels.
    y_pred (np.ndarray): Predicted cluster labels.

    Returns:
    dict: Dictionary with clustering metrics (ACC, ARI, NMI) multiplied by 100 for percentage format.
    """
    return {
        'ACC': clustering_accuracy_score(y_true, y_pred) * 100,
        'ARI': adjusted_rand_score(y_true, y_pred) * 100,
        'NMI': normalized_mutual_info_score(y_true, y_pred) * 100
    }

parser = argparse.ArgumentParser(description='evaluation for MTEB benchmark except its Retrieval category')
# [Parser argument definitions... omitted for brevity]
parser = argparse.ArgumentParser(description='evaluation for MTEB benchmark except its Retrieval category')
parser.add_argument('--input_path', default=None, type=str)
parser.add_argument('--output_path', default=None, type=str)
parser.add_argument('--model-name-or-path', default='tmp-outputs/',
                    type=str, metavar='N', help='which model to use')
parser.add_argument("--checkpoint", default=None, type=str)
parser.add_argument('--l2-normalize', action='store_true', help='whether to l2 normalize embeddings')
parser.add_argument('--pool-type', default='avg', help='pool type')
parser.add_argument('--prompt', default='query: ', help='prompt')
parser.add_argument("--measure", action="store_true",
                    help="if measure clustering performance")
parser.add_argument("--scale", default="small", type=str)

args = parser.parse_args()
logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
assert args.prompt in ['', 'query: ', 'passage: ']

def _transform_func(tokenizer: PreTrainedTokenizerFast, examples: Dict[str, List]) -> BatchEncoding:
    """
    Prepares text inputs for model encoding by adding prompts and tokenizing.

    Parameters:
    tokenizer (PreTrainedTokenizerFast): Tokenizer for text processing.
    examples (Dict[str, List]): Dictionary with input text.

    Returns:
    BatchEncoding: Tokenized and padded text data.
    """
    if args.prompt:
        examples['input_texts'] = [args.prompt + t for t in examples['input_texts']]
    batch_dict = tokenizer(
        examples['input_texts'],
        max_length=512,
        padding=True,
        truncation=True
    )
    return batch_dict

class DenseEncoder(torch.nn.Module):
    """
    Dense Encoder class to create embeddings from text using a pretrained model.
    """

    def __init__(self, **kwargs):
        """
        Initializes the DenseEncoder with specified model and tokenizer.

        Loads a pretrained model and tokenizer, optionally from a checkpoint.
        Sets up multi-GPU usage if available.
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path)
        if args.checkpoint is not None:
            self.encoder = AutoModel.from_pretrained(args.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.gpu_count = torch.cuda.device_count()
        self.encoder.eval().cuda()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

    @torch.no_grad()
    def encode(self, sentences, **kwargs) -> np.ndarray:
        """
        Encodes a list of sentences into embeddings using the model.

        Parameters:
        sentences (List[str]): List of sentences to encode.

        Returns:
        np.ndarray: Array of embeddings.
        """
        dataset: Dataset = Dataset.from_dict({'input_texts': sentences})
        dataset.set_transform(partial(_transform_func, self.tokenizer))
        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = DataLoader(
            dataset,
            batch_size=128 * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            collate_fn=data_collator,
            pin_memory=True
        )
        encoded_embeds = []
        for batch_dict in tqdm.tqdm(data_loader, desc='encoding', mininterval=10, disable=len(sentences) < 128):
            batch_dict = move_to_cuda(batch_dict)
            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], args.pool_type)
                if args.l2_normalize:
                    embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())
        return np.concatenate(encoded_embeds, axis=0)

def _convert_label_to_ids(labels):
    """
    Maps labels to unique IDs for clustering evaluation.
    
    Parameters:
    labels (List): List of labels.

    Returns:
    tuple: Array of label IDs and number of unique clusters.
    """
    unique_labels = list(set(labels))
    n_clusters = len(unique_labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    label_ids = [label_map[l] for l in labels]
    return np.asarray(label_ids), n_clusters

def eval_embeds(embeds, labels, args):
    """
    Evaluates embeddings using KMeans clustering, comparing cluster results to ground truth labels.
    
    Parameters:
    embeds (np.ndarray): Embeddings to evaluate.
    labels (List): Ground truth labels.
    args (argparse.Namespace): Parsed command-line arguments.

    Returns:
    dict: Dictionary of evaluation metrics (means and standard deviations).
    """
    if labels is not None:
        label_ids, n_clusters = _convert_label_to_ids(labels)
        all_measures = {'ACC': [], 'NMI': [], 'ARI': []}
        for seed in [100, 13, 21, 36, 42]:
            if args.scale == "small":
                preds = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(embeds)
            elif args.scale == "large":
                preds = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed).fit_predict(embeds)
            preds = np.asarray(preds)
            measures = clustering_score(label_ids, preds)
            for k in measures:
                all_measures[k].append(measures[k])
        for k in ['ACC', 'NMI', 'ARI']:
            mean = np.mean(all_measures[k])
            std = np.std(all_measures[k])
            all_measures[f'{k}_mean'] = mean
            all_measures[f'{k}_std'] = std
    else:
        all_measures = {}
    return all_measures

def main():
    """
    Main function for loading data, generating embeddings, performing clustering, and evaluating results.

    - Loads text and labels from input JSONL file.
    - Encodes text into embeddings using DenseEncoder.
    - Computes clustering metrics if specified.
    - Saves results and embeddings to HDF5 format for further analysis.
    """
    model = DenseEncoder()
    assert args.input_path.endswith("jsonl"), "input file must be jsonl"
    assert args.output_path.endswith("hdf5"), "output file must be hdf5"
    assert "e5" in args.output_path
    with open(args.input_path, 'r') as f:
        data = [json.loads(l) for l in f]
    texts, labels = [], []
    for datum in data:
        texts.append(datum['input'])
        if args.measure and 'label' in datum:
            labels.append(datum['label'])
        elif args.measure and 'label' not in datum:
            raise ValueError("Label not provided!")
        else:
            labels = None
    if not os.path.exists(args.output_path):
        embeds = model.encode(texts)
        measures = eval_embeds(embeds, labels, args)
        with h5py.File(args.output_path, 'w') as f:
            dset = f.create_dataset("embeds", data=embeds)
    else:
        with h5py.File(args.output_path, 'r') as f:
            embeds = np.asarray(f['embeds'])
        measures = eval_embeds(embeds, labels, args)
    if measures is not None and args.measure:
        with open(args.output_path.replace(".hdf5", "_measures.json"), 'w') as f:
            json.dump(measures, f)
    print(measures)
    print("--DONE--")

if __name__ == '__main__':
    main()
