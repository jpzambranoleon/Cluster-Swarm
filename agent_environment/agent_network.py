import os
import torch
import json
import h5py
import yaml
from pathlib import Path
import time
import numpy as np
import tqdm
from functools import partial
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import pipeline, AutoModel, AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerFast, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput
from typing import List, Dict, Optional

from e5_utils import logger, pool, move_to_cuda
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment


def _transform_func(tokenizer: PreTrainedTokenizerFast, examples: Dict[str, List], prompt=None) -> BatchEncoding:
    """Transforms input text using a tokenizer."""
    if prompt:
        examples['input_texts'] = [prompt + t for t in examples['input_texts']]
    return tokenizer(examples['input_texts'], max_length=512, padding=True, truncation=True)

class Network:
    def __init__(self):
        self.agents = {}

    def register_agent(self, name: str, agent):
        """Register an agent in the network."""
        self.agents[name] = agent
        agent.network = self

    def send_message(self, from_agent: str, to_agent: str, data):
        """Send data from one agent to another."""
        if to_agent in self.agents:
            self.agents[to_agent].receive_message(from_agent, data)
        else:
            raise ValueError(f"Agent '{to_agent}' not found in the network.")

class EncoderAgent(torch.nn.Module):
    def __init__(self, model_name_or_path, checkpoint=None, pool_type='avg', l2_normalize=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        if checkpoint:
            self.encoder = AutoModel.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.pool_type = pool_type
        self.l2_normalize = l2_normalize
        self.gpu_count = torch.cuda.device_count()

        self.encoder.eval()
        self.encoder.cuda()

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size=128) -> np.ndarray:
        """Encodes sentences into embeddings."""
        dataset = Dataset.from_dict({'input_texts': sentences})
        dataset.set_transform(partial(_transform_func, self.tokenizer))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            collate_fn=data_collator,
            pin_memory=True)

        encoded_embeds = []
        for batch_dict in tqdm.tqdm(data_loader, desc='encoding', mininterval=10, disable=len(sentences) < 128):
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], self.pool_type)
                if self.l2_normalize:
                    embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)
    
class ClusteringAgent:
    def __init__(self, scale="small"):
        self.scale = scale

    def fit_agent_predict(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Performs clustering on embeddings."""
        if self.scale == "test_json":
            return KMeans(n_clusters=n_clusters).fit_predict(embeddings)
        elif self.scale == "large":
            return MiniBatchKMeans(n_clusters=n_clusters).fit_predict(embeddings)
        
class FeedbackLoop:
    def __init__(self, encoder_agent: EncoderAgent, clustering_agent: ClusteringAgent, feedback_threshold=0.1, max_iterations=3):
        """
        Feedback loop for refining embeddings based on clustering results.
        
        Args:
            encoder_agent (EncoderAgent): The embedding generation agent.
            clustering_agent (ClusteringAgent): The clustering agent.
            feedback_threshold (float): Threshold for cluster coherence; triggers refinement.
            max_iterations (int): Maximum iterations for the feedback loop.
        """
        self.encoder_agent = encoder_agent
        self.clustering_agent = clustering_agent
        self.feedback_threshold = feedback_threshold
        self.max_iterations = max_iterations

    def evaluate_clusters(self, clusters: np.ndarray, embeddings: np.ndarray) -> float:
        """
        Evaluate the quality of clusters using metrics like Silhouette score.
        
        Args:
            clusters (np.ndarray): Cluster assignments.
            embeddings (np.ndarray): Embeddings used for clustering.
        
        Returns:
            float: Cluster coherence score (higher is better).
        """
        from sklearn.metrics import silhouette_score

        if len(set(clusters)) > 1:  # Ensure there is more than one cluster
            return silhouette_score(embeddings, clusters)
        else:
            return 0.0  # Invalid clustering scenario

    def refine_embeddings(self, texts: List[str], low_quality_indices: List[int]) -> np.ndarray:
        """
        Refines embeddings for identified low-quality clusters.
        
        Args:
            texts (List[str]): Input texts.
            low_quality_indices (List[int]): Indices of problematic clusters.
        
        Returns:
            np.ndarray: Refined embeddings for the problematic clusters.
        """
        problematic_texts = [texts[i] for i in low_quality_indices]
        refined_embeddings = self.encoder_agent.encode(problematic_texts)
        return refined_embeddings

    def run(self, texts: List[str], embeddings: np.ndarray, n_clusters: int):
        """
        Executes the feedback loop to refine embeddings based on clustering feedback.
        
        Args:
            texts (List[str]): Input texts.
            embeddings (np.ndarray): Initial embeddings.
            n_clusters (int): Number of clusters for clustering.
        
        Returns:
            np.ndarray: Refined embeddings.
        """
        for iteration in range(self.max_iterations):
            # Perform clustering
            clusters = self.clustering_agent.fit_agent_predict(embeddings, n_clusters)

            # Evaluate cluster quality
            cluster_score = self.evaluate_clusters(clusters, embeddings)
            print(f"Iteration {iteration + 1}: Cluster Score = {cluster_score}")

            # Check if refinement is needed
            if cluster_score >= self.feedback_threshold:
                print("Cluster quality is satisfactory. Exiting feedback loop.")
                break

            # Identify low-quality clusters
            low_quality_indices = [
                idx for idx, cluster in enumerate(clusters)
                if cluster == -1  # Example condition: unlabeled or poorly assigned
            ]

            if not low_quality_indices:
                print("No low-quality clusters identified. Exiting feedback loop.")
                break

            print(f"Refining embeddings for {len(low_quality_indices)} items.")

            # Refine embeddings for low-quality clusters
            refined_embeddings = self.refine_embeddings(texts, low_quality_indices)

            # Update embeddings
            for idx, new_embed in zip(low_quality_indices, refined_embeddings):
                embeddings[idx] = new_embed

        return embeddings

class SummarizationAgent:
    def __init__(self, model_name_or_path="facebook/bart-large-cnn"):
        """
        Initializes the SummarizationAgent with a pre-trained summarization model.
        
        Parameters:
            model_name_or_path (str): The pre-trained model used for summarization.
        """
        device = 0 if torch.cuda.is_available() else -1
        self.summarizer = pipeline("summarization", model=model_name_or_path, device=device)

    def generate_summaries(self, clusters: dict, texts: list, max_length=50, min_length=20) -> dict:
        """
        Generates textual summaries for each cluster.
        
        Parameters:
            clusters (dict): Mapping of cluster IDs to indices of associated texts.
            texts (list): List of text data corresponding to embeddings.
            max_length (int): Maximum length of generated summaries.
            min_length (int): Minimum length of generated summaries.
        
        Returns:
            summaries (dict): Mapping of cluster IDs to their summaries.
        """
        summaries = {}
        for cluster_id, indices in clusters.items():
            cluster_texts = [texts[i] for i in indices]
            concatenated_text = " ".join(cluster_texts)
            # Generate a summary with parameters to reduce redundancy
            summary = self.summarizer(
                concatenated_text,
                max_length=max_length,
                min_length=min_length,
                truncation=True,
                clean_up_tokenization_spaces=True,
            )[0]['summary_text']
            summaries[cluster_id] = summary
        return summaries

    def generate_summary_embeddings(self, summaries: dict, encoder_agent: EncoderAgent) -> np.ndarray:
        """
        Generates embeddings for the textual summaries.
        
        Parameters:
            summaries (dict): Mapping of cluster IDs to summaries.
            encoder_agent (EncoderAgent): Pre-trained encoder agent for embeddings.
        
        Returns:
            summary_embeddings (np.ndarray): Embeddings for the summaries.
        """
        summary_texts = list(summaries.values())
        summary_embeddings = encoder_agent.encode(summary_texts)
        return summary_embeddings

class AnomalyDetectionAgent:
    def __init__(self, n_neighbors=5, threshold_percentile=95):
        """
        Initialize the Anomaly Detection Agent using KNN.
        
        Parameters:
        - n_neighbors: int, number of neighbors for KNN.
        - threshold_percentile: float, percentile to define anomaly threshold.
        """
        self.n_neighbors = n_neighbors
        self.threshold_percentile = threshold_percentile

    def detect_anomalies(self, summary_embeddings, cluster_embeddings):
        """
        Detect anomalies by comparing summary embeddings with cluster embeddings.
        
        Parameters:
        - summary_embeddings: np.ndarray, embeddings of the summarized clusters.
        - cluster_embeddings: np.ndarray, embeddings of the original clusters.
        
        Returns:
        - anomalies: dict, mapping of anomaly type to indices:
            - 'summary_anomalies': anomalies within summary embeddings.
            - 'cluster_anomalies': anomalies within cluster embeddings.
        - scores: dict, containing scores for both summary and cluster embeddings.
        """
        summary_anomalies, summary_scores = self._knn_anomaly_detection(summary_embeddings)
        cluster_anomalies, cluster_scores = self._knn_anomaly_detection(cluster_embeddings)
        
        return {
            'summary_anomalies': summary_anomalies,
            'cluster_anomalies': cluster_anomalies
        }, {
            'summary_scores': summary_scores,
            'cluster_scores': cluster_scores
        }

    def _knn_anomaly_detection(self, embeddings):
        """
        Anomaly detection using KNN on given embeddings.
        
        Parameters:
        - embeddings: np.ndarray, embeddings to analyze.
        
        Returns:
        - anomalies: np.ndarray, indices of flagged anomalies.
        - scores: np.ndarray, average KNN distances (anomaly scores).
        """
        # Fit KNN model
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)

        # Compute average distance to neighbors (excluding self-distance)
        mean_distances = distances[:, 1:].mean(axis=1)
        
        # Calculate threshold for anomalies based on percentile
        threshold = np.percentile(mean_distances, self.threshold_percentile)
        anomalies = np.where(mean_distances > threshold)[0]

        return anomalies, mean_distances

    def refine_anomalies(self, feedback, embeddings):
        """
        Refine detected anomalies using feedback data.
        
        Parameters:
        - feedback: dict, contains information for refinement (e.g., confirmed anomalies).
        - embeddings: np.ndarray, embeddings to re-evaluate.
        
        Returns:
        - refined_anomalies: np.ndarray, updated anomaly indices.
        - refined_scores: np.ndarray, updated anomaly scores.
        """
        # Optionally implement additional logic here based on feedback
        anomalies, scores = self._knn_anomaly_detection(embeddings)
        return anomalies, scores

class RepresentativeSamplingAgent:
    def __init__(self, sampling_strategy: str = "diversity", top_k: int = 10):
        """
        Initializes the RepresentativeSamplingAgent with a specified sampling strategy.
        
        Args:
            sampling_strategy (str): Strategy for selecting anomalies ("diversity" or "uncertainty").
            top_k (int): Number of anomalies to select.
        """
        self.sampling_strategy = sampling_strategy
        self.top_k = top_k

    def _select_by_diversity(self, anomalies: List[List[float]]) -> List[int]:
        """
        Selects the most diverse anomalies using cosine similarity.
        
        Args:
            anomalies (List[List[float]]): Embeddings of anomalies.
        
        Returns:
            List[int]: Indices of selected anomalies.
        """
        selected_indices = []
        remaining_indices = list(range(len(anomalies)))
        embeddings = np.array(anomalies)

        # Start with a random anomaly
        if remaining_indices:
            selected_indices.append(remaining_indices.pop(0))

        while len(selected_indices) < self.top_k and remaining_indices:
            # Calculate pairwise cosine similarity
            selected_embeddings = embeddings[selected_indices]
            remaining_embeddings = embeddings[remaining_indices]
            similarity_matrix = cosine_similarity(selected_embeddings, remaining_embeddings)

            # Find the least similar point
            mean_similarity = similarity_matrix.mean(axis=0)
            least_similar_index = remaining_indices[np.argmin(mean_similarity)]
            selected_indices.append(least_similar_index)
            remaining_indices.remove(least_similar_index)

        return selected_indices

    def _select_by_uncertainty(self, anomalies: List[Dict], feedback_scores: List[float]) -> List[int]:
        """
        Selects anomalies with the highest uncertainty scores.
        
        Args:
            anomalies (List[Dict]): List of anomalies with associated data.
            feedback_scores (List[float]): Uncertainty scores from feedback.
        
        Returns:
            List[int]: Indices of selected anomalies.
        """
        sorted_indices = np.argsort(feedback_scores)[-self.top_k:]
        return sorted_indices.tolist()

    def sample_anomalies(
        self,
        anomalies: List[Dict],
        feedback: Optional[List[float]] = None,
        embeddings_key: str = "embedding"
    ) -> List[Dict]:
        """
        Selects the most informative anomalies based on the sampling strategy.
        
        Args:
            anomalies (List[Dict]): List of anomalies, each represented as a dictionary with relevant fields.
            feedback (Optional[List[float]]): Feedback scores for uncertainty-based sampling.
            embeddings_key (str): Key to access embeddings in anomaly dictionaries.
        
        Returns:
            List[Dict]: List of selected anomalies.
        """
        if not anomalies:
            return []

        if self.sampling_strategy == "diversity":
            embeddings = [anomaly[embeddings_key] for anomaly in anomalies]
            selected_indices = self._select_by_diversity(embeddings)
        elif self.sampling_strategy == "uncertainty" and feedback is not None:
            selected_indices = self._select_by_uncertainty(anomalies, feedback)
        else:
            raise ValueError(f"Invalid sampling strategy or missing feedback for strategy '{self.sampling_strategy}'.")

        return [anomalies[i] for i in selected_indices]
        

