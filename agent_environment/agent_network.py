import openai
import os
from dotenv import load_dotenv
import torch
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
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine


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

    def generate_summaries(self, cluster_texts: list, max_length=100, min_length=50) -> list:
        """
        Generates textual summaries for each cluster.
        
        Parameters:
            cluster_texts (list): List of concatenated texts for each cluster.
            max_length (int): Maximum length of generated summaries.
            min_length (int): Minimum length of generated summaries.
        
        Returns:
            list: A list of summaries corresponding to the input cluster texts.
        """
        summaries = []
        for text in cluster_texts:
            # Ensure text has enough content to summarize meaningfully
            if len(text.split()) < min_length:
                text = f"{text} {text}"  # Duplicate text for more content

            # Generate the summary
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                truncation=True,
                clean_up_tokenization_spaces=True,
            )[0]['summary_text']
            summaries.append(summary)
        return summaries

class AnomalyDetectionAgent:
    def __init__(self, method="knn", n_neighbors=5, contamination=0.05):
        """
        Initializes the Anomaly Detection Agent.

        Parameters:
            method (str): Algorithm for anomaly detection ('knn' or 'unsupervised').
            n_neighbors (int): Number of neighbors for K-NN anomaly detection.
            contamination (float): Expected proportion of anomalies (used in unsupervised methods).
        """
        self.method = method
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def detect_anomalies_knn(self, embeddings):
        """
        Detects anomalies using K-Nearest Neighbors (K-NN).

        Parameters:
            embeddings (np.array): Input embeddings (e.g., cluster or summary embeddings).

        Returns:
            np.array: Anomaly scores for each embedding.
        """
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, metric="cosine").fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        # Use the mean distance to neighbors as the anomaly score
        anomaly_scores = distances.mean(axis=1)
        return anomaly_scores

    def detect_anomalies_unsupervised(self, embeddings):
        """
        Detects anomalies using an unsupervised distance-based method.

        Parameters:
            embeddings (np.array): Input embeddings.

        Returns:
            np.array: Anomaly scores for each embedding.
        """
        pairwise_dists = pairwise_distances(embeddings, metric="cosine")
        # Compute anomaly score as the average distance to all other points
        anomaly_scores = pairwise_dists.mean(axis=1)
        return anomaly_scores

    def detect_anomalies(self, cluster_embeddings, summary_embeddings):
        """
        Detects anomalies based on the specified method.

        Parameters:
            cluster_embeddings (np.array): Cluster-level embeddings.
            summary_embeddings (np.array): Summary-level embeddings.

        Returns:
            list: Embeddings of detected anomalies.
        """
        # Combine cluster and summary embeddings for joint anomaly detection
        embeddings = np.vstack([cluster_embeddings, summary_embeddings])

        if self.method == "knn":
            anomaly_scores = self.detect_anomalies_knn(embeddings)
        else:
            anomaly_scores = self.detect_anomalies_unsupervised(embeddings)

        # Normalize scores to range [0, 1]
        normalized_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())

        # Determine the anomaly threshold
        threshold = np.percentile(normalized_scores, 100 * (1 - self.contamination))

        # Extract embeddings corresponding to significant anomalies
        anomalies = embeddings[normalized_scores >= threshold]

        return anomalies

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
        if not anomalies:
            return []

        selected_indices = []
        remaining_indices = list(range(len(anomalies)))
        embeddings = np.array(anomalies)

        # Start with a random anomaly
        if remaining_indices:
            selected_indices.append(remaining_indices.pop(0))

        while len(selected_indices) < min(self.top_k, len(embeddings)) and remaining_indices:
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
        if not feedback_scores or not anomalies:
            return []

        top_k = min(self.top_k, len(feedback_scores))
        sorted_indices = np.argsort(feedback_scores)[-top_k:]
        return sorted_indices.tolist()

    def sample_anomalies(
        self,
        anomalies: List[Dict],
        feedback: Optional[List[float]] = None,
        embeddings_key: str = "embedding",
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
    
class NovelClassNamingAgent:
    def __init__(self):
        """
        Initializes the Novel Class Naming Agent with GPT API.
        """

        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def name_novel_classes(self, novel_classes, summaries):
        """
        Assigns names to novel classes using GPT API.

        Args:
            novel_classes (list[dict]): Novel classes to be named, each with a "text" key.
            summaries (list[str]): Summaries for context, one for each novel class.

        Returns:
            list[dict]: Novel classes with assigned names.
        """
        named_classes = []

        for idx, novel_class in enumerate(novel_classes):
            anomaly_text = novel_class.get("text", "No text provided.")
            summary_text = summaries[idx]
            prompt = (
                f"You are tasked with naming a new category of data anomalies. "
                f"The following anomaly was detected: '{anomaly_text}'. "
                f"The summary of this anomaly is: '{summary_text}'. "
                "Please assign a meaningful and concise name to this novel class."
            )

            print(f"GPT prompt for naming novel class: {prompt}")  # Debugging line to see the prompt

            try:
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.7,
                )
                name = response.choices[0].message.content.strip()
                print(f"Generated name: {name}")  # Debugging line to see the response
                named_classes.append({"class": novel_class, "name": name})
            except Exception as e:
                print(f"Error generating name for novel class: {e}")
                named_classes.append({"class": novel_class, "name": "Error generating name"})

        return named_classes
        

