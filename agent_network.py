import openai
import os
from dotenv import load_dotenv
import torch
import numpy as np
from tqdm.auto import tqdm
from functools import partial
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import pipeline, AutoModel, AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerFast, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput
from typing import List, Dict, Optional

from e5_utils import logger, pool, move_to_cuda
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


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
        for batch_dict in tqdm(data_loader, desc='encoding', mininterval=10, disable=len(sentences) < 128):
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], self.pool_type)
                if self.l2_normalize:
                    embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())
        print("Embedding process complete.")
        return np.concatenate(encoded_embeds, axis=0)
    
class ClusteringAgent:
    def __init__(self, scale=None):
        self.scale = scale

    def fit_agent_predict(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Performs clustering on embeddings."""
        if self.scale == "small":
            print("Completed Clustering.")
            return KMeans(n_clusters=n_clusters).fit_predict(embeddings)
        elif self.scale == "large":
            print("Completed Clustering.")
            return MiniBatchKMeans(n_clusters=n_clusters).fit_predict(embeddings)
        else:
            print("Completed Clustering.")
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
    def __init__(self, model_name_or_path="facebook/bart-large-cnn", batch_size=4):
        """
        Initializes the SummarizationAgent with a pre-trained summarization model.

        Parameters:
            model_name_or_path (str): The pre-trained model used for summarization.
            batch_size (int): Number of texts to summarize per batch to prevent memory overload.
        """
        device = 0 if torch.cuda.is_available() else -1
        self.summarizer = pipeline("summarization", model=model_name_or_path, device=device)
        self.batch_size = batch_size  # Store batch size

    def generate_summaries(self, cluster_texts: list, max_length=100, min_length=50) -> list:
        """
        Generates textual summaries for each cluster in batches to prevent memory allocation issues.

        Parameters:
            cluster_texts (list): List of concatenated texts for each cluster.
            max_length (int): Maximum length of generated summaries.
            min_length (int): Minimum length of generated summaries.

        Returns:
            list: A list of summaries corresponding to the input cluster texts.
        """
        summaries = []
        num_batches = (len(cluster_texts) + self.batch_size - 1) // self.batch_size  # Calculate number of batches

        for i in tqdm(range(num_batches), desc="Summarizing Batches", unit="batch"):
            batch_texts = cluster_texts[i * self.batch_size:(i + 1) * self.batch_size]

            # Handle short texts
            batch_texts = [text if len(text.split()) >= min_length else f"{text} {text}" for text in batch_texts]

            # Generate summaries
            batch_summaries = self.summarizer(
                batch_texts,
                max_length=max_length,
                min_length=min_length,
                truncation=True,
                clean_up_tokenization_spaces=True,
            )

            summaries.extend([summary['summary_text'] for summary in batch_summaries])

            # Free CUDA memory after processing each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return summaries

class AnomalyDetectionAgent:
    def __init__(self, method="knn", initial_n_neighbors=5, initial_contamination=0.05, memory_decay=0.8, metric="cosine"):
        """
        Initializes the Anomaly Detection Agent with memory.

        Parameters:
            method (str): Algorithm for anomaly detection ('knn').
            initial_n_neighbors (int): Initial number of neighbors for K-NN anomaly detection.
            initial_contamination (float): Initial proportion of expected anomalies.
            memory_decay (float): Controls how much weight past refinements have (0.0 = no memory, 1.0 = full memory).
            metric (str): Distance metric for KNN ('cosine', 'euclidean', 'manhattan').
        """
        self.method = method
        self.n_neighbors = initial_n_neighbors
        self.contamination = initial_contamination
        self.memory_decay = memory_decay  # Weight for previous values
        self.previous_n_neighbors = initial_n_neighbors
        self.previous_contamination = initial_contamination
        self.metric = metric  # KNN metric (cosine, euclidean, etc.)

    def detect_anomalies_knn(self, embeddings):
        """
        Detects anomalies using K-Nearest Neighbors (K-NN).

        Parameters:
            embeddings (np.array): Input embeddings.

        Returns:
            np.array: Anomaly scores.
        """

        # Fit KNN model
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric).fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)

        # Use median instead of mean to be more robust to noise
        anomaly_scores = np.median(distances, axis=1)  
        return anomaly_scores

    def detect_anomalies(self, cluster_embeddings, summary_embeddings):
        """
        Detects anomalies using the selected method.

        Parameters:
            cluster_embeddings (np.array): Cluster-level embeddings.
            summary_embeddings (np.array): Summary-level embeddings.

        Returns:
            np.array: Embeddings of detected anomalies.
        """
        embeddings = np.vstack([cluster_embeddings, summary_embeddings])

        if self.method == "knn":
            anomaly_scores = self.detect_anomalies_knn(embeddings)
        else:
            raise ValueError("Unsupported method")

        # Normalize scores between 0 and 1
        normalized_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())

        # Use an adaptive threshold instead of a fixed percentile
        threshold = np.mean(normalized_scores) + (1.5 * np.std(normalized_scores))  # Mean + 1.5 Std Dev

        # Extract anomalies
        anomalies = embeddings[normalized_scores >= threshold]

        print(f"Detected {len(anomalies)} anomalies with threshold {threshold:.4f}")

        return anomalies

    def refine_anomaly_detection(self, detected_anomalies, selected_informative_anomalies):
        """
        Refines anomaly detection strategy using past refinements as memory.

        Parameters:
            detected_anomalies (np.ndarray): Initially detected anomalies.
            selected_informative_anomalies (np.ndarray): Informative anomalies chosen by the Representative Sampling Agent.
        """
        if len(selected_informative_anomalies) == 0:
            return  # No feedback, no need to refine

        # Compute similarity between detected anomalies and selected informative anomalies
        similarity_matrix = cosine_similarity(detected_anomalies, selected_informative_anomalies)
        max_similarities = np.max(similarity_matrix, axis=1)  # Find most similar anomaly for each detected one

        avg_similarity = np.mean(max_similarities)

        # Adjust n_neighbors and contamination rate based on feedback
        new_n_neighbors = self.n_neighbors
        new_contamination = self.contamination

        if avg_similarity < 0.8:  # Detected anomalies are too different from selected ones
            new_contamination += 0.02
            new_n_neighbors = min(self.n_neighbors + 2, 15)  # Expand search
        elif avg_similarity > 0.9:  # Detected anomalies are too similar
            new_contamination -= 0.02
            new_n_neighbors = max(self.n_neighbors - 2, 2)  # Focus search

        # Ensure contamination stays within reasonable bounds
        new_contamination = max(0.01, min(new_contamination, 0.2))

        # Apply memory decay using exponential smoothing
        self.n_neighbors = int(self.memory_decay * self.previous_n_neighbors + (1 - self.memory_decay) * new_n_neighbors)
        self.contamination = self.memory_decay * self.previous_contamination + (1 - self.memory_decay) * new_contamination

        # Update previous values for next iteration
        self.previous_n_neighbors = self.n_neighbors
        self.previous_contamination = self.contamination

        print(f"Updated Contamination Rate: {self.contamination:.4f}, Updated n_neighbors: {self.n_neighbors}")

class RepresentativeSamplingAgent:
    def __init__(self, selection_ratio=0.7, min_samples=5, max_samples=15):
        """
        Initializes the sampling agent.
        :param selection_ratio: Fraction of anomalies to select.
        :param min_samples: Minimum number of anomalies to select.
        :param max_samples: Maximum number of anomalies to select.
        """
        self.selection_ratio = selection_ratio
        self.min_samples = min_samples
        self.max_samples = max_samples

    def select_informative_anomalies(self, detected_anomalies):
        """
        Selects the most informative anomalies using clustering and diversity sampling.
        :param detected_anomalies: Numpy array of detected anomaly embeddings.
        :return: List of selected anomaly embeddings.
        """
        num_anomalies = len(detected_anomalies)
        if num_anomalies == 0:
            return []

        # Determine selection size dynamically
        selection_size = min(max(self.min_samples, int(num_anomalies * self.selection_ratio)), 
                             self.max_samples, num_anomalies)
        
        if num_anomalies <= selection_size:
            return detected_anomalies  # Return all if anomalies are few

        # Step 1: Cluster anomalies to group similar patterns
        num_clusters = max(2, selection_size // 2)  # At least 2 clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(detected_anomalies)

        # Step 2: Select representative anomalies from each cluster
        selected_anomalies = []
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            
            # Pick the most central anomaly (closest to cluster centroid)
            cluster_center = kmeans.cluster_centers_[cluster_id]
            similarities = cosine_similarity(detected_anomalies[cluster_indices], 
                                             cluster_center.reshape(1, -1))
            most_representative_idx = cluster_indices[np.argmax(similarities)]
            selected_anomalies.append(detected_anomalies[most_representative_idx])
            
            if len(selected_anomalies) >= selection_size:
                break

        return np.array(selected_anomalies)
    

class NovelClassNamingAgent:
    def __init__(self, similarity_threshold=0.5):  # Increased from 0.3
        """
        Initializes the Novel Class Naming Agent.

        Args:
            similarity_threshold (float): Threshold to determine if an anomaly belongs to an existing cluster.
        """
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.similarity_threshold = similarity_threshold

    def classify_anomalies(self, informative_anomalies, cluster_embeddings):
        labels = []
        for i, anomaly in enumerate(informative_anomalies):
            similarities = cosine_similarity([anomaly], cluster_embeddings)[0]
            max_similarity = np.max(similarities) if similarities.size > 0 else 0
            best_cluster_idx = np.argmax(similarities) if similarities.size > 0 else None

            print(f"\n🔍 **Anomaly {i}**")
            print(f"Similarity Scores: {similarities}")
            print(f"Max Similarity: {max_similarity:.4f} (Threshold: {self.similarity_threshold})")
            print(f"Best Matching Cluster: {best_cluster_idx} (🔍 Reference Only)")

            if max_similarity >= self.similarity_threshold:
                labels.append(("existing_cluster", best_cluster_idx))
                print(f"✅ Assigned to Cluster: {best_cluster_idx}")
            else:
                labels.append(("novel_class", None))
                print(f"🚀 Identified as a **Novel Class** (Best Match: {best_cluster_idx}, but similarity too low)")

        return labels
    
    def name_novel_classes(self, novel_classes, summaries):
        """
        Assigns names to novel classes using GPT-4.

        Args:
            novel_classes (list[dict]): Novel classes to be named, each with a "text" key.
            summaries (list[str]): Summaries for context.

        Returns:
            list[dict]: Novel classes with assigned names.
        """
        named_classes = []
        
        for idx, novel_class in enumerate(novel_classes):
            anomaly_text = novel_class.get("text", "No text provided.")
            summary_text = summaries[idx] if idx < len(summaries) else "No summary available."

            prompt = (
                f"You are an AI specializing in categorizing anomalies. "
                f"Detected anomaly: '{anomaly_text}'. "
                f"Summary: '{summary_text}'. "
                "Generate a short, meaningful category name for this anomaly."
            )

            try:
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=15,
                    temperature=0.5,
                )
                name = response.choices[0].message.content.strip()

                named_classes.append({"class": novel_class, "name": name})

            except Exception as e:
                print(f"Error generating name: {e}")
                named_classes.append({"class": novel_class, "name": "Error generating name"})

        return named_classes

    def process_anomalies(self, informative_anomalies, cluster_embeddings, summaries):
        """
        Classifies anomalies and integrates them into existing clusters or creates new clusters.
        
        Args:
            informative_anomalies (np.array): Embeddings of selected anomalies.
            cluster_embeddings (np.array): Embeddings of existing clusters.
            summaries (list[str]): Summaries for naming novel classes.

        Returns:
            dict: Updated clusters with novel classes added.
        """
        labels = self.classify_anomalies(informative_anomalies, cluster_embeddings)
        print("Labels:", labels)

        updated_clusters = cluster_embeddings.copy()
        novel_classes = []

        for idx, (label, cluster_idx) in enumerate(labels):
            if label == "existing_cluster" and cluster_idx is not None:
                # Merge into the closest existing cluster
                updated_clusters[cluster_idx] = (updated_clusters[cluster_idx] + informative_anomalies[idx]) / 2
            else:
                # Treat novel classes as new clusters
                updated_clusters = np.vstack([updated_clusters, informative_anomalies[idx]])  # Append new cluster
                novel_classes.append({
                    "text": summaries[idx],
                    "anomaly_text": informative_anomalies[idx].tolist()
                })

        # Name the novel clusters
        novel_class_names = self.name_novel_classes(novel_classes, summaries) if novel_classes else []

        return {
            "updated_clusters": updated_clusters,
            "novel_classes": novel_class_names,
        }