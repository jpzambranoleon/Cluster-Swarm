import h5py
import os
from dotenv import load_dotenv
import json
import requests
import uuid
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from agent_network import EncoderAgent, ClusteringAgent, SummarizationAgent

def fetch_data(source=None, api_key=None, is_api=False, count=None, output_file="output.json", text_key="text"):
    """
    Fetches text data from a static JSON file or a dynamic API and stores it in a JSON file.
    
    :param source: Path to a static JSON/JSONL file or API URL.
    :param base_url: Base URL of the API (only needed if is_api=True).
    :param api_key: API key for authentication (if needed).
    :param is_api: Boolean flag indicating if source is an API.
    :param output_file: Name of the output JSON file.
    :param text_key: Key to extract text data from the source.
    """
    
    results = []
    
    if is_api:
        # Load API key from environment if not provided
        load_dotenv()
        api_key = api_key or os.getenv("API_KEY")

        # Define search parameters
        params = {
            "api_key": api_key,
            "count": count
            }  # Adjust based on API needs

        try:
            response = requests.get(source, params=params, timeout=10)
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from API: {e}")
            return
    else:
        # Read from static JSON/JSONL file
        try:
            with open(source, "r", encoding="utf-8") as file:
                # Handle both JSON and JSONL formats
                if source.endswith(".jsonl"):
                    data = [json.loads(line) for line in file]
                else:
                    data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading file: {e}")
            return
    
    # Process each JSON object
    for entry in data:
        text_data = entry.get(text_key, "")  # Extract text using provided key
        results.append({
            "id": str(uuid.uuid4()),  # Generate a unique ID
            "text": text_data
        })
    
    # Save processed data to output file
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)
        print(f"Data saved to {output_file}")
    except IOError as e:
        print(f"Error writing to file: {e}")

def get_or_generate_embeddings(data, file_path, model_name_or_path, checkpoint=None, l2_normalize=False):
    """
    Get embeddings from an HDF5 file if it exists. Otherwise, generate and save embeddings to the file.

    Args:
        data (str or list): Path to input JSON file or list of text strings.
        file_path (str): Path to HDF5 file for storing/loading embeddings.
        model_name_or_path (str): Path or identifier of the model to generate embeddings.
        checkpoint (str, optional): Model checkpoint for fine-tuning.
        l2_normalize (bool, optional): Whether to L2 normalize embeddings.

    Returns:
        np.ndarray: Embeddings array loaded or generated.
    """
    if os.path.exists(file_path):
        print(f"Loading embeddings from {file_path}.")
        with h5py.File(file_path, 'r') as f:
            return np.asarray(f['embeds'])

    print(f"Generating embeddings and saving to {file_path}.")
    # Load data if input is a file path
    if isinstance(data, str) and os.path.exists(data):
        with open(data, 'r', encoding='utf-8') as f:
            input_data = json.load(f)  # Read the JSON array
        text_data = [entry["text"] for entry in input_data]  # Extract text field
    elif isinstance(data, list):
        text_data = data
    else:
        raise ValueError("Input must be a file path or a list of strings.")
    
    # Set environment variables for CUDA and threading
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    # Generate embeddings
    model = EncoderAgent(model_name_or_path, checkpoint, l2_normalize=l2_normalize)
    embeddings = model.encode(text_data)  # Encode extracted text data

    # Save to HDF5 file
    with h5py.File(file_path, 'w') as f:
        f.create_dataset("embeds", data=embeddings)

    return embeddings


def get_dynamic_n_clusters(data, method="hybrid", min_clusters=5, max_clusters=35):
    """
    Dynamically determines the optimal number of clusters for K-Means.

    Parameters:
        data (np.array): Data to cluster.
        method (str): Clustering method ('sqrt', 'elbow', 'silhouette', 'hybrid').
        min_clusters (int): Minimum number of clusters.
        max_clusters (int): Maximum number of clusters.

    Returns:
        int: Optimal number of clusters.
    """
    num_samples = len(data)

    if num_samples < min_clusters:
        return min_clusters  # Ensure at least some clusters

    if method == "sqrt":
        k = int(np.sqrt(num_samples))  # Simple square root rule

    elif method == "elbow":
        distortions = []
        for k in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
            distortions.append(kmeans.inertia_)  # Sum of squared distances to centroids
        k = np.argmin(np.diff(distortions, 2)) + min_clusters  # Find elbow point

    elif method == "silhouette":
        best_k = min_clusters
        best_score = -1
        for k in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
            score = silhouette_score(data, kmeans.labels_)
            if score > best_score:
                best_score = score
                best_k = k
        k = best_k

    else:  # Hybrid: Start with sqrt rule, refine with silhouette if needed
        k = int(np.sqrt(num_samples))
        if num_samples > 50:  # Only refine for larger datasets
            best_k = min_clusters
            best_score = -1
            for k_try in range(max(min_clusters, k - 2), min(max_clusters, k + 3)):  
                kmeans = KMeans(n_clusters=k_try, random_state=42, n_init=10).fit(data)
                score = silhouette_score(data, kmeans.labels_)
                if score > best_score:
                    best_score = score
                    best_k = k_try
            k = best_k

    return max(min_clusters, min(max_clusters, k))  # Keep within bounds

def get_clusters(scale, embeddings, n_clusters):
    cluster_agent = ClusteringAgent(scale=scale)
    cluster_labels = cluster_agent.fit_agent_predict(embeddings=embeddings, n_clusters=n_clusters)

    # Create a dictionary mapping cluster IDs to sample indices
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        clusters.setdefault(label, []).append(idx)

    return clusters

def refine_embeddings_with_clusters(text_embeddings, cluster_embeddings, clusters, learning_rate=0.01):
    """
    Refines embeddings based on cluster centroids.

    Parameters:
        embeddings (np.ndarray): Original semantic embeddings (shape: [n_samples, embedding_dim]).
        cluster_embeddings (np.ndarray): Embeddings for cluster centroids.
        clusters (dict): Mapping of cluster IDs to sample indices.
        learning_rate (float): Step size for updating embeddings.

    Returns:
        np.ndarray: Refined embeddings.
    """
    refined_embeddings = text_embeddings.copy()

    # Adjust embeddings to move closer to their cluster centroids
    for cluster_id, sample_indices in clusters.items():
        for sample_idx in sample_indices:
            refined_embeddings[sample_idx] += learning_rate * (
                cluster_embeddings[cluster_id] - refined_embeddings[sample_idx]
            )

    return refined_embeddings

def feedback_loop(text_embeddings, clusters, cluster_embeddings, max_iterations=10, convergence_threshold=0.01):
    """
    Feedback loop for refining embeddings based on provided clusters and cluster embeddings.

    Parameters:
        embeddings (np.ndarray): Initial semantic embeddings.
        clusters (dict): Mapping of cluster IDs to sample indices.
        cluster_texts (list): Texts representing each cluster for generating cluster embeddings.
        model_name_or_path (str): Path or identifier for the model to generate embeddings.
        max_iterations (int): Maximum number of refinement iterations.
        convergence_threshold (float): Threshold for stopping based on embedding changes.

    Returns:
        np.ndarray: Final refined embeddings.
    """
    for iteration in range(max_iterations):
        # Step 1: Refine text embeddings using cluster embeddings
        new_embeddings = refine_embeddings_with_clusters(
            text_embeddings, cluster_embeddings, clusters
        )

        # Step 2: Check convergence
        avg_change = np.mean(np.linalg.norm(new_embeddings - text_embeddings, axis=1))
        print(f"Iteration {iteration + 1}: Avg change in embeddings = {avg_change:.6f}")

        if avg_change < convergence_threshold:
            print(f"Converged after {iteration + 1} iterations with avg change: {avg_change:.6f}")
            break

        text_embeddings = new_embeddings

    return text_embeddings

def get_summaries(cluster_texts, model_name_or_path="facebook/bart-large-cnn", max_length=50, min_length=5):
    """
    Generates summaries for each cluster using pre-processed cluster texts.

    Parameters:
        cluster_texts (list): List of concatenated texts for each cluster.
        model_name_or_path (str): Pre-trained summarization model identifier.
        max_length (int): Maximum length of generated summaries.
        min_length (int): Minimum length of generated summaries.

    Returns:
        list: A list of summaries, one for each cluster.
    """
    # Step 1: Initialize the SummarizationAgent
    summarization_agent = SummarizationAgent(model_name_or_path=model_name_or_path)
    
    # Step 2: Generate summaries for the cluster texts
    summaries = summarization_agent.generate_summaries(cluster_texts, max_length=max_length, min_length=min_length)

    return summaries

def get_dynamic_n_neighbors(num_samples, min_neighbors=3, max_neighbors=50, method="hybrid"):
    """
    Dynamically determines the optimal number of neighbors for KNN anomaly detection.

    Parameters:
        num_samples (int): Total number of data points.
        min_neighbors (int): Minimum number of neighbors to use.
        max_neighbors (int): Maximum number of neighbors to use.
        method (str): Method for calculating neighbors ('sqrt', 'log', 'hybrid').

    Returns:
        int: Dynamically computed number of neighbors.
    """
    if num_samples < 10:  # Ensure at least some data
        return min_neighbors

    if method == "sqrt":
        n_neighbors = int(np.sqrt(num_samples))  # Square root scaling
    elif method == "log":
        n_neighbors = int(np.log(num_samples) * 5)  # Log-based scaling (adjust factor if needed)
    else:  # Hybrid approach
        n_neighbors = int((np.sqrt(num_samples) + np.log(num_samples) * 5) / 2)  

    return max(min_neighbors, min(max_neighbors, n_neighbors))  # Keep within bounds