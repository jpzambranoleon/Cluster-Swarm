import os
import json
import h5py
import numpy as np
from agent_network import EncoderAgent, ClusteringAgent, SummarizationAgent, AnomalyDetectionAgent

def refine_embeddings_with_clusters(embeddings, clusters, learning_rate=0.01):
    """
    Refines embeddings based on cluster information.

    Parameters:
        embeddings (np.ndarray): Original semantic embeddings (shape: [n_samples, embedding_dim]).
        clusters (dict): Clustering results with structure:
            {
                cluster_id: [list_of_sample_indices_in_cluster]
            }
        learning_rate (float): Step size for updating embeddings.

    Returns:
        np.ndarray: Refined embeddings.
    """
    refined_embeddings = embeddings.copy()

    # Calculate centroids for each cluster
    centroids = {
        cluster_id: np.mean(refined_embeddings[sample_indices], axis=0)
        for cluster_id, sample_indices in clusters.items()
    }

    # Adjust embeddings to move closer to their cluster centroids
    for cluster_id, sample_indices in clusters.items():
        for sample_idx in sample_indices:
            refined_embeddings[sample_idx] += learning_rate * (
                centroids[cluster_id] - refined_embeddings[sample_idx]
            )

    return refined_embeddings


def feedback_loop(embeddings, clusters, max_iterations=10, convergence_threshold=0.01):
    """
    Feedback loop for refining embeddings based on provided clusters.

    Parameters:
        embeddings (np.ndarray): Initial semantic embeddings.
        clusters (dict): Clustering results with structure:
            {
                cluster_id: [list_of_sample_indices_in_cluster]
            }
        max_iterations (int): Maximum number of refinement iterations.
        convergence_threshold (float): Threshold for stopping based on embedding changes.

    Returns:
        np.ndarray: Final refined embeddings.
    """
    for iteration in range(max_iterations):
        # Refine embeddings based on the provided clusters
        new_embeddings = refine_embeddings_with_clusters(embeddings, clusters)

        # Check convergence (average change in embeddings)
        avg_change = np.mean(np.linalg.norm(new_embeddings - embeddings, axis=1))
        print(f"Iteration {iteration + 1}: Avg change in embeddings = {avg_change:.6f}")

        if avg_change < convergence_threshold:
            print(f"Converged after {iteration + 1} iterations with avg change: {avg_change}")
            break

        embeddings = new_embeddings

    return embeddings

def get_embeddings(input_data, output_path, model_name_or_path, checkpoint=None, l2_normalize=False):
    """
    Generate embeddings from input data, which can be a JSON file path or a variable containing text data.

    Args:
        input_data (str or list): Path to the input JSON file or a list of strings as input.
        output_path (str): Path to save the embeddings in HDF5 format.
        model_name_or_path (str): Path to the model or model identifier.
        checkpoint (str, optional): Path to a checkpoint for model fine-tuning.
        l2_normalize (bool, optional): Whether to L2 normalize embeddings.
    """
    # Check if input_data is a file path
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            print(f"Input file {input_data} does not exist. Skipping.")
            return

        # Read the input JSONL file
        with open(input_data, 'r') as f:
            data = [json.loads(line) for line in f]

        # Extract texts from the input data
        texts = [datum['input'] for datum in data]

    elif isinstance(input_data, list):
        # Use input_data directly if it's a list of texts
        texts = input_data

    else:
        print(f"Invalid input type: {type(input_data)}. Must be a file path (str) or list of texts.")
        return

    # Check if embeddings file already exists
    if os.path.exists(output_path):
        print(f"Embeddings file {output_path} already exists. Skipping generation.")
        return

    # Set environment variables for CUDA and threading
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    # Initialize the model
    model = EncoderAgent(model_name_or_path, checkpoint, l2_normalize=l2_normalize)

    # Generate embeddings
    embeds = model.encode(texts)

    # Save embeddings to the HDF5 file
    with h5py.File(output_path, 'w') as f:
        f.create_dataset("embeds", data=embeds)

    print("--DONE--")
    return embeds

def get_clusters(scale, embeddings, n_clusters):
    cluster_agent = ClusteringAgent(scale=scale)
    cluster_labels = cluster_agent.fit_agent_predict(embeddings=embeddings, n_clusters=n_clusters)

    # Create a dictionary mapping cluster IDs to sample indices
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        clusters.setdefault(label, []).append(idx)

    return clusters

def get_summaries(input_path, clusters):
    # clusters is already a dictionary mapping cluster IDs to sample indices
    cluster_map = clusters

    # Read texts for summarization
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f]
    texts = [datum['input'] for datum in data]

    summarization_agent = SummarizationAgent(model_name_or_path="facebook/bart-large-cnn")
    summaries = summarization_agent.generate_summaries(cluster_map, texts, max_length=50, min_length=5)

    return summaries

def main():
    datasets = ["banking77"]
    scales = ["test_json"]

    n_clusters = 4

    for dataset in datasets:
        for scale in scales:
            input_path = f"datasets/{dataset}/{scale}.jsonl"
            output_path = f"datasets/{dataset}/{scale}_embeds_e5.hdf5"

            summary_path = f"datasets/{dataset}/{scale}_summary_embeds_e5.hf5"

            # Step 1: Generate or load embeddings
            print("Generating initial embeddings!")
            if not os.path.exists(output_path):
                embeddings = get_embeddings(input_path, output_path, model_name_or_path="intfloat/e5-large")

            else:
                with h5py.File(output_path, 'r') as f:
                    embeddings = np.asarray(f['embeds'])

            # Step 2: Generate clusters
            ("\n\nGenerating initial clusters!")
            clusters = get_clusters(scale, embeddings, n_clusters)
            print(f"Initial clusters:\n{clusters}")

            
            # Step 3: Perform feedback loop for refining embeddings
            print("\n\nStarting feedback loop for embedding refinement...")
            refined_embeddings = feedback_loop(
                embeddings=embeddings,
                clusters=clusters,
                max_iterations=10,
                convergence_threshold=0.01,
            )

            # Step 4: Generate final clusters using refined embeddings
            print("\n\nGenerating final clusters using refined embeddings")
            final_clusters = get_clusters(scale, refined_embeddings, n_clusters)
            print(f"Final clusters after refinement:\n{final_clusters}")  

            # Step 5: Generate summaries
            print("\n\nGenerating summaries!")
            summaries = get_summaries(input_path, final_clusters)
            summary_texts = list(summaries.values())
            for cluster_id, summary in summaries.items():
                print(f"Cluster {cluster_id}: {summary}")   

            print(f"Here are the summary texts:\n{summary_texts}")

            #Generate or load embeddings from summary
            print("\n\nGenerating embeddings for summaries!")
            if not os.path.exists(summary_path):
                summary_embeddings = get_embeddings(summary_texts, summary_path, model_name_or_path="intfloat/e5-large")

            else:
                with h5py.File(summary_path, 'r') as f:
                    summary_embeddings = np.asarray(f['embeds'])

            print(f"Summary Embeddings:\n{summary_embeddings}")

            # Step 6: Anomaly Detection
            print("\n\nPerforming anomaly detection!")
            # Initialize the agent
            anomaly_agent = AnomalyDetectionAgent(n_neighbors=3, threshold_percentile=95)

            # Detect anomalies
            anomalies, scores = anomaly_agent.detect_anomalies(summary_embeddings, embeddings)

            print("Summary Anomalies:", anomalies['summary_anomalies'])
            print("Cluster Anomalies:", anomalies['cluster_anomalies'])

if __name__ == '__main__':
    main()