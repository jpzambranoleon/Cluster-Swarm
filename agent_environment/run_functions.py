import os
import json
import h5py
import numpy as np
from agent_network import EncoderAgent, ClusteringAgent, SummarizationAgent, AnomalyDetectionAgent, RepresentativeSamplingAgent, NovelClassNamingAgent

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


def refine_summaries_with_feedback(
    cluster_texts,
    summary_embeddings,
    anomalies,
    summarization_agent,
    max_length=100,
    min_length=50,
):
    """
    Refines summaries based on anomaly feedback from the Anomaly Detection Agent.

    Parameters:
        cluster_texts (list): List of concatenated texts for each cluster.
        summary_embeddings (np.array): Summary embeddings corresponding to clusters.
        anomalies (dict): Detected anomalies as {index: anomaly_score}.
        summarization_agent (SummarizationAgent): Initialized summarization agent.
        max_length (int): Maximum length of generated summaries.
        min_length (int): Minimum length of generated summaries.

    Returns:
        list: Updated summaries for the refined clusters.
        list: New cluster texts after refinement.
    """
    refined_cluster_texts = []
    new_clusters = []

    for idx, texts in enumerate(cluster_texts):
        if idx in anomalies:
            # Example: Flagged cluster—adjust or split the cluster
            print(f"Anomaly detected in cluster {idx} (score: {anomalies[idx]:.2f}).")
            
            # Split or refine texts based on anomaly logic
            cluster_items = texts.split(". ")
            refined_texts = [item for item in cluster_items if len(item.split()) > 3]

            # If significant anomalies are found, treat as a new cluster
            if len(refined_texts) < len(cluster_items):
                print(f"Creating a new cluster for outliers in cluster {idx}.")
                outliers = set(cluster_items) - set(refined_texts)
                new_clusters.append(". ".join(outliers))
            
            refined_cluster_texts.append(". ".join(refined_texts))
        else:
            refined_cluster_texts.append(texts)

    # Add new clusters to the list of cluster texts
    refined_cluster_texts.extend(new_clusters)

    # Generate updated summaries for the refined clusters
    updated_summaries = summarization_agent.generate_summaries(
        refined_cluster_texts, max_length=max_length, min_length=min_length
    )

    return updated_summaries, refined_cluster_texts

def enrich_selected_anomalies(selected_anomalies, cluster_embeddings, cluster_texts):
    """
    Enrich selected anomalies with their corresponding text.

    Parameters:
        selected_anomalies (list): List of selected anomaly embeddings or dictionaries containing embeddings.
        cluster_embeddings (np.array): Cluster embeddings corresponding to cluster_texts.
        cluster_texts (list): List of cluster texts.

    Returns:
        list: Enriched anomalies with embeddings and corresponding text.
    """
    # Extract embeddings if anomalies are dictionaries
    if isinstance(selected_anomalies[0], dict) and "embedding" in selected_anomalies[0]:
        selected_anomalies = [anomaly["embedding"] for anomaly in selected_anomalies]

    enriched_anomalies = []
    
    for selected in selected_anomalies:
        # Find the closest matching cluster embedding for each selected anomaly
        distances = np.linalg.norm(cluster_embeddings - selected, axis=1)
        closest_index = np.argmin(distances)
        
        # Create an enriched anomaly with text
        enriched_anomaly = {
            "embedding": selected,
            "text": cluster_texts[closest_index]
        }
        enriched_anomalies.append(enriched_anomaly)
    
    return enriched_anomalies


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

def main():
    datasets = ["banking77"]
    scales = ["test_json"]

    n_clusters = 4

    for dataset in datasets:
        for scale in scales:
            input_path = f"datasets/{dataset}/{scale}.jsonl"
            output_path = f"datasets/{dataset}/{scale}_embeds_e5.hdf5"

            cluster_embeds_path = f"datasets/{dataset}/{scale}_cluster_embeds_e5.hdf5"
            summary_path = f"datasets/{dataset}/{scale}_summary_embeds_e5.hdf5"

            # Step 1: Generate or load embeddings
            print("Generating initial text embeddings!")
            if not os.path.exists(output_path):
                text_embeddings = get_embeddings(input_path, output_path, model_name_or_path="intfloat/e5-large")
            else:
                with h5py.File(output_path, 'r') as f:
                    text_embeddings = np.asarray(f['embeds'])

            # Step 2: Generate clusters
            ("\n\nGenerating initial clusters!")
            clusters = get_clusters(scale, text_embeddings, n_clusters)
            print(f"Initial clusters:\n{clusters}")

            # Read the input JSONL file
            with open(input_path, 'r') as f:
                data = [json.loads(line) for line in f]

            # Extract texts from the input data
            texts = [datum['input'] for datum in data]

            # Prepare cluster-level texts
            cluster_texts = []
            for cluster_id, indices in clusters.items():
                cluster_text = " ".join([texts[idx] for idx in indices])  # Combine texts in a cluster
                cluster_texts.append(cluster_text)

            print(f"Here are the Cluster Texts:\n{cluster_texts}")

            # Generate or load embeddings for clusters
            print("Generating cluster embeddings!")
            if not os.path.exists(cluster_embeds_path):
                cluster_embeddings = get_embeddings(cluster_texts, cluster_embeds_path, model_name_or_path="intfloat/e5-large")
            else:
                with h5py.File(cluster_embeds_path, 'r') as f:
                    cluster_embeddings = np.asarray(f['embeds'])

            # Step 3: Perform feedback loop for refining embeddings
            print("\n\nStarting feedback loop for embedding refinement...")
            refined_embeddings = feedback_loop(
                text_embeddings=text_embeddings,
                clusters=clusters,
                cluster_embeddings=cluster_embeddings,
                max_iterations=10,
                convergence_threshold=0.01,
            )

            # Step 4: Generate summaries
            print("\n\nGenerating summaries!")

            # Instantiate summarization agent
            summarization_agent = SummarizationAgent(model_name_or_path="t5-large")

            # Generate summaries for the pre-concatenated cluster texts
            summaries = summarization_agent.generate_summaries(cluster_texts, max_length=100, min_length=50)

            # Print summaries for debugging
            for idx, summary in enumerate(summaries):
                print(f"Cluster {idx}: {summary}")

            #Generate or load embeddings from summary
            print("\n\nGenerating embeddings for summaries!")
            if not os.path.exists(summary_path):
                summary_embeddings = get_embeddings(summaries, summary_path, model_name_or_path="intfloat/e5-large")

            else:
                with h5py.File(summary_path, 'r') as f:
                    summary_embeddings = np.asarray(f['embeds'])

            print(f"Summary Embeddings:\n{summary_embeddings}")

            # Step 5: Get Anomalies
            # Initialize the agent
            anomaly_agent = AnomalyDetectionAgent(method="knn", n_neighbors=3, contamination=0.2)

            # Detect anomalies
            detected_anomalies = anomaly_agent.detect_anomalies(cluster_embeddings, summary_embeddings)

            print("Detected Anomalies:")
            print(detected_anomalies)

            # Step 6
            updated_summaries, refined_cluster_texts = refine_summaries_with_feedback(
                cluster_texts,
                summary_embeddings,
                detected_anomalies,
                summarization_agent,
                max_length=100,
                min_length=50,
            )

            for idx, summary in enumerate(updated_summaries):
                print(f"Cluster {idx}: {summary}")

            # Convert detected anomalies to the expected input format for sampling
            formatted_anomalies = [{"embedding": emb.tolist()} for emb in detected_anomalies]

            # Step 7: Representative Sampling Agent
            sampler = RepresentativeSamplingAgent(sampling_strategy="diversity", top_k=1)
            selected_anomalies = sampler.sample_anomalies(formatted_anomalies)

            # Print selected anomalies
            print("Selected Anomalies:")
            print(selected_anomalies)

            # Enrich selected anomalies
            enriched_anomalies = enrich_selected_anomalies(selected_anomalies, cluster_embeddings, cluster_texts)
            print(enriched_anomalies)

            for anomaly in enriched_anomalies:
                distances = np.linalg.norm(cluster_embeddings - anomaly["embedding"], axis=1)
                closest_index = np.argmin(distances)
                print(f"Anomaly: {anomaly['text']}, Closest Cluster Text: {summaries[closest_index]}")

            # Instantiate and process
            """
            naming_agent = NovelClassAndNamingAgent(threshold=0.5)
            result = naming_agent.process_anomalies(selected_anomalies, cluster_embeddings, updated_summaries)

            print("Classified Anomalies:", result["classified_anomalies"])
            print("Named Novel Classes:", result["named_novel_classes"])
            """

            """
            # Example inputs
            informative_anomalies = [
                {"embedding": [0.1, 0.3, 0.4], "text": "This is a unique issue with card activation."},
                {"embedding": [0.9, 0.8, 0.7], "text": "Another anomaly around incorrect exchange rates."}
            ]
            cluster_embeddings = np.array([
                [0.2, 0.4, 0.5],
                [0.8, 0.9, 0.6]
            ])
            summaries = [
                "An issue related to card activation and validation.",
                "Anomaly regarding exchange rate discrepancies."
            ]
            """

            # Instantiate and process
            naming_agent = NovelClassNamingAgent()
            named_classes = naming_agent.name_novel_classes(enriched_anomalies, updated_summaries)

            print("Named Novel Classes:", named_classes)
            

if __name__ == '__main__':
    main()