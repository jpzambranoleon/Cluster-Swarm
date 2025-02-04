import os
import json
import numpy as np
from helper_functions import fetch_data, get_or_generate_embeddings, get_dynamic_n_clusters, get_clusters, feedback_loop, get_dynamic_n_neighbors
from agent_network import SummarizationAgent, AnomalyDetectionAgent, RepresentativeSamplingAgent, NovelClassNamingAgent


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

    summarization_agent = SummarizationAgent(model_name_or_path="t5-large")

    # Generate updated summaries for the refined clusters
    updated_summaries = summarization_agent.generate_summaries(
        refined_cluster_texts, max_length=max_length, min_length=min_length
    )

    return updated_summaries, refined_cluster_texts

def save_clusters_to_json(clusters, output_file):
    """
    Save generated clusters into the 'cluster' key of the JSON file.

    Args:
        data_file (str): Path to the input JSON file.
        clusters (dict): Dictionary where keys are cluster numbers and values are lists of indices.
        output_file (str): Path to save the updated JSON file.
    """
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    index_to_cluster = {}
    for cluster_num, indices in clusters.items():
        for index in indices:
            index_to_cluster[index] = str(cluster_num)  # Convert cluster number to string
    
    for i, entry in enumerate(data):
        entry["cluster"] = index_to_cluster.get(i, "")  # Assign cluster number or empty string if not found
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Updated JSON with clusters saved to {output_file}")

def group_text_by_cluster(json_file):
    """
    Groups text by cluster and returns a list of concatenated strings.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        list: A list where each entry contains concatenated text from the same cluster.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cluster_dict = {}
    
    for entry in data:
        cluster_num = entry["cluster"]
        text = entry["text"]
        
        if cluster_num in cluster_dict:
            cluster_dict[cluster_num].append(text)
        else:
            cluster_dict[cluster_num] = [text]
    
    grouped_texts = [" ".join(texts) for texts in cluster_dict.values()]
    
    return grouped_texts

def save_summaries_to_json(summaries, output_file):
    """
    Save summaries into the 'summary' key of the JSON file, matching them to the correct cluster.

    Args:
        json_file (str): Path to the input JSON file.
        summaries (list): List of summaries where each index corresponds to a cluster.
        output_file (str): Path to save the updated JSON file.
    """
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a mapping of cluster number to summary
    cluster_summary_map = {str(i): summaries[i] for i in range(len(summaries))}

    for entry in data:
        cluster_num = entry["cluster"]
        if cluster_num in cluster_summary_map:
            entry["cluster_summary"] = cluster_summary_map[cluster_num]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Updated JSON with summaries saved to {output_file}")

def identify_anomalous_clusters(cluster_embeddings, detected_anomalies):
    """
    Identify which clusters are anomalies based on the detected anomaly embeddings.

    Args:
        cluster_embeddings (dict): A dictionary where keys are cluster numbers, 
                                   and values are their corresponding embeddings.
        detected_anomalies (np.ndarray): An array of detected anomaly embeddings.

    Returns:
        list: A list of cluster numbers that are identified as anomalies.
    """
    anomalous_clusters = set()

    for cluster_num, embedding in cluster_embeddings.items():
        # Check if the cluster embedding is close to any detected anomaly
        for anomaly_embedding in detected_anomalies:
            if np.allclose(embedding, anomaly_embedding, atol=1e-5):  # Adjust tolerance if needed
                anomalous_clusters.add(cluster_num)

    return list(anomalous_clusters)


def main():

    # Fetch Data from Static File or API
    static = "datasets/banking77/test_json"
    base_url = "https://api.nasa.gov/planetary/apod"

    text_embeds_path = f"embeddings/text_data_embeds_e5.hdf5"

    cluster_embeds_path = f"embeddings/text_data_cluster_embeds_e5.hdf5"
    summary_embeds_path = f"embeddings/text_data_summary_embeds_e5.hdf5"

    fetch_data(
        source=base_url,
        api_key=os.getenv("NASA_API_KEY"),
        is_api=True,
        count=100,
        output_file="results/output.json",
        text_key="explanation"
    )

    dataset = "results/output.json"

    # Step 1: Generate or load embeddings
    text_embeddings = get_or_generate_embeddings(
        data=dataset,
        file_path=text_embeds_path,
        model_name_or_path="intfloat/e5-large"
    )

    n_clusters = get_dynamic_n_clusters(text_embeddings, method="elbow")
    print(f"Optimal number of clusters: {n_clusters}")

    # Step 2: Generate clusters
    print("\n\nGenerating initial clusters!")
    clusters = get_clusters(scale=None, embeddings=text_embeddings, n_clusters=n_clusters)

    save_clusters_to_json(clusters, dataset)

    clustered_texts = group_text_by_cluster(dataset)

    # Step 2: Generate or load cluster embeddings
    cluster_embeddings = get_or_generate_embeddings(
        data=clustered_texts,
        file_path=cluster_embeds_path,
        model_name_or_path="intfloat/e5-large"
    )

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
    summaries = summarization_agent.generate_summaries(clustered_texts, max_length=100, min_length=50)

    save_summaries_to_json(summaries, dataset)

    summary_embeddings = get_or_generate_embeddings(
        data=summaries,
        file_path=summary_embeds_path,
        model_name_or_path="intfloat/e5-large"
    )

    # Step 5: Find Anomalies
    # Initialize the Anomaly Detection Agent
    num_samples = len(cluster_embeddings) + len(summary_embeddings)
    initial_n_neighbors = get_dynamic_n_neighbors(num_samples)
    anomaly_agent = AnomalyDetectionAgent(method="knn", initial_n_neighbors=initial_n_neighbors, initial_contamination=0.05, memory_decay=0.8)
    representative_sampling_agent = RepresentativeSamplingAgent()

    # Track previous selected anomalies for comparison
    previous_anomalies = None  
    max_iterations = len(summary_embeddings)  
    convergence_threshold = 0.9  # Stop if 90% of anomalies are the same

    print("Initial N Neighbors:", initial_n_neighbors)

    for iteration in range(max_iterations):
        print(f"\n### Iteration {iteration + 1} ###")
        
        # Step 1: Detect anomalies
        detected_anomalies = anomaly_agent.detect_anomalies(cluster_embeddings, summary_embeddings)
        print("Detected anomalies:", len(detected_anomalies))
        
        # Step 2: Representative Sampling selects informative anomalies
        selected_informative_anomalies = representative_sampling_agent.select_informative_anomalies(detected_anomalies)
        
        # Step 3: Refine anomaly detection based on feedback
        anomaly_agent.refine_anomaly_detection(detected_anomalies, selected_informative_anomalies)
        
        # Step 4: Check stopping criteria
        if previous_anomalies is not None:
            # Calculate similarity between previous and current selected anomalies
            common_anomalies = set(map(tuple, previous_anomalies)) & set(map(tuple, selected_informative_anomalies))
            similarity = len(common_anomalies) / max(len(previous_anomalies), 1)  # Avoid division by zero
            
            if similarity >= convergence_threshold:
                print("Anomaly detection has stabilized.")
                break  # Stop iteration
        
        # Update previous anomalies for next iteration
        previous_anomalies = selected_informative_anomalies

    print("Final anomalies selected:", selected_informative_anomalies, len(selected_informative_anomalies))


    naming_agent = NovelClassNamingAgent()
    novel_classes = naming_agent.process_anomalies(selected_informative_anomalies, cluster_embeddings, summaries)

    print(novel_classes)
            

if __name__ == '__main__':
    main()