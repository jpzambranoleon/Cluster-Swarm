import subprocess

def run_triplet_sampling(dataset, scale, max_query, embed):
    """
    Run the triplet_sampling.py script with specified parameters.

    Args:
    dataset (str): The dataset to process.
    scale (str): The scale of the dataset.
    max_query (int): The maximum number of queries.
    embed (str): The embedding method.
    """
    feat_path = f"../datasets/{dataset}/{scale}_embeds.hdf5"
    command = [
        "python", "triplet_sampling.py",
        "--data_path", f"../datasets/{dataset}/{scale}.jsonl",
        "--feat_path", feat_path,
        "--dataset", dataset,
        "--embed_method", embed,
        "--max_query", str(max_query),
        "--filter_first_prop", "0.0",
        "--large_ent_prop", "0.2",
        "--out_dir", "sampled_triplet_results",
        "--max_distance", "67",
        "--scale", scale,
        "--shuffle_inds",
        "--seed", "100"
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Successfully processed {dataset} with {embed} embedding and max_query {max_query}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {dataset}: {e}")

def main():
    scale = "small"
    datasets = ["banking77"]
    max_queries = [1024]
    embeds = ["instructor"]

    for dataset in datasets:
        for max_query in max_queries:
            for embed in embeds:
                run_triplet_sampling(dataset, scale, max_query, embed)

if __name__ == "__main__":
    main()
