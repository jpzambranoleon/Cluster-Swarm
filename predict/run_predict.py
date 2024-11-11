import subprocess
import os

def run_predict_triplet(dataset, link_path, model_name):
    """
    Run the predict_triplet.py script with specified parameters.

    Args:
    dataset (str): The dataset to process.
    link_path (str): Path to the input data file.
    openai_org (str): OpenAI organization ID.
    model_name (str): Name of the model to use.
    temperature (float): Temperature parameter for the model.
    """
    command = [
        "python", "predict_triplet.py",
        "--dataset", dataset,
        "--data_path", link_path,
        "--llm", model_name,
    ]

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "OPENAI_API_KEY"  # Replace with your actual API key
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"

    try:
        subprocess.run(command, env=env, check=True)
        print(f"Successfully processed {dataset} with model {model_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {dataset}: {e}")

def main():
    datasets = ["banking77"]
    model_name = "gpt-4o-mini"
    #temperature = 0

    for dataset in datasets:
        link_path = f"sampled_triplet_results/{dataset}_embed=instructor_s=small_m=1024_d=67.0_sf_choice_seed=100.json"
        # Uncomment the following line if you want to use the large embed version
        # link_path = f"sampled_triplet_results/{dataset}_embed=instructor_s=large_m=1024_d=67.0_sf_choice_seed=100.json"
        
        run_predict_triplet(dataset, link_path, model_name)

if __name__ == "__main__":
    main()