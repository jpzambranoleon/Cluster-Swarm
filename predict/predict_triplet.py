"""
GAIA Tool-based Task-oriented Thinking (ToolTOT) Evaluation Script

This script runs an evaluation of the ToolTOT agent on a specified dataset,
typically for tasks like question answering or classification.

It uses the GAIA (General AI Agent) framework and the GPTSwarm system for processing.

Usage:
    python script_name.py [arguments]

Required packages:
    os, sys, argparse, json, yaml, time, tqdm, asyncio, pathlib
    Custom modules: tools, swarm.graph.swarm, swarm.environment.tools.reader.readers,
    swarm.environment.agents.io, swarm.environment.agents.gaia, swarm.environment.operations,
    swarm.memory.memory, swarm.utils.globals, swarm.utils.log
"""

import os
import sys
import argparse
import json
import yaml
import time
import asyncio
from tools import prepare_data, post_process
from pathlib import Path

from swarm.graph.swarm import Swarm
from swarm.environment.tools.reader.readers import JSONReader, YAMLReader
from swarm.environment.agents.io import IO
from swarm.environment.agents.gaia.normal_io import NormalIO
from swarm.environment.agents.gaia.tool_io import ToolIO
from swarm.environment.agents.gaia.web_io import WebIO
from swarm.environment.agents.gaia.tool_tot import ToolTOT
from swarm.environment.operations import DirectAnswer
from swarm.memory.memory import GlobalMemory
from swarm.utils.globals import Time, Cost
from swarm.utils.log import initialize_log_file, logger, swarmlog
from swarm.environment.domain.gaia import question_scorer
from swarm.environment.operations.final_decision import MergingStrategy

def dataloader(data_list):
    """
    Generator function to iterate over a list of data.

    Args:
        data_list (list): List of data items to be yielded.

    Yields:
        object: Each item in the data_list.
    """
    for data in data_list:
        yield data

def load_config(config_path):
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Loaded configuration as a dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

async def main(args):
    """
    Main asynchronous function to run the ToolTOT evaluation.

    Args:
        args (argparse.Namespace): Command-line arguments.

    This function:
    1. Sets up the environment (directories, logging)
    2. Loads the dataset and prepares the data
    3. Initializes the ToolTOT agent
    4. Processes each data point, running the agent and evaluating the results
    5. Saves the results to a JSON file
    """
    # Setup environment
    result_path = "result"
    os.makedirs(result_path, exist_ok=True)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

    log_file_path = initialize_log_file("GAIA", Time.instance().value)

    experiment_name = "ToolTOT"
    start_index = 0

    print(args.llm)

    # Load and prepare dataset
    dataset = JSONReader.parse_file(args.data_path)

    with open("prompts.json", "r") as f:
        prompts = json.load(f)
        task_prompt = prompts["banking77"]

    for d in dataset:
        if 'prepared' not in d:
            d['prepared'] = prepare_data(task_prompt, d)

    print(dataset[0])

    # Initialize ToolTOT agent
    agent = ToolTOT(domain="gaia", model_name=args.llm)
    agent.display()

    # Initialize result file
    result_dir = Path(f"result/eval")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{'_'.join(experiment_name.split())}_{args.llm}_{current_time}.json"

    if not result_file.exists():
        with open(result_file, 'w') as file:
            json.dump([], file)

    # Process each data point
    for idx, datum in enumerate(dataloader(dataset)):
        if idx == 0:
            print(datum['prepared'])

        if idx < start_index:
            print(f"Skipping index {idx}...")
            continue

        start_time = time.time()
        task = datum['prepared']
        inputs = {"task": task, "GT": str(datum["output"])}

        print(f"Input from dataset: {inputs}")

        swarmlog("🐝GPTSWARM SYS", f"Finish {idx} samples...", Cost.instance().value, log_file_path)

        # Agent
        answer = await agent.run(inputs=inputs)
        answer = answer[-1].split("FINAL ANSWER: ")[-1]

        end_time = time.time()
        exe_time =  end_time - start_time


        print("-----")
        print(f"AGENT ANSWER: {answer}")
        print("-----")

        # Update and save results
        with open(result_file, 'r') as file:
            data = json.load(file)

        total_solved, total_executed = (0, 0) if not data else (data[-1]["Total solved"], data[-1]["Total executed"])
        is_solved = question_scorer(answer, str(datum['output']))

        updated_item = {
            "Question": datum["input"],
            "GT": datum["output"],
            "Attempt answer": answer,
            "Solved": is_solved,
            "Total solved": total_solved + is_solved,
            "Total executed": total_executed + 1,
            "Accuracy": (total_solved + is_solved) / (total_executed + 1),
            "Time": exe_time,
            "Total Cost": Cost.instance().value,
        }
        data.append(updated_item)

        with open(result_file, 'w') as file:
            json.dump(data, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ToolTOT evaluation on a specified dataset.")
    parser.add_argument("--dataset", default="banking77", type=str, help="Name of the dataset to use.")
    parser.add_argument("--data_path", default="sampled_triplet_results/banking77_embed=instructor_s=small_m=1024_d=67.0_sf_choice_seed=100.json", type=str, help="Path to the dataset file.")
    parser.add_argument("--llm", default="gpt-4-1106-preview", type=str, help="Name of the language model to use.")
    #parser.add_argument("--result_file", type=str, default=None, help="Path to save the results file.")

    args = parser.parse_args()

    asyncio.run(main(args))