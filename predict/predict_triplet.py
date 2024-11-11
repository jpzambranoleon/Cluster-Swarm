import os
import sys
import argparse
import json
import yaml
import time
from tqdm import tqdm
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
    for data in data_list:
        yield data

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

async def main(args):

    result_path = "result"
    os.makedirs(result_path, exist_ok=True)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

    log_file_path = initialize_log_file("GAIA", Time.instance().value)

    experiment_name = "ToolTOT"

    start_index = 0

    print(args.llm)

    dataset = JSONReader.parse_file(args.data_path)

    with open(args.data_path, 'r') as f:
        data = json.load(f)

    with open("prompts.json", "r") as f:
        prompts = json.load(f)
        task_prompt = prompts["banking77"]

    for d in dataset:
        if 'prepared' not in d:
            d['prepared'] = prepare_data(task_prompt, d)

    

    print(dataset[0])

    agent = ToolTOT(domain="gaia", model_name=args.llm)
    agent.display()

    # Initialize result file
    experiment_name = "ToolTOT"
    result_dir = Path(f"result/eval")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{'_'.join(experiment_name.split())}_{args.llm}_{current_time}.json"

    # Initialize result file if it does not exist
    if not result_file.exists():
        with open(result_file, 'w') as file:
            json.dump([], file)

    for idx, datum in enumerate(dataloader(dataset)):
        if idx == 0:
            print(datum['prepared'])

        if idx < start_index:
            print(f"Skipping index {idx}...")
            continue

        start_time = time.time()
        task = datum['prepared']
        inputs = {"task": task}

        print(inputs)

        swarmlog("🐝GPTSWARM SYS", f"Finish {idx} samples...", Cost.instance().value, log_file_path)

        # Agent
        answer = await agent.run(inputs=inputs)
        answer = answer[-1].split("FINAL ANSWER: ")[-1]

        end_time = time.time()
        exe_time =  end_time - start_time


        print("-----")
        print(f"AGENT ANSWER: {answer}")
        print("-----")

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
        



"""
    result_path = "result"
    os.makedirs(result_path, exist_ok=True)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

    log_file_path = initialize_log_file("GAIA", Time.instance().value)

    start_index = 0

    for d in data:
        if 'prepared' not in d:
            d['prepared'] = prepare_data(task_prompt, d)

    dataset = JSONReader.parse_file(data)

    experiment_name = "ToolTOT"

    print(args.llm)

    agent = ToolTOT(domain="gaia", model_name=args.llm)

    agent.display()

    for idx, datum in tqdm(enumerate(dataloader(dataset)), total=len(data)):
        if idx < start_index:
            print(f"Skipping index {idx}...")
            continue

        start_time = time.time()
        if idx == 0:
            print(datum['prepared'])
        # breakpoint()
        if 'prediction' in datum:
            continue

        start_time = time.time()
        task = datum["prepared"]
        #files = [os.path.join(args.dataset_files, item["file_name"])] if item["file_name"] else item["file_name"]
        ground_truth = datum["output"]
        inputs = {"task": task, "GT": ground_truth}

        swarmlog("🐝GPTSWARM SYS", f"Finish {idx} samples...", Cost.instance().value, log_file_path)

        # Swarm
        # answer = await swarm.composite_graph.run(inputs)
        # answer = answer[-1].split("FINAL ANSWER: ")[-1]

        # end_time = time.time()
        # exe_time =  end_time - start_time

        # print("-----")
        # print(f"SWARM ANSWER: {answer}")
        # print("-----")

        # Agent
        answer = await agent.run(inputs=inputs)
        answer = answer[-1].split("FINAL ANSWER: ")[-1]

        end_time = time.time()
        exe_time =  end_time - start_time


        print("-----")
        print(f"AGENT ANSWER: {answer}")
        print("-----")

        result_dir = Path(f"result/eval")
        result_file = result_file or (result_dir / f"{'_'.join(experiment_name.split())}_{args.llm}_{current_time}.json")

        result_dir.mkdir(parents=True, exist_ok=True)

        if not result_file.exists():
            with open(result_file, 'w') as file:
                json.dump([], file)

        with open(result_file, 'r') as file:
            data = json.load(file)

        total_solved, total_executed = (0, 0) if not data else (data[-1]["Total solved"], data[-1]["Total executed"])
        is_solved = question_scorer(answer, datum['output'])

        updated_item = {
            "Question": datum["input"],
            "GT": datum['output'],
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
    
    print(data[0])
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="banking77", type=str)
    parser.add_argument("--data_path", default="sampled_triplet_results/banking77_embed=instructor_s=small_m=1024_d=67.0_sf_choice_seed=100.json", type=str)
    parser.add_argument("--llm", default="gpt-4-1106-preview", type=str)
    #parser.add_argument("--config", type=str, help="Path to configuration YAML file.")
    #parser.add_argument("--domain", type=str, default="gaia")
    #parser.add_argument("--agents", nargs='+', default=["IO"])
    #parser.add_argument("--dataset", default=None, type=str)
    #parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--result_file", type=str, default=None)

    args = parser.parse_args()

    asyncio.run(main(args))