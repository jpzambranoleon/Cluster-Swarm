import os
import sys
import argparse
import json
import yaml
import time
import asyncio
from tools import prepare_data, post_process
from pathlib import Path
from typing import List, Any, Optional

# Import Swarm framework modules
from swarm.graph.swarm import Swarm
from swarm.environment.tools.reader.readers import JSONReader
from swarm.environment.agents.gaia.tool_tot import ToolTOT
from swarm.utils.globals import Time, Cost
from swarm.utils.log import initialize_log_file, logger, swarmlog
from swarm.environment.domain.gaia import question_scorer
from swarm.llm.format import Message
from swarm.graph import Node, Graph
from swarm.environment.operations.cot_step import CoTStep
from swarm.environment.agents.agent_registry import AgentRegistry
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm import LLMRegistry

class CoTStep(Node):
    """
    A Chain-of-Thought (CoT) step operation node within the Swarm framework.

    Attributes:
        domain (str): The operational domain for the CoT step.
        model_name (Optional[str]): The name of the LLM model used for processing.
        is_last_step (bool): Indicates if this is the last step in the CoT process.
        operation_description (str): Description of the node operation.
    """

    def __init__(self, domain: str, model_name: Optional[str], is_last_step: bool,
                 operation_description: str = "Make one step of CoT", id=None):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.model_name = model_name
        self.is_last_step = is_last_step
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()

    @property
    def node_name(self):
        """Returns the name of the node."""
        return self.__class__.__name__

    async def _execute(self, inputs: List[Any] = [], **kwargs):
        """
        Executes the CoT step, processing inputs and generating outputs.

        Args:
            inputs (List[Any]): Input data for processing.
            **kwargs: Additional arguments.

        Returns:
            List[dict]: Outputs generated by the CoT step.
        """
        node_inputs = self.process_input(inputs)
        outputs = []
        for input_dict in node_inputs:
            role = self.prompt_set.get_role()
            constraint = self.prompt_set.get_constraint()

            # Construct system prompt based on step type
            if self.is_last_step:
                system_prompt = (
                    f"You are {role}. {constraint}. "
                    "Answer taking into consideration the provided sequence "
                    "of thoughts on the question at hand."
                )
            else:
                system_prompt = (
                    f"You are {role}. "
                    "Given the question, solve it step by step. "
                    "Answer your thoughts about the next step of the solution given "
                    "everything that has been provided to you so far. "
                    "Expand on the next step. "
                    "Answer in maximum 30 words. "
                    "Do not expect additional input. Make best use of whatever "
                    "knowledge you have been already provided."
                )

            task = input_dict.get('output', input_dict["task"])
            user_prompt = self.prompt_set.get_answer_prompt(question=task)
            message = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt)
            ]
            response = await self.llm.agen(message, max_tokens=50)

            concatenated_response = response if self.is_last_step else f"{task}. Here is the next thought. {response}. "

            execution = {
                "operation": self.node_name,
                "task": task,
                "files": input_dict.get("files", []),
                "input": task,
                "role": role,
                "constraint": constraint,
                "prompt": user_prompt,
                "output": concatenated_response,
                "ground_truth": input_dict.get("GT", []),
                "format": "natural language"
            }
            outputs.append(execution)
            self.memory.add(self.id, execution)

        return outputs

@AgentRegistry.register('CustomCOT')
class CustomCOT(Graph):
    """
    Custom Chain-of-Thought (CoT) graph for multi-step reasoning tasks.

    Methods:
        build_graph: Constructs the CoT graph with a defined number of steps.
    """

    def build_graph(self):
        """
        Builds the CoT graph with multiple sequential steps.
        """
        num_thoughts = 3
        assert num_thoughts >= 2

        thoughts = []
        for i_thought in range(num_thoughts):
            thought = CoTStep(self.domain, self.model_name, is_last_step=i_thought == num_thoughts - 1)
            if i_thought > 0:
                thoughts[-1].add_successor(thought)
            thoughts.append(thought)

        self.input_nodes = [thoughts[0]]
        self.output_nodes = [thoughts[-1]]

        for thought in thoughts:
            self.add_node(thought)

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
    

async def predict_swarm(dataset, result_file):
    # Initialize a log file with a unique identifier based on the current time.
    log_file_path = initialize_log_file("GAIA", Time.instance().value)
    start_index = 0  # Starting point for processing the dataset.

    # Initialize the ToolTOT agent with the specified domain and model name.
    agent = ToolTOT(domain="gaia", model_name=args.llm)
    agent.display()  # Display the agent configuration.

    # Iterate over each data point in the dataset.
    for idx, datum in enumerate(dataloader(dataset)):
        # Print the first data point's prepared field for verification.
        if idx == 0:
            print(datum['prepared'])

        # Skip data points before the start index.
        if idx < start_index:
            print(f"Skipping index {idx}...")
            continue

        # Record the start time for performance tracking.
        start_time = time.time()
        task = datum['prepared']  # Prepared input task for the agent.
        inputs = {"task": task, "GT": str(datum["output"])}  # Create input dictionary.

        # Log the input data for debugging.
        print(f"Input from dataset: {inputs}")

        # Log the progress in the system log.
        swarmlog("🐝GPTSWARM SYS", f"Finish {idx} samples...", Cost.instance().value, log_file_path)

        # Use the agent to process the inputs and retrieve the answer.
        answer = await agent.run(inputs=inputs)
        # Extract the final answer from the agent's response.
        answer = answer[-1].split("FINAL ANSWER: ")[-1]

        # Record the end time and calculate execution time.
        end_time = time.time()
        exe_time = end_time - start_time

        # Display the agent's answer for verification.
        print("-----")
        print(f"AGENT ANSWER: {answer}")
        print("-----")

        # Update and save the results in the output file.
        with open(result_file, 'r') as file:
            data = json.load(file)

        # Retrieve the total solved and executed counts from previous data or initialize them.
        total_solved, total_executed = (0, 0) if not data else (data[-1]["Total solved"], data[-1]["Total executed"])

        # Determine if the agent's answer matches the ground truth.
        is_solved = question_scorer(answer, str(datum['output']))

        # Construct an updated data item with the new metrics and results.
        updated_item = {
            "Question": datum["input"],
            "Prepared": datum["prepared"],
            "GT": datum["output"],
            "Attempt answer": answer,
            "Solved": is_solved,
            "Total solved": total_solved + is_solved,
            "Total executed": total_executed + 1,
            "Accuracy": (total_solved + is_solved) / (total_executed + 1),
            "Time": exe_time,
            "Total Cost": Cost.instance().value,
        }
        # Append the updated item to the dataset.
        data.append(updated_item)

    # Return the final updated data.
    return data

async def summary_swarm(dataset, result_file):
    # Initialize a log file with a unique identifier based on the current time.
    log_file_path = initialize_log_file("GAIA", Time.instance().value)
    start_index = 0  # Starting point for processing the dataset.

    # Initialize the Swarm agent with a predefined set of reasoning models.
    agent = Swarm(["CustomCOT", "CustomCOT"], "gaia", model_name=args.llm)

    # Ensure the result file exists by creating an empty JSON file if it doesn't.
    if not result_file.exists():
        with open(result_file, 'w') as file:
            json.dump([], file)

    # Iterate over each data point in the dataset.
    for idx, datum in enumerate(dataloader(dataset)):
        # Print the first data point's prepared field for verification.
        if idx == 0:
            print(datum['prepared'])

        # Skip data points before the start index.
        if idx < start_index:
            print(f"Skipping index {idx}...")
            continue

        # Record the start time for performance tracking.
        start_time = time.time()

        # Prepare the summarization task by combining problem and attempt details.
        task = (
            f"Summarize the reasoning behind the Attempt Answer and the Problem. \n"
            f"Problem: {datum['prepared']}\nAnswer choice: {datum['Attempt Answer']}"
        )
        inputs = {"task": task}  # Create the input dictionary.

        # Log the input data for debugging.
        print(f"Input from dataset: {inputs}")

        # Log the progress in the system log.
        swarmlog("🐝GPTSWARM SYS", f"Finish {idx} samples...", Cost.instance().value, log_file_path)

        # Use the agent to process the inputs and retrieve the summary.
        answer = await agent.run(inputs=inputs)
        # Extract the final answer from the agent's response.
        answer = answer[-1].split("FINAL ANSWER: ")[-1]

        # Record the end time and calculate execution time.
        end_time = time.time()
        exe_time = end_time - start_time

        # Display the agent's answer for verification.
        print("-----")
        print(f"AGENT ANSWER: {answer}")
        print("-----")

        # Load existing data from the result file.
        with open(result_file, 'r') as file:
            data = json.load(file)

        # Create an updated item containing the summary generated by the agent.
        updated_item = {
            "Summary": answer
        }
        # Append the updated item to the results dataset.
        data.append(updated_item)

    # Return the final updated data containing all processed summaries.
    return data

async def anomaly_swarm(dataset, result_file):
    # Initialize a log file with a unique identifier based on the current time.
    log_file_path = initialize_log_file("GAIA", Time.instance().value)
    start_index = 0  # Starting point for processing the dataset.

    # Initialize the Swarm agent with a predefined set of reasoning models.
    agent = Swarm(["CustomCOT", "CustomCOT"], "gaia", model_name=args.llm)

    # Ensure the result file exists by creating an empty JSON file if it doesn't.
    if not result_file.exists():
        with open(result_file, 'w') as file:
            json.dump([], file)

    # Iterate over each data point in the dataset.
    for idx, datum in enumerate(dataloader(dataset)):
        # Print the first data point's prepared field for verification.
        if idx == 0:
            print(datum['prepared'])

        # Skip data points before the start index.
        if idx < start_index:
            print(f"Skipping index {idx}...")
            continue

        # Record the start time for performance tracking.
        start_time = time.time()

        # Prepare the anomaly detection task by combining problem and attempt details.
        task = (
            f"Provide any anomaly detection for said task. \n"
            f"Problem: {datum['prepared']}\nAnswer choice: {datum['Attempt Answer']}"
        )
        inputs = {"task": task}  # Create the input dictionary.

        # Log the input data for debugging.
        print(f"Input from dataset: {inputs}")

        # Log the progress in the system log.
        swarmlog("🐝GPTSWARM SYS", f"Finish {idx} samples...", Cost.instance().value, log_file_path)

        # Use the agent to process the inputs and retrieve the anomaly detection result.
        answer = await agent.run(inputs=inputs)
        # Extract the final answer from the agent's response.
        answer = answer[-1].split("FINAL ANSWER: ")[-1]

        # Record the end time and calculate execution time.
        end_time = time.time()
        exe_time = end_time - start_time

        # Display the agent's answer for verification.
        print("-----")
        print(f"AGENT ANSWER: {answer}")
        print("-----")

        # Load existing data from the result file.
        with open(result_file, 'r') as file:
            data = json.load(file)

        # Create an updated item containing the detected anomaly.
        updated_item = {
            "Anomaly Detection": answer
        }
        # Append the updated item to the results dataset.
        data.append(updated_item)

    # Return the final updated data containing all detected anomalies.
    return data

async def main(args):
    # Setup environment
    result_path = "result"
    os.makedirs(result_path, exist_ok=True)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

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

    # Initialize result file
    result_dir = Path(f"../result/eval")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{'_'.join(experiment_name.split())}_{args.llm}_{current_time}.json"

    data = asyncio.run(predict_swarm(dataset=dataset, result_file=result_file))
    data = asyncio.run(summary_swarm(dataset=data, result_file=result_file))
    data = asyncio.run(anomaly_swarm(dataset=data, result_file=result_file))

    with open(result_file, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ToolTOT evaluation on a specified dataset.")
    parser.add_argument("--dataset", default="banking77", type=str, help="Name of the dataset to use.")
    parser.add_argument("--data_path", default="sampled_triplet_results/banking77_embed=instructor_s=small_m=1024_d=67.0_sf_choice_seed=100.json", type=str, help="Path to the dataset file.")
    parser.add_argument("--llm", default="gpt-4-1106-preview", type=str, help="Name of the language model to use.")
    #parser.add_argument("--result_file", type=str, default=None, help="Path to save the results file.")

    args = parser.parse_args()

    main(args)