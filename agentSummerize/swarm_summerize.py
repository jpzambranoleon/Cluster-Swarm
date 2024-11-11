from swarm.llm.format import Message
from swarm.graph import Node
from swarm.graph import Graph
from typing import List, Any, Optional
from swarm.environment.operations.cot_step import CoTStep
from swarm.environment.agents.agent_registry import AgentRegistry
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm.format import Message
from swarm.llm import LLMRegistry
from swarm.graph.swarm import Swarm
import asyncio

class CoTStep(Node):
    def __init__(self,
                 domain: str,
                 model_name: Optional[str],
                 is_last_step: bool,
                 operation_description: str = "Make one step of CoT",
                 id=None):
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
        return self.__class__.__name__
    
    async def _execute(self, inputs: List[Any] = [], **kwargs):

        node_inputs = self.process_input(inputs)
        outputs = []
        for input_dict in node_inputs:

            role = self.prompt_set.get_role()
            constraint = self.prompt_set.get_constraint()
            if self.is_last_step:
                system_prompt = (
                    f"You are {role}. {constraint}. "
                    "Answer taking into consideration the provided sequence "
                    "of thoughts on the question at hand.")
            else:
                system_prompt = (
                    f"You are {role}. "
                    "Given the question, solve it step by step. "
                    "Answer your thoughts about the next step of the solution given "
                    "everything that has been provided to you so far. "
                    "Expand on the next step. "
                    "Do not try to provide the answer straight away, instead expand "
                    "on your thoughts about the next step of the solution."
                    "Aswer in maximum 30 words. "
                    "Do not expect additional input. Make best use of whatever "
                    "knowledge you have been already provided.")
            if 'output' in input_dict:
                task = input_dict['output']
            else:
                task = input_dict["task"]
            user_prompt = self.prompt_set.get_answer_prompt(question=task)
            message = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt)]
            response = await self.llm.agen(message, max_tokens=50)
            if self.is_last_step:
                concatenated_response = response
            else:
                concatenated_response = f"{task}. Here is the next thought. {response}. "

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

    def build_graph(self):
        num_thoughts = 3

        assert num_thoughts >= 2

        thoughts = []
        for i_thought in range(num_thoughts):
            thought = CoTStep(self.domain,
                              self.model_name,
                              is_last_step=i_thought==num_thoughts-1)
            if i_thought > 0:
                thoughts[-1].add_successor(thought)
            thoughts.append(thought)

        self.input_nodes = [thoughts[0]]
        self.output_nodes = [thoughts[-1]]

        for thought in thoughts:
            self.add_node(thought)

swarm = Swarm(["CustomCOT", "CustomCOT"], "gaia")
task = "What is the text representation of the last digit of twelve squared?"
inputs = {"task": task}
answer = asyncio.run(swarm.arun(inputs))
print(answer)