import os
import json
import dspy
import hashlib
from copy import deepcopy
import time
import re
import numpy as np
import subprocess
import tiktoken
import litellm

import openhands
from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

from .utils import batch_inference, run_model, construct_prompt_removal
from .optimizers import BayesianOptimizer, COPRO, OpenAIPromptOptimizer

from dotenv import load_dotenv
load_dotenv(override=True)
litellm.suppress_debug_info = True

USER_PROMPT = """# User Instruction
{instruction}

# Generated codebase
{output}

# Your task
Evaluate the above codebase based on the provided instruction and guidelines. First, provide a detailed analysis based on the guidelines, pointing out both strengths and areas for improvement. Then, give a final score from 1 to 10, where 1 is poor quality and 10 is excellent quality.

Format your response as follows:

```
# Analysis: [analysis here]
# Score: X/10
```"""


def num_tokens_from_string(string: str, model_name: str = "gpt-4o") -> int:
    """Returns the number of tokens in a text string for a given model."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class CodebaseEvaluator:
    def __init__(self, prompt, input_key, lm, n=1, max_retries=3):
        self.lm = lm
        self.n = n
        self.prompt = prompt
        self.input_key = input_key
        self.max_retries = max_retries
    
    def __parse_response(self, response):
        pattern = r"(?:^|\n)\s*(?:#+\s*|\*\*.*?\b)?Score:\s*(?:\*\*)?\s*((?:10(?:\.0+)?|[0-9](?:\.\d+)?))/10"
        try:
            match = re.search(pattern, response)
            if match:
                score = float(match.group(1))
                return score
        except Exception as e:
            print(f"Error parsing response: {e}")
        assert False, f"Could not parse score from response: {response}"

    def __evaluate_single_with_retry(self, messages):
        for attempt in range(self.max_retries):
            # print(f"Evaluation attempt {attempt + 1}/{self.max_retries}")
            responses = self.lm(messages=messages, n=self.n, rollout_id=attempt)
            if all(r is not None for r in responses):
                return responses
        assert False, f"Failed to get valid response after {self.max_retries} attempts."

    def __call__(self, example):
        input = example.get(self.input_key)

        example['evaluation_responses'] = []
        example['evaluation_scores'] = []
        for output in example.get("outputs"):
            messages = [
                {
                    "role": "system",
                    "content": self.prompt,
                },
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        instruction=input,
                        output=output,
                    ),
                },
            ]

            responses = self.__evaluate_single_with_retry(messages)
            example['evaluation_responses'].append(responses)
            example['evaluation_scores'].append([self.__parse_response(response) for response in responses])
        return np.mean(example['evaluation_scores']), example
    
    def evaluate_batch(self, program, devset, max_workers=32):
        examples = deepcopy(devset)
        
        examples = run_model(program, examples, use_process=True, max_workers=max_workers // program.n // 2)

        results = batch_inference(
            self.__call__,
            [{"example": example} for example in examples],
            max_workers=max_workers
        )
        scores = [score for score, _ in results]
        examples = [example for _, example in results]
        return np.mean(scores), examples


class AgenticTaskProgram:
    def __init__(self, prompt, input_key, lm, n=1, max_retries=3):
        self.lm = lm
        self.n = n
        self.prompt = prompt
        self.input_key = input_key
        self.max_retries = max_retries

    def __run_agent(self, agent, temp_dir, input):
        conversation = Conversation(agent=agent, workspace=temp_dir, visualizer=None)
        additional_instr = "\nWrite all the code under `app/`."
        conversation.send_message(input + additional_instr)
        conversation.run()

    def __initialize(self, temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        prompt_path = os.path.join(temp_dir, "system_prompt.j2")
        with open(prompt_path, "w") as f:
            f.write(self.prompt)

    def __cleanup(self, temp_dir):
        if os.path.exists(temp_dir):
            os.system(f"rm -rf {temp_dir}/")
        self.__initialize(temp_dir)

    def __summarize_workspace(self, temp_dir):
        subprocess.run(
            [
                "repomix", "--no-gitignore", "--quiet",
                "--ignore", "system_prompt.j2,**/.cache/,**/public/",
                "--output", "repomix-output.xml"
            ],
            cwd=temp_dir,
            check=True,
            # stdout=subprocess.DEVNULL,
        )
        output_path = os.path.join(temp_dir, "repomix-output.xml")
        with open(output_path, "r") as f:
            output = f.read()
        return output

    def __evaluate_repomix(self, output):
        pattern = r"<directory_structure>(.*?)</directory_structure>"
        match = re.search(pattern, output, re.DOTALL)
        if match:
            directory_structure = match.group(1).strip()
            # check if directory_structure is non-empty
            if directory_structure:
                n = num_tokens_from_string(output)
                if n > 1_000_000:
                    print(f"Output too long: {n} tokens.")
                    return False
                return True
        return False

    def run(self, temp_dir, input):
        # get absolute path
        temp_dir = os.path.abspath(temp_dir)
        prompt_path = os.path.join(temp_dir, "system_prompt.j2")

        output_path = os.path.join(temp_dir, "repomix-output.xml")
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                output = f.read()
            if self.__evaluate_repomix(output):
                print(f"Found existing output at {output_path}, skipping execution.")
                return output
        
        self.__cleanup(temp_dir)

        llm = LLM(
            model=self.lm.model,
        )

        agent = Agent(
            llm=llm,
            system_prompt_filename=prompt_path,
            tools=[
                Tool(name=TerminalTool.name),
                Tool(name=FileEditorTool.name),
                Tool(name=TaskTrackerTool.name),
            ],
        )

        for attempt in range(self.max_retries):
            try:
                self.__run_agent(agent, temp_dir, input)
                output = self.__summarize_workspace(temp_dir)
                if self.__evaluate_repomix(output):
                    return output
            except openhands.sdk.conversation.exceptions.ConversationRunError as e:
                print(f"Conversation Error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(5)
            self.__cleanup(temp_dir)
        
        assert False, f"Failed to generate valid output after {self.max_retries} attempts: {temp_dir}"

    def __call__(self, **kwargs):
        input = kwargs.get(self.input_key)

        prompt_hash = int.from_bytes(hashlib.shake_128(self.prompt.encode()).digest(8), "big")

        workspace_id = f'{prompt_hash}_example_{kwargs.get("id")}'
        temp_dir = f"./data/results/webgen_react/{workspace_id}"
        
        outputs = batch_inference(
            self.run,
            [{"temp_dir": f"{temp_dir}_rollout_{rollout_id}", "input": input} for rollout_id in range(self.n)],
            use_process=True,
        )

        return dspy.Example(**{self.input_key: input}, output=outputs[0], outputs=outputs)
    
    def deepcopy(self):
        return deepcopy(self)

    def predictors(self):
        return [self]
    
if __name__ == "__main__":
    from .load_data import prepare_data, load_data
    from .utils import run_model, LM_DICT

    task_description, TaskProgram, trainset, valset, requirements, prompts = prepare_data(
        task_name="webgen_react",
    )

    task_program = TaskProgram(
        lm=LM_DICT["qwen3-coder-30b-a3b"],
    )
    task_program.n = 3

    trainset = [dspy.Example(**{**example, "id": i}).with_inputs("id", "instruction") for i, example in enumerate(trainset)]
    valset = [dspy.Example(**{**example, "id": i + len(trainset)}).with_inputs("id", "instruction") for i, example in enumerate(valset)]

    eval_prompt_path = "./data/prompts/reactdev_eval.j2"
    with open(eval_prompt_path, "r") as f:
        eval_prompt = f.read()

    evaluator = CodebaseEvaluator(
        prompt=eval_prompt,
        input_key="instruction",
        lm=LM_DICT["gemini-2.5-flash"],
        n=1,
    )

    prompt_path = "./data/prompts/reactdev.j2"
    with open(prompt_path, "r") as f:
        prompt = f.read()

    requirement_path = "./data/requirements/reactdev.json"
    with open(requirement_path, "r") as f:
        requirements = [item["requirement"] for item in json.load(f)]

    # # Optimization of prompt using Bayesian Optimizer
    # optimizer = BayesianOptimizer(
    #     task_description=prompt,
    #     requirements=requirements,
    #     evaluate=evaluator.evaluate_batch,
    #     construct_prompt_fn=construct_prompt_removal,
    #     num_trials=9,
    #     seed=42
    # )

    # optimizer = OpenAIPromptOptimizer(seed=42)

    # optimizer = COPRO(
    #     prompt_model=LM_DICT["gpt-4o"],
    #     evaluate=evaluator.evaluate_batch,
    #     breadth=3,
    #     depth=3,
    #     init_temperature=0.5,
    # )

    # compiled_task_module = optimizer.compile(task_program, trainset[:30])

    # optimized_prompt = compiled_task_module.prompt

    # with open("./data/prompts/reactdev_optimized_openai.j2", "w") as f:
    #     f.write(optimized_prompt)

    # # Evaluation of original prompt
    # _, results = evaluator.evaluate_batch(
    #     task_program,
    #     valset[:100],
    #     max_workers=32,
    # )
    # with open("./data/results/webgen_react/valset_evaluated.json", "w") as f:
    #     json.dump([example.toDict() for example in results], f, indent=4)

    # # Evaluation of optimized prompt
    # prompt_path = "./data/prompts/reactdev_optimized_bayesian_r9.j2"
    # with open(prompt_path, "r") as f:
    #     prompt = f.read()
    # task_program.prompt = prompt
    
    # _, results = evaluator.evaluate_batch(
    #     task_program,
    #     valset[:100],
    #     max_workers=32,
    # )
    # with open("./data/results/webgen_react/valset_evaluated_optimized_bayesian.json", "w") as f:
    #     json.dump([example.toDict() for example in results], f, indent=4)


    # # Evaluation of optimized prompt
    # prompt_path = "./data/prompts/reactdev_optimized_copro_r.j2"
    # with open(prompt_path, "r") as f:
    #     prompt = f.read()
    # task_program.prompt = prompt
    
    # _, results = evaluator.evaluate_batch(
    #     task_program,
    #     valset[:100],
    #     max_workers=32,
    # )
    # with open("./data/results/webgen_react/valset_evaluated_optimized_copro_r.json", "w") as f:
    #     json.dump([example.toDict() for example in results], f, indent=4)

    # Evaluation of optimized prompt
    prompt_path = "./data/prompts/reactdev_optimized_openai.j2"
    with open(prompt_path, "r") as f:
        prompt = f.read()
    task_program.prompt = prompt
    
    _, results = evaluator.evaluate_batch(
        task_program,
        valset[:100],
        max_workers=32,
    )
    with open("./data/results/webgen_react/valset_evaluated_optimized_openai.json", "w") as f:
        json.dump([example.toDict() for example in results], f, indent=4)

    # # Evaluation of under-specified prompt
    # task_program.prompt = ""
    # _, results = evaluator.evaluate_batch(
    #     task_program,
    #     valset[:100],
    #     max_workers=32,
    # )
    # with open("./data/results/webgen_react/valset_evaluated_no_req.json", "w") as f:
    #     json.dump([example.toDict() for example in results], f, indent=4)