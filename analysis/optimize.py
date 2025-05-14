import pandas as pd
import numpy as np
import json
import logging
import re
import argparse
import yaml

from .optimizers.copro import COPRO
from .optimizers.textgrad import TextGradOptimizer
from .optimizers.reqaware import ReqAwareOptimizer
from .optimizers.openai import OpenAIPromptOptimizer

from .load_data import prepare_data
from .utils import LM_DICT, batch_inference
from .judge import LLMJudge

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     filename='my_log_file.log',  # <--- file path here
#     filemode='a'  # 'a' = append, 'w' = overwrite
# )

def gather_results(evaluate_examples, requirements):
    def gather_result(example):
        score = judge.calculate_scores_aggregate([example], requirements)
        return (
            example,
            example.output,
            judge.aggregate_feedback(
                example,
                requirements=requirements,
            ),
            score,
        )
    results = batch_inference(
        gather_result,
        [{"example": example} for example in evaluate_examples],
        max_workers=32,
    )
    return results

def generic_evaluate(module, devset):
    evaluate_examples = judge.evaluate_outputs(devset, program=module)
    score = judge.calculate_scores_aggregate(evaluate_examples)
    return score, evaluate_examples

def evaluate_requirements(requirements):
    def evaluate(module, devset, gather_feedback=False):
        evaluate_examples = judge.evaluate(devset, requirements=requirements, program=module)
        score = judge.calculate_scores_aggregate(evaluate_examples, requirements)
        if gather_feedback:
            results = gather_results(evaluate_examples, requirements)
            return score, results
        return score, evaluate_examples
    return evaluate


def gather_relevant_requirements(prompt_name, requirements):
    if prompt_name.startswith("fixed"):
        indices = prompt_name.split("_")[1].split("+")
        indices = [int(i) for i in indices]
        relevant_requirements = [requirements[i] for i in indices]
        return relevant_requirements
    elif prompt_name.startswith("task_only"):
        return []
    return []

def construct_optimizer(
    optimizer_name,
    task_description,
    local_requirements,
):
    if optimizer_name.endswith("textgrad"):
        optimizer = TextGradOptimizer(
            evaluate=evaluate_requirements(local_requirements),
            prompt_model=LM_DICT["gpt-4o"],
            num_epochs=3,
            batch_size=10,
        )
    elif optimizer_name == "openai":
        optimizer = OpenAIPromptOptimizer(seed=42)
    elif optimizer_name == "reqaware":
        optimizer = ReqAwareOptimizer(
            task_description=task_description,
            requirements=[requirement["requirement"] for requirement in local_requirements],
            evaluate=evaluate_requirements(local_requirements),
            num_trials=9,
            seed=42
        )
    # elif optimizer_name == "textgrad_reqaware":
    #     # first find the prompt with textgrad
    #     textgrad_prompt = prompts_copy[f"optimized_{model_name}_textgrad_" + prompt_name]
    #     intro_match = re.search(r"^.*?(?=\n-)", textgrad_prompt, re.DOTALL)
    #     task_description = intro_match.group(0).strip() if intro_match else None
    #     requirement_components = re.findall(r"^- (.+)", textgrad_prompt, re.MULTILINE)

    #     # then find a good combination
    #     optimizer = ReqAwareOptimizer(
    #         task_description=task_description,
    #         requirements=requirement_components,
    #         evaluate=evaluate_requirements(local_requirements),
    #         num_trials=9,
    #         seed=42
    #     )
    elif optimizer_name == "copro_baseline":
        optimizer = COPRO(
            prompt_model=LM_DICT["gpt-4o"],
            evaluate=generic_evaluate,
            breadth=3,
            depth=3,
            init_temperature=0.5,
        )
    elif optimizer_name == "copro":
        optimizer = COPRO(
            prompt_model=LM_DICT["gpt-4o"],
            evaluate=evaluate_requirements(local_requirements),
            breadth=3,
            depth=3,
            init_temperature=0.5,
        )
    return optimizer

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="The path to the config file.", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    task = config["task_name"]
    task_description, TaskProgram, trainset, valset, requirements, prompts = prepare_data(
        task_name=task,
        configs={
            "prompt_paths": config["prompt_paths"],
        }
    )
    model_name = config["model_names"][0]
    task_program = TaskProgram(lm=LM_DICT[model_name])

    optimizer_names = config["optimizer_names"]
    
    judge = LLMJudge(task_description=task_description, lm=LM_DICT["4.1-mini-eval"], max_workers=64, omit_input=False)

    prompt_subsets = config["prompt_subsets"]

    key_in_subset = lambda key: any([key.startswith(subset) for subset in prompt_subsets])

    prompts_copy = prompts.copy()
    prompts = {
        k: v for k, v in prompts.items() if key_in_subset(k)
    }
    optimized_prompts = {}
    for i, (prompt_name, prompt) in list(enumerate(prompts.items()))[:]:
        for optimizer_name in optimizer_names:
            print(f"Optimizing {prompt_name} with {optimizer_name}...")
            
            local_requirements = gather_relevant_requirements(prompt_name, requirements)

            task_program.prompt = prompt

            optimizer = construct_optimizer(
                optimizer_name=optimizer_name,
                task_description=task_description,
                local_requirements=local_requirements,
            )

            compiled_task_module = optimizer.compile(task_program, trainset[:30])

            optimized_prompts["optimized_" + model_name + "_" + optimizer_name + "_" + prompt_name] = compiled_task_module.prompt

            # print(json.dumps(optimized_prompts, indent=4))

    output_path = config["prompt_paths"][0].replace(".json", "_optimized.json")
    with open(output_path, "w") as f:
        f.write(json.dumps(optimized_prompts, indent=4))