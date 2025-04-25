import pandas as pd
import numpy as np
import json
import yaml
import argparse
import os
import copy
import dspy
import litellm
import mlflow
from analysis.load_data import prepare_data, load_data
from analysis.utils import run_model, LM_DICT
from analysis.judge import LLMJudge

def run_evaluation(task, model_name, prompt_name, task_program, trainset, valset, n_samples=1, requirements=None, judge=None):
    if not os.path.exists(f"data/results/{task}"):
        os.makedirs(f"data/results/{task}")

    valset_2 = load_data(task, model_name, prompt_name, task_program, {"valset": valset})["valset"]
    
    if requirements is None:
        return
    
    requirements_evaluated = []
    if os.path.exists(f"data/results/{task}/{model_name}_{prompt_name}_valset_evaluated.json"):
        print("Loading evaluation data from cache...")
        with open(f"data/results/{task}/{model_name}_{prompt_name}_valset_evaluated.json", "r") as f:
            valset_2 = [dspy.Example(**row).with_inputs(task_program.input_key) for row in json.load(f)]
        for example in valset_2:
            example.requirements = example.requirements_dict["0"]
        requirements_evaluated = list(valset_2[0].requirements_dict["0"].keys())

    # identify requirement not evaluated yet
    requirements_not_evaluated = [requirement for requirement in requirements if requirement["requirement"] not in requirements_evaluated]
    
    if len(requirements_not_evaluated) > 0:
        for i in range(n_samples):
            print("Evaluating model on sample", i)
            for example in valset_2:
                example.output = example.outputs[i]
            valset_2 = judge.evaluate(valset_2, requirements=requirements_not_evaluated)
            for example in valset_2:
                for requirement in requirements_not_evaluated:
                    requirement = requirement["requirement"]
                if not hasattr(example, "requirements_dict"):
                    example.requirements_dict = {}
                example.requirements_dict[str(i)] = copy.deepcopy(example.requirements)
                example.requirements = {}

            with open(f"data/results/{task}/{model_name}_{prompt_name}_valset_evaluated.json", "w") as f:
                json.dump([example.toDict() for example in valset_2], f)
    
    return valset_2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="The path to the config file.", required=True)
    parser.add_argument("--experiment", type=str, help="The name of the experiment to log to.")
    args = parser.parse_args()

    if args.experiment:
        mlflow.litellm.autolog()
        # mlflow.dspy.autolog()
        experiment = mlflow.set_experiment(args.experiment)
        print(experiment.experiment_id)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    task = config["task_name"]
    task_description, TaskProgram, trainset, valset, requirements, prompts = prepare_data(
        task_name=task,
        configs={
            "prompt_paths": config["prompt_paths"],
        }
    )
    judge = LLMJudge(task_description=task_description, lm=LM_DICT["4.1-mini-eval"], max_workers=64, omit_input=False)

    model_names = config["model_names"]
    prompt_subsets = config["prompt_subsets"]

    key_in_subset = lambda key: any([key.startswith(subset) for subset in prompt_subsets])

    prompts = {
        k: v for k, v in prompts.items() if key_in_subset(k)
    }
    
    n_samples = 1
    if "n_samples" in config:
        n_samples = config["n_samples"]

    all_pass_rates = {}
    for model_name in model_names:
        task_model = LM_DICT[model_name]
        task_program = TaskProgram(lm=task_model, n=n_samples)
        for prompt_name, prompt in prompts.items():
            task_program.prompt = prompt
            if n_samples > 1:
                prompt_name += f"_samples_{n_samples}"
            print("Running evaluation for", model_name, prompt_name)
            run_evaluation(
                task, 
                model_name, prompt_name, task_program, 
                trainset, valset, 
                n_samples=n_samples, 
                requirements=requirements, 
                judge=judge,
            )