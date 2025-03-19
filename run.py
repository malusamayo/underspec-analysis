import pandas as pd
import numpy as np
import json
import argparse
import os
import copy
from dotenv import load_dotenv
import dspy
import litellm
import mlflow
from analysis.load_data import prepare_data
from analysis.utils import use_lm, run_model, requirements_to_str

load_dotenv()

lm_dict = {
    "gpt-4o": dspy.LM('openai/gpt-4o-2024-08-06', temperature=1.0),
    "gpt-4o-mini": dspy.LM('openai/gpt-4o-mini-2024-07-18', temperature=1.0),
    "o3-mini": dspy.LM('openai/o3-mini', temperature=1.0, max_tokens=10000),
    "gemini-1.5-pro": dspy.LM('openai/gemini-1.5-pro-002', temperature=1.0, api_base=os.environ.get("CMU_API_BASE"), api_key=os.environ.get("LITELLM_API_KEY")),
    "claude-3.5-sonnet": dspy.LM('openai/claude-3-5-sonnet-20241022', temperature=1.0, api_base=os.environ.get("CMU_API_BASE"), api_key=os.environ.get("LITELLM_API_KEY")),
    "llama3-2-11b-instruct": dspy.LM('openai/llama3-2-11b-instruct', temperature=1.0, api_base=os.environ.get("CMU_API_BASE"), api_key=os.environ.get("LITELLM_API_KEY")),
    "mixtral-8x7b": dspy.LM('bedrock/mistral.mixtral-8x7b-instruct-v0:1', temperature=1.0),
    "qwen2.5-7b": dspy.LM('hosted_vllm/Qwen/Qwen2.5-7B-Instruct', temperature=0.7, api_base=os.environ.get("BABEL_API_BASE")),
    "ministral-8b": dspy.LM('hosted_vllm/mistralai/Ministral-8B-Instruct-2410', temperature=0, api_base=os.environ.get("BABEL_API_BASE")),
    "llama3.1-8b": dspy.LM('hosted_vllm/meta-llama/Llama-3.1-8B-Instruct', temperature=0.6, api_base=os.environ.get("BABEL_API_BASE")),
}

def run_evaluation(task, model_name, task_program, trainset, valset, n_samples=5):
    if not os.path.exists(f"data/results/{task}"):
        os.makedirs(f"data/results/{task}")

    if not os.path.exists(f"data/results/{task}/{model_name}_valset.json"):
        print("Running model...")
        trainset_2 = run_model(task_program, trainset, max_workers=32)
        valset_2 = run_model(task_program, valset, max_workers=32)
        with open(f"data/results/{task}/{model_name}_trainset.json", "w") as f:
            json.dump([example.toDict() for example in trainset_2], f)
        with open(f"data/results/{task}/{model_name}_valset.json", "w") as f:
            json.dump([example.toDict() for example in valset_2], f)
    else:
        print("Loading data from cache...")
        with open(f"data/results/{task}/{model_name}_trainset.json", "r") as f:
            trainset_2 = [dspy.Example(**row).with_inputs(task_program.input_key) for row in json.load(f)]
        with open(f"data/results/{task}/{model_name}_valset.json", "r") as f:
            valset_2 = [dspy.Example(**row).with_inputs(task_program.input_key) for row in json.load(f)]

    score_matrix = {requirement: [[] for _ in range(5)] for requirement in requirements} # requirement -> score for each output
    for i in range(n_samples):
        for example in valset_2:
            example.output = example.outputs[i]
        valset_2 = judge.evaluate(valset_2, requirements=requirements)
        for example in valset_2:
            for requirement in requirements:
                score_matrix[requirement][i].append(example.requirements[requirement]["meets_requirement"])
            if not hasattr(example, "requirements_dict"):
                example.requirements_dict = {}
            example.requirements_dict[i] = copy.deepcopy(example.requirements)
            example.requirements = {}

    with open(f"data/results/{task}/{model_name}_valset_evaluated.json", "w") as f:
        json.dump([example.toDict() for example in valset_2], f)

    with open(f"data/results/{task}/{model_name}_scores.json", "w") as f:
        json.dump(score_matrix, f)
    
    print(selected_index)
    for i in range(n_samples):
        pass_rates = {}
        for requirement in requirements:
            pass_rates[requirement] = str(sum(score_matrix[requirement][i]) / len(valset_2))
        print(i)
        print(','.join(list(pass_rates.values())))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, help="The name of the experiment to log to.")
    parser.add_argument("--elicitation_mode", action='store_true', help="Whether to run in elicitation mode.")
    args = parser.parse_args()

    if args.experiment:
        mlflow.litellm.autolog()
        mlflow.dspy.autolog()
        experiment = mlflow.set_experiment(args.experiment)
        print(experiment.experiment_id)

    task_description, TaskProgram, trainset, valset, requirements = prepare_data(
        'lc',
    )

    task_model = lm_dict["llama3-2-11b-instruct"]
    task_program = use_lm(lm=task_model)(TaskProgram)
    results = run_model(task_program, trainset[:2])