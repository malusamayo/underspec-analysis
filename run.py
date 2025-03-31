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
from analysis.utils import run_model, LM_DICT

load_dotenv()

def run_evaluation(task, model_name, prompt_name, task_program, trainset, valset, n_samples=5, requirements=None, judge=None):
    if not os.path.exists(f"data/results/{task}"):
        os.makedirs(f"data/results/{task}")

    if not os.path.exists(f"data/results/{task}/{model_name}_{prompt_name}_valset.json"):
        print("Running model...")
        trainset_2 = run_model(task_program, trainset, max_workers=32)
        valset_2 = run_model(task_program, valset, max_workers=32)
        with open(f"data/results/{task}/{model_name}_{prompt_name}_trainset.json", "w") as f:
            json.dump([example.toDict() for example in trainset_2], f)
        with open(f"data/results/{task}/{model_name}_{prompt_name}_valset.json", "w") as f:
            json.dump([example.toDict() for example in valset_2], f)
    else:
        print("Loading data from cache...")
        with open(f"data/results/{task}/{model_name}_{prompt_name}_trainset.json", "r") as f:
            trainset_2 = [dspy.Example(**row).with_inputs(task_program.input_key) for row in json.load(f)]
        with open(f"data/results/{task}/{model_name}_{prompt_name}_valset.json", "r") as f:
            valset_2 = [dspy.Example(**row).with_inputs(task_program.input_key) for row in json.load(f)]

    if requirements is None:
        return
    
    if os.path.exists(f"data/results/{task}/{model_name}_{prompt_name}_valset_evaluated.json"):
        print("Loading evaluation data from cache...")
        with open(f"data/results/{task}/{model_name}_{prompt_name}_valset_evaluated.json", "r") as f:
            valset_2 = [dspy.Example(**row).with_inputs(task_program.input_key) for row in json.load(f)]
        for example in valset_2:
            example.requirements = example.requirements_dict["0"]
        
        with open(f"data/results/{task}/{model_name}_{prompt_name}_scores.json", "r") as f:
            score_matrix = json.load(f)
        score_matrix |= {requirement: [[] for _ in range(n_samples)] for requirement in requirements if requirement not in score_matrix}
    else:
        score_matrix = {requirement: [[] for _ in range(n_samples)] for requirement in requirements}

    # identify requirement not evaluated yet
    requirements_not_evaluated = [requirement for requirement in requirements if requirement not in score_matrix or len(score_matrix[requirement][0]) < len(valset_2)]
    
    if len(requirements_not_evaluated) > 0:
        for i in range(n_samples):
            print("Evaluating model on sample", i)
            for example in valset_2:
                example.output = example.outputs[i]
            valset_2 = judge.evaluate(valset_2, requirements=requirements_not_evaluated)
            for example in valset_2:
                for requirement in requirements_not_evaluated:
                    score_matrix[requirement][i].append(example.requirements[requirement]["meets_requirement"])
                if not hasattr(example, "requirements_dict"):
                    example.requirements_dict = {}
                example.requirements_dict[str(i)] = copy.deepcopy(example.requirements)
                example.requirements = {}

            with open(f"data/results/{task}/{model_name}_{prompt_name}_valset_evaluated.json", "w") as f:
                json.dump([example.toDict() for example in valset_2], f)

            with open(f"data/results/{task}/{model_name}_{prompt_name}_scores.json", "w") as f:
                json.dump(score_matrix, f)
    
    for i in range(n_samples):
        pass_rates = {}
        for requirement in requirements:
            pass_rates[requirement] = str(sum(score_matrix[requirement][i]) / len(valset_2))
        # print(i)
        print(','.join(list(pass_rates.values())))
    return pass_rates

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