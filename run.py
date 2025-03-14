import pandas as pd
import numpy as np
import json
import argparse
import os
from dotenv import load_dotenv
import dspy
import litellm
import mlflow
from analysis.load_data import prepare_data
from analysis.utils import run_model, requirements_to_str

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