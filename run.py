import argparse
import os
from dotenv import load_dotenv
import dspy
import litellm
import mlflow
from analysis.load_data import prepare_data
from analysis.utils import use_lm, run_model

lm_dict = {
    "gpt-4o": dspy.LM('openai/gpt-4o-2024-08-06', temperature=1.0),
    "gpt-4o-mini": dspy.LM('openai/gpt-4o-mini-2024-07-18', temperature=1.0),
    "o3-mini": dspy.LM('openai/o3-mini', temperature=1.0),
    "gemini-1.5-pro": dspy.LM('openai/gemini-1.5-pro-002', temperature=1.0, api_base = "https://cmu-aiinfra.litellm-prod.ai/", api_key = os.environ.get("LITELLM_API_KEY")),
    "llama3-2-11b-instruct": dspy.LM('openai/llama3-2-11b-instruct', temperature=1.0, api_base = "https://cmu-aiinfra.litellm-prod.ai/", api_key = os.environ.get("LITELLM_API_KEY")),
    "mixtral-8x7b": dspy.LM('bedrock/mistral.mixtral-8x7b-instruct-v0:1', temperature=1.0),
}

lm_dict['o3-mini'].kwargs['max_completion_tokens'] = lm_dict['o3-mini'].kwargs.pop('max_tokens')

if __name__ == "__main__":

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, help="The name of the experiment to log to.")
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