import pandas as pd
import numpy as np
import json
import logging

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
        relevant_requirements = [req for i, req in enumerate(requirements) if i in indices]
        return relevant_requirements
    elif prompt_name.startswith("task_only"):
        return []
    raise ValueError(f"Unknown prompt name: {prompt_name}")

if __name__ == "__main__":

    task = "trip"
    task_description, TaskProgram, trainset, valset, requirements, prompts = prepare_data(
        task_name=task,
        configs={
            "prompt_path": f"data/prompts/trip_n10.json",
        }
    )
    task_program = TaskProgram(lm=LM_DICT["gpt-4o"])

    
    judge = LLMJudge(task_description=task_description, lm=LM_DICT["4.1-mini-eval"], max_workers=64, omit_input=False)

    prompt_subsets = [
        "fixed",
    ]

    key_in_subset = lambda key: any([key.startswith(subset) for subset in prompt_subsets])

    prompts = {
        k: v for k, v in prompts.items() if key_in_subset(k)
    }
    optimized_prompts = {}
    for i, (prompt_name, prompt) in enumerate(prompts.items()):
        print(f"Optimizing {prompt_name}...")
        # if prompt_name != "fixed_0+1+2+3+4+5+6+7+8+9":
        #     continue
        # if i <= 5:
        #     continue
        local_requirements = gather_relevant_requirements(prompt_name, requirements)

        task_program.prompt = prompt
        
        # optimizer = COPRO(
        #     prompt_model=LM_DICT["gpt-4o"],
        #     evaluate=evaluate_requirements(requirements),
        #     breadth=3,
        #     depth=3,
        #     init_temperature=0.5,
        # )    
        optimizer = TextGradOptimizer(
            evaluate=evaluate_requirements(local_requirements),
            prompt_model=LM_DICT["gpt-4o"],
            num_epochs=3,
            batch_size=10,
        )
        optimizer_name = "textgrad"

        # optimizer = OpenAIPromptOptimizer(seed=42)
        # optimizer_name = "openai"

        compiled_task_module = optimizer.compile(task_program, trainset[:30])

        optimized_prompts["optimized_" + optimizer_name + "_" + prompt_name] = compiled_task_module.prompt

    
        print(json.dumps(optimized_prompts, indent=4))

    # compiled_task_module = optimizer.compile(task_program, trainset=trainset[:30])
    # print(compiled_task_module.prompt)
    # evaluate(compiled_task_module, trainset[:30])
    # evaluate(compiled_task_module, trainset[30:])


    # compiled_task_module = optimizer.compile(task_program, trainset[:30])
    # all_prompts = {
    #     "optimized_" + str(k-1): v['prompt']
    #     for k, v in compiled_task_module.prompt_history.items() if k > 0
    # }
    # print(json.dumps(compiled_task_module.prompt_history))
    # print(json.dumps(all_prompts, indent=4))

    # optimizer = MIPROv2(
    #     evaluate=evaluate,
    #     prompt_model=LM_DICT["gpt-4o"],
    #     auto="light"
    # )
    # compiled_task_module = optimizer.compile(task_program, trainset=trainset[:30])

    # optimizer = ReqAwareOptimizer(
    #     task_description=task_description,
    #     requirements=[requirement["requirement"] for requirement in requirements],
    #     evaluate=evaluate,
    #     num_trials=10,
    #     seed=42
    # )
    # compiled_task_module = optimizer.compile(task_program, trainset[:30])

    # print(json.dumps({
    #     "optimized": compiled_task_module.prompt,
    # }, indent=4))