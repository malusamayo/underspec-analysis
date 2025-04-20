import pandas as pd
import numpy as np
import json
from .optimizers.copro import COPRO
from .optimizers.textgrad import TextGradOptimizer
from copy import deepcopy

from .load_data import prepare_data
from .utils import LM_DICT, batch_inference
from .judge import LLMJudge


if __name__ == "__main__":

    task = "trip"
    task_description, TaskProgram, trainset, valset, requirements, prompts = prepare_data(
        task_name=task,
    )
    task_program = TaskProgram(lm=LM_DICT["gpt-4o-mini"])

    
    judge = LLMJudge(task_description=task_description, lm=LM_DICT["4.1-mini-eval"], max_workers=64, omit_input=False)

    requirements = [requirement for requirement in requirements if requirement["source"] == "prompt"]

    def metric(example, output):
        pass
    
    def evaluate(module, devset, gather_feedback=False):
        evaluate_examples = judge.evaluate(devset, requirements=requirements, program=module)
        score = judge.calculate_scores_aggregate(evaluate_examples, requirements)
        if gather_feedback:
            results = gather_results(evaluate_examples)
            return score, results
        return score, evaluate_examples
    
    def gather_results(evaluate_examples):
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

    teleprompter = COPRO(
        prompt_model=LM_DICT["gpt-4o"],
        evaluate=evaluate,
        breadth=2,
        depth=3,
        init_temperature=1.0,
    )

    # compiled_prompt_opt = teleprompter.compile(task_program, trainset=trainset[:30])
    # print(compiled_prompt_opt.prompt)
    # evaluate(compiled_prompt_opt, trainset[:30])
    # evaluate(compiled_prompt_opt, trainset[30:])


    optimizer = TextGradOptimizer(
        evaluate=evaluate,
        prompt_model=LM_DICT["gpt-4o"],
        num_epochs=3,
        batch_size=10,
    )
    compiled_task_module = optimizer.compile(task_program, trainset[:30])
    # all_prompts = {
    #     "optimized_" + str(k-1): v['prompt']
    #     for k, v in compiled_task_module.prompt_history.items() if k > 0
    # }
    # print(json.dumps(compiled_task_module.prompt_history))
    # print(json.dumps(all_prompts, indent=4))