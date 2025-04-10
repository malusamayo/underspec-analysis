import numpy as np
import copy
import dspy
from .utils import use_lm, batch_inference, run_model, find_nearest_requirement
from typing import List, Any
from pydantic import BaseModel

class EvalResult(BaseModel):
    """Stores evaluation results for a single example when using a particular prompt."""
    requirement: str
    evaluation_plan: str
    plan_execution: str
    meets_requirement: bool


class EvaluateRequirement(dspy.Signature):
    """You are a reviewer who is evaluating whether a model output satisfies the given requirement. 
    Given a task description, model input, model output, and requirement, first generate a step-by-step evaluation plan for the requirement, then execute the evaluation plan to evaluate if the model output meets the requirement.
    Note that sometimes a requirement may not be directly applicable, the evaluation plan should include a step to check if the requirement is applicable to the task. If the requirement is not applicable, return True for meets_requirement."""
    
    task_description = dspy.InputField(desc="Description of the task")
    model_input = dspy.InputField(desc="The model input")
    model_output = dspy.InputField(desc="The model output")
    requirement = dspy.InputField(desc="The requirement to evaluate")
    evaluation_plan: str = dspy.OutputField(desc="The evaluation plan for the requirement")
    plan_execution: str = dspy.OutputField(desc="The execution of the evaluation plan")
    is_applicable: bool = dspy.OutputField(desc="Whether the requirement is applicable to the task, True or False")
    meets_requirement: bool = dspy.OutputField(desc="Whether the model output meets the requirement, True or False")

class EvaluateGuideline(dspy.Signature):
    """You are a reviewer who is evaluating whether a model output satisfies the given guideline.
Given a task description, model input, model output, and guideline, evaluate the model output using the requirements in the guideline one by one. 
For each requirement in the guideline, do the evaluation in two steps: First generate a step-by-step evaluation plan for the requirement, then execute the evaluation plan to evaluate if the model output meets the requirement. Finally, generate a final judgment on whether the model output meets the requirement."""

    task_description: str = dspy.InputField(desc="Description of the task")
    model_input: str = dspy.InputField(desc="The model input")
    model_output: str = dspy.InputField(desc="The model output")
    guideline: List[str] = dspy.InputField(desc="The guideline for evaluation")
    results: List[EvalResult] = dspy.OutputField(desc="The evaluation results for each requirement")

class IdentifyMistakes(dspy.Signature):
    """You are a reviewer who is evaluating whether a model output satisfies the given guideline.
Given a task description, model input, model output, and guideline, first evaluate the model output using the requirements in the guideline one by one to identify mistakes. For each requirement in the guideline, do the evaluation step-by-step. 
Then, list all requirements that the model output does not satisfy."""

    task_description: str = dspy.InputField(desc="Description of the task")
    model_input: str = dspy.InputField(desc="The model input")
    model_output: str = dspy.InputField(desc="The model output")
    guideline: List[str] = dspy.InputField(desc="The guideline for evaluation")
    evaluation_execution: str = dspy.OutputField(desc="The execution of the evaluation guideline")
    unsatisfied_requirements: List[str] = dspy.OutputField(desc="A list of requirements that the model output does not satisfy")

class CompareModelOutputsWithGuideline(dspy.Signature):
    """You are a reviewer who is comparing two model outputs to determine which one better satisfies the given guideline.
Given a task description, two model outputs, and a guideline, evaluate each model output using the requirements in the guideline one by one. For each requirement in the guideline, do the evaluation step-by-step.
Then, compare the two model outputs based on how many requirements each output satisfies."""

    task_description: str = dspy.InputField(desc="Description of the task")
    model_input: str = dspy.InputField(desc="The model input")
    model_output_a: str = dspy.InputField(desc="The first model output")
    model_output_b: str = dspy.InputField(desc="The second model output")
    guideline: List[str] = dspy.InputField(desc="The guideline for evaluation")
    execution_a: str = dspy.OutputField(desc="The execution of the evaluation of the first model output")
    execution_b: str = dspy.OutputField(desc="The execution of the evaluation of the second model output")
    reasoning: str = dspy.OutputField(desc="The reasoning for the comparison")
    better_output: str = dspy.OutputField(desc="The model output that better satisfies the guideline, either 'A' or 'B', or 'tie' if they are equal")


class LLMJudge(dspy.Module):
    def __init__(self, task_description, lm, max_workers=32, omit_input=False):
        self.lm = lm
        self.task_description = task_description
        self.max_workers = max_workers
        self.omit_input = omit_input
        self.evaluator = use_lm(self.lm)(dspy.Predict(EvaluateRequirement))
        self.aggregate_evaluator = use_lm(self.lm)(dspy.Predict(EvaluateGuideline))
        self.compare_evaluator = use_lm(self.lm)(dspy.Predict(CompareModelOutputsWithGuideline))

    def evaluate_requirement(self, example, requirement):
        return self.evaluator(task_description=self.task_description, 
                            model_input="" if self.omit_input else example.inputs().toDict(), 
                            model_output=example.output, 
                            requirement=requirement)
    
    def evaluate_guideline(self, example, guideline):
        return self.aggregate_evaluator(task_description=self.task_description, 
                            model_input=example.inputs().toDict(), 
                            model_output=example.output, 
                            guideline=guideline)
    
    def compare_outputs(self, example_a, example_b, guideline):
        return self.compare_evaluator(task_description=self.task_description, 
                            model_input=example_a.inputs().toDict(), 
                            model_output_a=example_a.output, 
                            model_output_b=example_b.output, 
                            guideline=guideline)
    
    def compare(self, examples_a, examples_b, requirements):
        def random_permute(example_a, example_b):
            if np.random.rand() > 0.5:
                return example_a, example_b, 0
            else:
                return example_b, example_a, 1
        
        np.random.seed(42)
        permuations = [random_permute(example_a, example_b) for example_a, example_b in zip(examples_a, examples_b)]

        results = batch_inference(
            self.compare_outputs,
            [{"example_a": example_a, "example_b": example_b, "guideline": requirements} for example_a, example_b, _ in permuations],
            max_workers=self.max_workers
        )
        # calculate win rate, mapping permutations back
        win_rate_A = sum([(result.better_output == 'A' and not is_permutated) or (result.better_output == 'B' and is_permutated) 
                            for result, (_, _, is_permutated) in zip(results, permuations)]) / len(results)
        win_rate_B = sum([(result.better_output == 'B' and not is_permutated) or (result.better_output == 'A' and is_permutated)
                            for result, (_, _, is_permutated) in zip(results, permuations)]) / len(results)
        print(f"Win rate for A: {win_rate_A}")
        print(f"Win rate for B: {win_rate_B}")

        eval_results = []
        for result, (example_a, example_b, is_permutated) in zip(results, permuations):
            eval_results.append({
                "input": example_a.inputs().toDict(),
                "output_a": example_a.output,
                "output_b": example_b.output,
                "permutated": is_permutated,
                "execution_a": result.execution_a,
                "execution_b": result.execution_b,
                "reasoning": result.reasoning,
                "better_output": result.better_output,
            }) 

        return eval_results

    def forward(self, examples, requirements, aggregate=False):
        if aggregate:        
            results = batch_inference(
                self.evaluate_guideline,
                [{"example": example, "guideline": requirements} for example in examples],
                max_workers=self.max_workers
            )
            for example, result in zip(examples, results):
                if not hasattr(example, "requirements"):
                    example.requirements = {}
                for requirement, eval_result in zip(requirements, result.results):
                    if requirement != eval_result.requirement:
                        print(f"Requirement mismatch: {requirement} != {eval_result.requirement}")
                    example.requirements[requirement] = {
                        "requirement": requirement,
                        "evaluation_plan": eval_result.evaluation_plan,
                        "plan_execution": eval_result.plan_execution,
                        "is_applicable": eval_result.is_applicable,
                        "meets_requirement": eval_result.meets_requirement
                    }
        
        else:
            results = batch_inference(
                self.evaluate_requirement,
                [{"example": example, "requirement": requirement} for requirement in requirements for example in examples],
                max_workers=self.max_workers
            )
            
            for i, result in enumerate(results):
                requirement_id = i // len(examples)
                example_id = i % len(examples)
                requirement = requirements[requirement_id]
                # create the requirements field if it doesn't exist
                if not hasattr(examples[example_id], "requirements"):
                    examples[example_id].requirements = {}
                examples[example_id].requirements[requirement] = {
                    "requirement": requirement,
                    "evaluation_plan": result.evaluation_plan,
                    "plan_execution": result.plan_execution,
                    "is_applicable": result.is_applicable,
                    "meets_requirement": result.meets_requirement
                }

        return examples

    def calculate_score(self, evaluate_examples, requirement):
        applicable_examples = [example for example in evaluate_examples if example.requirements[requirement]["is_applicable"]]
        applicable_ratio = len(applicable_examples) / len(evaluate_examples)
        if applicable_ratio == 0:
            score = 1
        else:
            score = sum([example.requirements[requirement]['meets_requirement'] for example in applicable_examples]) / len(applicable_examples)
        return applicable_ratio, score

    def evaluate(self, examples, requirements, program=None, aggregate=False):
        examples = copy.deepcopy(examples)

        # run the program to generate the output if provided
        if program is not None:
            examples = run_model(program, examples)

        evaluate_examples = self.forward(examples, requirements, aggregate=aggregate)
        pass_rates = []
        for requirement in requirements:
            applicable_ratio, pass_rate = self.calculate_score(evaluate_examples, requirement)
            print(f"{requirement} | {applicable_ratio} | {pass_rate}")
            pass_rates.append(pass_rate)
        print(f"Average pass rate: {sum(pass_rates) / len(pass_rates)}")
        return evaluate_examples


