import numpy as np
import copy
import dspy
from .utils import use_lm, batch_inference, run_model, find_nearest_requirement
from typing import List, Any

class EvaluateRequirement(dspy.Signature):
    """You are a reviewer who is evaluating whether a model output satisfies the given requirement. 
    Given a task description, model input, model output, and requirement, first generate a step-by-step evaluation plan for the requirement, then execute the evaluation plan to evaluate if the model output meets the requirement."""
    
    task_description = dspy.InputField(desc="Description of the task")
    model_input = dspy.InputField(desc="The model input")
    model_output = dspy.InputField(desc="The model output")
    requirement = dspy.InputField(desc="The requirement to evaluate")
    evaluation_plan: str = dspy.OutputField(desc="The evaluation plan for the requirement")
    plan_execution: str = dspy.OutputField(desc="The execution of the evaluation plan")
    meets_requirement: bool = dspy.OutputField(desc="Whether the model output meets the requirement, True or False")

class EvaluateGuideline(dspy.Signature):
    """You are a reviewer who is evaluating whether a model output satisfies the given guideline.
Given a task description, model input, model output, and guideline, first evaluate the model output using the requirements in the guideline one by one. For each requirement in the guideline, do the evaluation step-by-step. 
Then, calculate an overall score for how many requirements the model output satisfies."""

    task_description: str = dspy.InputField(desc="Description of the task")
    model_input: str = dspy.InputField(desc="The model input")
    model_output: str = dspy.InputField(desc="The model output")
    guideline: List[str] = dspy.InputField(desc="The guideline for evaluation")
    evaluation_execution: str = dspy.OutputField(desc="The execution of the evaluation guideline")
    score: int = dspy.OutputField(desc="A score indicating how many requirements in the guideline the model output satisfies")

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
    def __init__(self, task_description, lm, max_workers=32):
        self.lm = lm
        self.task_description = task_description
        self.max_workers = max_workers
        self.evaluator = use_lm(self.lm)(dspy.Predict(EvaluateRequirement))
        self.aggregate_evaluator = use_lm(self.lm)(dspy.Predict(IdentifyMistakes))
        self.compare_evaluator = use_lm(self.lm)(dspy.Predict(CompareModelOutputsWithGuideline))

    def evaluate_requirement(self, example, requirement, omit_input=False):
        return self.evaluator(task_description=self.task_description, 
                            model_input="" if omit_input else example.inputs().toDict(), 
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
                example.evaluation_result = {
                    "evaluation_execution": result.evaluation_execution,
                    "unsatisfied_requirements": result.unsatisfied_requirements,
                    "score": 1 - len(result.unsatisfied_requirements) / len(requirements)
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
                    "meets_requirement": result.meets_requirement
                }

        return examples

    def evaluate(self, examples, requirements, program=None, aggregate=False):
        examples = copy.deepcopy(examples)

        # run the program to generate the output if provided
        if program is not None:
            examples = run_model(program, examples)

        if aggregate:
            evaluate_examples = self.forward(examples, requirements, aggregate=aggregate)
            requirements_unsat = { requirement: [] for requirement in requirements }
            for example in evaluate_examples:
                example.requirements = {}
                for requirement in example.evaluation_result['unsatisfied_requirements']:
                    if requirement not in requirements_unsat:
                        # print(f"Requirement not found: {requirement}")
                        requirement = find_nearest_requirement(requirement, requirements)
                        # print(f"Nearest requirement: {requirement}")

                    requirements_unsat[requirement].append({
                        "input": example.inputs().toDict(),
                        "output": example.output,
                        "execution": example.evaluation_result['evaluation_execution']
                    })

                    example.requirements[requirement] = {
                        "requirement": requirement,
                        "meets_requirement": False,
                    }
                for requirement in requirements:
                    if requirement not in example.requirements:
                        example.requirements[requirement] = {
                            "requirement": requirement,
                            "meets_requirement": True,
                        }
            for requirement, examples in requirements_unsat.items():
                print(f"Requirement: {requirement}")
                pass_rate = 1 - len(examples) / len(evaluate_examples)
                print(f"Pass rate for requirement: {pass_rate}")
            print(f"Average score: {sum([example.evaluation_result['score'] for example in evaluate_examples]) / len(evaluate_examples)}")
        else:
            evaluate_examples = self.forward(examples, requirements)
            pass_rates = []
            for requirement in requirements:
                print(f"Requirement: {requirement}")
                pass_rate = sum([example.requirements[requirement]['meets_requirement'] for example in evaluate_examples]) / len(evaluate_examples)
                print(f"Pass rate for requirement: {pass_rate}")
                pass_rates.append(pass_rate)
            print(f"Average pass rate: {sum(pass_rates) / len(pass_rates)}")
        return evaluate_examples


