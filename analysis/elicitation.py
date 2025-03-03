import dspy
import os
import copy
import dspy.predict
import litellm
from typing import List, Dict, Any
from .textgrad_optimizer import TextGradOptimizer
from .utils import use_lm, batch_inference
from .judge import IdentifyMistakes

class JustifyResponseAndExtractRequirements(dspy.Signature):
    """You are an LLM working on a user-given task. First, provide justification for the model output you produce. The justification should explain why the model output is appropriate for the given task. 
Then extract task-level requirements for the task based on the justification. Make sure the requirements are task-level and not specific to the given example."""

    task_description = dspy.InputField(desc="Description of the task")
    model_input = dspy.InputField(desc="The model input")
    model_output = dspy.InputField(desc="The model output")
    justification: str = dspy.OutputField(desc="Justification for producing the model output")
    requirements: List[str] = dspy.OutputField(desc="Task-level requirements.")

class CritiqueResponseAndSuggestRequirements(dspy.Signature):
    """You are an experienced requirement engineer for an LLM application. Given the task description, model input, model output, first critique the model output, then suggest additional requirements for the task.
The suggested requirements should be applicable beyond the specific example provided."""

    task_description = dspy.InputField(desc="Description of the task")
    model_input = dspy.InputField(desc="The model input")
    model_output = dspy.InputField(desc="The model output")
    critique: str = dspy.OutputField(desc="Critique of the model output")
    suggested_requirements: List[str] = dspy.OutputField(desc="Suggested additional requirement for the LLM")

class ClassifyRequirement(dspy.Signature):
    """Classify whether the requirement contains example-specific information. Example-specific information is information that is specific to the given example and not generalizable to other examples. This is in contrast to task-level requirements, which are generalizable to other examples."""

    task_description = dspy.InputField(desc="Description of the task")
    model_input = dspy.InputField(desc="The model input")
    requirement = dspy.InputField(desc="The requirement")
    input_specific: bool = dspy.OutputField(desc="Whether the requirement contains example-specific information.")

class GroupRequirements(dspy.Signature):
    """Group requirements into different clusters. Make sure that all provided requirements are put into one of the clusters."""

    task_description = dspy.InputField(desc="Description of the task")
    requirements = dspy.InputField(desc="List of requirements")
    groups: Dict[str, List[str]] = dspy.OutputField(desc="Grouped requirements")

# class GenerateEvaluationPlan(dspy.Signature):
#     """You are a reviewer who is evaluating a model output. Given a task description and a requirement, generate a step-by-step evaluation plan."""

#     task_description = dspy.InputField(desc="Description of the task")
#     requirement = dspy.InputField(desc="The requirement")
#     evaluation_plan = dspy.OutputField(desc="Evaluation plan for the requirement")

class CompareModelOutputs(dspy.Signature):
    """You are a reviewer who is comparing two model outputs. List all differences of these two outputs -- consider different aspects like length, style, content, structure, etc.

Examples:
- Model Output A mention references but Model Output B does not
- Model Output A uses significantly more formulas and equations than Model Output B"""

    task_description = dspy.InputField(desc="Description of the task")
    model_output_a = dspy.InputField(desc="The first model output")
    model_output_b = dspy.InputField(desc="The second model output")
    differences: List[str] = dspy.OutputField(desc="List of differences between the two model outputs")

class SummarizeDifferences(dspy.Signature):
    """Given a list of differences between two model outputs, first summarize the differences into key high-level differences. Then expand the differences into a list of requirements for each model respectively."""
    
    task_description = dspy.InputField(desc="Description of the task")
    differences = dspy.InputField(desc="List of example-level differences between two model outputs")
    summary: List[str] = dspy.OutputField(desc="A summary list of the key high-level differences between the two model outputs")
    requirements_a: List[str] = dspy.OutputField(desc="A list of requirements for the first model output based on the summary")
    requirements_b: List[str] = dspy.OutputField(desc="A list of requirements for the second model output based on the summary")

class BrainstormRequirements(dspy.Signature):
    """Given a task description, brainstorm a list of requirements that a model output should satisfy when performing the task."""

    task_description: str = dspy.InputField(desc="Description of the task")
    n: int = dspy.InputField(desc="Number of requirements to brainstorm")
    requirements: List[str] = dspy.OutputField(desc="A list of requirements")

class SuggestDesignDecisions(dspy.Signature):
    """You are a developer working on LLM application. Given a task description, suggest potential design choices to make for model's outputs."""

    task_description: str = dspy.InputField(desc="Description of the task")
    n: int = dspy.InputField(desc="Number of design chocies to suggest")
    design_choices: List[str] = dspy.OutputField(desc="A list of suggested design chocies")

class InferRequirementsFromTask(dspy.Module):

    def __init__(self, task_description, lm):
        self.lm = lm
        self.task_description = task_description
        self.suggest = use_lm(self.lm)(dspy.Predict(BrainstormRequirements))
    
    def forward(self, n=10):
        return self.suggest(task_description=self.task_description, n=n).requirements
    

class InferRequirementsFromData(dspy.Module):

    def __init__(self, task_description, lm, judge_lm):
        self.lm = lm
        self.judge_lm = judge_lm
        self.task_description = task_description
        self.extract = use_lm(self.lm)(dspy.Predict(JustifyResponseAndExtractRequirements))
        self.suggest = use_lm(self.lm)(dspy.Predict(CritiqueResponseAndSuggestRequirements))
        self.classify = use_lm(self.judge_lm)(dspy.ChainOfThought(ClassifyRequirement))
        self.group = use_lm(self.lm)(dspy.Predict(GroupRequirements))


    def forward(self, examples, n=10):
        
        results = batch_inference(self.suggest, [
            {"task_description": self.task_description, 
             "model_input": example.inputs().toDict(), 
             "model_output": example.output} for example in examples
        ])

        for example, result in zip(examples, results):
            example.critique, example.requirements = result.critique, result.suggested_requirements
            
        all_requirements = [req for example in examples for req in example.requirements]

        # arg_list = [{
        #     "task_description": self.task_description, 
        #     "model_input": example.inputs().toDict(),
        #     "requirement": requirement
        # } for example in examples for requirement in example.requirements]

        # results = batch_inference(self.classify, arg_list)
        # # filter out example-specific requirements
        # filtered_requirements = []
        # accum = 0
        # for i, example in enumerate(examples):
        #     example.requirements = [req for j, req in enumerate(example.requirements) if not results[accum + j].input_specific]
        #     accum += len(example.requirements)
        #     filtered_requirements.extend(example.requirements)

        # grouped_requirements = self.group(task_description=self.task_description, requirements=all_requirements).groups
        
        return all_requirements
class InferRequirementsFromCompareData(dspy.Module):

    def __init__(self, task_description, lm, judge_lm):
        self.lm = lm
        self.judge_lm = judge_lm
        self.task_description = task_description
        self.compare = use_lm(self.lm)(dspy.Predict(CompareModelOutputs))
        self.summarize = use_lm(self.judge_lm)(dspy.Predict(SummarizeDifferences))


    def forward(self, examples_a, examples_b, n=10):
        
        results = batch_inference(self.compare, [
            {"task_description": self.task_description, 
             "model_output_a": example_a.output,
            "model_output_b": example_b.output} for (example_a, example_b) in zip(examples_a, examples_b)
        ])

        all_differences = []

        for (example_a, example_b), result in zip(zip(examples_a, examples_b), results):
            example_a.differences = result.differences
            example_b.differences = result.differences
            all_differences.extend(result.differences)

        # summarize the differences
        result = self.summarize(task_description=self.task_description, differences=all_differences)

        return result.summary, result.requirements_a + result.requirements_b
        
class InferRequirements(dspy.Module):

    def __init__(self, task_description, lm):
        self.lm = lm
        self.task_description = task_description
        self.suggest = use_lm(self.lm)(dspy.Predict(BrainstormRequirements))
        self.identify = use_lm(self.lm)(dspy.Predict(IdentifyMistakes))
    
    def forward(self, examples, requirements=None, n=10):
        if requirements == None:
            requirements = self.suggest(task_description=self.task_description, n=n).requirements
        requirements_unsat_result = {
            requirement: [] for requirement in requirements
        }
        arg_list = [{
            "task_description": self.task_description, 
            "model_input": example.inputs().toDict(),
            "model_output": example.output,
            "guideline": requirements
        } for example in examples]
        results = batch_inference(self.identify, arg_list)
        for example, result in zip(examples, results):
            for requirement in result.unsatisfied_requirements:
                if requirement not in requirements_unsat_result:
                    requirements_unsat_result[requirement] = []
                requirements_unsat_result[requirement].append({
                    "input": example.inputs().toDict(),
                    "output": example.output,
                    "execution": result.evaluation_execution,
                })
        return requirements_unsat_result
    
    def rank_and_filter(self, examples, requirements=None, n=10):
        MIN_UNSAT_CUTOFF = 5
        requirements_unsat_result = self.forward(examples, requirements=requirements, n=n)
        # sort the requirements by the number of examples that don't meet them
        requirements_unsat_result = {k: v for k, v in sorted(requirements_unsat_result.items(), key=lambda item: len(item[1]), reverse=True)}
        filtered_requirements = [k for k, v in requirements_unsat_result.items() if len(v) >= MIN_UNSAT_CUTOFF]
        return filtered_requirements
    
