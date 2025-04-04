import dspy
import os
import copy
import dspy.predict
import litellm
import json
import argparse
import mlflow
import numpy as np
from typing import List, Dict, Any
from pydantic import BaseModel
from .textgrad_optimizer import TextGradOptimizer
from .utils import use_lm, batch_inference, run_model, LM_DICT
from .judge import IdentifyMistakes, LLMJudge
from .load_data import prepare_data

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

class CompareTwoModelOutputs(dspy.Signature):
    """"You are a reviewer who is comparing a few model outputs. Analyze what is consistent across all outputs and what is different across them -- consider different aspects like length, style, content, structure, format, etc."""

    task_description = dspy.InputField(desc="Description of the task")
    model_output_a = dspy.InputField(desc="The first model output")
    model_output_b = dspy.InputField(desc="The second model output")
    similarities_analysis: str = dspy.OutputField(desc="Analysis of consistencies across the model outputs")
    similarities: List[str] = dspy.OutputField(desc="List of consistencies across the model outputs")
    differences_analysis: str = dspy.OutputField(desc="Analysis of differences across the model outputs")
    differences: List[str] = dspy.OutputField(desc="List of broad distinctions across the model outputs")

class CompareMultiModelOutputs(dspy.Signature):
    """You are a reviewer who is comparing a few model outputs. Analyze what is consistent across all outputs and what is different across them -- consider different aspects like length, style, content, structure, format, etc."""

    task_description: str = dspy.InputField(desc="Description of the task")
    model_outputs: List[str] = dspy.InputField(desc="List of model outputs")
    similarities_analysis: str = dspy.OutputField(desc="Analysis of consistencies across the model outputs")
    similarities: List[str] = dspy.OutputField(desc="List of consistencies across the model outputs")
    differences_analysis: str = dspy.OutputField(desc="Analysis of differences across the model outputs")
    differences: List[str] = dspy.OutputField(desc="List of broad distinctions across the model outputs")

class DiffItem(BaseModel):
    """A difference between two model outputs."""
    difference_description: str
    mentions: List[int]

class SummarizeDifferences(dspy.Signature):
    """Given a list of differences across model outputs, extract the most frequent 10 key differences. For each key difference, provide a description and the examples where it occurs."""
    
    task_description: str = dspy.InputField(desc="Description of the task")
    differences: List[str] = dspy.InputField(desc="List of example-level differences across model outputs")
    key_differences: List[DiffItem] = dspy.OutputField(desc="A summary of the key differences across two model outputs")
    # requirements_a: List[str] = dspy.OutputField(desc="A list of requirements for the first model output based on the summary")
    # requirements_b: List[str] = dspy.OutputField(desc="A list of requirements for the second model output based on the summary")

class ExtractRequirementsFromPrompt(dspy.Signature):
    """You are an experienced requirements engineer. Your goal is to extract a list of atomic requirements from the provided prompt.

Guidelines:
- Each requirement should test exactly ONE requirement
- Requirements should be easily verifiable, almost as if writing a Boolean condition in Python
- Requirements should not be overly general (i.e. they should not be universal requirements that might apply to any reasonable reasponse)
- Requirements should be generally applicable for responses to that task, not referring to any specific response
- Focus only on objective, measurable requirements
- Use concise and unambiguous language

Here are some bad requirements:
- The output should be interesting. - This is subjective
- The output should discuss cats in fewer than 280 characters. - This overloads multiple aspects
- The output should be helpful and harmless. - This is overly general

Here are some good atomic requirements:
- The output should discuss cats.
- The output should be fewer than 280 characters.
- The output should contain at least 3 references."""

    prompt: str = dspy.InputField(desc="Task prompt")
    requirements: List[str] = dspy.OutputField(desc="A list of requirements")

class RephraseDiffsToRequirements(dspy.Signature):
    """You are an experienced requirements engineer. You observe that there are lots of divergent behaviors among model outputs.
From this list of differences across model outputs, your goal is to extract a list of atomic requirements that specify desired behaviors.
These requirements should be consistent with each other without contradictions and complementary to existing requirements.

Guidelines:
- Each requirement should test exactly ONE requirement
- Requirements should be easily verifiable, almost as if writing a Boolean condition in Python
- Requirements should not be overly general (i.e. they should not be universal requirements that might apply to any reasonable reasponse)
- Requirements should be generally applicable for responses to that task, not referring to any specific response
- Focus only on objective, measurable requirements
- Use concise and unambiguous language
- The requirements should be consistent with each other without contradictions
- The requirements should not overlap with existing requirements

Here are some bad requirements:
- The output should be interesting. - This is subjective
- The output should discuss cats in fewer than 280 characters. - This overloads multiple aspects
- The output should be helpful and harmless. - This is overly general

Here are some good atomic requirements:
- The output should discuss cats.
- The output should be fewer than 280 characters.
- The output should contain at least 3 references."""

    task_description: str = dspy.InputField(desc="Description of the task")
    existing_requirements: List[str] = dspy.InputField(desc="List of existing requirements")
    differences: List[DiffItem] = dspy.InputField(desc="List of differences")
    n: int = dspy.InputField(desc="Number of requirements to extract")
    requirements: List[str] = dspy.OutputField(desc="List of requirements")

class BrainstormRequirements(dspy.Signature):
    """Given a task description, brainstorm a list of requirements that a model output should satisfy when performing the task."""

    task_description: str = dspy.InputField(desc="Description of the task")
    n: int = dspy.InputField(desc="Number of requirements to brainstorm")
    requirements: List[str] = dspy.OutputField(desc="A list of requirements")

class BrainstormAdditionalRequirements(dspy.Signature):
    """You are an experienced requirements engineer. Your goal is to brainstorm a list of atomic requirements that augment the existing requirements.
These requirements should be consistent with each other without contradictions and complementary to existing requirements.
    
Guidelines:
- Each requirement should test exactly ONE requirement
- Requirements should be easily verifiable, almost as if writing a Boolean condition in Python
- Requirements should not be overly general (i.e. they should not be universal requirements that might apply to any reasonable reasponse)
- Requirements should be generally applicable for responses to that task, not referring to any specific response
- Focus only on objective, measurable requirements
- Use concise and unambiguous language
- Never generate similar requirements to the existing requirements

Here are some bad requirements:
- The output should be interesting. - This is subjective
- The output should discuss cats in fewer than 280 characters. - This overloads multiple aspects
- The output should be helpful and harmless. - This is overly general

Here are some good atomic requirements:
- The output should discuss cats.
- The output should be fewer than 280 characters.
- The output should contain at least 3 references."""

    task_description: str = dspy.InputField(desc="Description of the task")
    existing_requirements: List[str] = dspy.InputField(desc="List of existing requirements")
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
        self.suggest_with_prompt = use_lm(self.lm)(dspy.ChainOfThought(BrainstormAdditionalRequirements))
    
    def forward(self, existing_requirements=[], n=20):
        if existing_requirements == []:
            return self.suggest(task_description=self.task_description, n=n).requirements
        else:
            return self.suggest_with_prompt(
                task_description=self.task_description, 
                existing_requirements=existing_requirements, n=n).requirements
    

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
        self.compare = use_lm(self.lm)(dspy.Predict(CompareTwoModelOutputs))
        self.compare_mul = use_lm(self.lm)(dspy.Predict(CompareMultiModelOutputs))
        self.summarize = use_lm(self.lm)(dspy.Predict(SummarizeDifferences))
        self.extract = use_lm(self.lm)(dspy.Predict(ExtractRequirementsFromPrompt))
        self.rephrase = use_lm(self.lm)(dspy.ChainOfThought(RephraseDiffsToRequirements))


    def forward(self, examples_a, examples_b=None, existing_requirements=[], n=10):
        all_differences = []
        ## if examples_b is not provided, compare the examples to themselves
        if examples_b == None:
            results = batch_inference(self.compare_mul, [
                {"task_description": self.task_description, 
                 "model_outputs": example.outputs, } for example in examples_a # [example.output for example in examples_a[i:i+5]]} for i in range(0, len(examples_a), 5)
            ])

            for diff in results:
                all_differences.extend(diff.differences)
        
        else:
        
            results = batch_inference(self.compare, [
                {"task_description": self.task_description, 
                "model_output_a": example_a.output,
                "model_output_b": example_b.output} for (example_a, example_b) in zip(examples_a, examples_b)
            ])

            for (example_a, example_b), result in zip(zip(examples_a, examples_b), results):
                example_a.differences = result.differences
                example_b.differences = result.differences
                all_differences.extend(result.differences)

        # summarize the differences
        result = self.summarize(task_description=self.task_description, differences=all_differences)

        requirements = self.rephrase(
            task_description=self.task_description, 
            existing_requirements=existing_requirements, 
            differences=result.key_differences,
            n=n).requirements

        return requirements
        
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

class IterativeRequirementsSearch(dspy.Module):
    """Iteratively search for requirements by comparing model outputs and identifying requirements 
    where there are significant differences in pass rates between models."""

    def __init__(self, task_description, lm, judge_lm):
        self.lm = lm
        self.judge_lm = judge_lm
        self.task_description = task_description
        
        self.extract = use_lm(self.lm)(dspy.Predict(ExtractRequirementsFromPrompt))
        self.compare_module = InferRequirementsFromCompareData(task_description, lm, judge_lm)
        self.judge = LLMJudge(task_description, judge_lm)

    def evaluate_requirements(self, examples_a, examples_b, requirements):
        """Evaluate pass rates for requirements on both sets of examples."""
        results_a = self.judge.evaluate(examples_a, requirements)
        results_b = self.judge.evaluate(examples_b, requirements)
        
        pass_rates = {}
        for requirement in requirements:
            pass_rate_a = sum([example.requirements[requirement]['meets_requirement'] for example in results_a]) / len(results_a)
            pass_rate_b = sum([example.requirements[requirement]['meets_requirement'] for example in results_b]) / len(results_b)
            pass_rates[requirement] = {
                'pass_rate_a': pass_rate_a,
                'pass_rate_b': pass_rate_b,
            }
        return pass_rates

    def filter_requirements(self, pass_rates, delta=0.1):
        """Filter requirements based on pass rate gap."""
        return [
            req for req, rates in pass_rates.items() 
            if abs(rates['pass_rate_a'] - rates['pass_rate_b']) >= delta
        ]

    def forward(self, examples_a, examples_b, prompt="", requirements=[], budget=15, batch_size=5):
        """
        Iteratively search for requirements with significant pass rate differences.
        
        Args:
            examples_a: First set of examples
            examples_b: Second set of examples
            prompt: Initial prompt to extract base requirements
            budget: Maximum number of requirements to generate
            batch_size: Number of examples to analyze in each iteration
            
        Returns:
            List of requirements with significant pass rate differences
        """
        discovered_requirements = self.extract(prompt=prompt).requirements if requirements == [] else requirements
        final_requirements = []
        
        # Split examples into batches for iterative analysis
        num_examples = len(examples_a)
        for start_idx in range(0, num_examples, batch_size):
            if len(final_requirements) >= budget:
                break
                
            end_idx = min(start_idx + batch_size, num_examples)
            batch_a = examples_a[start_idx:end_idx]
            batch_b = examples_b[start_idx:end_idx]
            
            # Generate new requirements through comparative analysis
            new_requirements = self.compare_module(
                examples_a=batch_a,
                examples_b=batch_b,
                existing_requirements=discovered_requirements,
                n=10
            )

            dspy.inspect_history(n=1)
            
            # Add new requirements to discovered list
            discovered_requirements += [req for req in new_requirements if req not in discovered_requirements]
            
            # Evaluate new requirements    
            pass_rates = self.evaluate_requirements(examples_a, examples_b, new_requirements)

            print(pass_rates)
            
            # Filter requirements with significant gaps
            significant_reqs = self.filter_requirements(pass_rates)
            final_requirements.extend(significant_reqs)
            print(final_requirements)
            
        return final_requirements[:budget]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, help="The name of the experiment to log to.")
    args = parser.parse_args()

    if args.experiment:
        mlflow.litellm.autolog()
        # mlflow.dspy.autolog()
        experiment = mlflow.set_experiment(args.experiment)
        print(experiment.experiment_id)

    task = "commitpack"
    # task = "arxiv"
    # task = "product"
    task_description, TaskProgram, trainset, valset, requirements, prompts = prepare_data(
        task_name=task,
    )

    prompt_model = LM_DICT["gpt-4o"]

    model_names = ["gpt-4o-mini", "gemini-1.5-flash"]
    prompt_name = "original"
    trainsets = []
    for model_name in model_names:
        task_model = LM_DICT[model_name]
        task_program = TaskProgram(lm=task_model, n=1)
        if not os.path.exists(f"data/results/{task}/{model_name}_{prompt_name}_trainset.json"):
            print("Running model...")
            trainset_2 = run_model(task_program, trainset, max_workers=32)
        else:
            print("Loading data from cache...")
            with open(f"data/results/{task}/{model_name}_{prompt_name}_trainset.json", "r") as f:
                trainset_2 = [dspy.Example(**row).with_inputs(task_program.input_key) for row in json.load(f)]
        trainsets.append(trainset_2)
    
    # merge trainsets output into outputs
    trainset = [
        dspy.Example(**{task_program.input_key: examples[0][task_program.input_key]}, outputs=[row["output"] for row in examples]).with_inputs(task_program.input_key)
        for examples in zip(*trainsets)
    ]

    from analysis.elicitation import (
        ExtractRequirementsFromPrompt,
        InferRequirementsFromCompareData,
        InferRequirementsFromTask,
        IterativeRequirementsSearch,
    )

    infer = IterativeRequirementsSearch(task_description=task_description, lm=prompt_model, judge_lm=LM_DICT["gpt-4o-mini"])
    requirements = infer(examples_a=trainsets[0], examples_b=trainsets[1], prompt=task_program.prompt)
                        #  requirements = requirements["known"] + requirements["unseen"])
    print(json.dumps({"unseen": requirements}, indent=4))


    exit(0)
    
    extract = use_lm(prompt_model)(dspy.Predict(ExtractRequirementsFromPrompt))
    existing_requirements = extract(prompt=task_program.prompt).requirements
    # infer = InferRequirementsFromCompareData(task_description=task_description, lm=prompt_model, judge_lm=prompt_model)
    # summary = infer(trainset[:20], existing_requirements=existing_requirements)
    # dspy.inspect_history(n=1)

    existing_requirements += [
        "The output should use structured sections with headings to organize the explanation.",
        "The output should not explain import statements.",
        "The output should include a walkthrough example.",
        "The output should interleave explanations with code snippets.",
        "The output should avoid line-by-line explanations of code.",
    ]

    infer = InferRequirementsFromTask(task_description=task_description, lm=prompt_model)
    requirements = infer(existing_requirements=existing_requirements, n=20)
    dspy.inspect_history(n=1)