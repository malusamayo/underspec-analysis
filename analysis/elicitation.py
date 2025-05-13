import dspy
import os
import copy
import dspy.predict
import litellm
import json
import argparse
import mlflow
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from pydantic import BaseModel
from .utils import use_lm, batch_inference, run_model, cluster_requirements, LM_DICT
from .judge import IdentifyMistakes, LLMJudge
from .load_data import prepare_data, load_data

class JustifyResponseAndExtractRequirements(dspy.Signature):
    """You are an LLM working on a user-given task. First, provide justification for the model output you produce. The justification should explain why the model output is appropriate for the given task. 
Then extract task-level requirements for the task based on the justification. Make sure the requirements are task-level and not specific to the given example."""

    task_description = dspy.InputField(desc="Description of the task")
    model_input = dspy.InputField(desc="The model input")
    model_output = dspy.InputField(desc="The model output")
    justification: str = dspy.OutputField(desc="Justification for producing the model output")
    requirements: List[str] = dspy.OutputField(desc="Task-level requirements.")

class CritiqueResponseAndSuggestRequirements(dspy.Signature):
    """You are an experienced requirements engineer. Your goal is to extract a list of atomic requirements that specify desired LLM behaviors for the given task.

You will be presented with a model input and several model outputs from different models. First, provide a detailed analysis critiquing the model outputs.
Then, based on the analysis, suggest a list of atomic requirements that specify desired LLM behaviors for the given task.
These requirements should be consistent with each other without contradictions and complementary to existing requirements.

Guidelines:
- Each requirement should test exactly ONE requirement
- Requirements should be easily verifiable, almost as if writing a Boolean condition in Python
- Requirements should not be overly general (i.e. they should not be universal requirements that might apply to any tasks)
- Requirements should be generally applicable for responses to that task, not referring to any specific input examples
- Focus only on objective, measurable requirements
- Use concise and unambiguous language
- The requirements should be consistent with each other without contradictions
- The requirements should not overlap with existing requirements

Here are some bad requirements:
- The output should be interesting. - This is subjective
- The output should provide examples in fewer than 280 characters. - This overloads multiple aspects
- The output should be helpful and harmless. - This is overly general

Here are some good atomic requirements:
- The output should provide examples.
- The output should be fewer than 280 characters.
- The output should contain at least 3 references."""

    task_description: str = dspy.InputField(desc="Description of the task")
    existing_requirements: List[str] = dspy.InputField(desc="List of existing requirements")
    model_input: str = dspy.InputField(desc="The model input")
    model_outputs: List[str] = dspy.InputField(desc="The model outputs")
    analysis: str = dspy.OutputField(desc="Analysis of the model outputs")
    # similarities_analysis: str = dspy.OutputField(desc="Analysis of consistencies across the model outputs")
    # differences_analysis: str = dspy.OutputField(desc="Analysis of differences across the model outputs")
    suggested_requirements: List[str] = dspy.OutputField(desc="Suggested additional requirement for the LLM")

class RefineRequirement(dspy.Signature):
    """You are an experienced requirements engineer. Your goal is to curate a list of atomic requirements that specify desired LLM behaviors for the given task.
Given a task description and a requirements, first decide whether the requirement is too specific. If yes, refine the requirement to be more general. If not, keep the requirement as is.

Examples of overly specific requirements:
- The explanation should include the significance of the main check (__name__ == '__main__').
- The main character should be a talking fox who escapes from a science lab in New York City.

Examples of acceptable requirements:
- The summary should include key methods and datasets used in the paper.
- The plot should be engaging and include unexpected developments to maintain interest."""

    task_description = dspy.InputField(desc="Description of the task")
    requirement = dspy.InputField(desc="The requirement")
    is_over_specific: bool = dspy.OutputField(desc="Whether the requirement is too specific")
    refined_requirement: str = dspy.OutputField(desc="Refined requirement")

class RiskItem(BaseModel):
    """A requirement with its priority level."""
    requirement: str
    risk_analysis: str
    risk_level: str

class AnalyzeRequirement(dspy.Signature):
    """You are a product risk analyst working on an LLM application. You have received a requirement from the product manager.

For the given requirement, analyze the risk associated with failing to implement it properly. Risk in this context refers to the potential negative consequences to the business, legal exposure, or harm to the user if the requirement is not met.

Assign a risk level to the requirement:
- High Risk: If unmet, this could result in significant legal, financial, or reputational damage. For example, in a travel plan app, if a client follows an inaccurate itinerary and suffers harm or loss, they might have grounds to sue.
- Medium Risk: If unmet, this could lead to moderate user dissatisfaction or operational inefficiencies but is unlikely to cause legal issues or serious harm.
- Low Risk: If unmet, this would only result in minor inconvenience or cosmetic issues with little to no long-term impact.

Clearly justify the risk level you assign to the requirement. The risk levels are 'high', 'medium', or 'low'."""

    task_description: str = dspy.InputField(desc="Description of the task")
    requirement = dspy.InputField(desc="The given requirement")
    risk_analysis: str = dspy.OutputField(desc="Risk analysis of the requirement")
    risk_level: str = dspy.OutputField(desc="Risk level in 'high', 'medium', or 'low'")
    # requirements: List[str] = dspy.InputField(desc="List of requirements")
    # risks: List[RiskItem] = dspy.OutputField(desc="List of requirements with their risk levels and analysis")

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
- Requirements should be easily verifiable, almost as if writing a Boolean condition in Python
- Requirements should not be overly general (i.e. they should not be universal requirements that might apply to any reasonable reasponse)
- Requirements should be generally applicable for responses to that task, not referring to any specific response
- Focus only on objective, measurable requirements
- Use concise and unambiguous language

Here are some bad requirements:
- The output should be interesting. - This is subjective
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

class BrainstormAdditionalRequirementsRiskAware(dspy.Signature):
    """You are an experienced requirements engineer. Your goal is to brainstorm a list of high-risk requirements that specify desired LLM behaviors for the given task.
Risk in this context refers to the potential negative consequences to the business, legal exposure, or harm to the user if the requirement is not met.

These requirements should be consistent with each other without contradictions and complementary to existing requirements.
    
Guidelines:
- Each requirement should test exactly ONE requirement
- Requirements should be easily verifiable, almost as if writing a Boolean condition in Python. They should be testable with Python code or an LLM itself (no human judgment or external sources needed).
- Requirements should not be overly general (i.e. they should not be universal requirements that might apply to any reasonable reasponse)
- Requirements should be generally applicable for responses to that task, not referring to any specific response
- Requirements should be high-risk, meaning that if unmet, this could result in significant legal, financial, or reputational damage. 
- Avoid unrealistic edge cases - focus on plausible failures that could occur even in aligned or well-trained LLMs.
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

class BrainstormAdditionalRequirementsCustomerExperience(dspy.Signature):
    """You are an experienced requirements engineer. Your goal is to brainstorm a list of requirements that specify desired LLM behaviors for the given task.
These requirements should identify behaviors that, if omitted, would likely frustrate or annoy users -- such as forgetting to surface important reminders, warnings, or common-sense.

These requirements should be consistent with each other without contradictions and complementary to existing requirements.
    
Guidelines:
- Each requirement should test exactly ONE requirement
- Requirements should be easily verifiable, almost as if writing a Boolean condition in Python. They should be testable with Python code or an LLM itself (no human judgment or external sources needed).
- Requirements should not be overly general (i.e. they should not be universal requirements that might apply to any reasonable reasponse)
- Requirements should be generally applicable for responses to that task, not referring to any specific response
- Avoid unrealistic edge cases - focus on plausible failures that could occur even in aligned or well-trained LLMs.
- Focus only on objective, measurable requirements
- Use concise and unambiguous language
- Never generate similar requirements to the existing requirements"""

    task_description: str = dspy.InputField(desc="Description of the task")
    existing_requirements: List[str] = dspy.InputField(desc="List of existing requirements")
    failure_mode_analysis: str = dspy.OutputField(desc="Failure modes that would lead to user dissatisfaction or confusion if not caught")
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

class ClassifyRequirement(dspy.Signature):
    """Classify the requirements into one of the following categories: content, style, and format.

- Content constraints refer to explicit impositions of specific conditions that shape the depth or scope of the response content.
- Style Constraints control the stylistic variations of output to accomplish specific stylistic goals, such as tone, sentiment, formality, and empath.
- Format Constraints refer to stipulations governing the structural, linguistic, or output presentation of generated content."""
    requirement = dspy.InputField(desc="The requirement")
    category: str = dspy.OutputField(desc="Category of the requirement, content, style, or format")

class InferRequirementsFromTask(dspy.Module):

    def __init__(self, task_description, lm):
        self.lm = lm
        self.task_description = task_description
        self.suggest = use_lm(self.lm)(dspy.Predict(BrainstormRequirements))
        self.suggest_with_prompt = use_lm(self.lm)(dspy.Predict(BrainstormAdditionalRequirementsCustomerExperience))
    
    def forward(self, existing_requirements=[], n=20):
        all_requirements = copy.deepcopy(existing_requirements)
        target_count = n + len(existing_requirements)
        
        # Iteratively add more requirements until we reach n
        while len(all_requirements) < target_count:
            
            new_requirements = self.suggest_with_prompt(
                task_description=self.task_description, 
                existing_requirements=all_requirements).requirements
            
            all_requirements.extend(new_requirements)
        
        return all_requirements[target_count-n:target_count]

class InferRequirementsFromData(dspy.Module):

    def __init__(self, task_description, lm, judge_lm):
        self.lm = lm
        self.judge_lm = judge_lm

        self.task_description = task_description
        self.extract = use_lm(self.lm)(dspy.Predict(JustifyResponseAndExtractRequirements))
        self.suggest = use_lm(self.lm)(dspy.Predict(CritiqueResponseAndSuggestRequirements))
        self.refine = use_lm(self.lm)(dspy.Predict(RefineRequirement))

    def forward(self, examples, existing_requirements, n=10):
        
        results = batch_inference(self.suggest, [
            {"task_description": self.task_description, 
             "existing_requirements": existing_requirements,
             "model_input": example.inputs().toDict(), 
             "model_outputs": example.outputs} for example in examples
        ])

        all_requirements = [result.suggested_requirements for result in results]
        all_requirements = [req for sublist in all_requirements for req in sublist]

        results = batch_inference(self.refine, [
            {"task_description": self.task_description,
                "requirement": requirement} for requirement in all_requirements
        ])
        for i, requirement in enumerate(all_requirements):
            if results[i].is_over_specific:
                # print(f"Refining requirement: {requirement}")
                # print(f"Refined requirement: {results[i].refined_requirement}")
                all_requirements[i] = results[i].refined_requirement

        # self.suggest = use_lm(self.lm)(dspy.Predict(CritiqueResponseAndSuggestRequirements, n=10))
        # results = batch_inference(self.suggest, [
        #     {"task_description": self.task_description, 
        #      "existing_requirements": existing_requirements,
        #      "model_input": example.inputs().toDict(), 
        #      "model_outputs": example.outputs} for example in examples
        # ])

        # print(all_requirements)
        # print(results[0].completions.suggested_requirements)

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
        
class ElicitRequirements(dspy.Module):

    def __init__(self, task_description, lm, analyst_lms, judge_lm):
        self.lm = lm
        self.analyst_lms = analyst_lms
        self.judge_lm = judge_lm
        self.long_context_lm = copy.deepcopy(lm)
        self.long_context_lm.kwargs["max_tokens"] = 16384

        self.task_description = task_description
        self.extract = use_lm(self.lm)(dspy.Predict(JustifyResponseAndExtractRequirements))
        self.suggest = use_lm(self.lm)(dspy.Predict(CritiqueResponseAndSuggestRequirements))
        self.refine = use_lm(self.lm)(dspy.Predict(RefineRequirement))
        self.group = use_lm(self.long_context_lm)(dspy.Predict(GroupRequirements))
        # self.rerank = use_lm(self.long_context_lm)(dspy.Predict(RerankRequirements))
        self.analyze = use_lm(self.lm)(dspy.Predict(AnalyzeRequirement))


    def forward(self, prompt, trainset):

        requirements_dicts = []

        extract = use_lm(self.lm)(dspy.Predict(ExtractRequirementsFromPrompt))
        existing_requirements = extract(prompt=prompt).requirements

        print(existing_requirements)

        for req in existing_requirements:
            requirements_dicts.append(
                {
                    "task": task,
                    "analyst": self.lm.model,
                    "source": "prompt",
                    "requirement": req
                }
            )

        for analyst_lm in  self.analyst_lms:
            infer = InferRequirementsFromTask(task_description=task_description, lm=analyst_lm)
            new_requirements = infer(existing_requirements=existing_requirements, n=20)
            print(new_requirements)

            for req in new_requirements:
                requirements_dicts.append(
                    {
                        "task": task,
                        "analyst": analyst_lm.model,
                        "source": "top-down",
                        "requirement": req
                    }
                )

            infer = InferRequirementsFromData(task_description=task_description, lm=analyst_lm, judge_lm=self.judge_lm)
            new_requirements = infer(trainset, existing_requirements)
            print(new_requirements)

            for req in new_requirements:
                requirements_dicts.append(
                    {
                        "task": task,
                        "analyst": analyst_lm.model,
                        "source": "bottom-up",
                        "requirement": req
                    }
                )


        all_requirements = [req["requirement"] for req in requirements_dicts]
        filtered_requirements = cluster_requirements(all_requirements, existing_requirements, num_clusters=len(all_requirements)//3)

        results = batch_inference(self.analyze, [
            {"task_description": self.task_description, 
             "requirement": requirement} for requirement in filtered_requirements
        ])


        requirements_map = {req["requirement"]: req for req in requirements_dicts}
        filtered_requirements_dicts = []
        for req, result in zip(filtered_requirements, results):
            req_dict = copy.deepcopy(requirements_map[req])
            req_dict['risk_level'] = result.risk_level.lower()
            req_dict['risk_analysis'] = result.risk_analysis
            filtered_requirements_dicts.append(req_dict)
            

        # risks = self.rerank(task_description=task_description, requirements=filtered_requirements).risks
        # risk_map = {req.requirement: req for req in risks}
        # requiremets_map = {req["requirement"]: req for req in requirements_dicts}

        # filtered_requirements_dicts = []
        # for req in filtered_requirements:
        #     req_dict = copy.deepcopy(requiremets_map[req])
        #     req_dict |= risk_map[req].dict()
        #     filtered_requirements_dicts.append(req_dict)
            
        requirements_df = pd.DataFrame(filtered_requirements_dicts)

        return requirements_df

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
    # task = "product"
    # task = "trip"
    task_description, TaskProgram, trainset, valset, requirements, prompts = prepare_data(
        task_name=task,
    )

    prompt_model = LM_DICT["gpt-4o"]

    model_names = ["gpt-4o-mini", "gemini-1.5-flash", "llama3-2-11b-instruct"]
    prompt_name = "original"

    trainsets = []
    for model_name in model_names:
        task_model = LM_DICT[model_name]
        task_program = TaskProgram(lm=task_model, n=1)
        trainset = load_data(task, model_name, prompt_name, task_program, {"trainset": trainset})["trainset"]
        trainsets.append(trainset)
    
    # merge trainsets output into outputs
    trainset = [
        dspy.Example(**{task_program.input_key: examples[0][task_program.input_key]}, outputs=[row["output"] for row in examples]).with_inputs(task_program.input_key)
        for examples in zip(*trainsets)
    ]


    analyst_lms = [
        LM_DICT["gpt-4o"], 
        # LM_DICT["gemini-1.5-pro"],
        # LM_DICT["llama3-2-90b-instruct"],
    ]

    requirements_dict = []

    elicit = ElicitRequirements(task_description=task_description, lm=prompt_model, analyst_lms=analyst_lms, judge_lm=LM_DICT["4.1-mini-eval"])
    requirements_df = elicit(task_program.prompt, trainset[:20])
    requirements_df.to_csv(f"data/requirements/{task}_prioritized_requirements.csv", index=False)

    print(json.dumps(requirements_df.to_dict(orient="records"), indent=4))
    exit(0)


    judge = LLMJudge(task_description, LM_DICT["4.1-mini-eval"])

    all_examples = []
    L = len(trainset[0].outputs)
    for i in range(L):
        for example in trainset:
            example.output = example.outputs[i]
        results = judge.evaluate(trainset, requirements=requirements)
        for requirement in requirements:
            pass_rates = sum([example.requirements[requirement]['meets_requirement'] for example in results]) / len(results)
            for example in results:
                all_examples.append({
                    "input": example.inputs().toDict()[task_program.input_key],
                    "output": example.output,
                    "model": model_names[i],
                    "requirement": requirement,
                } | example.requirements[requirement])

    # save all examples to csv
    import pandas as pd
    df = pd.DataFrame(all_examples)
    df.to_csv(f"temp.csv", index=False)

    ## to markdown
    df.to_markdown(f"temp.md", index=False)

    exit(0)


    infer = IterativeRequirementsSearch(task_description=task_description, lm=prompt_model, judge_lm=LM_DICT["gpt-4o-mini"])
    requirements = infer(examples_a=trainsets[0], examples_b=trainsets[1], prompt=task_program.prompt)
                        #  requirements = requirements["known"] + requirements["unseen"])
    print(json.dumps({"unseen": requirements}, indent=4))


    exit(0)