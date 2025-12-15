import dspy
from typing import List, Dict, Any
import difflib
import json
import random
from .utils import use_lm
from .load_data import prepare_data
from .utils import requirements_to_str

class ParaphrasePrompt(dspy.Signature):
    """Generate a paraphrase of a given prompt. Make sure no existing requirements are removed and no new requirements are added."""

    prompt: str = dspy.InputField(desc="The prompt to paraphrase.")
    requirements: List[str] = dspy.InputField(desc="The requirements to keep in the paraphrase.")
    paraphrased_prompt: str = dspy.OutputField(desc="The paraphrased prompt.")

class CheckEquivalence(dspy.Signature):
    """Check if two prompts are equivalent with respect to the given requirements."""

    prompt1: str = dspy.InputField(desc="The first prompt.")
    prompt2: str = dspy.InputField(desc="The second prompt.")
    requirements: List[str] = dspy.InputField(desc="The requirements to check for equivalence.")
    is_equivalent: bool = dspy.OutputField(desc="Whether the two prompts are equivalent.")

def generate_paraphrase(lm, prompt, requirements, n=10):
    paraphrase = use_lm(lm)(dspy.Predict(ParaphrasePrompt, n=n))
    check_equivalence = use_lm(lm)(dspy.ChainOfThought(CheckEquivalence))
    paraphrased_prompts = []
    while len(paraphrased_prompts) < n:
        new_paraphrases = paraphrase(prompt=prompt, requirements=requirements).completions.paraphrased_prompt
        new_paraphrases = [p for p in new_paraphrases if check_equivalence(prompt1=prompt, prompt2=p, requirements=requirements).is_equivalent]
        paraphrased_prompts.extend(new_paraphrases)
        # Remove duplicates while preserving order
        paraphrased_prompts = list(dict.fromkeys(paraphrased_prompts))
        print(paraphrased_prompts)
        print(len(paraphrased_prompts))
    return paraphrased_prompts[:n]

class EditPrompt(dspy.Signature):
    """Make small edits to the given prompt. Edit at most a few words and make sure no existing requirements are removed and no new requirements are added."""

    prompt: str = dspy.InputField(desc="The prompt to edit.")
    example_edits: List[str] = dspy.InputField(desc="The example edited prompts.")
    requirements: List[str] = dspy.InputField(desc="The requirements to keep.")
    edited_prompt: str = dspy.OutputField(desc="The edited prompt.")

def count_char_diffs(a, b):
    diff = difflib.ndiff(list(a), list(b))
    count = sum(1 for d in diff if d.startswith('- ') or d.startswith('+ '))
    return count

def generate_edits(lm, prompt, requirements, n=10):
    edit = use_lm(lm)(dspy.Predict(EditPrompt, n=n))
    example_edits = []
    while len(example_edits) < n:
        new_edits = edit(prompt=prompt, example_edits=example_edits, requirements=requirements).completions.edited_prompt
        example_edits.extend([e for e in new_edits if count_char_diffs(prompt, e) <= 20 and count_char_diffs(prompt, e) > 0])
        # Remove duplicates while preserving order
        example_edits = list(dict.fromkeys(example_edits))
        print(example_edits)
        print(len(example_edits))
    return example_edits[:n]

class RemoveRequirementFromPrompt(dspy.Signature):
    """Make minimal edits to the given prompt to remove the given requirement."""

    prompt: str = dspy.InputField(desc="The prompt to edit.")
    requirement: str = dspy.InputField(desc="The requirement to remove.")
    edited_prompt: str = dspy.OutputField(desc="The edited prompt.")

def remove_requirements(lm, prompt, requirements):
    remove = use_lm(lm)(dspy.Predict(RemoveRequirementFromPrompt))
    prompts = {}
    for idx, requirement in enumerate(requirements):
        prompts["removed_" + str(idx)] = remove(prompt=prompt, requirement=requirement).edited_prompt
    return prompts


def generate_fixes(prompt, requirements, select_indices):
    prompts = {}
    for selected_index in select_indices:
        requirements_selected = [requirements[i] for i in selected_index]
        prompts["fixed_" + "+".join([str(i) for i in selected_index])] = prompt + requirements_to_str(requirements_selected)
    return prompts

def generate_pbdesign(requirements):
    from pyDOE2 import pbdesign
    design = pbdesign(len(requirements))
    select_indices = [
        [i for i, _ in enumerate(requirements) if r[i] == 1]
        for r in design
    ]
    return select_indices

def generate_cyclic_designs(requirements, k):
    """Generate cyclic designs for the given requirements."""
    
    L = len(requirements)
    select_indices = [
        [j%L for j in range(i, i+k)]
        for i in range(L)
    ]
    return select_indices


if __name__ == "__main__":
    lm = dspy.LM('openai/gpt-4o-2024-08-06', temperature=1.0, cache=False)

    # task = "commitpack"
    # task = "trip"
    # task = "product"
    # task = "health"
    # task = "education"
    task = "webgen"
    task_description, TaskProgram, trainset, valset, requirements, prompts = prepare_data(
        task_name=task,
    )
    prompt = prompts["original"]


    # shuffle the requirements
    # random.seed(42)
    # random.shuffle(requirements)
    # print(json.dumps(requirements, indent=4))
    # with open(f"data/requirements/{task}", "w") as f:
    #     json.dump(requirements, f, indent=4)
    # exit(0)

    requirements = [requirement["requirement"] for requirement in requirements]
    
    for n in [1, 5, 10, 19]:
        select_indices = generate_cyclic_designs(requirements, n)
        # select_indices = generate_pbdesign(requirements)

        d = generate_fixes(task_description, requirements, select_indices)
        print(json.dumps(d, indent=4))
        with open(f"data/prompts/" + task + f"_n{n}.json", "w") as f:
            json.dump(d, f, indent=4)

    # d = remove_requirements(lm, prompt, requirements_base)
    # print(json.dumps(d, indent=4))

    # example_edits = generate_edits(lm, prompt, requirements_base)
    # d = ({"edited_" + str(idx): prompt for idx, prompt in enumerate(example_edits)})
    # print(json.dumps(d, indent=4))
    
    # paraphrases = generate_paraphrase(lm, prompt, requirements_base)
    # d = ({"paraphrased_" + str(idx): prompt for idx, prompt in enumerate(paraphrases)})
    # print(json.dumps(d, indent=4))