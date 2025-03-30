import dspy
from typing import List, Dict, Any
import difflib
import json
from .utils import use_lm


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
    check_equivalence = use_lm(lm)(dspy.Predict(CheckEquivalence))
    paraphrased_prompts = []
    while len(paraphrased_prompts) < n:
        new_paraphrases = paraphrase(prompt=prompt, requirements=requirements).completions.paraphrased_prompt
        new_paraphrases = [p for p in new_paraphrases if check_equivalence(prompt1=prompt, prompt2=p, requirements=requirements).is_equivalent]
        paraphrased_prompts.extend(new_paraphrases)
        # Remove duplicates while preserving order
        paraphrased_prompts = list(dict.fromkeys(paraphrased_prompts))
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

if __name__ == "__main__":
    lm = dspy.LM('openai/gpt-4o-2024-08-06', temperature=1.0, cache=False)

    prompt = (
        "Your task is to take the code snippet provided and explain it in simple, easy-to-understand language. " 
        "Break down the code's functionality, purpose, and key components. Use analogies, examples, and plain terms to make the explanation accessible to someone with minimal coding knowledge. "
        "Avoid using technical jargon unless absolutely necessary, and provide clear explanations for any jargon used. The goal is to help the reader understand what the code does and how it works at a high level."
    )
    requirements_base = [
        "The output should use simple, easy-to-understand language.",
        "The output should break down the code's functionality, purpose, and key components.",
        "The output should use analogies, examples, and plain terms.",
        "The output should be accessible to someone with minimal coding knowledge.",
        "The output should avoid using technical jargon unless absolutely necessary.",
        "The output should provide clear explanations for any jargon used.",
    ]
    example_edits = generate_edits(lm, prompt, requirements_base)
    d = ({"edited_" + str(idx): prompt for idx, prompt in enumerate(example_edits)})
    print(json.dumps(d, indent=4))
    
    paraphrases = generate_paraphrase(lm, prompt, requirements_base)
    d = ({"paraphrased_" + str(idx): prompt for idx, prompt in enumerate(paraphrases)})
    print(json.dumps(d, indent=4))