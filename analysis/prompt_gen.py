import dspy
from typing import List, Dict, Any
from .utils import use_lm


class ParaphrasePrompt(dspy.Signature):
    """Generate a paraphrase of a given prompt. Make sure no existing requirements are removed and no new requirements are added."""

    prompt: str = dspy.InputField(desc="The prompt to paraphrase.")
    requirements: List[str] = dspy.InputField(desc="The requirements to keep in the paraphrase.")
    paraphrased_prompt: str = dspy.OutputField(desc="The paraphrased prompt.")


if __name__ == "__main__":
    lm = dspy.LM('openai/gpt-4o-2024-08-06', temperature=1.0)
    paraphrase = use_lm(lm)(dspy.Predict(ParaphrasePrompt, n=10))

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
    completions = paraphrase(prompt=prompt, requirements=requirements_base).completions
    print({"paraphrased_" + str(idx): prompt for idx, prompt in enumerate(completions.paraphrased_prompt)})