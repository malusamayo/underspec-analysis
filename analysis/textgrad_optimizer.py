import dspy
from dspy.teleprompt import Teleprompter

from tqdm import tqdm
from copy import deepcopy
import random
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Tuple, Optional, Any


class EvalResult(BaseModel):
    """Stores evaluation results for a single example when using a particular prompt."""
    example: dict
    output: str
    feedback: str
    score: float

class PromptComparator(dspy.Signature):
    """After evaluating the model with the current prompt, you will have a set of positive and negative examples. Your task is to analyze these examples and provide feedback on how to improve the prompt."""

    current_prompt: str = dspy.InputField(
        prefix="Current Prompt: ",
        desc="The current version of the prompt used to generate outputs",
    )
    pos_input_with_metrics: List[EvalResult] = dspy.InputField(
        prefix="Positive Inputs: ",
        desc="Positive examples (with high scores), their outputs, and feedback",
    )
    neg_input_with_metrics: List[EvalResult] = dspy.InputField(
        prefix="Negative Inputs: ",
        desc="Negative examples (with low scores), their outputs, and feedback",
    )
    feedback: str = dspy.OutputField(
        prefix="Prompt Feedback: ",
        desc="Suggestions or modifications needed for the prompt to handle negative examples better",
    )

class PromptFeedbackBasedInstruction(dspy.Signature):
    """There is an existing prompt used to guide a user-specified task. We have evaluated the performance of this prompt on a set of examples, identifying both positive examples where the prompt worked well, and negative examples where improvements are needed.

Based on the feedback for handling negative examples and retaining the strengths observed in positive ones, we want you to revise the prompt. The new prompt should integrate the suggestions from the feedback while preserving the core guidelines from the previous prompt. The final prompt must be no longer than three paragraphs."""

    previous_prompt: str = dspy.InputField(
        prefix="Previous Prompt: ",
        desc="The older version of the prompt used previously",
    )
    feedback: str = dspy.InputField(
        prefix="Feedback: ",
        desc="Feedback for improving the prompt",
    )
    new_prompt: str = dspy.OutputField(
        prefix="New Prompt: ",
        desc="Refined prompt incorporating the feedback",
    )



DEFAULT_MAX_EXAMPLES = 10

class TextGradOptimizer(Teleprompter):
    def __init__(
        self,
        metric: Callable,
        prompt_model: Optional[Any] = None,
        task_model: Optional[Any] = None,
        max_iters: int = 10,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        max_positive_inputs: int = None,
        max_negative_inputs: int = None,
        optimize_for: str = "max",  # or "min"
    ):
        """
        Args:
            metric (Callable): A function that takes (example, model_output) and returns a numeric score.
            max_iters (int): Max number of optimization iterations.
            lower_bound (float): Score <= lower_bound is considered negative.
            upper_bound (float): Score >= upper_bound is considered positive.
            max_positive_inputs (int): Max number of positive examples to include in prompt feedback each round.
            max_negative_inputs (int): Max number of negative examples to include in prompt feedback each round.
            optimize_for (str): "max" or "min" depending on whether higher or lower metric is better.
        """
        assert metric is not None, "`metric` argument cannot be None. Please provide a metric function."
        self.metric = metric
        self.optimize_for = optimize_for

        self.task_model = task_model if task_model else dspy.settings.lm
        self.prompt_model = prompt_model if prompt_model else dspy.settings.lm

        self.max_iters = max_iters
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.max_positive_inputs = max_positive_inputs or DEFAULT_MAX_EXAMPLES
        self.max_negative_inputs = max_negative_inputs or DEFAULT_MAX_EXAMPLES

        # We will use these two typed predictors to generate feedback & new prompts
        self.comparator = dspy.Predict(PromptComparator)
        self.feedback_instruction = dspy.Predict(PromptFeedbackBasedInstruction)

    def process_example(self, model: dspy.Module, example, return_outputs=False):
        model = deepcopy(model)

        try:

            with dspy.context(lm=self.task_model):
                output = model(example.inputs().toDict())

            # Evaluate
            feedback, score = self.metric(example, output)

            if return_outputs:
                return example, output, feedback, score
            else:
                return score

        except Exception as e:
            print(e)
            if return_outputs:
                return example, None, None, 0
            else:
                return 0

    def thread_safe_evaluator(self, devset, model, return_outputs=False, num_threads=16):
        """
        Evaluate all examples in `devset` using a ThreadPool for speed.
        """
        total_score = 0
        total_examples = len(devset)
        results = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.process_example, model, example, return_outputs)
                for example in devset
            ]

            for future in tqdm(futures, total=total_examples, desc="Processing examples"):
                result = future.result()
                if return_outputs:
                    example, output, feedback, score = result
                    total_score += score
                    results.append((example, output, feedback, score))
                else:
                    total_score += result

        avg_metric = total_score / total_examples if total_examples > 0 else 0
        return (avg_metric, results) if return_outputs else avg_metric

    def _get_pos_neg_results(self, model, trainset):
        """
        Returns:
          - avg_score
          - list of EvalResult for positives
          - list of EvalResult for negatives
        """
        avg_score, results = self.thread_safe_evaluator(trainset, model, return_outputs=True)
        print(f"Average Score: {avg_score}")

        pos_inputs = []
        neg_inputs = []

        for example, output, feedback, score in results:
            if score >= self.upper_bound:
                pos_inputs.append(
                    EvalResult(
                        example=example.inputs().toDict(),
                        output=output,
                        feedback=feedback,
                        score=score,
                    )
                )
            elif score <= self.lower_bound:
                neg_inputs.append(
                    EvalResult(
                        example=example.inputs().toDict(),
                        output=output,
                        feedback=feedback,
                        score=score,
                    )
                )

        # if len(pos_inputs) == 0:
        #     raise ValueError(
        #         "No positive examples found. Adjust your `upper_bound` or ensure you have enough data."
        #     )
        # if len(neg_inputs) == 0:
        #     raise ValueError(
        #         "No negative examples found. Adjust your `lower_bound` or ensure you have enough data."
        #     )

        return avg_score, pos_inputs, neg_inputs

    def compile(self, model: dspy.Module, trainset: List[dspy.Example]):
        """
        The main loop:
        1. Evaluate using the current best prompt (one evaluation per iteration)
        2. Collect positives and negatives
        3. Generate feedback
        4. Propose a refined prompt
        5. Accept and store the refined prompt (no second evaluation in this iteration)
            - The next iteration will evaluate the newly accepted prompt
        """
        # Keep a copy of the original model so we can modify the prompt
        current_model = deepcopy(model)
        random.seed(42)

        # Initialize best_score according to whether we are maximizing or minimizing
        best_score = float("-inf") if self.optimize_for == "max" else float("inf")

        for i in range(self.max_iters):
            print("=" * 20)
            print(f"Iteration {i+1}/{self.max_iters}")
            print(f"Current Prompt:\n{current_model.model.signature.instructions}")

            # 1) Evaluate using the current best prompt
            score, pos_inputs, neg_inputs = self._get_pos_neg_results(current_model, trainset)
            print(f"Average Score (current prompt): {score}")
            print(f"Positive examples: {len(pos_inputs)}")
            print(f"Negative examples: {len(neg_inputs)}")

            # Subsample if we have too many pos/neg examples
            if len(pos_inputs) > self.max_positive_inputs:
                pos_inputs = random.sample(pos_inputs, self.max_positive_inputs)
            if len(neg_inputs) > self.max_negative_inputs:
                neg_inputs = random.sample(neg_inputs, self.max_negative_inputs)

            with dspy.context(lm=self.prompt_model):

                # 2) Generate feedback to improve the prompt
                feedback = self.comparator(
                    current_prompt=current_model.model.signature.instructions,
                    pos_input_with_metrics=pos_inputs,
                    neg_input_with_metrics=neg_inputs
                ).feedback

                # 3) Propose a refined prompt using the feedback
                new_prompt = self.feedback_instruction(
                    previous_prompt=current_model.model.signature.instructions,
                    feedback=feedback
                ).new_prompt

            print(f"Feedback:\n{feedback}\n")
            # print(f"Proposed new prompt:\n{new_prompt}\n")

            # 4) Update best_score and accept the new prompt
            #    We do NOT evaluate 'new_prompt' in this iteration.
            #    The newly accepted prompt will be evaluated at the start of the next iteration.
            if (self.optimize_for == "max" and score > best_score) or (self.optimize_for == "min" and score < best_score):
                best_score = score
                current_model.best_model = deepcopy(current_model.model) # Store the best model
            
            # Update the prompt
            current_model.model.signature = current_model.model.signature.with_instructions(new_prompt) 

        print(f"\nOptimization complete.")
        print(f"Last Prompt Score = {best_score}")
        print(f"Final Prompt:\n{current_model.best_model.signature.instructions}")
        return current_model