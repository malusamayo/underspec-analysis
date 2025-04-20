import dspy
from dspy.teleprompt import Teleprompter

from tqdm import tqdm
from copy import deepcopy
import random
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Tuple, Optional, Any
import json

class EvalResult(BaseModel):
    """Stores evaluation results for a single example when using a particular prompt."""
    example: dict
    output: str
    feedback: str
    score: float

class ProposeNewPromptWithEvalResults(dspy.Signature):
    """There is an existing prompt used to guide a user-specified task. We have evaluated the performance of this prompt on a set of examples, and gathered feedback from multiple examples. 
We want you to first consolidate the feedback into a single coherent statement. Based on this feedback, we want you to revise the prompt. 
The new prompt should integrate the suggestions from the feedback while preserving the core guidelines from the previous prompt."""

    current_prompt: str = dspy.InputField(
        prefix="Current Prompt: ",
        desc="The current version of the prompt used to generate outputs",
    )
    eval_results: List[EvalResult] = dspy.InputField(
        prefix="Feedbacks: ",
        desc="List of feedback statements from different examples",
    )
    consolidated_feedback: str = dspy.OutputField(
        prefix="Consolidated Feedback: ",
        desc="A single coherent feedback statement that summarizes the suggestions",
    )
    new_prompt: str = dspy.OutputField(
        prefix="New Prompt: ",
        desc="Refined prompt incorporating the feedback",
    )

DEFAULT_MAX_EXAMPLES = 10

class TextGradOptimizer(Teleprompter):
    def __init__(
        self,
        evaluate: Callable,
        prompt_model: Optional[Any],
        num_epochs: int = 3,
        batch_size: int = 10,
        optimize_for: str = "max",  # or "min"
    ):
        self.evaluate = evaluate
        self.optimize_for = optimize_for

        self.prompt_model = prompt_model

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # We will use these two typed predictors to generate feedback & new prompts
        self.feedback_instruction = dspy.Predict(ProposeNewPromptWithEvalResults)

    def compile(self, model, trainset: List[dspy.Example]):
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
        current_model.prompt_history = {}
        iteration_count = 0

        for epoch in range(self.num_epochs):
            print("=" * 20)
            print(f"Epoch {epoch+1}/{self.num_epochs}")

            # Shuffle the dataset for this epoch
            shuffled_indices = list(range(len(trainset)))
            random.shuffle(shuffled_indices)
            
            # Process the dataset in batches
            num_batches = 0
            
            for batch_start in range(0, len(trainset), self.batch_size):
                print(f"Current Prompt:\n{current_model.prompt}")
                batch_end = min(batch_start + self.batch_size, len(trainset))
                batch_indices = shuffled_indices[batch_start:batch_end]
                batch = [trainset[idx] for idx in batch_indices]
                
                # Evaluate on this batch
                batch_score, batch_results = self.evaluate(model, batch, gather_feedback=True)
                num_batches += 1
                
                # Collect evaluation results for this batch
                batch_eval_results = []
                for example, output, feedback, score in batch_results:
                    batch_eval_results.append(
                        EvalResult(
                            example=example.inputs().toDict(),
                            output=output,
                            feedback=feedback,
                            score=score,
                        )
                    )
                
                # Generate feedback and update prompt after each batch
                with dspy.context(lm=self.prompt_model):
                    result = self.feedback_instruction(
                        current_prompt=current_model.prompt,
                        eval_results=batch_eval_results
                    )
                    feedback, new_prompt = result.consolidated_feedback, result.new_prompt
                
                print(f"Batch Feedback:\n{feedback}\n")
                
                # Update the prompt for the next batch
                current_model.prompt = new_prompt

                # Compute full eval score
                full_eval_score, _ = self.evaluate(current_model, trainset)
                print(f"Batch {num_batches}: Score = {full_eval_score}")

                # Update best_score and accept the new prompt
                if (self.optimize_for == "max" and full_eval_score > best_score) or (self.optimize_for == "min" and full_eval_score < best_score):
                    best_score = full_eval_score
                    current_model.best_prompt = deepcopy(current_model.prompt) # Store the best model
                
                # Record this iteration
                iteration_count += 1
                current_model.prompt_history[iteration_count] = {
                    "prompt": current_model.prompt,
                    "score": full_eval_score,
                    "epoch": epoch,
                    "batch": num_batches
                }
            

        print(f"\nOptimization complete.")
        print(f"Best Prompt Score = {best_score}")
        print(f"Final Prompt:\n{current_model.best_prompt}")
        return current_model