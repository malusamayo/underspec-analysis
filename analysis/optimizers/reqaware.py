import logging
import random
from typing import List, Dict, Any, Optional
import optuna
from optuna.distributions import CategoricalDistribution
import numpy as np
from dspy.teleprompt.teleprompt import Teleprompter
from ..utils import requirements_to_str

logger = logging.getLogger(__name__)

class ReqAwareOptimizer(Teleprompter):
    def __init__(
        self,
        task_description: str,
        requirements: List[str],
        evaluate: Any,
        num_trials: int = 30,
        seed: Optional[int] = None,
    ):
        """
        Initialize the ReqAwareOptimizer.
        
        Args:
            task_description: The description of the task to optimize for
            requirements: List of requirements to try different combinations of
            evaluate: Evaluation function that takes a program and dataset and returns a score
            num_trials: Number of trials to run for Bayesian optimization
            seed: Random seed for reproducibility
        """
        self.task_description = task_description
        self.requirements = requirements
        self.evaluate = evaluate
        self.num_trials = num_trials
        self.seed = seed or random.randint(0, 1000)
        
        # Set random seeds
        self.rng = random.Random(self.seed)
        np.random.seed(self.seed)

    def compile(self, program: Any, trainset: List) -> Any:
        """
        Optimize the program by trying different combinations of requirements using Bayesian optimization.
        
        Args:
            program: The program to optimize
            trainset: Training dataset for evaluation
            
        Returns:
            The optimized program
        """
        print("Starting requirement-aware optimization...")
        print(f"Task description: {self.task_description}")
        print(f"Number of requirements: {len(self.requirements)}")
        
        # Create a study for Bayesian optimization
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        
        # Keep track of best program and score
        best_score = float("-inf")
        best_program = None
        
        def objective(trial):
            nonlocal best_score, best_program
            
            # Create a new candidate program
            candidate_program = program.deepcopy()
            
            # For each requirement, decide whether to include it or not
            selected_requirements = []
            for i, req in enumerate(self.requirements):
                if trial.suggest_categorical(f"req_{i}", [0, 1]):
                    selected_requirements.append(req)
            
            print(f"Selected requirements: {selected_requirements}")

            # Update program with selected requirements
            candidate_program.prompt = self.task_description + requirements_to_str(selected_requirements)
            
            # Evaluate the candidate program
            score, _ = self.evaluate(candidate_program, trainset)
            
            # Update best score and program if better
            if score > best_score:
                best_score = score
                best_program = candidate_program.deepcopy()
                print(f"New best score: {score}")
            
            return score
        
        # Run optimization
        study.optimize(objective, n_trials=self.num_trials)
        
        print("Optimization complete!")
        print(f"Best score achieved: {best_score}")
        print(f"Final prompt: {best_program.prompt}")
        
        return best_program
