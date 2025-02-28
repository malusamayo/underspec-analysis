import dspy
from .utils import use_lm, batch_inference
from .judge import EvaluateRequirement

class RefineResponseWithFeedback(dspy.Signature):
    """Given the task description, model input, model output, and feedback, refine the model output based on the feedback."""

    task_description = dspy.InputField(desc="Description of the task")
    model_input = dspy.InputField(desc="The model input")
    model_output = dspy.InputField(desc="The model output")
    feedback = dspy.InputField(desc="Feedback on the model output")
    refined_output: str = dspy.OutputField(desc="Refined model output based on the feedback")


class RefinePromptWithFeedback(dspy.Signature):
    """You are a prompt re-writer for large language models. I will give you a task description, and a prompt that is used to generate the model output. 
Your task is to refine the prompt based on the feedback provided. The refined prompt should lead a good language model to perform the task well and meet the requirement. Don't be afraid to be creative."""

    task_description = dspy.InputField(desc="Description of the task")
    previous_prompt = dspy.InputField(desc="The previous prompt")
    model_input = dspy.InputField(desc="The model input")
    model_output = dspy.InputField(desc="The model output")
    requirement = dspy.InputField(desc="The requirement")
    feedback = dspy.InputField(desc="Feedback on the model output")
    prompt = dspy.OutputField(desc="The proposed prompt")

class IterativeRefine(dspy.Module):

    def __init__(self, task_description, input_variable, lm, task_program):
        self.lm = lm
        self.judge_lm = dspy.LM('openai/o3-mini', temperature=1.0, max_tokens=10000)
        self.judge_lm.kwargs['max_completion_tokens'] = self.judge_lm.kwargs.pop('max_tokens')
        self.task_description = task_description
        self.pred = use_lm(self.lm)(dspy.Predict(task_program))
        self.evaluator = use_lm(self.lm)(dspy.Predict(EvaluateRequirement))
        self.refine = use_lm(self.lm)(dspy.Predict(RefineResponseWithFeedback))
        self.prompt_refine = use_lm(self.lm)(dspy.Predict(RefinePromptWithFeedback))

    def generate_and_refine(self, example, requirement):
        # generate the response
        response = self.pred(**example.inputs().toDict()).output
        # get the feedback
        result = self.evaluator(task_description=self.task_description, model_input=example.inputs().toDict(), model_output=response, requirement=requirement)
        i = 0
        while not result.meets_requirement and i < 3:
            feedback = result.plan_execution
            # refine the response
            response = self.refine(task_description=self.task_description, model_input=example.inputs().toDict(), model_output=response, feedback=feedback).refined_output
            result = self.evaluator(task_description=self.task_description, model_input=example.inputs().toDict(), model_output=response, requirement=requirement)
            i += 1
        return response

    def forward(self, examples, requirement):
        results = batch_inference(
            self.generate_and_refine,
            [{"example": example, "requirement": requirement} for example in examples]
        )
        return results
