import pandas as pd
import dspy
import json
import litellm

def prepare_dataset(data_path, input_key, data_size=100):

    df = pd.read_csv(data_path)[:data_size]
    df_train = df.sample(frac=0.2, random_state=42)
    df_val = df.drop(df_train.index)
    trainset = []
    for i, row in df_train.iterrows():
        example = dspy.Example(**{input_key: row[input_key]}, output="").with_inputs(input_key)
        trainset.append(example)

    valset = []
    for i, row in df_val.iterrows():
        example = dspy.Example(**{input_key: row[input_key]}, output="").with_inputs(input_key)
        valset.append(example)

    return trainset, valset


def prepare_data_quiz_making():
    data_path = "data/mmlu_econ_100.csv"
    input_key = "question"
    requirement_path = "data/requirements/requirements_quiz_v6.json"

    trainset, valset = prepare_dataset(data_path, input_key)

    task_description = "Write quiz answer options for the following question."

    class TaskProgram(dspy.Signature):
        """Write four quiz answer options for the following question. Label the quiz answer options with (A) (B) (C) (D)."""

        question = dspy.InputField(desc="The question for which options are needed.")
        output = dspy.OutputField(desc="List of four possible answers for a quiz question.")

    class TaskProgramV2(dspy.Signature):
        """Write quiz answer options for the following question."""

        question = dspy.InputField(desc="The question for which options are needed.")
        output = dspy.OutputField(desc="List of possible answers for a quiz question.")
    
    return task_description, TaskProgram, trainset, valset, requirement_path

def prepare_data_product_descrp_gen():
    data_path = "data/esci_100.csv"
    input_key = "product_bullet_point"
    requirement_path = "data/requirements/requirements_product_v0.json"

    trainset, valset = prepare_dataset(data_path, input_key)

    task_description = "Generate engaging product description from the bullet points."

    class TaskProgram(dspy.Signature):
        """Generate engaging product description from the bullet points."""

        product_bullet_point = dspy.InputField(desc="The bullet points of the product.")
        output = dspy.OutputField(desc="The generated product description.")
    
    return task_description, TaskProgram, trainset, valset, requirement_path

def prepare_data_recipe_gen():
    data_path = "data/recipe.csv"
    input_key = "ingredients"
    requirement_path = "data/requirements/requirements_recipe_v0.json"

    trainset, valset = prepare_dataset(data_path, input_key)

    task_description = "Generate a recipe given the list of ingredients."

    class TaskProgram(dspy.Signature):
        """Generate a recipe given the list of ingredients."""

        ingredients = dspy.InputField(desc="A list of ingredients.")
        output = dspy.OutputField(desc="Generated recipe.")
    
    return task_description, TaskProgram, trainset, valset, requirement_path

def prepare_data_story():
    data_path = "data/stories.csv"
    input_key = "application"
    requirement_path = "data/requirements/requirements_story_v0.json"

    trainset, valset = prepare_dataset(data_path, input_key)

    task_description = "For the user-given application, generate a 200-word story that causes potential harms to relevant stakeholders."

    class TaskProgram(dspy.Signature):
        """For the user-given application, generate a 200-word story that causes potential harms to relevant stakeholders."""

        application = dspy.InputField(desc="A description of user-given application.")
        output = dspy.OutputField(desc="Generated story.")
    
    return task_description, TaskProgram, trainset, valset, requirement_path


def prepare_data_arxiv_summarization(configs):
    data_path = "data/arxiv.csv"
    input_key = "article"
    requirement_path = "data/requirements/requirements_arxiv_v0.json"

    trainset, valset = prepare_dataset(data_path, input_key)

    task_description = "Generate a summary of the arXiv paper."

    class TaskProgram(dspy.Signature):
        """Generate a summary of the arXiv paper."""

        article = dspy.InputField(desc="The content of the arXiv paper.")
        output = dspy.OutputField(desc="Generated summary.")

    class TaskProgramV2(dspy.Signature):
        """You will be given the content of a newly published arXiv paper and asked to write a summary of it.

Here are some things to keep in mind:
- Summarize the paper in a way that is understandable to the general public
- Use a professional tone
- Don't use the word "quest" or similar flowery language
- Don't say this is a recent paper, since this summary may be referenced in the future
- Limit your summary to about 4 paragraphs
- Do not prefix the article with a title
- Do not mention the author's names
- You can use the following markdown tags in your summary: ordered list, unordered list, and h3 headings
- Divide the summary into sections using markdown h3 headings
- Do not include a title for the summary; only include headings to divide the summary into sections
- The first line should be an h3 heading as well.
- Assume readers know what common AI acronyms stand for like LLM and AI
- Don't mention any part of this prompt"""

        article = dspy.InputField(desc="The content of the arXiv paper.")
        output = dspy.OutputField(desc="Generated summary.")

    final_task_program = TaskProgramV2 if configs and configs.get("version") == "v2" else TaskProgram

    return task_description, final_task_program, trainset, valset, requirement_path

def prepare_data_explain_code_dep():
    data_path = "data/humaneval.csv"
    input_key = "code"
    requirement_path = "data/requirements/requirements_code_v0.json"

    trainset, valset = prepare_dataset(data_path, input_key)

    task_description = "Explain the code snippet."

    class TaskProgram(dspy.Signature):
        """Explain the code snippet."""

        code = dspy.InputField(desc="The code snippet.")
        output = dspy.OutputField(desc="Explanation of the code snippet.")

    return task_description, TaskProgram, trainset, valset, requirement_path

def prepare_data_explain_code():
    data_path = "data/leetcode-py-all.csv"
    input_key = "python_solution"
    requirement_path = ""

    trainset, valset = prepare_dataset(data_path, input_key, data_size=200)

    task_description = "Explain the code snippet."

    prompt = (
        "Your task is to take the code snippet provided and explain it in simple, easy-to-understand language. " 
        "Break down the code's functionality, purpose, and key components. Use analogies, examples, and plain terms to make the explanation accessible to someone with minimal coding knowledge. "
        "Avoid using technical jargon unless absolutely necessary, and provide clear explanations for any jargon used. The goal is to help the reader understand what the code does and how it works at a high level."
    )

    def generate_explanation(lm, python_solution):
        response = litellm.completion(
            model=lm.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": python_solution}
            ],
            **lm.kwargs
        )
        output = response.choices[0].message.content
        return dspy.Example(**{input_key: python_solution}, output=output)

    return task_description, generate_explanation, trainset, valset, requirement_path

def prepare_data(
    task_name,
    configs=None,
):

    match task_name:
        case "quiz_making":
            task_description, TaskProgram, trainset, valset, requirement_path = prepare_data_quiz_making()
        case "product_descrp_gen":
            task_description, TaskProgram, trainset, valset, requirement_path = prepare_data_product_descrp_gen()
        case "recipe":
            task_description, TaskProgram, trainset, valset, requirement_path = prepare_data_recipe_gen()
        case "story":
            task_description, TaskProgram, trainset, valset, requirement_path = prepare_data_story()
        case "arxiv":
            task_description, TaskProgram, trainset, valset, requirement_path = prepare_data_arxiv_summarization(configs)
        case "code":
            task_description, TaskProgram, trainset, valset, requirement_path = prepare_data_explain_code_dep()
        case "lc":
            task_description, TaskProgram, trainset, valset, requirement_path = prepare_data_explain_code()
        case _:
            task_description, TaskProgram, trainset, valset, requirement_path = "", None, [], [], ""
    
    if configs and configs.get("requirement"):
        requirement_path = configs.get("requirement")

    if requirement_path:
        with open(requirement_path, "r") as f:
            requirements = json.load(f)
    else:
        requirements = {}

    return task_description, TaskProgram, trainset, valset, requirements