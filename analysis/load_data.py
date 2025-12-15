import pandas as pd
import dspy
import json
import os
import litellm
from functools import partial
from copy import deepcopy
from analysis.utils import run_model

class TaskProgram:
    def __init__(self, prompt, input_key, lm, n=1):
        self.lm = lm
        self.n = n
        self.prompt = prompt
        self.input_key = input_key

    def __call__(self, **kwargs):
        input = kwargs.get(self.input_key)
        response = litellm.completion(
            model=self.lm.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": input}
            ],
            n=self.n,
            **self.lm.kwargs
        )
        output = response.choices[0].message.content
        outputs = [response.choices[i].message.content for i in range(len(response.choices))]
        return dspy.Example(**{self.input_key: input}, output=output, outputs=outputs)
    
    def deepcopy(self):
        return deepcopy(self)

    def predictors(self):
        return [self]

def prepare_dataset(data_path, input_key, data_size=100):

    df = pd.read_csv(data_path)[:data_size]
    df_train = df.sample(frac=0.5, random_state=42)
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
    data_path = "data/esci_500.csv"
    input_key = "product_bullet_point"
    requirement_path = "data/requirements/requirements_product_v0.json"

    trainset, valset = prepare_dataset(data_path, input_key, data_size=200)

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


def prepare_data_arxiv_summarization_v0(configs):
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

def prepare_data_explain_code_v0():
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

def prepare_data_explain_code_v1():
    data_path = "data/leetcode-py-all.csv"
    input_key = "python_solution"

    trainset, valset = prepare_dataset(data_path, input_key, data_size=200)

    task_description = "Explain the code snippet."

    prompt = (
        "Your task is to take the code snippet provided and explain it in simple, easy-to-understand language. " 
        "Break down the code's functionality, purpose, and key components. Use analogies, examples, and plain terms to make the explanation accessible to someone with minimal coding knowledge. "
        "Avoid using technical jargon unless absolutely necessary, and provide clear explanations for any jargon used. The goal is to help the reader understand what the code does and how it works at a high level."
    )

    return task_description, partial(TaskProgram, prompt=prompt, input_key=input_key), trainset, valset

def prepare_data_explain_code():
    data_path = "data/commitpack_357.csv"
    input_key = "code"

    trainset, valset = prepare_dataset(data_path, input_key, data_size=200)

    task_description = "Explain the code snippet."

    prompt = (
        "Your task is to take the code snippet provided and explain it in simple, easy-to-understand language. " 
        "Break down the code's functionality, purpose, and key components. Use analogies, examples, and plain terms to make the explanation accessible to someone with minimal coding knowledge. "
        "Avoid using technical jargon unless absolutely necessary, and provide clear explanations for any jargon used. The goal is to help the reader understand what the code does and how it works at a high level."
    )

    return task_description, partial(TaskProgram, prompt=prompt, input_key=input_key), trainset, valset

def prepare_data_arxiv_summarization():
    data_path = "data/ai_arxiv.csv"
    input_key = "content"

    trainset, valset = prepare_dataset(data_path, input_key, data_size=200)

    task_description = "Summarize the arXiv paper."

    prompt = """You will be given the content of a newly published arXiv paper and asked to write a summary of it.

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

    return task_description, partial(TaskProgram, prompt=prompt, input_key=input_key), trainset, valset

def prepare_data_product_descrp_gen():
    data_path = "data/esci.csv"
    input_key = "product_text"

    trainset, valset = prepare_dataset(data_path, input_key, data_size=200)

    task_description = "Write engaging product description from the context."

    prompt = """You are a leading digital marketer working for a top retail organization. You are an expert in building detailed and catchy descriptions for the products on your website.

Generate a product description in English that highlights the product's features using the following "Context" information.
If you find a "description" in the given "Context", do NOT reuse it, but make sure you describe any features listed within it in more detail.
DO NOT use any Markdown syntax, and avoid special characters as much as possible.
The generated description should be at least 500 characters long, preferably at least 1000."""

    return task_description, partial(TaskProgram, prompt=prompt, input_key=input_key), trainset, valset

def prepare_data_trip_advisor():
    # data_path = "data/travel_planner.csv"
    data_path = "data/travel_conversations.csv"
    input_key = "query"

    trainset, valset = prepare_dataset(data_path, input_key, data_size=200)

    task_description = "Provide personalized travel recommendations, itineraries, and tips."

    prompt = """1. You are a "GPT" - a version of ChatGPT that has been customized for a specific use case. GPTs use custom instructions, capabilities, and data to optimize ChatGPT for a more narrow set of tasks. You yourself are a GPT created by a user, and your name is Trip Advisor. Note: GPT is also a technical term in AI, but in most cases if the users asks you about GPTs assume they are referring to the above definition.
2. Here are instructions from the user outlining your goals and how you should respond:
   - As a trip advisor, your role is to provide personalized travel recommendations, itineraries, and tips with a focus on user preferences. 
   - Specialize in destinations, accommodations, activities, dining, and cultural insights, considering budget, travel dates, and specific interests. 
   - Ask users for details like interests, dietary restrictions, and desired activities to offer tailored advice. 
   - Avoid booking or transaction handling. 
   - Your approach should be friendly, casual, and enthusiastic about travel, ensuring responses are personalized to user goals. 
   - Be clear and engaging, with a tone that's helpful, casual, and culturally sensitive. 
   - Clarify any ambiguous preferences. 
   - Show enthusiasm for exploring new cultures and experiences."""

    return task_description, partial(TaskProgram, prompt=prompt, input_key=input_key), trainset, valset

# https://github.com/linexjlin/GPTs/blob/3adfb7b38423b64a995057483c1f9007ed5f4da5/prompts/Medical%20Diagnosis%20Assistant.md
def prepare_data_health_consulting():
    data_path = "data/healthbench_consulting.csv"
    input_key = "query"

    trainset, valset = prepare_dataset(data_path, input_key, data_size=200)

    task_description = "Offer symptom guidance, medical info, first-aid tips, and health advice."

    prompt = """Features and Capabilities:

Symptom Checker: Users can describe their symptoms, and MedDiagnose ChatGPT will provide possible explanations and suggest potential conditions or illnesses associated with those symptoms.

General Medical Information: The AI offers easy-to-understand explanations of various medical terms, procedures, and concepts, making it a valuable educational tool.

First Aid Tips: MedDiagnose ChatGPT can provide basic first aid instructions for common injuries and illnesses, such as burns, cuts, or allergies, to help users take immediate action when needed.

Prevention and Lifestyle Advice: It offers guidance on maintaining a healthy lifestyle, including diet recommendations, exercise tips, and advice for preventing common health issues.

Medical Resources: The AI can direct users to trusted medical resources, reputable websites, and recommend healthcare professionals for further assistance.

Medication Information: Users can inquire about over-the-counter and prescription medications, including side effects, dosages, and potential interactions.

Emergency Response Guidance: In case of emergencies, MedDiagnose ChatGPT can provide step-by-step instructions for CPR, choking, and other life-saving procedures.

User Interaction:

Users can describe their symptoms or ask questions about medical conditions in plain language. MedDiagnose ChatGPT responds with clear and informative answers, providing explanations, suggestions, and recommendations. Ethical Considerations:

The AI should emphasize that it is not a substitute for professional medical advice, diagnosis, or treatment. It encourages users to consult healthcare professionals for accurate medical assessments. Privacy and data security should be a top priority to protect users' sensitive medical information. Audience:

Individuals seeking general medical information, preliminary symptom assessments, and advice on common health concerns. Benefits:

Provides accessible and reliable medical information. Offers guidance for users to make informed decisions about their health. Promotes health awareness and preventive measures. Disclaimer: MedDiagnose ChatGPT should always include a clear disclaimer stating that it is not a licensed medical practitioner and should not be used as a substitute for professional medical advice."""

    return task_description, partial(TaskProgram, prompt=prompt, input_key=input_key), trainset, valset

# https://platform.claude.com/docs/en/resources/prompt-library/lesson-planner
def prepare_data_plan_lesson():
    data_path = "data/lesson_plans.csv"
    input_key = "input"

    trainset, valset = prepare_dataset(data_path, input_key, data_size=200)

    task_description = "Generate a lesson plan for the given user query."

    prompt = (
        "Your task is to create a comprehensive, engaging, and well-structured lesson plan on the given subject. "
        "The lesson plan should be designed for a 60-minute class session and should cater to a specific grade level or age group. "
        "Begin by stating the lesson objectives, which should be clear, measurable, and aligned with relevant educational standards. "
        "Next, provide a detailed outline of the lesson, breaking it down into an introduction, main activities, and a conclusion. "
        "For each section, describe the teaching methods, learning activities, and resources you will use to effectively convey the content and engage the students. "
        "Finally, describe the assessment methods you will employ to evaluate students' understanding and mastery of the lesson objectives. "
        "The lesson plan should be well-organized, easy to follow, and promote active learning and critical thinking."
    )

    return task_description, partial(TaskProgram, prompt=prompt, input_key=input_key), trainset, valset

# 
def prepare_data_generate_website():
    data_path = "data/webgen.csv"
    input_key = "instruction"

    trainset, valset = prepare_dataset(data_path, input_key, data_size=200)

    task_description = "Generate a one-page website based on the given specifications."

    prompt = (
        "Your task is to create a one-page website based on the given specifications, delivered as an HTML file with embedded JavaScript and CSS. "
        "The website should incorporate a variety of engaging and interactive design features, such as drop-down menus, dynamic text and content, clickable buttons, and more. "
        "Ensure that the design is visually appealing, responsive, and user-friendly. "
        "The HTML, CSS, and JavaScript code should be well-structured, efficiently organized, and properly commented for readability and maintainability."
    )

    return task_description, partial(TaskProgram, prompt=prompt, input_key=input_key), trainset, valset


def prepare_data(
    task_name,
    configs=None,
):

    match task_name:
        case "quiz_making":
            task_description, TaskProgram, trainset, valset, requirement_path = prepare_data_quiz_making()
        case "recipe":
            task_description, TaskProgram, trainset, valset, requirement_path = prepare_data_recipe_gen()
        case "story":
            task_description, TaskProgram, trainset, valset, requirement_path = prepare_data_story()
        case "code":
            task_description, TaskProgram, trainset, valset, requirement_path = prepare_data_explain_code_v0()
        case "lc":
            task_description, TaskProgram, trainset, valset, requirement_path = prepare_data_explain_code_v1()
        case "commitpack":
            task_description, TaskProgram, trainset, valset = prepare_data_explain_code()
        case "arxiv":
            task_description, TaskProgram, trainset, valset = prepare_data_arxiv_summarization()
        case "product":
            task_description, TaskProgram, trainset, valset = prepare_data_product_descrp_gen()
        case "trip":
            task_description, TaskProgram, trainset, valset = prepare_data_trip_advisor()
        case "health":
            task_description, TaskProgram, trainset, valset = prepare_data_health_consulting()
        case "education":
            task_description, TaskProgram, trainset, valset = prepare_data_plan_lesson()
        case "webgen":
            task_description, TaskProgram, trainset, valset = prepare_data_generate_website()
        case _:
            task_description, TaskProgram, trainset, valset = "", None, [], []
    
    requirement_path = "data/requirements/" + task_name + ".json"
    prompt_path = "data/prompts/" + task_name + ".json"

    prompt_paths = []
    if configs:
        requirement_path = configs.get("requirement_path") or requirement_path
        prompt_path = configs.get("prompt_path") or prompt_path
        prompt_paths = configs.get("prompt_paths") or []

    def safe_load_json(path):
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            return json.load(f)

    if requirement_path:
        requirements = safe_load_json(requirement_path)
    else:
        requirements = {}

    if prompt_paths:
        prompts = {}
        for prompt_path in prompt_paths:
            prompts.update(safe_load_json(prompt_path))
    else:
        prompts = safe_load_json(prompt_path)
    
    if not prompts:
        prompts = {"original": TaskProgram(lm=None).prompt}

    return task_description, TaskProgram, trainset, valset, requirements, prompts


def load_data(task, model_name, prompt_name, task_program, datasets):
    loaded_datasets = {}
    for dataset_name, dataset in datasets.items():
        if not os.path.exists(f"data/results/{task}"):
            os.makedirs(f"data/results/{task}")
        if not os.path.exists(f"data/results/{task}/{model_name}_{prompt_name}_{dataset_name}.json"):
            print(f"Running model on {dataset_name} split...")
            loaded_datasets[dataset_name] = run_model(task_program, dataset, max_workers=32)
            with open(f"data/results/{task}/{model_name}_{prompt_name}_{dataset_name}.json", "w") as f:
                json.dump([example.toDict() for example in loaded_datasets[dataset_name]], f)
        else:
            print(f"Loading {dataset_name} data from cache...")
            with open(f"data/results/{task}/{model_name}_{prompt_name}_{dataset_name}.json", "r") as f:
                loaded_datasets[dataset_name] = [dspy.Example(**row).with_inputs(task_program.input_key) for row in json.load(f)]
    return loaded_datasets