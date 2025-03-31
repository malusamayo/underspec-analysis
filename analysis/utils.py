import dspy
import litellm
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.cluster import KMeans
import os
import numpy as np
import pandas as pd
import tqdm
import copy
import time

LM_DICT = {
    "gpt-4o": dspy.LM('openai/gpt-4o-2024-08-06', temperature=1.0),
    "gpt-4o-2024-11-20": dspy.LM('openai/gpt-4o-2024-11-20', temperature=1.0),
    "gpt-4o-2024-05-13": dspy.LM('openai/gpt-4o-2024-05-13', temperature=1.0),
    "gpt-4o-mini": dspy.LM('openai/gpt-4o-mini-2024-07-18', temperature=1.0),
    "o3-mini": dspy.LM('openai/o3-mini', temperature=1.0, max_tokens=10000),
    "gemini-1.5-pro": dspy.LM('openai/gemini-1.5-pro-002', temperature=1.0, api_base=os.environ.get("CMU_API_BASE"), api_key=os.environ.get("LITELLM_API_KEY")),
    "gemini-1.5-flash": dspy.LM('openai/gemini-1.5-flash-002', temperature=1.0, api_base=os.environ.get("CMU_API_BASE"), api_key=os.environ.get("LITELLM_API_KEY")),
    "claude-3.5-sonnet": dspy.LM('openai/claude-3-5-sonnet-20241022', temperature=1.0, api_base=os.environ.get("CMU_API_BASE"), api_key=os.environ.get("LITELLM_API_KEY")),
    "llama3-2-11b-instruct": dspy.LM('openai/llama3-2-11b-instruct', temperature=1.0, api_base=os.environ.get("CMU_API_BASE"), api_key=os.environ.get("LITELLM_API_KEY")),
    "mixtral-8x7b": dspy.LM('bedrock/mistral.mixtral-8x7b-instruct-v0:1', temperature=1.0),
    "qwen2.5-7b": dspy.LM('hosted_vllm/Qwen/Qwen2.5-7B-Instruct', temperature=0.7, api_base=os.environ.get("BABEL_API_BASE")),
    "ministral-8b": dspy.LM('hosted_vllm/mistralai/Ministral-8B-Instruct-2410', temperature=0, api_base=os.environ.get("BABEL_API_BASE")),
    "llama3.1-8b": dspy.LM('hosted_vllm/meta-llama/Llama-3.1-8B-Instruct', temperature=0.6, api_base=os.environ.get("BABEL_API_BASE")),
}


def use_lm(lm, n=1):
    def decorator(program):
        def wrapper(*args, **kwargs):
            max_retries = 3
            initial_delay = 1
            delay = initial_delay
            
            for attempt in range(max_retries):
                try:
                    with dspy.context(lm=lm):
                        return program(*args, **kwargs)
                except litellm.APIError as e:
                    if "502 Bad Gateway" in str(e) and attempt < max_retries - 1:
                        print(f"API Error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        raise
                except Exception as e:
                    print(f"Error: {e}")
                    return dspy.Example(output="")
        return wrapper
    return decorator

def batch_inference(program, args_list, max_workers=32) -> List[Any]:
    futures = {}
    results = [None] * len(args_list)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, args in enumerate(args_list):
            future = executor.submit(
                program,
                **args
            )
            futures[future] = i

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            index = futures[future]
            results[index] = result
    return results

def run_model(program, examples, max_workers=32):
    examples = copy.deepcopy(examples)
    results = batch_inference(
        program,
        [example.inputs().toDict() for example in examples],
        max_workers=max_workers,
    )
    for example, result in zip(examples, results):
        example.output = result.output
        example.outputs = result.outputs
    return examples

def find_nearest_requirement(requirement, requirements):
    distances = []
    requirement_embedding = litellm.embedding(model='openai/text-embedding-ada-002', input=[requirement]).data[0]['embedding']
    requirements_embedding = batch_inference(
        lambda requirement: litellm.embedding(model='openai/text-embedding-ada-002', input=[requirement]).data[0]['embedding'],
        [{"requirement": req} for req in requirements]
    )
    # calculate the cosine similarity
    for req_embedding in requirements_embedding:
        distances.append(np.dot(requirement_embedding, req_embedding) / (np.linalg.norm(requirement_embedding) * np.linalg.norm(req_embedding)))
    return requirements[np.argmax(distances)]

def cluster_requirements(requirements, num_clusters=40):
    
    requirement_df = pd.DataFrame({"requirements": requirements})
    requirement_df['ada_embedding'] =  batch_inference(
        lambda requirement: litellm.embedding(model='openai/text-embedding-ada-002', input=[requirement]).data[0]['embedding'],
        [{"requirement": req} for req in requirements]
    )

    # cluster the requirements
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(np.vstack(requirement_df['ada_embedding'].to_list()))
    requirement_df['cluster'] = labels

    # remove clusters with only 1 requirements
    cluster_counts = requirement_df['cluster'].value_counts()
    clusters_to_remove = cluster_counts[cluster_counts < 2].index
    requirement_df = requirement_df[~requirement_df['cluster'].isin(clusters_to_remove)]

    # print the clusters
    for i in range(num_clusters):
        subset = requirement_df[requirement_df['cluster'] == i]['requirements']
        if len(subset) == 0:
            continue
        print(f"Cluster {i}:")
        for req in subset:
            print(f"  - {req}")
        print()

def requirements_to_str(requirements):
    if len(requirements) == 0:
        return ""
    # if len(requirements) == 1:
    #     return f" {requirements[0]}"
    return "\n\nFollow the following requirements:\n" + "\n".join([f"- {req}" for req in requirements])