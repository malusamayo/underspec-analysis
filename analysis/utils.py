import dspy
import litellm
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import tqdm
import copy

def use_lm(lm, n=1):
    def decorator(program):
        def wrapper(*args, **kwargs):
            try:
                if type(program) == dspy.Module:
                    with dspy.context(lm=lm):
                        return program(*args, **kwargs)
                else:
                    lm.kwargs['n'] = n
                    return program(lm=lm, *args, **kwargs)
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
    if len(requirements) == 1:
        return f" {requirements[0]}"
    return "\n\nFollow the following requirements:\n" + "\n".join([f"- {req}" for req in requirements])