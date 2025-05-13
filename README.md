# Under-specification in LLM Uses
This is a repository for analyzing the impacts of under-specification to LLM behaviors.

## Data available
We share all experiment configurations in `data/configs`, all prompts in `data/prompts`, all curated requirements in `data/requirements`, and the results in []().

## Steps to reproduce the experiments

First, install the dependencies with poetry: `poetry install`. 

Next, add your OpenAI key for running the OpenAI models, and Bedrock key for running the Llama3 models.


### Experiment 3.1 / 3.2

```bash
poetry run python3 run.py --config=data/configs/commitpack_main.yaml 
poetry run python3 run.py --config=data/configs/trip_main.yaml 
poetry run python3 run.py --config=data/configs/product_main.yaml 
```

### Experiment 4.1

```bash
poetry run python3 run.py --config=data/configs/commitpack_fix.yaml 
poetry run python3 run.py --config=data/configs/trip_fix.yaml 
poetry run python3 run.py --config=data/configs/product_fix.yaml 
```

### Experiment 4.2

To rerun prompt optimization, use

```bash
poetry run python3 -m analysis.optimize  --config=data/configs/commitpack_optimizer_gen.yaml 
poetry run python3 -m analysis.optimize --config=data/configs/trip_optimizer_gen.yaml 
poetry run python3 -m analysis.optimize --config=data/configs/product_optimizer_gen.yaml 
```


To reused the optimized prompts, use

```bash
poetry run python3 run.py --config=data/configs/commitpack_prioritize.yaml 
poetry run python3 run.py --config=data/configs/trip_prioritize.yaml 
poetry run python3 run.py --config=data/configs/product_prioritize.yaml 
```

### Other utilities provided in this repository

To generate new requirements, use

```bash
poetry run python3 -m analysis.elicitation 
```

To generate new prompts, use

```bash
poetry run python3 -m analysis.prompt_gen 
```

To generate new evaluators, use

```bash
poetry run python3 -m analysis.judge 
```