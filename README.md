# Under-specification in LLM Uses
This is a repository for analyzing the impacts of under-specification on LLM behaviors.

## Data available
We share all experiment configurations in `data/configs`, all prompts in `data/prompts`, all curated requirements in `data/requirements`, and the evaluation results [here](https://figshare.com/s/38acdc02f9cae8c39198).

## Steps to reproduce the analysis
Download evaluation data from [here](https://figshare.com/s/38acdc02f9cae8c39198). Create three repositores `data/results/commitpack`, `data/results/trip`, `data/results/product`, and uncompress the evaluation results into each repository.

Run steps in `analysis-reproduction.ipynb`.

## Steps to reproduce the full experiments

First, add your OpenAI key for running the OpenAI models, and Bedrock key for running the Llama3 models.


### Experiment 3.2 / 3.3

```bash
uv run python3 run.py --config=data/configs/commitpack_main.yaml 
uv run python3 run.py --config=data/configs/trip_main.yaml 
uv run python3 run.py --config=data/configs/product_main.yaml 
```

### Experiment 3.5

```bash
uv run python3 run.py --config=data/configs/commitpack_fix.yaml 
uv run python3 run.py --config=data/configs/trip_fix.yaml 
uv run python3 run.py --config=data/configs/product_fix.yaml 
```

### Experiment 4.1 / 4.2

To rerun prompt optimization, use

```bash
uv run python3 -m analysis.optimize  --config=data/configs/commitpack_optimizer_gen.yaml 
uv run python3 -m analysis.optimize --config=data/configs/trip_optimizer_gen.yaml 
uv run python3 -m analysis.optimize --config=data/configs/product_optimizer_gen.yaml 
```


To reused the optimized prompts, use

```bash
uv run python3 run.py --config=data/configs/commitpack_prioritize.yaml 
uv run python3 run.py --config=data/configs/trip_prioritize.yaml 
uv run python3 run.py --config=data/configs/product_prioritize.yaml 
```

### Other utilities provided in this repository

To generate new requirements, use

```bash
uv run python3 -m analysis.elicitation 
```

To generate new evaluators, use

```bash
uv run python3 -m analysis.judge 
```

To generate new prompts, use

```bash
uv run python3 -m analysis.prompt_gen 
```