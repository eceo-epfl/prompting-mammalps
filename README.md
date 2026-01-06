# Prompting MammAlps
Development codebase for text-to-video retrieval benchmark on MammAlps.

## Getting started for development
```bash
conda create -n prompting-mammalps python=3.12 uv
uv pip install -e .
pre-commit install
```

## Benchmark 
- Generate queries and associate ground truth videos to them
- Dataset split at the query and event level for 2023 and 2024 annotated data
- Dataset analysis and visualization

## Models
Each model has it's unique python environment (DockerFile) to run on RCP, please check the associated READMEs
- SALMA (TBD): Spatio-Temporal Action Localization
- llm: Automatic generation of code to verify if a generated JSON file from SALMA corresponds to a given query or not

## Evaluation
- Evaluation at different stages of the prediction pipeline
- TrackEval (TBD): Submodule for tracking evaluation of SALMA