#!/bin/bash

python3 run.py \
    -IQ ../../benchmark/queries_and_videos.json \
    -O /home/eceo_scratch/eceo-staff/valentin/results/prompting_mammalps-v2-LLM/Qwen3-8B \
    --llm "mistral"

