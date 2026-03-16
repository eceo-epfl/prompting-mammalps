#!/bin/bash

python3 run_vanilla.py \
    -IQ ../../benchmark/queries_and_videos.json \
    -O /home/eceo_scratch/eceo-staff/valentin/results/prompting_mammalps-v2-LLM/Qwen3-8B_vanilla \
    --llm "qwen" \
    --yaml "prompt_vanilla.yaml"

