#!/bin/bash

python3 run_no_label_constrain.py \
    -IQ ../../benchmark/queries_and_videos.json \
    -O /home/eceo_scratch/eceo-staff/valentin/results/prompting_mammalps-v2-LLM/Qwen3-8B_no_label \
    --llm "qwen" \
    --yaml "prompt_no_label_constrain.yaml"

