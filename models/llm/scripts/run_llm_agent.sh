#!/bin/bash
DATA_ROOT="$1"

python3 run.py \
    -IQ "$DATA_ROOT/metadata/queries_and_videos_train.json" \
    -D "$DATA_ROOT" \
    -O ./results/ \
    --llm "qwen"
