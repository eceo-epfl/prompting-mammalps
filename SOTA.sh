#!/usr/bin/env bash
# commands.sh
#nohup bash commands.sh > 

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p logs

VIDEOS_TRAIN="/media/EVO870/datasets/prompting-mammalps/videos/train"
VIDEOS_TEST="/media/EVO870/datasets/prompting-mammalps/videos/test"

BENCH_TRAIN="benchmark/queries_and_videos_train.json"
BENCH_TEST="benchmark/queries_and_videos_test.json"

GRAM_CKPT="models/GRAM/pretrained_weights/GRAM_pretrained_4modalities"
CLIP_CROSS="models/CLIP4Clip/modules/cross-base"

# Fine-tuned CLIP4Clip checkpoint produced by STEP 2 (pytorch_model.bin.<epoch>)
CLIP_CKPT_DIR="zero_shot/CLIP4Clip/checkpoints"
CLIP_CKPT="${CLIP_CKPT_DIR}/pytorch_model.bin.1"   # epoch 1 by default


# Writes queries.txt, ground_truth.json, pairs.csv
# for both models and both splits in one call each.
echo "1 prepare dataset  (TRAIN)"
uv run python zero_shot/prepare_dataset.py \
  --input        "$BENCH_TRAIN" \
  --output_dirs  zero_shot/CLIP4Clip/dataset/train \
                 zero_shot/GRAM/dataset/train \
  --video_folder "$VIDEOS_TRAIN"


echo "1 prepare dataset  (TEST)"
uv run python zero_shot/prepare_dataset.py \
  --input        "$BENCH_TEST" \
  --output_dirs  zero_shot/CLIP4Clip/dataset/test \
                 zero_shot/GRAM/dataset/test \
  --video_folder "$VIDEOS_TEST"


# Produces zero_shot/CLIP4Clip/checkpoints/pytorch_model.bin.1
echo "2 CLIP4Clip fine-tuning  (TRAIN set)"
uv run python zero_shot/CLIP4Clip/training_clip4clip.py \
  --train_csv       zero_shot/CLIP4Clip/dataset/train/pairs.csv \
  --train_video_dir "$VIDEOS_TRAIN" \
  --output_dir      "$CLIP_CKPT_DIR" \
  --cross_model     "$CLIP_CROSS" \
  --epochs          5 \
  --batch_size      8 \
  --lr              1e-5 \
  --max_frames      12 \
  --sim_header      tightTransf


# Train split threshold calibration
# Test  split final similarity scores

echo "3 CLIP4Clip eval  (TRAIN  for threshold)"

uv run python zero_shot/CLIP4Clip/eval_zeroshot_clip4clip.py \
  --queries_file  zero_shot/CLIP4Clip/dataset/train/queries.txt \
  --features_path "$VIDEOS_TRAIN" \
  --ground_truth  zero_shot/CLIP4Clip/dataset/train/ground_truth.json \
  --output_dir    zero_shot/CLIP4Clip/eval_dataset/train \
  --cross_model   "$CLIP_CROSS" \
  --init_model    "$CLIP_CKPT" \
  --max_frames    16 \
  --batch_size    32

echo "3  CLIP4Clip eval  (TEST  final scores)"
uv run python zero_shot/CLIP4Clip/eval_zeroshot_clip4clip.py \
  --queries_file  zero_shot/CLIP4Clip/dataset/test/queries.txt \
  --features_path "$VIDEOS_TEST" \
  --ground_truth  zero_shot/CLIP4Clip/dataset/test/ground_truth.json \
  --output_dir    zero_shot/CLIP4Clip/eval_dataset/test \
  --cross_model   "$CLIP_CROSS" \
  --init_model    "$CLIP_CKPT" \
  --max_frames    16 \
  --batch_size    32



echo "4  CLIP4Clip threshold validation  (TRAIN split)"

uv run python zero_shot/CLIP4Clip/validation_thres.py \
  --results_dir  zero_shot/CLIP4Clip/eval_dataset/train \
  --ground_truth zero_shot/CLIP4Clip/dataset/train/ground_truth.json \
  --output_dir   zero_shot/CLIP4Clip/validation_results/train




echo "5  GRAM eval  (TEST  final scores)"

uv run python zero_shot/GRAM/eval_zeroshot_gram.py \
  --queries_file  zero_shot/GRAM/dataset/test/queries.txt \
  --video_path    "$VIDEOS_TEST" \
  --ground_truth  zero_shot/GRAM/dataset/test/ground_truth.json \
  --output_dir    zero_shot/GRAM/eval_results/test \
  --checkpoint    "$GRAM_CKPT" \
  --batch_size    4


echo "5  GRAM eval  (TRAIN  for threshold)"

uv run python zero_shot/GRAM/eval_zeroshot_gram.py \
  --queries_file  zero_shot/GRAM/dataset/train/queries.txt \
  --video_path    "$VIDEOS_TRAIN" \
  --ground_truth  zero_shot/GRAM/dataset/train/ground_truth.json \
  --output_dir    zero_shot/GRAM/eval_results/train \
  --checkpoint    "$GRAM_CKPT" \
  --batch_size    4

echo "6  GRAM threshold validation  (TRAIN split)"

uv run python zero_shot/GRAM/validation_threshold_gram.py \
  --results_dir  zero_shot/GRAM/eval_results/train \
  --ground_truth zero_shot/GRAM/dataset/train/ground_truth.json \
  --output_dir   zero_shot/GRAM/validation_results/train


