#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# support tp1 train eagle3 for qwen2.5-vl-7b-instruct
NUM_GPUS=${2:-2}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path /scratch/yh5961/RLSD/SpecForge/qwen3-vl \
    --draft-model-config $ROOT_DIR/configs/qwen3-vl-8b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --output-dir $ROOT_DIR/outputs/Qwen3-VL-8B-eagle3-text \
    --num-epochs 2 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --max-length 10240 \
    --dist-timeout 360 \
    --chat-template qwen2-vl \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.language_model.embed_tokens.weight \
    --tp-size 1 \
    --dp-size 2 \
    --ttt-length 8 \
    --is-vlm \
    --resume
