#!/bin/bash
# Task addition evaluation over all AutoencoderBasisMethod bases for a given model/scenario.
#
# Usage: CUDA_VISIBLE_DEVICES=0 MODEL=ViT-B-32 TASKS=8 bash scripts/run_addition.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL=${MODEL:-ViT-B-32}
TASKS=${TASKS:-8}

BASE_PATH="checkpoints/${MODEL}/_basis_runs"

echo "Running AutoencoderBasisMethod task addition for ${MODEL} (${TASKS}-task pattern)..."

for dir in "$BASE_PATH"/AutoencoderBasisMethod_*_${TASKS}task_*; do
    if [ -d "$dir" ]; then
        folder_name=$(basename "$dir")
        echo "Processing: $folder_name"
        python src/eval_task_addition.py \
            --model "$MODEL" \
            --task-scenario "$TASKS" \
            --finetuning-mode standard \
            --basis "$folder_name"
        echo "Completed: $folder_name"
        echo "----------------------------------------"
    fi
done

echo "All task addition evaluations (${TASKS}-task) completed!"
