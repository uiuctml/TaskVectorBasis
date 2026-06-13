#!/bin/bash
# Task negation evaluation over basis folders for a given model/scenario.
#
# Usage: CUDA_VISIBLE_DEVICES=0 MODEL=ViT-B-16 TASKS=8 bash scripts/run_negation.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL=${MODEL:-ViT-B-16}
TASKS=${TASKS:-8}

BASE_PATH="checkpoints/${MODEL}/_basis_runs"

# Basis methods to evaluate (leave empty to run all available)
BASIS_METHODS=("AutoencoderBasisMethod" "PCABasisMethod" "RandomBasisMethod")

echo "=== Task negation for ${MODEL}, ${TASKS}-task scenario ==="

if [ ! -d "$BASE_PATH" ]; then
    echo "Base path $BASE_PATH does not exist. Build a basis first (scripts/build_ae_basis.py)."
    exit 1
fi

# Find all basis folders matching the task scenario, then filter by method
all_basis_folders=$(find "$BASE_PATH" -maxdepth 1 -type d -name "*_${TASKS}task_*" -exec basename {} \; | sort)

if [ ${#BASIS_METHODS[@]} -gt 0 ]; then
    basis_folders=""
    for method in "${BASIS_METHODS[@]}"; do
        method_folders=$(echo "$all_basis_folders" | grep "^${method}_")
        if [ -n "$method_folders" ]; then
            basis_folders="${basis_folders}${method_folders}"$'\n'
        fi
    done
    basis_folders=$(echo "$basis_folders" | sed '/^$/d' | sort)
else
    basis_folders="$all_basis_folders"
fi

if [ -z "$basis_folders" ]; then
    echo "No matching basis folders found in $BASE_PATH for ${TASKS}-task scenario."
    exit 1
fi

while IFS= read -r basis_folder; do
    if [ -n "$basis_folder" ]; then
        echo "--- Evaluating basis: $basis_folder ---"
        python src/eval_task_negation.py \
            --model "$MODEL" \
            --task-scenario "$TASKS" \
            --finetuning-mode standard \
            --basis "$basis_folder"
        echo "Completed: $basis_folder"
    fi
done <<< "$basis_folders"

echo "All negation evaluations completed!"
