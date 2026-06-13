#!/bin/bash
# Single-task evaluation: zero-shot and standard fine-tuned accuracies.
# These produce zeroshot_accuracies.json / ft_accuracies.json used to normalize
# task addition results.
#
# Usage: CUDA_VISIBLE_DEVICES=0 MODEL=ViT-B-32 TASKS=8 bash scripts/run_single_task.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL=${MODEL:-ViT-B-32}
TASKS=${TASKS:-8}

echo "=== Zero-shot evaluation (${MODEL}, ${TASKS}-task) ==="
python src/eval_single_task.py --model "$MODEL" --task-scenario "$TASKS" --finetuning-mode none

echo "=== Standard fine-tuned evaluation (${MODEL}, ${TASKS}-task) ==="
python src/eval_single_task.py --model "$MODEL" --task-scenario "$TASKS" --finetuning-mode standard
