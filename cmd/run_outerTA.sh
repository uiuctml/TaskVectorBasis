#!/bin/bash

# Define the different values for --exp_name
exp_names=("ties" "topk" "mean")
splits=("E-MRS" "E-R-MS" "E-S-MR" "M-E-RS" "M-ERS" "M-R-ES" "M-S-E-R" "M-S-ER" "ME-RS" "MERS" "MR-ES" "MS-ER" "R-MES" "S-MER" "S-R-ME")

# Loop through each value and run the Python script with --exp_name and split arguments
for exp_name in "${exp_names[@]}"; do
    for split_value in "${splits[@]}"; do
        CUDA_VISIBLE_DEVICES=3 python tangent_task_arithmetic/src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=standard --pretrain_openclip_ckpt_name=laion2b_e16 --exp_name=MERS-$exp_name/$split_value
        CUDA_VISIBLE_DEVICES=3 python tangent_task_arithmetic/src/eval_task_addition.py --model=ViT-B-32 --finetuning-mode=standard --pretrain_openclip_ckpt_name=laion2b_e16 --exp_name=MERS-$exp_name/$split_value
    done
done