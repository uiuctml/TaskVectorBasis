import json

import sys
sys.path.append('/home/cindy2000_sh/TaskVectorBasis')

from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.linearize import LinearizedImageEncoder
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector, BasisTaskVector, LinearizedBasisTaskVector

import torch

args = parse_arguments()
if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"/home/cindy2000_sh/TaskVectorBasis/checkpoints_{args.pretrain_openclip_ckpt_name}/{args.model}"

accuracies = {}


print("*" * 100)
if args.finetuning_mode == "none":
    print("Evaluating pretrained models.")
elif args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
elif args.finetuning_mode == "basis":
    print("Evaluating basis models.")
elif args.finetuning_mode == "linearbasis":
    print("Evaluating linear basis models.")
elif args.finetuning_mode == 'continual':
    print("Evaluating continual models.")
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
elif args.finetuning_mode == "posthoc":
    print("Evaluating post-hoc linearized models.")

for dataset in [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]:
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    pretrained_checkpoint = f"{args.save}/zeroshot.pt"
    if args.finetuning_mode == 'linear':
        pretrained_checkpoint = f"{args.save}/linear_zeroshot.pt"

    finetuned_checkpoint = (
        f"{args.save}/{dataset}Val/linear_finetuned.pt"
        if args.finetuning_mode == "linear"
        else f"{args.save}/{dataset}Val/finetuned.pt"
    )


    if args.finetuning_mode == 'basis':
        # Load the saved task-to-cluster label mapping
        mapping_path = f'{args.save}/{args.exp_name}_checkpoints/task_to_label_mapping.json'
        with open(mapping_path, 'r') as f:
            task_to_label_mapping = json.load(f)
        k = task_to_label_mapping.get(dataset)
        if k is None:
            continue
        else:
            if args.mergeB is not None:
                task_vector = (BasisTaskVector(vector=torch.load(f'{args.save}/{args.exp_name}_checkpoints/merged_centroid_{args.mergeB}.pt', map_location=torch.device('cpu'))))
            else:
                task_vector = (BasisTaskVector(vector=torch.load(f'{args.save}/{args.exp_name}_checkpoints/centroid_{k}.pt')))
    elif args.finetuning_mode == 'linearbasis':
        # Load the saved task-to-cluster label mapping
        pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
        mapping_path = f'{args.save}/{args.exp_name}_checkpoints/task_to_label_mapping.json'
        with open(mapping_path, 'r') as f:
            task_to_label_mapping = json.load(f)
        k = task_to_label_mapping.get(dataset)
        if k is None:
            continue
        else:
            task_vector = (LinearizedBasisTaskVector(vector=torch.load(f'{args.save}/{args.exp_name}_checkpoints/centroid_{k}.pt')))
    elif args.finetuning_mode == 'continual':
        # Load the saved task-to-cluster label mapping
        mapping_path = f'{args.save}/{args.exp_name}_checkpoints/gtsrb_most_similar_mapping.json'
        with open(mapping_path, 'r') as f:
            task_to_label_mapping = json.load(f)
        k = task_to_label_mapping.get(dataset)
        if k is None:
            continue
        else: # mtl
            # pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            # finetuned_checkpoint = f'{args.save}/{args.exp_name}_checkpoints/centroid_{k}.pt'
            # task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            
            # basis
            task_vector = (BasisTaskVector(vector=torch.load(f'{args.save}/{args.exp_name}_checkpoints/centroid_{k}.pt')))
    elif args.finetuning_mode == 'mtl':
        mapping_path = f'{args.save}/{args.exp_name}_checkpoints/task_to_label_mapping.json'
        with open(mapping_path, 'r') as f:
            task_to_label_mapping = json.load(f)
        k = task_to_label_mapping.get(dataset)
        if k is None:
            continue
        else:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            finetuned_checkpoint = f'{args.save}/{args.exp_name}_checkpoints/centroid_{k}.pt'
            task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
    else:
        try:
            task_vector = (
                LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
                if args.finetuning_mode == "linear"
                else NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )
        except FileNotFoundError:
            print(f"Error: Could not find {finetuned_checkpoint}.")
            continue

    if args.finetuning_mode == "none":
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
    elif args.finetuning_mode == "standard" or args.finetuning_mode == "linear" or args.finetuning_mode == 'basis' or args.finetuning_mode == 'mtl' or args.finetuning_mode == 'continual' or args.finetuning_mode == 'linearbasis':
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
    elif args.finetuning_mode == "posthoc":
        zs_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
        ft_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        image_encoder = LinearizedImageEncoder(
            init_encoder=zs_encoder, image_encoder=ft_encoder, args=args
        )

    for split in ["test", "val"]:
        # Evaluate
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"

        accuracies[eval_dataset] = eval_single_dataset(
            image_encoder, eval_dataset, args
        )["top1"]


if args.finetuning_mode == "none":
    # Evaluate zero-shot accuracy on ImageNet
    for split in ["ImageNetVal", "ImageNet"]:
        accuracies[split] = eval_single_dataset(image_encoder, split, args)["top1"]

# Save results
if args.finetuning_mode == "none":
    save_path = f"/home/cindy2000_sh/TaskVectorBasis/checkpoints_{args.pretrain_openclip_ckpt_name}/{args.model}/zeroshot_accuracies.json"
elif args.finetuning_mode == "standard":
    save_path = f"{args.save}/ft_accuracies.json"
elif args.finetuning_mode == "basis" or args.finetuning_mode == 'linearbasis':
    if args.mergeB is not None:
        save_path = f"{args.save}/{args.exp_name}_checkpoints/merged_centroid_{args.mergeB}_accuracies.json"
    else:
        save_path = f"{args.save}/{args.exp_name}_checkpoints/basis_accuracies.json"
elif args.finetuning_mode == "continual":
    save_path = f"{args.save}/{args.exp_name}_checkpoints/continual_accuracies.json"
elif args.finetuning_mode == "mtl":
    save_path = f"{args.save}/{args.exp_name}_checkpoints/mtl_accuracies.json"
elif args.finetuning_mode == "linear":
    save_path = f"{args.save}/linear_ft_accuracies.json"
elif args.finetuning_mode == "posthoc":
    save_path = f"{args.save}/posthoc_ft_accuracies.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)
