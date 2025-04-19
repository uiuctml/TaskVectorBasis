import json
import os

from utils import find_optimal_coef

import sys
sys.path.append('/home/cindy2000_sh/TaskVectorBasis')

from src.args import parse_arguments
from src.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector, BasisTaskVector, LinearizedBasisTaskVector

import torch
import glob


args = parse_arguments()

if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"/home/cindy2000_sh/TaskVectorBasis/checkpoints_{args.pretrain_openclip_ckpt_name}/{args.model}"


print("*" * 100)
if args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "ft_accuracies.json")
elif args.finetuning_mode == "basis" or args.finetuning_mode == 'linearbasis':
    print(f"Evaluating {args.finetuning_mode} models.")
    ft_accuracies_path = args.save + f"/{args.exp_name}_checkpoints/basis_accuracies.json"
elif args.finetuning_mode == "mtl":
    print("Evaluating mtl models.")
    ft_accuracies_path = args.save + f"/{args.exp_name}_checkpoints/mtl_accuracies.json"
elif args.finetuning_mode == "continual":
    print("Evaluating continual models.")
    ft_accuracies_path = args.save + f"/{args.exp_name}_checkpoints/continual_accuracies.json"
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "linear_ft_accuracies.json")
elif args.finetuning_mode == "posthoc":
    print("Evaluating post-hoc linearized models.")
    ft_accuracies_path = os.path.join(args.save, "posthoc_ft_accuracies.json")
else:
    raise ValueError(f"Invalid finetuning mode: {args.finetuning_mode}")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

eval_datasets = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SVHN",
    "SUN397",
]

task_vectors = []

if args.finetuning_mode == "basis":
    centroid_files = glob.glob(f"{args.save}/{args.exp_name}_checkpoints/centroid_*.pt")
    for file in centroid_files:
        task_vectors.append(
            BasisTaskVector(vector=torch.load(file, map_location="cpu"))
        )
    mapping_path = f'{args.save}/{args.exp_name}_checkpoints/task_to_label_mapping.json'
    with open(mapping_path, 'r') as f:
        task_to_label_mapping = json.load(f)
    eval_datasets = list(task_to_label_mapping.keys())
elif args.finetuning_mode == "linearbasis":
    centroid_files = glob.glob(f"{args.save}/{args.exp_name}_checkpoints/centroid_*.pt")
    for file in centroid_files:
        task_vectors.append(
            LinearizedBasisTaskVector(vector=torch.load(file, map_location="cpu"))
        )
    mapping_path = f'{args.save}/{args.exp_name}_checkpoints/task_to_label_mapping.json'
    with open(mapping_path, 'r') as f:
        task_to_label_mapping = json.load(f)
    eval_datasets = list(task_to_label_mapping.keys())
elif args.finetuning_mode == "mtl":
    centroid_files = glob.glob(f"{args.save}/{args.exp_name}_checkpoints/centroid_*.pt")
    pretrained_checkpoint = f"{args.save}/zeroshot.pt"
    for file in centroid_files:
        task_vectors.append(
            NonLinearTaskVector(pretrained_checkpoint, file)
        )
    mapping_path = f'{args.save}/{args.exp_name}_checkpoints/task_to_label_mapping.json'
    with open(mapping_path, 'r') as f:
        task_to_label_mapping = json.load(f)
    eval_datasets = list(task_to_label_mapping.keys())
elif args.finetuning_mode == "continual":
    with open(f'{args.save}/{args.exp_name}_checkpoints/gtsrb_most_similar_mapping.json', 'r') as f:
        task_to_label_mapping = json.load(f)
    k = task_to_label_mapping.get('GTSRB')
    # mtl
    # pretrained_checkpoint = f"{args.save}/{'GTSRB'}Val/zeroshot.pt"
    # finetuned_checkpoint = f'{args.save}/{args.exp_name}_checkpoints/centroid_{k}.pt'
    # task_vectors.append(NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint))
    task_vectors.append(
            BasisTaskVector(vector=torch.load(f'{args.save}/{args.exp_name}_checkpoints/centroid_{k}.pt', map_location="cpu"))
        )
    eval_datasets = ['GTSRB']
else:
    for dataset in eval_datasets:
        if args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
            task_vectors.append(
                LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )
        else:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
            task_vectors.append(
                NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )

task_vector = sum(task_vectors)

args.eval_datasets = [dataset  for dataset in eval_datasets] 
args.control_dataset = None

# reload pretrained checkpoint from the parent folder
if args.finetuning_mode.startswith('linear'):
    start_pretrained_checkpoint = f"{args.save}/linear_zeroshot.pt"
else:
    start_pretrained_checkpoint = f"{args.save}/zeroshot.pt"

val_metrics = evaluate_task_vector(
    task_vector,
    start_pretrained_checkpoint,
    args,
    posthoc_linearization=args.finetuning_mode == "posthoc",
)

optimal_coef = find_optimal_coef(
    val_metrics,
    metric="avg_normalized_top1",
    minimize=False,
)

args.eval_datasets = [dataset for dataset in eval_datasets]
test_metrics = evaluate_task_vector_at_coef(
    task_vector,
    start_pretrained_checkpoint,
    args,
    float(optimal_coef),
    posthoc_linearization=args.finetuning_mode == "posthoc",
)

print("=" * 100)
print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
additive_accuracies = {"test": test_metrics, "val": val_metrics}

if args.finetuning_mode == "standard":
    save_file = f"{args.save}/additions.json"
elif args.finetuning_mode == "basis" or args.finetuning_mode == "linearbasis":
    save_file = f"{args.save}/{args.exp_name}_checkpoints/basis_additions.json"
elif args.finetuning_mode == "mtl":
    save_file = f"{args.save}/{args.exp_name}_checkpoints/mtl_additions.json"
elif args.finetuning_mode == "continual":
    save_file = f"{args.save}/{args.exp_name}_checkpoints/continual_additions.json"
elif args.finetuning_mode == "linear":
    save_file = f"{args.save}/linear_additions.json"
elif args.finetuning_mode == "posthoc":
    save_file = f"{args.save}/posthoc_additions.json"
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)
