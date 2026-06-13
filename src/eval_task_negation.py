import json
import os
import sys

from src.utils import find_optimal_coef
from src.args import parse_arguments, get_eval_datasets
from src.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.task_vectors import NonLinearTaskVector
from src.basis_pipeline import load_and_recover_from_saved_basis
from src.basis_utils import _discover_finetuned_ckpts

args = parse_arguments()
args.control_threshold = 0.95 # following Task Arithmetic paper

checkpoints_root = "checkpoints"
args.save = f"checkpoints/{args.model}"

with open(os.path.join(args.save, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

# Get evaluation datasets based on task scenario    
eval_datasets = get_eval_datasets(args.task_scenario)
print(f"Using {args.task_scenario}-task scenario with {len(eval_datasets)} datasets:")
print(f"Datasets: {eval_datasets}")

print("*" * 100)
if args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "ft_accuracies.json")
else:
    raise ValueError(f"Invalid finetuning mode: {args.finetuning_mode}")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

# Determine the save file path first to check if it already exists
if args.basis:
    save_folder = args.save + f'/_basis_runs/{args.basis}'
else:
    save_folder = args.save

save_file = f"{save_folder}/negations_{args.task_scenario}task.json"

# Check if the results file already exists
if os.path.exists(save_file):
    print(f"Results file {save_file} already exists. Skipping evaluation.")
    print("If you want to re-run the evaluation, please delete the existing file first.")
    sys.exit(0)

control_dataset = "ImageNet"
negation_accuracies = {}

# Load all task vectors at once if using basis
if args.basis:
    print("Loading all task vectors from basis artifacts...")
    
    # Discover finetuned checkpoints for basis reconstruction
    # Temporarily set args.save to root directory for _discover_finetuned_ckpts
    original_save = args.save
    args.save = checkpoints_root
    finetuned_ckpts = _discover_finetuned_ckpts(args, eval_datasets)
    args.save = original_save
    pretrained_ckpt_for_basis = os.path.join(args.save, "zeroshot.pt")
    basis_run_dir = args.save + f'/_basis_runs/{args.basis}'

    # Load all recovered task vectors at once
    all_recovered_task_vectors = load_and_recover_from_saved_basis(
        run_dir=basis_run_dir,
    )

    print(f"Loaded {len(all_recovered_task_vectors)} task vectors from basis.")

for dataset in eval_datasets:
    parent_folder = args.save
    
    if args.basis:
        # === NEW: Use pre-loaded task vector from basis ===
        dataset_idx = eval_datasets.index(dataset)
        reconstructed_task_vector = all_recovered_task_vectors[dataset_idx]

        pretrained_checkpoint = f"{parent_folder}/zeroshot.pt"
        task_vector = -NonLinearTaskVector(vector=reconstructed_task_vector)

    else:
        # === ORIGINAL: Load from pretrained and finetuned checkpoints ===
        checkpoint_folder = args.save
        finetuned_name = 'finetuned'

        pretrained_checkpoint = f"{parent_folder}/zeroshot.pt"
        finetuned_checkpoint = f"{checkpoint_folder}/{dataset}Val/{finetuned_name}.pt"
        task_vector = -NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    # We use the validation set to choose the optimal coefficient.
    args.eval_datasets = [dataset + "Val"]
    args.control_dataset = control_dataset + "Val"

    val_metrics = evaluate_task_vector(
        task_vector,
        pretrained_checkpoint,
        args,
    )

    optimal_coef = find_optimal_coef(
        val_metrics,
        metric=f"{dataset}Val:top1",
        minimize=True,
        control_metric=f"{control_dataset}Val:top1",
        control_metric_threshold=args.control_threshold
        * pretrained_accuracies[control_dataset + "Val"],
    )

    # Evaluate on the test set with the optimal coefficient.
    args.eval_datasets = [dataset]
    args.control_dataset = control_dataset
    test_metrics = evaluate_task_vector_at_coef(
        task_vector,
        pretrained_checkpoint,
        args,
        optimal_coef,
    )

    print("=" * 100)
    print(f"Test accuracy: {test_metrics[f'{dataset}:top1']}")

    negation_accuracies[dataset] = {
        "test": test_metrics[f"{dataset}:top1"],
        "test_control": test_metrics[f"{control_dataset}:top1"],
        "val": val_metrics,
    }

with open(save_file, "w") as f:
    json.dump(negation_accuracies, f, indent=4)
