import json
import os
import sys
import time

from src.utils import find_optimal_coef
from src.args import parse_arguments, get_eval_datasets
from src.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.task_vectors import NonLinearTaskVector
from src.basis_vectors import BasisMethod

args = parse_arguments()

checkpoints_root = "checkpoints"
if getattr(args, 'seed', None) is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"checkpoints/{args.model}"


print("*" * 100)
if args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "ft_accuracies.json")
else:
    raise ValueError(f"Invalid finetuning mode: {args.finetuning_mode}")

if getattr(args, 'seed', None) is not None:
    print(f"Experiment seed: {args.seed}")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

with open(os.path.join(args.save, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

# Determine the save file path and folder to handle basis runs
if args.basis:
    save_folder = args.save + f'/_basis_runs/{args.basis}'
else:
    save_folder = args.save

base_filename = f"additions_{args.task_scenario}task.json"

save_file = f"{save_folder}/{base_filename}"

# Check if the results file already exists
if os.path.exists(save_file):
    print(f"Results file {save_file} already exists. Skipping evaluation.")
    print("If you want to re-run the evaluation, please delete the existing file first.")
    sys.exit(0)

# Get evaluation datasets based on task scenario
eval_datasets = get_eval_datasets(args.task_scenario)
print(f"Using {args.task_scenario}-task scenario with {len(eval_datasets)} datasets:")
print(f"Datasets: {eval_datasets}")

task_vectors = []
pretrained_checkpoint = f"{args.save}/zeroshot.pt"

if args.basis:
    print("Loading basis vectors...")
    basis_run_dir = args.save + f'/_basis_runs/{args.basis}'

    # Load basis vectors directly
    basis_vectors = BasisMethod.load_basis_vectors(basis_run_dir)
    print(f"Loaded {len(basis_vectors)} basis vectors from {basis_run_dir}")

    # Create task vectors from basis vectors and sum them
    for basis_vector in basis_vectors:
        task_vectors.append(NonLinearTaskVector(vector=basis_vector))
else:
    # Original approach: load from individual dataset checkpoints
    for dataset in eval_datasets:
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        task_vectors.append(
            NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )

task_vector = sum(task_vectors)

args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
args.control_dataset = None

# Start timing for the validation phase
print("Starting validation evaluation...")
validation_start_time = time.time()

# We use the validation set to choose the optimal coefficient.
val_metrics = evaluate_task_vector(
    task_vector,
    pretrained_checkpoint,
    args,
)

validation_end_time = time.time()
validation_time = validation_end_time - validation_start_time
print(f"Validation evaluation completed in {validation_time:.2f} seconds")

optimal_coef = find_optimal_coef(
    val_metrics,
    metric="avg_normalized_top1",
    minimize=False,
)

# Evaluate on the test set with the optimal coefficient.
args.eval_datasets = [dataset for dataset in eval_datasets]

print("Starting test evaluation...")
test_start_time = time.time()

# Always evaluate on all datasets for test (no sampling)
test_metrics = evaluate_task_vector_at_coef(
    task_vector,
    pretrained_checkpoint,
    args,
    float(optimal_coef),
)

test_end_time = time.time()
test_time = test_end_time - test_start_time
print(f"Test evaluation completed in {test_time:.2f} seconds")

total_time = validation_time + test_time

print("=" * 100)
print(f"FINAL RESULTS")
print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
print(f"Optimal coefficient: {optimal_coef}")
print(f"Validation time: {validation_time:.2f} seconds")
print(f"Test time: {test_time:.2f} seconds")
print(f"Total time: {total_time:.2f} seconds")
print("=" * 100)

# Save results with timing information
additive_accuracies = {
    "test": test_metrics,
    "val": val_metrics,
    "timing": {
        "validation_time": validation_time,
        "test_time": test_time,
        "total_time": total_time
    },
    "optimal_coefficient": float(optimal_coef),
    "seed": args.seed,
    }


with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)
