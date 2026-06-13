import json

from src.args import parse_arguments, get_eval_datasets
from src.eval import eval_single_dataset
from src.task_vectors import NonLinearTaskVector

args = parse_arguments()
if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"checkpoints/{args.model}"

accuracies = {}


print("*" * 100)
if args.finetuning_mode == "none":
    print("Evaluating pretrained models.")
elif args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")

# Get evaluation datasets based on task scenario
eval_datasets = get_eval_datasets(args.task_scenario)
print(f"Using {args.task_scenario}-task scenario with {len(eval_datasets)} datasets:")
print(f"Datasets: {eval_datasets}")

for dataset in eval_datasets:
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    pretrained_checkpoint = f"{args.save}/zeroshot.pt"
    finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"

    try:
        task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
    except FileNotFoundError:
        print(f"Error: Could not find {finetuned_checkpoint}.")
        continue

    if args.finetuning_mode == "none":
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
    else:
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)

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
    save_path = f"{args.save}/zeroshot_accuracies.json"
else:
    save_path = f"{args.save}/ft_accuracies.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)
