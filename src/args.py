import argparse
import os

import torch


def get_eval_datasets(task_scenario="8"):
    """
    Get evaluation datasets based on task scenario.

    Args:
        task_scenario: String indicating number of tasks ("8", "14", or "20").

    Returns:
        List of dataset names for the specified scenario.
    """
    # Base 8 datasets (standard)
    base_8_datasets = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SVHN",
        "SUN397",
    ]

    # Additional 6 datasets for 14-task scenario
    additional_6_datasets = [
        "CIFAR100",
        "STL10",
        "Flowers102",
        "OxfordIIITPet",
        "PCAM",
        "FER2013",
    ]

    # Additional 6 datasets for 20-task scenario
    additional_6_more_datasets = [
        "EMNIST",
        "CIFAR10",
        "Food101",
        "FashionMNIST",
        "RenderedSST2",
        "KMNIST",
    ]

    if task_scenario == "8":
        return base_8_datasets
    elif task_scenario == "14":
        return base_8_datasets + additional_6_datasets
    elif task_scenario == "20":
        return base_8_datasets + additional_6_datasets + additional_6_more_datasets
    else:
        raise ValueError(f"Invalid task scenario: {task_scenario}. Must be '8', '14', or '20'.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser("~/data"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. ",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num-grad-accumulation",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",  # noqa: E501
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
        help="Directory for caching models from OpenCLIP",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=-1,
        help="How often to checkpoint the model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--finetuning-mode",
        choices=["standard", "none"],
        default="standard",
        help="Finetuning mode: 'standard' (non-linear FT) or 'none' (zero-shot).",
    )
    parser.add_argument(
        "--n-eval-points",
        type=int,
        default=21,
        help="Number of evaluation points used to find optimal coefficient in task arithmetic.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default=None,
        help="Whether to use basis reconstructed experiments or not in negation experiments. If yes, provide the experiment folder name, such as standard_M5_lr0.01_tau1_steps4k_bcFull_seed0",
    )
    parser.add_argument(
        "--subsample-eval",
        action="store_true",
        help="If set, use subsampled evaluation when using basis.",
    )
    parser.add_argument(
        "--task-scenario",
        type=str,
        choices=["8", "14", "20"],
        default="8",
        help="Choose task scenario: 8 (standard), 14 (standard + 6 extra), or 20 (standard + 12 extra) tasks.",
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
