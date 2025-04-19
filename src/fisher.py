import os

import torch

import sys
sys.path.append('/home/cindy2000_sh/ntk-llm/tangent_task_arithmetic')

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier, ImageEncoder

import torch
from tqdm import tqdm

def compute_fisher_for_batch(
    batch, model, variables, parameter_names, expectation_wrt_logits=True
):
    assert expectation_wrt_logits, "Sampling from logits is not implemented yet."

    batch = maybe_dictionarize(batch)

    inputs = batch["images"].cuda()

    logits = model(inputs) 
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    weighted_log_probs = (probs * log_probs).sum()  

    grads = torch.autograd.grad(
        weighted_log_probs, variables, retain_graph=False, create_graph=False, allow_unused=True
    )
    grads = [
        grad if grad is not None else torch.zeros_like(variable)
        for grad, variable in zip(grads, variables)
    ]

    fishers = {name: torch.zeros_like(param) for name, param in zip(parameter_names, variables)}

    for name, grad in zip(parameter_names, grads):
        fishers[name] += grad ** 2 

    del logits, log_probs, probs, grads
    torch.cuda.empty_cache()

    return fishers

def compute_fisher_for_model(
    model, dataloader, expectation_wrt_logits=True
):
    model.eval()  

    variables = [param for param in model.parameters() if param.requires_grad]
    parameter_names = [name for name, param in model.named_parameters() if param.requires_grad]
    fishers = {name: torch.zeros_like(param) for name, param in zip(parameter_names, variables)}

    n_examples = 0

    for i, batch in enumerate(tqdm(dataloader)):
        batch = maybe_dictionarize(batch)
        batch_size = batch["images"].size(0)
        n_examples += batch_size

        batch_fishers = compute_fisher_for_batch(
            batch, model, variables, parameter_names, expectation_wrt_logits
        )

        for name in fishers.keys():
            fishers[name] += batch_fishers[name]
    for name in fishers.keys():
        fishers[name] /= n_examples

    return fishers

def save_fisher_info(base_dir, dataset_name, fishers):
    dataset_dir = os.path.join(base_dir, f"{dataset_name}")
    os.makedirs(dataset_dir, exist_ok=True)

    fisher_save_path = os.path.join(dataset_dir, "fisher_train.pth")
    torch.save(fishers, fisher_save_path)
    print(f"Fisher information saved to {fisher_save_path}")


def finetune(args):
    train_dataset = args.train_dataset

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    assert train_dataset is not None, "Please provide a training dataset."



    if args.load is not None and args.load.endswith("pt"):
        args.load = args.save + f'/{train_dataset}/' + args.load
        image_encoder = torch.load(args.load)
        print(f'Checkpoint loaded from {args.load}.')
    else:
        print("Building image encoder.")
        image_encoder = (
            LinearizedImageEncoder(args, keep_lang=False)
            if linearized_finetuning
            else ImageEncoder(args)
        )

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()

    preprocess_fn = model.train_preprocess

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)

    fishers = compute_fisher_for_model(model, data_loader)
    base_dir = "/home/cindy2000_sh/ntk-llm/tangent_task_arithmetic/checkpoints_laion2b_e16/ViT-B-32"
    save_fisher_info(base_dir, train_dataset, fishers)



if __name__ == "__main__":
    train_datasets = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
    ]

    
    for dataset in train_datasets:
        args = parse_arguments()
        args.lr = 1e-5
        args.epochs = 1
        args.train_dataset = dataset + "Val"

        args.batch_size = 64 if args.model == "ViT-L-14" else 128
        args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1

        if args.seed is not None:
            args.save = f"/home/cindy2000_sh/ntk-llm/tangent_task_arithmetic/checkpoints_{args.seed}/{args.model}"
        else:
            args.save = f"/home/cindy2000_sh/ntk-llm/tangent_task_arithmetic/checkpoints_{args.pretrain_openclip_ckpt_name}/{args.model}"

        finetune(args)