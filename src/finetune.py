import os
import time

import torch

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import eval_single_dataset
from src.heads import get_classification_head
from src.modeling import ImageClassifier, ImageEncoder
from src.utils import LabelSmoothing, cosine_lr


def finetune(args):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    assert args.finetuning_mode == "standard", "Only standard fine-tuning is supported."

    # Check if checkpoints already exist
    ft_path = os.path.join(args.save, train_dataset, "finetuned.pt")
    zs_path = os.path.join(args.save, train_dataset, "zeroshot.pt")
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f"Skipping fine-tuning because {ft_path} exists.")
        return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."

    if args.load is not None and args.load.endswith("pt"):
        image_encoder = ImageEncoder.load(args.load)
    else:
        print("Building image encoder.")
        image_encoder = ImageEncoder(args)

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(dataset.train_loader)

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    # Saving zero-shot model
    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        model.image_encoder.save(os.path.join(ckpdir, "zeroshot.pt"))

    for epoch in range(args.epochs):
        model.train()

        for i, batch in enumerate(data_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            labels = batch["labels"].cuda()
            data_time = time.time() - start_time

            logits = model(inputs)

            loss = loss_fn(logits, labels)

            loss.backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
            ):
                print("Saving checkpoint.")
                model.image_encoder.save(os.path.join(ckpdir, f"checkpoint_{step}.pt"))

            if step % print_every == 0 and ((i + 1) % args.num_grad_accumulation == 0):
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    flush=True,
                )

    # Evaluate
    image_encoder = model.image_encoder
    eval_single_dataset(image_encoder, train_dataset, args)

    if args.save is not None:
        zs_path = os.path.join(ckpdir, "zeroshot.pt")
        ft_path = os.path.join(ckpdir, "finetuned.pt")
        image_encoder.save(ft_path)
        return zs_path, ft_path


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
    epochs = {
        "Cars": 35,
        "DTD": 76,
        "EuroSAT": 12,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SUN397": 14,
        "SVHN": 4,
    }

    for dataset in train_datasets:
        args = parse_arguments()

        # HACK: Some command line arguments are overwritten by defaults here.
        args.lr = 1e-5
        args.epochs = epochs[dataset]
        args.train_dataset = dataset + "Val"

        # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
        args.batch_size = 64 if args.model == "ViT-L-14" else 128
        args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1

        if args.seed is not None:
            args.save = f"checkpoints_{args.seed}/{args.model}"
        else:
            args.save = f"checkpoints/{args.model}"
        print("=" * 100)
        print(f"Finetuning {args.model} on {dataset}")
        print("=" * 100)

        finetune(args)
