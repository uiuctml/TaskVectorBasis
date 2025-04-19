import os
import json
import shutil
import time

import torch

import sys
sys.path.append('/home/cindy2000_sh/TaskVectorBasis')

from src.args import parse_arguments
from src.datasets.registry import get_dataset
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageEncoder, MultiHeadImageClassifier
from src.utils import LabelSmoothing, cosine_lr

from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = [dataset.train_dataset for dataset in datasets]
        self.dataset_indices = []
        
        current_index = 0
        for dataset in datasets:
            dataset_len = len(dataset.train_dataset)
            self.dataset_indices.extend([current_index] * dataset_len)
            current_index += 1
        
        self.total_len = sum(len(dataset.train_dataset) for dataset in datasets)
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        dataset_idx = self.dataset_indices[idx]
        
        adjusted_idx = idx - sum(len(d) for d in self.datasets[:dataset_idx])
        
        sample = self.datasets[dataset_idx][adjusted_idx]

        inputs = torch.as_tensor(sample[0]) if not isinstance(sample[0], torch.Tensor) else sample[0]
        labels = torch.as_tensor(sample[1]) if not isinstance(sample[1], torch.Tensor) else sample[1]

        dataset_idx_tensor = torch.as_tensor(dataset_idx, dtype=torch.long)
        
        return inputs, labels, dataset_idx_tensor


def finetune(rank, label, args):
    setup_ddp(rank, args.world_size, port=args.port)

    ckpdir = args.save

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")
    
    if args.load is not None and args.load.endswith("pt"):
        image_encoder = (
            LinearizedImageEncoder.load(args.load)
            if linearized_finetuning
            else ImageEncoder.load(args.load)
        )
    else:
        print("Building shared image encoder.")
        image_encoder = (
            LinearizedImageEncoder(args, keep_lang=False)
            if linearized_finetuning
            else ImageEncoder(args)
        )

    classification_heads = [get_classification_head(args, dataset) for dataset in args.train_datasets]

    model = MultiHeadImageClassifier(image_encoder, classification_heads)

    model.freeze_head()
    model = model.cuda()

    preprocess_fn = model.train_preprocess
    print_every = 100

    combined_dataset = CombinedDataset([get_dataset(name, preprocess_fn, location=args.data_location,
        batch_size=args.batch_size,) for name in args.train_datasets])
    
    data_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)
    num_batches = len(combined_dataset) // args.batch_size

    ddp_loader = distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=True,
        output_device=rank,
    )

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    for epoch in range(args.epochs):
        ddp_model.train()

        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            inputs, labels, dataset_idx = batch
            inputs = inputs.cuda()
            labels = labels.cuda()
            dataset_idx = dataset_idx.cuda() 

            data_time = time.time() - start_time

            optimizer.zero_grad()

            unique_datasets = torch.unique(dataset_idx)
            
            total_loss = 0.0
            for ds_idx in unique_datasets:
                ds_mask = dataset_idx == ds_idx

                ds_inputs = inputs[ds_mask]
                ds_labels = labels[ds_mask]

                logits = ddp_model(ds_inputs, head_idx=ds_idx.item())

                loss = loss_fn(logits, ds_labels)

                task_loss_weight = 1.0 / len(ds_mask) 
                total_loss += task_loss_weight * loss

                loss.backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            batch_time = time.time() - start_time

            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
                and is_main_process()
            ):
                print("Saving checkpoint.")
                model_path = (
                    os.path.join(ckpdir, f"linear_checkpoint_{step}.pt")
                    if linearized_finetuning
                    else os.path.join(ckpdir, f"checkpoint_{step}.pt")
                )
                ddp_model.module.image_encoder.save(model_path)

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * i / len(ddp_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(ddp_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    flush=True,
                )

    if args.save is not None and is_main_process():
        ft_path = args.save + f'/{args.exp_name}_checkpoints/centroid_{label}.pt'
        image_encoder.save(ft_path)
        return ft_path

    cleanup_ddp()


if __name__ == "__main__":
    args = parse_arguments()

    with open(f'/home/cindy2000_sh/TaskVectorBasis/checkpoints_{args.pretrain_openclip_ckpt_name}/{args.model}/{args.exp_name}_checkpoints/task_to_label_mapping.json') as f:
        task_to_label = json.load(f)
    
    label_to_datasets = {}
    for dataset, label in task_to_label.items():
        if label not in label_to_datasets:
            label_to_datasets[label] = []
        label_to_datasets[label].append(dataset)

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

    for label, datasets in label_to_datasets.items():
        if len(datasets) == 1 and args.finetuning_mode == 'standard':
            dataset = datasets[0]
            print(f"Copying finetuned model for {dataset}")
            src_path = f'/home/cindy2000_sh/TaskVectorBasis/{args.model}/{dataset}Val/finetuned.pt'
            dst_path = f'/home/cindy2000_sh/TaskVectorBasis/{args.model}/{args.exp_name}_checkpoints/centroid_{label}.pt'
            shutil.copyfile(src_path, dst_path)
        else:
            
            print(f"Performing multi-task fine-tuning for label {label}: {datasets}")

            total_epochs = sum([epochs[dataset] for dataset in datasets])

            args.train_datasets = datasets
            args.epochs = total_epochs
            args.lr = 1e-5
            args.warmup_length = 200

            args.batch_size = 64 if args.model == "ViT-L-14" else 128
            args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1

            if args.seed is not None:
                args.save = f"/home/cindy2000_sh/TaskVectorBasis/checkpoints_{args.seed}/{args.model}"
            else:
                args.save = f"/home/cindy2000_sh/TaskVectorBasis/checkpoints_{args.pretrain_openclip_ckpt_name}/{args.model}"
            
            print("=" * 100)
            print(f"Finetuning {args.model} for cluster {label} with datasets: {datasets}")
            print("=" * 100)
            
            torch.multiprocessing.spawn(finetune, args=(label, args,), nprocs=args.world_size)

