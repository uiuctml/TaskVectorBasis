"""Build a single AE basis for a given model, task scenario, and seed.

Usage:
    python scripts/build_ae_basis.py --model ViT-B-16 --task-scenario 8 --seed 0 --finetuning-mode standard
"""
import os

from src.args import parse_arguments, get_eval_datasets
from src.basis_vectors import AutoencoderBasisMethod
from src.basis_utils import _discover_finetuned_ckpts
from src.basis_pipeline import run_pipeline

args = parse_arguments()
args.save = "./checkpoints"

train_datasets = get_eval_datasets(args.task_scenario)
print(f"Building AE basis: model={args.model}, task_scenario={args.task_scenario}, seed={args.seed}")
print(f"Datasets ({len(train_datasets)}): {train_datasets}")

finetuned_ckpts = _discover_finetuned_ckpts(args, train_datasets)

pretrained_ckpt = os.path.join(args.save, args.model, "zeroshot.pt")

# M = T // 2 (matches existing convention)
M = len(train_datasets) // 2
steps = 6000
lr = 0.01
tau = 5.0

print(f"M={M}, steps={steps}, lr={lr}, tau={tau}, seed={args.seed}")

method = AutoencoderBasisMethod(M=M, steps=steps, lr=lr, tau=tau, device="cuda", seed=args.seed)

run_dir = run_pipeline(
    pretrained_ckpt=pretrained_ckpt,
    finetuned_ckpts=finetuned_ckpts,
    dataset_names=train_datasets,
    args=args,
    basis_method=method,
    M=M, steps=steps, lr=lr, tau=tau, device="cuda",
    seed=args.seed,
)

print(f"Basis saved to: {run_dir}")
