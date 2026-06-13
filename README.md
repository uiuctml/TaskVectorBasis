# Task Vector Basis

This is the official repository implementing the task vector **basis** algorithms from
[Task Vector Bases: A Unified and Scalable Framework for Compressed Task Arithmetic](https://openreview.net/forum?id=zkc7u3mIaE).

## Introduction

Task arithmetic, representing downstream tasks through linear operations on task vectors, has emerged as a simple yet powerful paradigm for transferring knowledge across diverse settings. However, maintaining a large collection of task vectors introduces scalability challenges in both storage and computation. We propose Task Vector Bases, a framework compressing $T$ task vectors into $M < T$ basis vectors while preserving the functionality of task arithmetic. By representing each task vector as a structured linear combination of basis atoms, our approach supports standard operations such as addition, negation, as well as more advanced arithmetic ones. The framework is orthogonal to other efficiencyoriented improvements in task arithmetic and can be used in combination with them. We provide theoretical analysis showing that basis compression retains addition generalization guarantees and provides unlearning error bounds that depend on reconstruction quality. Empirically, our proposed basis construction methods consistently outperform heuristic basis construction baselines and, in some cases, even surpass the performance of full task vector collections across diverse downstream applications while reducing storage and computational requirements.

## Installation

```bash
git clone https://github.com/uiuctml/TaskVectorBasis.git
cd TaskVectorBasis
conda env create -n task-vector-basis --file environment.yml
conda activate task-vector-basis
export PYTHONPATH="$PYTHONPATH:$PWD"
```

## Datasets

To download and prepare the vision datasets, follow the instructions in
[this issue](https://github.com/mlfoundations/task_vectors/issues/1) or in
[task_singular_vectors](https://github.com/AntoAndGar/task_singular_vectors).
The task scenarios used in the paper are defined in `src/args.py` (`get_eval_datasets`):

- **8 tasks:** Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SVHN, SUN397
- **14 tasks:** the 8 above + CIFAR100, STL10, Flowers102, OxfordIIITPet, PCAM, FER2013
- **20 tasks:** the 14 above + EMNIST, CIFAR10, Food101, FashionMNIST, RenderedSST2, KMNIST

## Repository structure

```
src/
  args.py                  # CLI arguments and task-scenario dataset lists
  finetune.py              # standard (non-linear) fine-tuning of image encoders
  task_vectors.py          # NonLinearTaskVector and arithmetic
  basis_vectors.py         # basis methods: Autoencoder, PCA, Random, RandomSample
  basis_utils.py           # task-vector discovery / construction helpers
  basis_pipeline.py        # pipeline to build/save a basis and recover task vectors
  eval_single_task.py      # zero-shot / fine-tuned single-task accuracies
  eval_task_addition.py    # multi-task addition via a basis
  eval_task_negation.py    # task negation via a basis
scripts/
  build_ae_basis.py        # build one Autoencoder basis for a model/scenario/seed
  run_single_task.sh       # zero-shot + fine-tuned single-task evaluation
  run_addition.sh          # task addition over all bases of a model/scenario
  run_negation.sh          # task negation over bases of a model/scenario
  load_basis.py            # load a saved basis (local or from the Hugging Face Hub)
```

## Pre-trained bases

Pre-built bases for ViT-B/16, ViT-B/32, and ViT-L/14 (Autoencoder and PCA, for the 8/14/20-task
scenarios) are available on the Hugging Face Hub:

**[cindy2000sh/TaskVectorBasis-checkpoints](https://huggingface.co/cindy2000sh/TaskVectorBasis-checkpoints)**

Download and load one without rebuilding it:

```bash
python scripts/load_basis.py --hf-repo cindy2000sh/TaskVectorBasis-checkpoints --hf-subdir ViT-B-32_AE_M4_8task
```

See the repo card for the naming scheme and a Python loading snippet.

## Workflow

### 1. Fine-tune single-task models

Produces `zeroshot.pt` and one `finetuned.pt` per dataset under
`checkpoints/{model}/{dataset}Val/`.

```bash
python src/finetune.py --model ViT-B-32 --finetuning-mode standard --seed 0
```

### 2. Single-task accuracies (normalization denominators)

Generates `zeroshot_accuracies.json` and `ft_accuracies.json`, used to normalize
task-addition results.

```bash
CUDA_VISIBLE_DEVICES=0 MODEL=ViT-B-32 TASKS=8 bash scripts/run_single_task.sh
```

### 3. Build a task vector basis

Bases are saved under `checkpoints/{model}/_basis_runs/`. To build an Autoencoder basis:

```bash
python scripts/build_ae_basis.py --model ViT-B-32 --task-scenario 8 --seed 0
```

PCA / Random / RandomSample bases can be built through `src/basis_pipeline.py`
(see the `__main__` examples in that file).

### 4. Task addition

```bash
CUDA_VISIBLE_DEVICES=0 MODEL=ViT-B-32 TASKS=8 bash scripts/run_addition.sh
```

### 5. Task negation

```bash
CUDA_VISIBLE_DEVICES=0 MODEL=ViT-B-16 TASKS=8 bash scripts/run_negation.sh
```

## Acknowledgement

This repository builds upon
[tangent_task_arithmetic](https://github.com/gortizji/tangent_task_arithmetic) and
[task_vectors](https://github.com/mlfoundations/task_vectors).

## Reference

If you find this code useful, please cite:

```bibtex
@article{zeng2025task,
  title={Task Vector Bases: A Unified and Scalable Framework for Compressed Task Arithmetic},
  author={Zeng, Siqi and He, Yifei and Liu, Meitong and You, Weiqiu and Hao, Yifan and Tsai, Yao-Hung Hubert and Yamada, Makoto and Zhao, Han},
  journal={arXiv preprint arXiv:2502.01015},
  year={2025}
}
```

## Contact

Feel free to open an Issue or contact siqi6@illinois.edu for any questions or comments.
