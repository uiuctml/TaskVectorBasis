# Task vector basis pipeline:
#   1. extract task vectors from finetuned checkpoints
#   2. train / build a task vector basis and save it to disk (run_pipeline)
#   3. load a saved basis and recover the per-task vectors (load_and_recover_from_saved_basis)
#
# How the eval scripts consume the saved basis:
#   - eval_task_addition.py: loads the basis vectors directly (BasisMethod.load_basis_vectors)
#       and sums them; it does NOT use this module.
#   - eval_task_negation.py: uses load_and_recover_from_saved_basis to reconstruct each
#       per-task vector, then negates it.
# Either way, the basis must first be built by run_pipeline (see scripts/build_ae_basis.py).

import os
from typing import Dict, Tuple, Optional

from src.args import parse_arguments, get_eval_datasets
from src.basis_vectors import BasisMethod, RandomBasisMethod, PCABasisMethod, AutoencoderBasisMethod, RandomSampleMethod

from src.basis_utils import _discover_finetuned_ckpts, _build_task_vectors_with_class

# ============================================================================
# PIPELINE WITH BASIS METHOD SELECTION
# ============================================================================

def run_pipeline(
    pretrained_ckpt: str,
    finetuned_ckpts: Dict[str, str],  # map dataset_name -> finetuned_ckpt_path
    dataset_names: list[str],
    args,
    basis_method: Optional[BasisMethod] = None,
    # For backward compatibility with autoencoder method:
    M: int = 8,
    steps: int = 4000,
    lr: float = 1e-2,
    tau: float = 1.0,
    weight_decay: float = 1e-6,
    grad_clip: float = None,  # 1.0,
    batch_cols: int = None,   # e.g., 512 or 1024 to reduce T^2 temps in loss
    device: str = "cuda",
    seed: int = 0,
    anneal_tau: Tuple[int,float] = None,  # (500, 0.9) every 500 steps, tau *= 0.9 (min 0.1)
    softmax_w: bool = False,  # whether to apply softmax to W columns,
    encoder: str = "softmax",
    w_activation: str = "linear",
):
    """
    Generic pipeline that supports different basis methods and automatically saves basis vectors.
    
    If basis_method is None, defaults to AutoencoderBasisMethod for backward compatibility.
    
    Available basis methods:
    - AutoencoderBasisMethod: Original Gram matrix approach (default)
    - PCABasisMethod: Simple PCA decomposition (storage-efficient)
    - RandomBasisMethod: Random orthogonal basis (storage-efficient)

    IMPORTANT FEATURE: This pipeline now automatically saves:
    1. All method-specific artifacts (A/W/B matrices, PCA coefficients, etc.)
    2. The generated basis vectors themselves (basis_vectors.pt)
    3. Method configuration and metadata (method_info.json)
    
    This enables you to later call recover_task_vectors_from_basis() directly
    without having to rebuild the basis vectors, saving significant time!
    
    Usage patterns:
    
    # Pattern 1: Build and save basis (run once)
    run_dir = run_pipeline(pretrained_ckpt, finetuned_ckpts, dataset_names, args, basis_method=PCABasisMethod(M=4))
    
    # Pattern 2: Load and reuse saved basis (run many times for evaluation)
    recovered = load_and_recover_from_saved_basis(run_dir, pretrained_ckpt, finetuned_ckpts, dataset_names, args)
    
    Returns:
        run_dir: Path to the saved artifacts directory
    """
    
    # 1) Build task vectors
    T = len(dataset_names)
    task_vecs, float_keys = _build_task_vectors_with_class(args, pretrained_ckpt, finetuned_ckpts)
   
    # 2) Use specified basis method or default to autoencoder
    if basis_method is None:
        basis_method = AutoencoderBasisMethod(
            M=M, steps=steps, lr=lr, tau=tau, weight_decay=weight_decay,
            grad_clip=grad_clip, batch_cols=batch_cols, device=device,
            seed=seed, anneal_tau=anneal_tau, softmax_w=softmax_w,
            encoder=encoder, w_activation=w_activation
        )
    
    print(f"Using basis method: {basis_method.__class__.__name__}")
    
    # 3) Build basis vectors
    basis = basis_method.build_basis_vectors(task_vecs, float_keys)
    
    # 4) Recover task vectors
    # recovered = basis_method.recover_task_vectors_from_basis(basis, float_keys)
    
    # 5) Save training artifacts (now unified across all methods)
    print("Saving training artifacts ...")
    run_id, run_dir = basis_method.save_training_artifacts(
        checkpoints_root=getattr(args, "save", "./checkpoints"),
        model=args.model,
        dataset_names=dataset_names,
        M=M,
        seed=seed,
        basis_vectors=basis,  # Pass the generated basis vectors for saving
        finetuning_mode=args.finetuning_mode,
        task_scenario=args.task_scenario,  # Include task scenario info in directory name
        # Include any additional method-specific parameters
        **{k: v for k, v in locals().items() 
           if k in ['steps', 'lr', 'tau', 'weight_decay', 'grad_clip', 'batch_cols', 'device', 'anneal_tau', 'softmax_w', 'encoder', 'w_activation']}
    )

    # 6) Save encoders - DISABLED for storage efficiency
    # Instead of saving individual finetuned_recovered.pt files for each dataset,
    # we rely on the saved basis vectors and recovery method to reconstruct task vectors on-demand
    # This saves significant storage space (typically GB per experiment)
    print("Skipping individual encoder saves for storage efficiency...")
    print("Task vectors can be reconstructed on-demand using saved basis artifacts.")
    
    # ORIGINAL CODE (now disabled):
    # print("Saving recovered ImageEncoders ...")
    # _save_recovered_image_encoders(args, pretrained_ckpt,
    #                                 recovered, float_keys,
    #                                 dataset_names, finetuned_ckpts,
    #                                 run_dir)

    print("Done.")
    return run_dir

# ============================================================================
# UTILITY FUNCTIONS FOR LOADING AND USING PRE-BUILT BASIS
# ============================================================================

def load_and_recover_from_saved_basis(run_dir: str) -> list[Dict]:
    """
    Load a pre-built basis method and use it to recover task vectors
    without having to run build_basis_vectors again.
    
    This is the efficient approach for evaluation: load all task vectors at once
    rather than reconstructing them individually.
    
    Args:
        run_dir: Directory containing saved basis method artifacts
        pretrained_ckpt: Path to pretrained checkpoint 
        finetuned_ckpts: Mapping of dataset names to finetuned checkpoints
        dataset_names: List of dataset names
        args: Arguments object
        
    Returns:
        List of recovered task vectors (one per dataset, in same order as dataset_names)
    """
    print(f"Loading basis method from: {run_dir}")
    
    # 1) Load the basis method with all its trained parameters
    basis_method = BasisMethod.load_method_from_artifacts(run_dir)
    print(f"Loaded: {basis_method.__class__.__name__}")
    
    # 2) Load the pre-built basis vectors
    basis_vectors = BasisMethod.load_basis_vectors(run_dir)
    print(f"Loaded {len(basis_vectors)} basis vectors")
    
    # 3) Extract parameter keys from the basis vectors (no need to rebuild task vectors!)
    float_keys = [k for k, v in basis_vectors[0].items() if v.dtype.is_floating_point]
    print(f"Extracted {len(float_keys)} parameter keys from basis vectors")
    
    # 4) Recover all task vectors using the saved basis (efficient batch operation)
    print("Recovering all task vectors from saved basis...")
    recovered = basis_method.recover_task_vectors_from_basis(basis_vectors, float_keys)
    
    print("Recovery completed!")
    return recovered


# === Usage with different basis methods =========================
if __name__ == "__main__":
    args = parse_arguments()
    if getattr(args, 'seed', None) is not None:
        args.save = f"./checkpoints_{args.seed}"
    else:
        args.save = "./checkpoints"

    # Get training datasets based on task scenario
    train_datasets = get_eval_datasets(args.task_scenario)
    print(f"Using vision experiment with {args.task_scenario}-task scenario")
    print(f"Training datasets ({len(train_datasets)}): {train_datasets}")

    # auto-discover finetuned paths under ./checkpoints/{model}/{ds}/...
    finetuned_ckpts = _discover_finetuned_ckpts(args, train_datasets)

    pretrained_ckpt = os.path.join(args.save, args.model, "zeroshot.pt")

    
    for m in range(len(train_datasets)//4, len(train_datasets)//4+1):

        # Example 1: Default autoencoder method (backward compatible)
        print("=== Running with default AutoencoderBasisMethod ===")
        autoencoder_method = AutoencoderBasisMethod(M=m, steps=20000, lr=1e-2, tau=5.0, device="cuda", seed=args.seed)
        run_pipeline(
            pretrained_ckpt = pretrained_ckpt,
            finetuned_ckpts = finetuned_ckpts,
            dataset_names   = train_datasets,
            args            = args,
            basis_method    = autoencoder_method,
            M=m, steps=20000, lr=1e-2, tau=5.0, device="cuda",
            seed=args.seed,
        )

        # Example 2: PCA-based method
        print("\n=== Running with PCABasisMethod ===")
        pca_method = PCABasisMethod(M=m)
        run_pipeline(
            pretrained_ckpt = pretrained_ckpt,
            finetuned_ckpts = finetuned_ckpts,
            dataset_names   = train_datasets,
            args            = args,
            basis_method    = pca_method,
            M=m,
            seed=args.seed,
        )

        # Example 3: Random orthogonal basis method
        print("\n=== Running with RandomBasisMethod ===")
        random_method = RandomBasisMethod(M=m, seed=args.seed)
        run_pipeline(
            pretrained_ckpt = pretrained_ckpt,
            finetuned_ckpts = finetuned_ckpts,
            dataset_names   = train_datasets,
            args            = args,
            basis_method    = random_method,
            M=m,
            seed=args.seed,
        )

        # Example 4: Random sample method
        print("\n=== Running with RandomSampleMethod ===")
        random_sample_method = RandomSampleMethod(M=m, seed=args.seed)
        run_pipeline(
            pretrained_ckpt = pretrained_ckpt,
            finetuned_ckpts = finetuned_ckpts,
            dataset_names   = train_datasets,
            args            = args,
            basis_method    = random_sample_method,
            M=m,
            seed=args.seed,
        )

