"""Load a saved task vector basis and use it for addition / negation.

A basis directory contains:
    basis_vectors.pt     # the M basis vectors
    method_info.json     # method + hyperparameters
    <method artifact>    # AWB.pt (Autoencoder) or pca_components.pt (PCA), etc.

Usage:
    # from a local directory
    python scripts/load_basis.py --basis-dir checkpoints/ViT-B-32/_basis_runs/<name>

    # download from the Hugging Face Hub first, then load
    python scripts/load_basis.py --hf-repo uiuctml/TaskVectorBasis --hf-subdir ViT-B-32_AE_M4_8task
"""
import argparse
import os

from src.basis_vectors import BasisMethod
from src.task_vectors import NonLinearTaskVector
from src.basis_pipeline import load_and_recover_from_saved_basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis-dir", help="local path to a saved basis run directory")
    ap.add_argument("--hf-repo", help="Hugging Face repo id to download the basis from (optional)")
    ap.add_argument("--hf-subdir", help="subfolder inside the HF repo holding the basis (optional)")
    args = ap.parse_args()

    basis_dir = args.basis_dir
    if args.hf_repo:
        from huggingface_hub import snapshot_download
        patterns = [f"{args.hf_subdir}/*"] if args.hf_subdir else None
        local = snapshot_download(repo_id=args.hf_repo, allow_patterns=patterns)
        basis_dir = os.path.join(local, args.hf_subdir) if args.hf_subdir else local

    assert basis_dir, "Provide --basis-dir or --hf-repo (+ --hf-subdir)."
    print(f"Loading basis from: {basis_dir}")

    # (1) Task addition: load the M basis vectors and sum them into one merged task vector.
    basis_vectors = BasisMethod.load_basis_vectors(basis_dir)
    merged = sum(NonLinearTaskVector(vector=bv) for bv in basis_vectors)
    print(f"[addition] loaded {len(basis_vectors)} basis vectors -> merged task vector "
          f"with {len(merged.vector)} parameters")
    # Apply to a pretrained encoder with, e.g.:
    #   image_encoder = merged.apply_to('checkpoints/ViT-B-32/zeroshot.pt', scaling_coef=0.4)

    # (2) Task negation: reconstruct the per-task vectors from the basis.
    recovered = load_and_recover_from_saved_basis(run_dir=basis_dir)
    print(f"[negation] recovered {len(recovered)} per-task vectors")
    # Negate one task with, e.g.:
    #   neg = -NonLinearTaskVector(vector=recovered[0])


if __name__ == "__main__":
    main()
