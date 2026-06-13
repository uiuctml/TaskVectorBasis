import os
from typing import Dict
from collections import OrderedDict

import torch


from src.modeling import ImageEncoder
from src.task_vectors import NonLinearTaskVector

# ============================================================================
# UTILITY FUNCTIONS (Math operations, Load/Save models, etc.)
# ============================================================================

def _discover_finetuned_ckpts(args, dataset_names):
    """
    Auto-build the mapping {dataset: finetuned_path} for vision experiments:
      ./checkpoints/{args.model}/{DATASET}Val/finetuned.pt
    """
    fname = "finetuned.pt"
    root  = getattr(args, "save", "./checkpoints")
    base  = os.path.join(root, args.model)

    paths = {}
    for ds in dataset_names:
        ds = ds + 'Val'
        p = os.path.join(base, ds, fname)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Expected finetuned checkpoint not found: {p}")
        paths[ds] = p
    return paths

def _build_task_vectors_with_class(args, pretrained_ckpt: str,
                                  finetuned_ckpts: dict) -> tuple[list[dict], list[str]]:
    """
    Uses NonLinearTaskVector to compute each task vector.
    Returns:
      - task_vecs: list of dicts (one per dataset) mapping param key -> delta tensor
      - float_keys: canonical order of keys (taken from the first vector)
    """
    # Construct one task vector per dataset
    tv_objs = []
    for _, ft_path in finetuned_ckpts.items():
        tv = NonLinearTaskVector(pretrained_checkpoint=pretrained_ckpt,
                                finetuned_checkpoint=ft_path)
        tv_objs.append(tv)

    # Extract raw dicts from the class
    task_vecs = [tv.vector for tv in tv_objs]

    # Choose a stable key order from the first vector
    float_keys = [k for k, v in task_vecs[0].items() if v.dtype.is_floating_point]

    return task_vecs, float_keys

def _accum_add_(dst: Dict[str, torch.Tensor], src: Dict[str, torch.Tensor], scale: float = 1.0):
    """dst[k] += scale * src[k] (in place)."""
    for k, t in src.items():
        if k not in dst:
            dst[k] = torch.zeros_like(t)
        dst[k].add_(t, alpha=scale)

@torch.no_grad()
def _save_recovered_image_encoders(args,
                                  pretrained_ckpt: str,
                                  recovered: list[dict[str, torch.Tensor]],
                                  keys: list[str],                    # unused; keep if API needed
                                  dataset_names: list[str],
                                  finetuned_ckpts: dict[str, str],
                                  run_dir: str):
    """
    Save recovered encoders under {run_dir}/{DatasetSubdir}/finetuned*_recovered.pt,
    preserving subfolder structure relative to checkpoints/{model}.
    """

    def _to_statedict(obj):
        sd = obj
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        return sd

    def _load_sd(path):
        obj = torch.load(path, map_location="cpu")
        return _to_statedict(obj)

    def _strip_prefix(sd: dict, prefix: str):
        if not prefix:
            return sd
        plen = len(prefix)
        return { (k[plen:] if k.startswith(prefix) else k): v for k, v in sd.items() }

    def _add_prefix(sd: dict, prefix: str):
        if not prefix:
            return sd
        return { f"{prefix}{k}": v for k, v in sd.items() }

    def _dominant_prefix(keys, candidate):
        """Return True if a majority of keys start with candidate."""
        cnt = sum(1 for k in keys if k.startswith(candidate))
        return cnt > (len(keys) // 2)

    def _align_keys_to_target(src_sd: dict, target_sd: dict) -> dict:
        """
        Align prefixes between src_sd and target_sd.
        Handles 'model.' and 'module.' transparently.
        """
        src_keys = list(src_sd.keys())
        tgt_keys = list(target_sd.keys())
        if not src_keys or not tgt_keys:
            return src_sd

        # Detect dominant prefixes
        src_has_model  = _dominant_prefix(src_keys,  "model.")
        src_has_module = _dominant_prefix(src_keys,  "module.")
        tgt_has_model  = _dominant_prefix(tgt_keys,  "model.")
        tgt_has_module = _dominant_prefix(tgt_keys,  "module.")

        out = src_sd

        # Normalize src by stripping whichever it mostly has
        if src_has_model:
            out = _strip_prefix(out, "model.")
        elif src_has_module:
            out = _strip_prefix(out, "module.")

        # Now add target's dominant prefix back
        if tgt_has_model:
            out = _add_prefix(out, "model.")
        elif tgt_has_module:
            out = _add_prefix(out, "module.")

        return out

    # 1) Load pretrained state dict (as a plain dict)
    pre_sd = _load_sd(pretrained_ckpt)

    # Decide recovered filename
    out_name = "finetuned_recovered.pt"

    for j, ds in enumerate(dataset_names):
        finetuned_path = finetuned_ckpts[ds + "Val"]  # e.g. ./checkpoints/{model}/{ds}Val/finetuned.pt
        subdir = os.path.basename(os.path.dirname(finetuned_path))  # e.g. DTDVal
        out_subdir = os.path.join(run_dir, subdir)
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, out_name)

        # 2) Build new_sd = pre_sd + delta (float tensors only)
        delta = recovered[j]
        # ensure delta is plain dict of tensors
        if hasattr(delta, "state_dict"):
            delta = delta.state_dict()

        new_sd = OrderedDict()
        for k, v in pre_sd.items():
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                d = delta.get(k)
                if d is not None:
                    new_sd[k] = v + d.to(dtype=v.dtype, device=v.device)
                else:
                    new_sd[k] = v.clone()
            else:
                # copy non-float params/buffers as-is
                new_sd[k] = v.clone() if isinstance(v, torch.Tensor) else v

        # 3) Build encoder and align keys to its expected format
        enc = ImageEncoder(args, keep_lang=False)
        target_sd = enc.model.state_dict()  # what the module expects
        aligned_sd = _align_keys_to_target(new_sd, target_sd)

        # Optional: warn about unexpected/missing keys
        missing = [k for k in target_sd.keys() if k not in aligned_sd]
        unexpected = [k for k in aligned_sd.keys() if k not in target_sd]
        if missing:
            print(f"[warn] {len(missing)} missing keys (showing first 5): {missing[:5]}")
        if unexpected:
            print(f"[warn] {len(unexpected)} unexpected keys (showing first 5): {unexpected[:5]}")

        # 4) Load & save
        enc.model.load_state_dict(aligned_sd, strict=False)
        enc.save(out_path)
        print(f"[saved] {ds} -> {out_path}")