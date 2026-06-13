import time, json, os
from dataclasses import asdict, dataclass
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

import torch

from src.basis_utils import _accum_add_

# === Reorganized Basis Methods Framework ====================================
#
# STRUCTURE:
# 1. ABSTRACT BASIS METHOD CLASS
#    - BasisMethod: Abstract base class requiring:
#      * build_basis_vectors() and recover_task_vectors_from_basis() (abstract)
#      * save_training_artifacts() (default implementation, can be overridden)
#
# 2. AUTOENCODER-LIKE BASIS METHOD (Original Gram Matrix Approach)
#    - AutoencoderBasisMethod: Implements the original approach
#    - GramRunConfig: Configuration for Gram matrix training
#    - GramTrainer: Training logic for Gram matrix
#    - Related functions: build_gram_from_vectors, _build_basis_vectors_impl,
#      _recover_task_vectors_impl, _make_run_id, save_awb_snapshot
#    - Overrides save_training_artifacts() for Gram-specific artifacts
#
# 3. EXAMPLE BASIS METHODS
#    - PCABasisMethod: Simple PCA decomposition approach
#    - RandomBasisMethod: Random orthogonal basis for comparison
#    - All override save_training_artifacts() to save method-specific data
#
# 4. UTILITY FUNCTIONS
#    - Mathematical operations, load/save models, etc.
#
# 5. GENERIC PIPELINE
#    - run_pipeline(): Generic function supporting any BasisMethod
#    - Uses unified save_training_artifacts() interface for all methods
#    - Backward compatible: defaults to AutoencoderBasisMethod if no method specified
#
# TO ADD A NEW BASIS METHOD:
# 1. Inherit from BasisMethod
# 2. Implement build_basis_vectors() and recover_task_vectors_from_basis()
# 3. Optionally override save_training_artifacts() for custom artifacts
# 4. Pass your method instance to run_pipeline(basis_method=your_method)
#
# EXAMPLE:
# class MyMethod(BasisMethod):
#     def build_basis_vectors(self, task_vecs, keys, **kwargs):
#         # Your basis building logic
#         return basis_vectors
#     
#     def recover_task_vectors_from_basis(self, basis, keys, **kwargs):
#         # Your recovery logic  
#         return recovered_vectors
#     
#     def save_training_artifacts(self, checkpoints_root, model, dataset_names, M, seed=0, **kwargs):
#         # Optional: custom artifact saving
#         run_id, run_dir = super().save_training_artifacts(checkpoints_root, model, dataset_names, M, seed, **kwargs)
#         # Save your custom artifacts here
#         return run_id, run_dir
#
# # Usage:
# my_method = MyMethod(your_params)
# run_pipeline(..., basis_method=my_method)
#
# ============================================================================

class BasisMethod(ABC):
    """Abstract base class for task vector basis methods."""
    
    @abstractmethod
    def build_basis_vectors(self, 
                           task_vecs: List[Dict[str, torch.Tensor]], 
                           keys: List[str],
                           **kwargs) -> List[Dict[str, torch.Tensor]]:
        """
        Build basis vectors from task vectors.
        
        Args:
            task_vecs: List of task vectors (each as dict of parameter name -> tensor)
            keys: List of parameter keys to process
            **kwargs: Method-specific parameters
            
        Returns:
            List of M basis vectors (each as dict of parameter name -> tensor)
        """
        pass
    
    @abstractmethod
    def recover_task_vectors_from_basis(self,
                                       basis: List[Dict[str, torch.Tensor]],
                                       keys: List[str],
                                       **kwargs) -> List[Dict[str, torch.Tensor]]:
        """
        Recover task vectors from basis vectors.
        
        Args:
            basis: List of basis vectors (each as dict of parameter name -> tensor)
            keys: List of parameter keys to process
            **kwargs: Method-specific parameters (e.g., reconstruction coefficients)
            
        Returns:
            List of T recovered task vectors (each as dict of parameter name -> tensor)
        """
        pass
    
    def get_basis_weights(self, dataset_names: List[str], use_uniform: bool = False) -> torch.Tensor:
        """
        Get the basis weights matrix for given datasets.
        
        Args:
            dataset_names: List of dataset names in order
            use_uniform: If True, use uniform weights instead of method-specific weights
            
        Returns:
            torch.Tensor: Basis weights matrix of shape (M, T) where M is number of basis vectors
                         and T is number of tasks/datasets. Entry (i,j) represents the weight of 
                         basis vector i for task j.
        """
        if use_uniform:
            # Get the number of basis vectors from the method-specific implementation
            method_weights = self._get_method_specific_weights(dataset_names)
            M, T = method_weights.shape
            
            # Create uniform weights: each task gets equal weight (1/T) from each basis vector
            uniform_weight = 1.0 / T
            weights = torch.full((M, T), uniform_weight, device=method_weights.device, dtype=method_weights.dtype)
            print(f"Using uniform weights ({uniform_weight:.3f}) for {self.__class__.__name__}")
            return weights
        else:
            return self._get_method_specific_weights(dataset_names)
    
    @abstractmethod
    def _get_method_specific_weights(self, dataset_names: List[str]) -> torch.Tensor:
        """
        Get method-specific basis weights. This replaces the old get_basis_weights method.
        
        Args:
            dataset_names: List of dataset names in order
            
        Returns:
            torch.Tensor: Basis weights matrix of shape (M, T)
        """
        pass
    
    def save_training_artifacts(self, 
                               checkpoints_root: str, 
                               model: str, 
                               dataset_names: List[str],
                               M: int,
                               seed: int = 0,
                               basis_vectors: Optional[List[Dict[str, torch.Tensor]]] = None,
                               **kwargs) -> Tuple[str, str]:
        """
        Save training artifacts and method configuration.
        
        Args:
            checkpoints_root: Root directory for checkpoints
            model: Model name
            dataset_names: List of dataset names used
            M: Number of basis vectors
            seed: Random seed used
            basis_vectors: The basis vectors to save (optional)
            **kwargs: Method-specific parameters
            
        Returns:
            Tuple of (run_id, run_dir)
        """
        # Default implementation for simple methods
        import time
        import os
        import json
        method_name = self.__class__.__name__
        finetuning_mode = kwargs.get('finetuning_mode', 'standard')
        
        # Include task scenario in run_id if provided
        task_scenario = kwargs.get('task_scenario', None)
        if task_scenario is not None:
            run_id = f"{method_name}_{finetuning_mode}_M{M}_seed{seed}_{task_scenario}task_{int(time.time())}"
        else:
            run_id = f"{method_name}_{finetuning_mode}_M{M}_seed{seed}_{int(time.time())}"
            
        run_dir = os.path.join(checkpoints_root, model, "_basis_runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Save basic info about the method
        info = {
            "method": method_name,
            "M": M,
            "seed": seed,
            "datasets": dataset_names,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **kwargs  # Include any additional method-specific parameters
        }
        with open(os.path.join(run_dir, "method_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        
        # Save basis vectors if provided
        if basis_vectors is not None:
            basis_path = os.path.join(run_dir, "basis_vectors.pt")
            torch.save({
                "basis_vectors": basis_vectors,
                "num_basis": len(basis_vectors),
                "method": method_name,
            }, basis_path)
            print(f"[Basis vectors saved] {basis_path}")
        
        print(f"[Method artifacts saved] {run_dir}")
        return run_id, run_dir

    @staticmethod
    def load_basis_vectors(run_dir: str) -> List[Dict[str, torch.Tensor]]:
        """Load saved basis vectors from artifacts directory."""
        basis_path = os.path.join(run_dir, "basis_vectors.pt")
        if not os.path.exists(basis_path):
            raise FileNotFoundError(f"Basis vectors not found: {basis_path}")
        
        data = torch.load(basis_path, map_location='cpu')
        return data["basis_vectors"]
    
    @staticmethod  
    def load_method_from_artifacts(run_dir: str, **kwargs):
        """Load a basis method instance from saved artifacts with all parameters."""
        # Load method info
        info_path = os.path.join(run_dir, "method_info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Method info not found: {info_path}")
        
        with open(info_path, "r") as f:
            info = json.load(f)
        
        method_name = info["method"]
        
        # Create the appropriate method instance
        if method_name == "AutoencoderBasisMethod":
            # For autoencoder, we need to load the trained matrices
            awb_files = [f for f in os.listdir(run_dir) if f.startswith("AWB") and f.endswith(".pt")]
            if not awb_files:
                raise FileNotFoundError(f"No AWB file found in {run_dir}")
            
            awb_path = os.path.join(run_dir, awb_files[0])
            awb_data = torch.load(awb_path, map_location='cpu')
            metadata = awb_data["metadata"]
            
            method = AutoencoderBasisMethod(
                M=metadata["M"],
                steps=metadata["steps"],
                lr=metadata["lr"],
                tau=metadata["tau_init"],
                weight_decay=metadata["weight_decay"],
                grad_clip=metadata["grad_clip"],
                batch_cols=metadata["batch_cols"],
                device=metadata["device"],
                seed=metadata["seed"],
                anneal_tau=(metadata["anneal_every"], metadata["anneal_factor"]) if metadata["anneal_every"] else None,
                softmax_w=metadata["softmax_w"]
            )
            # Restore trained parameters
            method.A = awb_data["A"]
            method.W = awb_data["W"] 
            method.B = awb_data["B"]
            method.tau_final = metadata.get("tau_final", metadata["tau_init"])
            
        elif method_name == "PCABasisMethod":
            method = PCABasisMethod(M=info["M"])
            # Load PCA-specific data
            pca_path = os.path.join(run_dir, "pca_components.pt")
            if os.path.exists(pca_path):
                pca_data = torch.load(pca_path, map_location='cpu')
                method.pca_coefficients = pca_data["coefficients"]
                method.mean_vector = pca_data["mean_vector"]
                method.singular_values = pca_data["singular_values"]
                method.total_dim = pca_data["total_dim"]
                method.keys_order = pca_data["keys_order"]
                method.param_shapes = pca_data["param_shapes"]
                
        elif method_name == "RandomBasisMethod":
            method = RandomBasisMethod(M=info["M"], seed=info["seed"])
            # Load random basis specific data
            random_path = os.path.join(run_dir, "random_basis.pt")
            if os.path.exists(random_path):
                random_data = torch.load(random_path, map_location='cpu')
                method.projection_coefficients = random_data["projection_coefficients"]
                method.total_dim = random_data["total_dim"]
                method.keys_order = random_data["keys_order"]
                method.param_shapes = random_data["param_shapes"]
                
        elif method_name == "RandomSampleMethod":
            method = RandomSampleMethod(M=info["M"], seed=info["seed"])
            # Load random selection info
            random_path = os.path.join(run_dir, "random_selection.json")
            if os.path.exists(random_path):
                with open(random_path, "r") as f:
                    selection_data = json.load(f)
                method.selected_indices = selection_data["selected_indices"]
                
        else:
            raise ValueError(f"Unknown method type: {method_name}")
        
        return method

@dataclass
class GramRunConfig:
    # model/data
    model: str
    finetuning_mode: str                # "standard" or "linear"
    datasets: list                      # list of dataset names (T order)
    # training hyperparams
    M: int
    steps: int
    lr: float
    tau_init: float
    anneal_every: int | None
    anneal_factor: float | None
    weight_decay: float
    grad_clip: float | None
    batch_cols: int | None
    device: str
    seed: int
    softmax_w: bool = False             # kept for backward-compat (maps to w_activation="softmax")
    # gram details
    encoder: str = "softmax"            # "softmax" or "linear" for A -> B
    w_activation: str = "linear"        # "linear" or "softmax"
    gram_dtype: str = "float32"
    notes: str | None = None
    task_scenario: int | None = None    # Task scenario (8, 14, 20) for directory naming


class GramTrainer:
    """Training on Gram matrix with ONE optimizer (pure SGD/Adam on A and W)."""

    def __init__(self, G: torch.Tensor, T: int, M: int,
                 tau: float = 1.0, device: str = "cuda",
                 lr: float = 1e-2, weight_decay: float = 1e-6,
                 grad_clip: float = 1.0, batch_cols: int | None = None, seed: int = 0,
                 softmax_w: bool = False,
                 encoder: str = "softmax",            # "softmax" | "linear"
                 w_activation: str = "linear"         # "linear" | "softmax"
                 ):
        torch.manual_seed(seed)
        self.device = device
        self.T, self.M = T, M
        self.tau = tau
        self.G = G.to(device)
        self.A = torch.randn(T, M, device=device, requires_grad=True)
        self.W = torch.zeros(M, T, device=device, requires_grad=True)
        self.grad_clip = grad_clip
        self.batch_cols = batch_cols

        if softmax_w:  # backward-compat switch
            w_activation = "softmax"

        self.encoder = encoder.lower()
        self.w_activation = w_activation.lower()
        assert self.encoder in {"softmax", "linear"}
        assert self.w_activation in {"linear", "softmax"}

        self.opt = torch.optim.Adam(
            [{"params": [self.A, self.W], "lr": lr, "weight_decay": weight_decay}]
        )

    def _encode_B(self):
        Z = self.A / self.tau
        if self.encoder == "softmax":
            return torch.softmax(Z, dim=0)     # (T, M), column-wise
        else:  # linear
            return Z                            # (T, M), no activation

    def _activate_W(self):
        if self.w_activation == "linear":
            return self.W                       # (M, T)
        Z = self.W / self.tau
        return torch.softmax(Z, dim=0)          # (M, T), column-wise (over rows M)

    def _loss(self):
        B = self._encode_B()               # (T, M)
        Wp = self._activate_W()            # (M, T)

        if self.batch_cols is None:
            E  = B @ Wp - torch.eye(self.T, device=self.device)  # (T, T)
            GE = self.G @ E
            loss = torch.sum(E * GE) / (self.T * self.T)
        else:
            J   = torch.randint(0, self.T, (self.batch_cols,), device=self.device)
            EJ  = B @ Wp[:, J] - torch.eye(self.T, device=self.device)[:, J]  # (T, b)
            GEJ = self.G @ EJ
            loss = torch.sum(EJ * GEJ) / (self.T * self.batch_cols)
        return loss

    def _full_loss(self):
        """Compute full (non-minibatched) Gram loss with the SAME activations & current τ."""
        B  = self._encode_B()        # (T, M)
        Wp = self._activate_W()      # (M, T)
        E  = B @ Wp - torch.eye(self.T, device=self.device)            # (T, T)
        GE = self.G @ E
        return torch.sum(E * GE) / (self.T * self.T)

    def train(self,
              steps: int = 4000,
              log_every: int = 200,
              anneal_tau: Optional[Tuple[int, float]] = None):
        """
        anneal_tau: (every_n_steps, factor), e.g. (500, 0.9) to sharpen softmax over time.
        """
        for t in range(1, steps + 1):
            loss = self._loss()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([self.A, self.W], self.grad_clip)
            self.opt.step()

            # τ-annealing
            if anneal_tau and (t % anneal_tau[0] == 0):
                # keep a small positive floor so logits scaling doesn't explode
                self.tau = max(0.1, self.tau * anneal_tau[1])

            # logging (batch loss + full loss)
            if (t % log_every) == 0:
                with torch.no_grad():
                    full = self._full_loss()
                print(f"[{t:5d}] loss(batch)={loss.item():.6f}  loss(full)={float(full):.6f}  tau={self.tau:.3f}")

        return self.A.detach().cpu(), self.W.detach().cpu(), self.tau

class AutoencoderBasisMethod(BasisMethod):
    """Autoencoder-like basis method using Gram matrix training."""
    
    def __init__(self, 
                 M: int = 8,
                 steps: int = 4000,
                 lr: float = 1e-2,
                 tau: float = 1.0,
                 weight_decay: float = 1e-6,
                 grad_clip: Optional[float] = None,
                 batch_cols: Optional[int] = None,
                 device: str = "cuda",
                 seed: int = 0,
                 anneal_tau: Optional[Tuple[int, float]] = None,
                 softmax_w: bool = False,
                 encoder: str = "softmax", 
                 w_activation: str = "linear",):
        self.M = M
        self.steps = steps
        self.lr = lr
        self.tau = tau
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.batch_cols = batch_cols
        self.device = device
        self.seed = seed
        self.anneal_tau = anneal_tau
        self.softmax_w = softmax_w
        self.w_activation = w_activation
        self.encoder = encoder
        
        # Will store training results
        self.A = None
        self.W = None
        self.B = None
        self.tau_final = None
    
    def build_basis_vectors(self, 
                           task_vecs: List[Dict[str, torch.Tensor]], 
                           keys: List[str],
                           **kwargs) -> List[Dict[str, torch.Tensor]]:
        """Build basis vectors using Gram matrix training."""
        T = len(task_vecs)
        
        # Build Gram matrix
        print("Building Gram matrix G = V^T V ...")
        G = self._build_gram_from_vectors(task_vecs, keys, dtype=torch.float32)
        print(f"Gram built: shape={tuple(G.shape)}, dtype={G.dtype}")

        # G is our Gram (float32 on CPU/GPU)
        evals = torch.linalg.eigvalsh(G.cpu())  # ascending
        evals = torch.relu(evals)               # guard tiny negatives
        evals_sorted = torch.flip(evals, dims=[0])  # descending

        for M_try in [self.M, max(1,self.M-1), self.M+1, min(len(evals_sorted), 2*self.M)]:
            lb = evals_sorted[M_try:].sum().item() / (G.shape[0]**2)
            print(f"M={M_try:3d}  spectral lower bound ~ {lb:.6f}")

        # Train on Gram matrix
        print("Training A,W on Gram with one optimizer ...")
        trainer = GramTrainer(G, T=T, M=self.M, tau=self.tau, device=self.device, 
                             lr=self.lr, weight_decay=self.weight_decay, 
                             grad_clip=self.grad_clip, batch_cols=self.batch_cols, 
                             seed=self.seed, softmax_w=self.softmax_w, encoder=self.encoder,
                             w_activation = self.w_activation)
        self.A, self.W, self.tau_final = trainer.train(steps=self.steps, log_every=200,
                                                       anneal_tau=self.anneal_tau)
        # Apply same encoder as training to get final B
        if self.encoder == "softmax":
            self.B = torch.softmax(self.A / self.tau_final, dim=0)  # (T,M)
        else:  # linear
            self.B = self.A / self.tau_final                        # (T,M)
        
        # Build basis vectors
        print("Forming basis S = V B ...")
        basis = self._build_basis_vectors_impl(task_vecs, keys, self.B)
        return basis
    
    def recover_task_vectors_from_basis(self,
                                       basis: List[Dict[str, torch.Tensor]],
                                       keys: List[str],
                                       **kwargs) -> List[Dict[str, torch.Tensor]]:
        """Recover task vectors from basis using learned W matrix."""
        if self.W is None:
            raise ValueError("Must call build_basis_vectors first to train W matrix")
        
        # Apply softmax to final W if enabled
        W_final = torch.softmax(self.W / self.tau_final, dim=0) if self.softmax_w else self.W
        
        print("Recovering vectors Vhat = S W ...")
        recovered = self._recover_task_vectors_impl(basis, keys, W_final)
        return recovered
    
    def save_training_artifacts(self, 
                               checkpoints_root: str, 
                               model: str, 
                               dataset_names: List[str],
                               M: int,
                               seed: int = 0,
                               basis_vectors: Optional[List[Dict[str, torch.Tensor]]] = None,
                               **kwargs) -> Tuple[str, str]:
        """Save A, W, B matrices and Gram-specific config."""
        if self.A is None or self.W is None or self.B is None:
            raise ValueError("Must call build_basis_vectors first")
        
        # Build config for snapshot
        cfg = GramRunConfig(
            model=model,
            finetuning_mode=kwargs.get('finetuning_mode', 'standard'),
            datasets=dataset_names,
            M=M,
            steps=self.steps,
            lr=self.lr,
            tau_init=self.tau,
            anneal_every=self.anneal_tau[0] if self.anneal_tau else None,
            anneal_factor=self.anneal_tau[1] if self.anneal_tau else None,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            batch_cols=self.batch_cols,
            device=self.device,
            seed=seed,
            softmax_w=self.softmax_w,
            gram_dtype="float32",
            notes=kwargs.get('notes', None),
            encoder=self.encoder,
            w_activation=self.w_activation,
            task_scenario=kwargs.get('task_scenario', None),
        )
        
        W_final = torch.softmax(self.W / self.tau_final, dim=0) if self.softmax_w else self.W
        run_id, run_dir = self._save_awb_snapshot(self.A, W_final, self.B, cfg, checkpoints_root, model)
        
        # Save basis vectors if provided
        if basis_vectors is not None:
            basis_path = os.path.join(run_dir, "basis_vectors.pt")
            torch.save({
                "basis_vectors": basis_vectors,
                "num_basis": len(basis_vectors),
                "method": "AutoencoderBasisMethod",
            }, basis_path)
            print(f"[Basis vectors saved] {basis_path}")
        
        return run_id, run_dir
    
    @torch.no_grad()
    def _build_gram_from_vectors(self, task_vecs: List[Dict[str, torch.Tensor]],
                                keys: List[str],
                                chunk_rows: int = 2_000_000,
                                dtype=torch.float32) -> torch.Tensor:
        """
        G = V^T V where columns of V are flattened task vectors.
        Never materializes full V; accumulates per-parameter (and per-chunk if needed).
        """
        T = len(task_vecs)
        G = torch.zeros((T, T), dtype=dtype)  # stays on CPU; move to GPU later if you want
        for k in keys:
            # shape: (numel_k, T) but chunked along numel_k
            numel = task_vecs[0][k].numel()
            # plan chunks
            rows_done = 0
            while rows_done < numel:
                rows = min(chunk_rows, numel - rows_done)
                X = torch.stack([tv[k].flatten()[rows_done:rows_done+rows] for tv in task_vecs], dim=1)  # (rows, T)
                G.add_(X.t().mm(X))  # (T,T) += (T,rows)@(rows,T)
                rows_done += rows
                del X
        return G

    @torch.no_grad()
    def _build_basis_vectors_impl(self, task_vecs: List[Dict[str, torch.Tensor]],
                                keys: List[str],
                                B: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Returns a list of M basis vectors (each a dict[k]->tensor with original shapes),
        computed as convex combos of the T task vectors: s_m = sum_j B[j,m] * v_j.
        """
        T, M = B.shape
        assert T == len(task_vecs)
        # init zero dicts
        basis = [ {} for _ in range(M) ]
        for m in range(M):
            for k in keys:
                basis[m][k] = torch.zeros_like(task_vecs[0][k])
        # accumulate
        for j in range(T):
            wj = B[j, :]   # (M,)
            tvj = task_vecs[j]
            for m in range(M):
                w = float(wj[m])
                if w == 0.0: 
                    continue
                _accum_add_(basis[m], tvj, scale=w)
        return basis  # list length M

    @torch.no_grad()
    def _recover_task_vectors_impl(self, basis: List[Dict[str, torch.Tensor]],
                                keys: List[str],
                                W: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Given M bases and W (M x T), form T recovered task vectors: vhat_j = sum_m W[m,j] * s_m.
        """
        M, T = W.shape
        rec = [ {} for _ in range(T) ]
        # init zeros
        for j in range(T):
            rec[j] = {k: torch.zeros_like(basis[0][k]) for k in keys}
        # accumulate
        for m in range(M):
            sm = basis[m]
            wm = W[m, :]   # (T,)
            for j in range(T):
                coeff = float(wm[j])
                if coeff == 0.0:
                    continue
                _accum_add_(rec[j], sm, scale=coeff)
        return rec  # list length T

    def _make_run_id(self, cfg: GramRunConfig) -> str:
        """
        Compact, filename-safe id that captures mode + core hparams.
        Example: AutoencoderBasisMethod_linear_M6_lr1e-2_tau1.0_steps4k_bc512_seed0_smW_8task_1756595359
        """
        s_steps = f"{cfg.steps//1000}k" if cfg.steps % 1000 == 0 else str(cfg.steps)
        bc = f"bc{cfg.batch_cols}" if cfg.batch_cols else "bcFull"
        mode = "linear" if cfg.finetuning_mode=="linear" else "standard"
        softmax_w_suffix = "_smW" if cfg.softmax_w else ""
        task_scenario_suffix = f"_{cfg.task_scenario}task" if cfg.task_scenario is not None else ""
        timestamp = int(time.time())
        return f"AutoencoderBasisMethod_{mode}_M{cfg.M}_lr{cfg.lr:g}_tau{cfg.tau_init:g}_steps{s_steps}_{bc}_seed{cfg.seed}{softmax_w_suffix}{task_scenario_suffix}_{timestamp}"

    def _save_awb_snapshot(self, A, W, B, cfg: GramRunConfig,
                        checkpoints_root: str,
                        model: str) -> tuple[str,str]:
        """
        Save AWB (A,W,B + metadata) under ./checkpoints/{model}/_basis_runs/{run_id}/.
        Returns (run_id, run_dir).
        """
        run_id = self._make_run_id(cfg)
        run_dir = os.path.join(checkpoints_root, model, "_basis_runs", run_id)
        os.makedirs(run_dir, exist_ok=True)

        filename = f"AWB_{run_id}.pt"
        save_path = os.path.join(run_dir, filename)

        payload = {"A": A, "W": W, "B": B,
                "metadata": asdict(cfg) | {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }}
        payload["metadata"]['method'] = 'AutoencoderBasisMethod'
        torch.save(payload, os.path.join(run_dir, "AWB.pt"))
        with open(os.path.join(run_dir, "method_info.json"), "w") as f:
            json.dump(payload["metadata"], f, indent=2)

        print(f"[AWB saved] {save_path}")
        return run_id, run_dir
    
    def _get_method_specific_weights(self, dataset_names: List[str]) -> torch.Tensor:
        """
        Get basis weights from the learned B matrix (T, M) -> return W.T = (M, T).
        For AutoencoderBasisMethod, the weights are stored in self.B (T, M) from the optimization.
        """
        if self.B is None:
            raise ValueError("Must call build_basis_vectors first to get B matrix")
        
        T_learned, M = self.B.shape
        T_requested = len(dataset_names)
        
        # Handle dimension mismatch with learned weights
        if T_learned != T_requested:
            print(f"Warning: Learned B matrix has {T_learned} tasks, but {T_requested} datasets requested")
            if T_learned > T_requested:
                # Truncate to match requested datasets
                B_truncated = self.B[:T_requested, :]  # (T_requested, M)
            else:
                # Pad with small weights for extra datasets
                padding = torch.zeros(T_requested - T_learned, M, device=self.B.device, dtype=self.B.dtype)
                B_truncated = torch.cat([self.B, padding], dim=0)  # (T_requested, M)
        else:
            B_truncated = self.B
        
        # Return transpose to get (M, T) format
        return B_truncated.t()  # Shape: (M, T)


class PCABasisMethod(BasisMethod):
    """Storage-efficient PCA-based basis method - stores only coefficients and metadata."""
    
    def __init__(self, M: int = 8):
        self.M = M
        # Instead of storing full components, store only what we need for reconstruction
        self.pca_coefficients = None  # (M, T) - coefficients for each task vector
        self.mean_vector = None       # (D,) - mean of all task vectors  
        self.singular_values = None   # (M,) - singular values for the M components
        self.total_dim = None         # D - total dimension
        self.keys_order = None        # Parameter keys in order
        self.param_shapes = None      # Original shapes for each parameter

    def svd_tall_cpu(self, V, k=None, row_chunk=8192):
        """
        SVD for tall matrices V (d x T) with d >> T on CPU.
        Returns U (d x k), S (k), Vh (k x T).
        Memory-friendly: builds Gram and U in chunks over rows of V.
        """
        V = V.to(torch.float32)
        assert V.dim() == 2
        d, T = V.shape
        k = min(k if k is not None else T, T)

        # 1) Build Gram in chunks: G = V^T V (T x T)
        G = torch.zeros(T, T, dtype=V.dtype, device=V.device)
        for i in range(0, d, row_chunk):
            j = min(i + row_chunk, d)
            B = V[i:j]                      # (chunk x T), contiguous slice
            # Accumulate without extra copies
            G += B.transpose(0, 1).matmul(B)

        # 2) Eigendecomposition of symmetric Gram
        evals, Vsmall = torch.linalg.eigh(G)       # ascending
        idx = torch.argsort(evals, descending=True)[:k]
        S = torch.sqrt(torch.clamp(evals[idx], min=0))
        Vh = Vsmall[:, idx].T                      # (k x T)  -> right singular vectors

        # 3) Build U in chunks: U = V @ Vsmall[:, idx] / S
        U = torch.empty(d, k, dtype=V.dtype, device=V.device)
        denom = S.clamp_min(torch.finfo(V.dtype).eps).unsqueeze(0)   # (1 x k)
        Rt = Vh.transpose(0, 1)                                      # (T x k)
        for i in range(0, d, row_chunk):
            j = min(i + row_chunk, d)
            U[i:j] = V[i:j].matmul(Rt) / denom

        return U, S, Vh
    
    def build_basis_vectors(self, 
                        task_vecs: List[Dict[str, torch.Tensor]], 
                        keys: List[str],
                        **kwargs) -> List[Dict[str, torch.Tensor]]:
        """Build PCA 'variance-scaled' basis: basis_m = U[:,m]*S[m], coeffs = V^T."""
        T = len(task_vecs)
        self.keys_order = keys
        self.param_shapes = {k: task_vecs[0][k].shape for k in keys}

        # Concatenate all task vectors into a matrix V \in R^{D x T}
        V_list = []
        for tv in task_vecs:
            v_flat = torch.cat([tv[k].flatten() for k in keys])
            V_list.append(v_flat)
        V = torch.stack(V_list, dim=1)  # (D, T)

        self.total_dim = V.shape[0]
        device = V.device
        dtype = V.dtype

        # Center data
        self.mean_vector = V.mean(dim=1)  # (D,)
        V_centered = V - self.mean_vector.unsqueeze(1)

        # Economy SVD: V_centered = U S Vh, with U:(D x r), S:(r,), Vh:(r x T), r=min(D,T)
        # U, S, Vh = torch.linalg.svd(V_centered, full_matrices=False)
        target_dtype = V_centered.dtype
        U, S, Vh = self.svd_tall_cpu(V_centered, k=self.M, row_chunk=8192)
        U  = U.to(target_dtype)
        S  = S.to(target_dtype)
        Vh = Vh.to(target_dtype)

        # Choose number of components
        r = S.shape[0]
        M_actual = min(self.M, r, T)
        self.singular_values = S[:M_actual]  # (M,)
        components = U[:, :M_actual]         # (D, M)

        # Store coefficients as plain V^T
        # coeffs shape: (M, T)
        self.pca_coefficients = Vh[:M_actual, :]  # equals V^T_M

        # Build variance-scaled basis dicts: basis_m = U[:,m] * S[m]
        basis = []
        for m in range(M_actual):
            component_flat = components[:, m] * self.singular_values[m]
            basis_dict = self._unflatten_vector(component_flat, keys)
            basis.append(basis_dict)

        # Storage info (coeffs + mean + S); 4 bytes per float32
        bytes_per_elem = 4 if dtype in (torch.float32, torch.int32) else V.element_size()
        compact_size_mb = (self.pca_coefficients.numel() 
                        + self.mean_vector.numel() 
                        + self.singular_values.numel()) * bytes_per_elem / 1024**2
        full_matrix_gb = self.total_dim * M_actual * bytes_per_elem / 1024**3

        print(f"PCA basis built with {M_actual} components (variance-scaled basis; coeffs = V^T).")
        print(f"Storage: {compact_size_mb:.1f} MB (coeffs/mean/S) vs {full_matrix_gb:.1f} GB (full U*S).")
        return basis
    
    def recover_task_vectors_from_basis(self,
                                    basis: List[Dict[str, torch.Tensor]],
                                    keys: List[str],
                                    **kwargs) -> List[Dict[str, torch.Tensor]]:
        """Recover task vectors using stored V^T coefficients and variance-scaled basis."""
        if self.pca_coefficients is None:
            raise ValueError("Must call build_basis_vectors first")

        M, T = self.pca_coefficients.shape
        recovered = []

        for t in range(T):
            v_flat = self.mean_vector.clone()
            for m in range(M):
                coeff = self.pca_coefficients[m, t]  # V^T_{m,t}
                component_dict = basis[m]            # U[:,m] * S[m]
                component_flat = torch.cat([component_dict[k].flatten() for k in keys])
                v_flat += coeff * component_flat     # sum_m U[:,m]*S[m]*V^T_{m,t}
            rec_dict = self._unflatten_vector(v_flat, keys)
            recovered.append(rec_dict)

        print("PCA reconstruction completed using V^T coefficients and variance-scaled basis.")
        return recovered
    
    def _unflatten_vector(self, v_flat: torch.Tensor, keys: List[str]) -> Dict[str, torch.Tensor]:
        """Convert flattened vector back to dict format."""
        result = {}
        start_idx = 0
        for k in keys:
            shape = self.param_shapes[k]
            numel = int(torch.prod(torch.tensor(shape, dtype=torch.int64)).item())
            result[k] = v_flat[start_idx:start_idx+numel].reshape(shape)
            start_idx += numel
        return result
    
    def save_training_artifacts(self, 
                               checkpoints_root: str, 
                               model: str, 
                               dataset_names: List[str],
                               M: int,
                               seed: int = 0,
                               basis_vectors: Optional[List[Dict[str, torch.Tensor]]] = None,
                               **kwargs) -> Tuple[str, str]:
        """Save PCA-specific artifacts - only coefficients and metadata."""
        run_id, run_dir = super().save_training_artifacts(
            checkpoints_root, model, dataset_names, M, seed, basis_vectors=basis_vectors, **kwargs
        )
        
        # Save only the compact representation
        if self.pca_coefficients is not None:
            pca_path = os.path.join(run_dir, "pca_components.pt")
            
            # Compute storage savings
            full_size_gb = self.total_dim * M * 4 / (1024**3)
            compact_size_mb = (self.pca_coefficients.numel() + self.mean_vector.numel() + 
                              self.singular_values.numel()) * 4 / (1024**2)
            
            torch.save({
                "coefficients": self.pca_coefficients,     # (M, T)
                "mean_vector": self.mean_vector,           # (D,)  
                "singular_values": self.singular_values,   # (M,)
                "M_actual": self.pca_coefficients.shape[0],
                "total_dim": self.total_dim,
                "keys_order": self.keys_order,
                "param_shapes": self.param_shapes,
                "dataset_names": dataset_names,
                # Storage info for reference
                "storage_info": {
                    "compact_size_mb": compact_size_mb,
                    "full_matrix_size_gb": full_size_gb,
                    "compression_ratio": full_size_gb * 1024 / compact_size_mb
                }
            }, pca_path)
            print(f"[PCA coefficients saved] {pca_path}")
            print(f"Storage: {compact_size_mb:.1f} MB (vs {full_size_gb:.1f} GB full matrix)")
        
        return run_id, run_dir
    
    def _get_method_specific_weights(self, dataset_names: List[str], use_pca_based_weights: bool = False, tau_w: float = 1.0) -> torch.Tensor:
        """
        Get basis weights for PCA method. 
        
        Args:
            dataset_names: List of dataset names
            use_pca_based_weights: If True, use PCA coefficient-based weights instead of uniform
            tau_w: Temperature parameter for softmax normalization (only used when use_pca_based_weights=True)
        
        Returns:
            weights: Tensor of shape (M, T) containing basis weights
        """
        if self.pca_coefficients is None:
            raise ValueError("Must call build_basis_vectors first to get PCA coefficients")
        
        M, T_learned = self.pca_coefficients.shape
        T_requested = len(dataset_names)
        
        # Handle dimension mismatch
        if T_learned != T_requested:
            print(f"Warning: PCA coefficients have {T_learned} tasks, but {T_requested} datasets requested")
            if T_learned > T_requested:
                # Truncate to match requested datasets
                coeffs = self.pca_coefficients[:, :T_requested]  # (M, T_requested)
            else:
                # Pad with zeros for extra datasets
                padding = torch.zeros(M, T_requested - T_learned, device=self.pca_coefficients.device, dtype=self.pca_coefficients.dtype)
                coeffs = torch.cat([self.pca_coefficients, padding], dim=1)  # (M, T_requested)
        else:
            coeffs = self.pca_coefficients
        
        if use_pca_based_weights:
            # Apply PCA-based weighting scheme
            print(f"Using PCA-based task weights with temperature tau_w={tau_w}")
            
            # For each basis vector m, compute weights w^(m) = softmax(max(v_m, 0) / tau_w)
            weights = torch.zeros_like(coeffs)  # (M, T)
            
            for m in range(M):
                v_m = coeffs[m, :]  # Shape: (T,) - coefficients for basis m
                
                # Keep only positive entries: z^(m) = max(v_m, 0)
                z_m = torch.clamp(v_m, min=0.0)  # Shape: (T,)
                
                # Find positive positions
                positive_mask = z_m > 0  # Boolean mask for positive elements
                
                if positive_mask.sum() > 0:  # Only apply softmax if there are positive entries
                    # Apply softmax only to positive elements
                    positive_elements = z_m[positive_mask] / tau_w  # Extract positive elements and scale
                    softmax_positive = torch.softmax(positive_elements, dim=0)  # Softmax only on positive
                    
                    # Create sparse weight vector: softmax values at positive positions, 0 elsewhere
                    w_m = torch.zeros_like(z_m)  # Initialize with zeros
                    w_m[positive_mask] = softmax_positive  # Fill in softmax values at positive positions
                else:
                    # If all coefficients are negative, fall back to uniform weights
                    w_m = torch.ones_like(z_m) / len(z_m)
                    print(f"Warning: All coefficients for basis {m} are negative, using uniform weights")
                
                weights[m, :] = w_m

            print(weights)
            
            print(f"Applied PCA-based weighting transformation (sparse softmax):")
            print(f"  - Original coefficients range: [{coeffs.min().item():.4f}, {coeffs.max().item():.4f}]")
            print(f"  - Final weights range: [{weights.min().item():.4f}, {weights.max().item():.4f}]")
            print(f"  - Sparsity: {(weights == 0).sum().item()}/{weights.numel()} elements are zero ({100 * (weights == 0).float().mean().item():.1f}%)")
            print(f"  - Weights sum per basis: {weights.sum(dim=1).tolist()}")
            
            return weights
        else:
            # Use uniform weights (default behavior for PCA when not using coefficient-based weighting)
            print("Using uniform weights for PCA method")
            uniform_weights = torch.ones_like(coeffs) / coeffs.shape[1]  # Shape: (M, T), each row sums to 1
            
            return uniform_weights  # Shape: (M, T)


class RandomBasisMethod(BasisMethod):
    """Storage-efficient random orthogonal basis method - regenerates basis from seed."""
    
    def __init__(self, M: int = 8, seed: int = 0):
        self.M = M
        self.seed = seed
        # Instead of storing the full basis matrix, store only reconstruction info
        self.projection_coefficients = None  # (M, T) - how each task projects onto random basis
        self.total_dim = None
        self.keys_order = None
        self.param_shapes = None

    def _generate_random_basis(self, D, device=None, dtype=None):
        g = torch.Generator(device=device if device else "cpu")
        g.manual_seed(self.seed)             # reset every call
        A = torch.randn(D, self.M, generator=g, device=device, dtype=dtype or torch.float32)
        Q, _ = torch.linalg.qr(A, mode="reduced")
        return Q
    
    def build_basis_vectors(self, 
                           task_vecs: List[Dict[str, torch.Tensor]], 
                           keys: List[str],
                           **kwargs) -> List[Dict[str, torch.Tensor]]:
        """Build random orthogonal basis vectors - store only coefficients."""
        T = len(task_vecs)
        self.keys_order = keys
        self.param_shapes = {k: task_vecs[0][k].shape for k in keys}
        
        # Get dimension
        self.total_dim = sum(task_vecs[0][k].numel() for k in keys)
        
        # Generate random orthogonal matrix (temporarily)
        random_basis = self._generate_random_basis(self.total_dim)  # (D, M)
        
        # Flatten task vectors and compute projections
        V_list = []
        for tv in task_vecs:
            v_flat = torch.cat([tv[k].flatten() for k in keys])
            V_list.append(v_flat)
        V = torch.stack(V_list, dim=1)  # (D, T)
        
        # Store projection coefficients instead of full basis
        self.projection_coefficients = random_basis.t() @ V  # (M, T)
        
        # Convert random basis to dict format for return (but don't store permanently)
        basis = []
        for m in range(self.M):
            component = random_basis[:, m]
            basis_dict = self._unflatten_vector(component, keys)
            basis.append(basis_dict)
        
        # Calculate storage savings
        full_size_gb = self.total_dim * self.M * 4 / (1024**3)
        compact_size_mb = self.projection_coefficients.numel() * 4 / (1024**2)
        
        print(f"Random orthogonal basis built with {self.M} vectors (storage-efficient)")
        print(f"Storage: {compact_size_mb:.1f} MB (coefficients) vs {full_size_gb:.1f} GB (full matrix)")
        return basis
    
    def recover_task_vectors_from_basis(self,
                                       basis: List[Dict[str, torch.Tensor]],
                                       keys: List[str],
                                       **kwargs) -> List[Dict[str, torch.Tensor]]:
        """Recover using stored projection coefficients and regenerated basis."""
        if self.projection_coefficients is None:
            raise ValueError("Must call build_basis_vectors first")
        
        M, T = self.projection_coefficients.shape
        
        # Regenerate the random basis (cheap operation)
        random_basis = self._generate_random_basis(self.total_dim)  # (D, M)
        
        # Reconstruct: V_reconstructed = random_basis @ coefficients
        V_reconstructed = random_basis @ self.projection_coefficients  # (D, T)
        
        # Convert back to dict format
        recovered = []
        for t in range(T):
            v_flat = V_reconstructed[:, t]
            rec_dict = self._unflatten_vector(v_flat, keys)
            recovered.append(rec_dict)
        
        print(f"Random basis reconstruction completed using stored coefficients")
        return recovered
    
    def _unflatten_vector(self, v_flat: torch.Tensor, keys: List[str]) -> Dict[str, torch.Tensor]:
        """Convert flattened vector back to dict format."""
        result = {}
        start_idx = 0
        for k in keys:
            shape = self.param_shapes[k]
            numel = int(torch.prod(torch.tensor(shape, dtype=torch.int64)).item())
            result[k] = v_flat[start_idx:start_idx+numel].reshape(shape)
            start_idx += numel
        return result
    
    def save_training_artifacts(self, 
                               checkpoints_root: str, 
                               model: str, 
                               dataset_names: List[str],
                               M: int,
                               seed: int = 0,
                               basis_vectors: Optional[List[Dict[str, torch.Tensor]]] = None,
                               **kwargs) -> Tuple[str, str]:
        """Save random basis artifacts - only coefficients and metadata."""
        run_id, run_dir = super().save_training_artifacts(
            checkpoints_root, model, dataset_names, M, seed, basis_vectors=basis_vectors, **kwargs
        )
        
        # Save only the compact representation
        if self.projection_coefficients is not None:
            basis_path = os.path.join(run_dir, "random_basis.pt")
            
            # Compute storage savings
            full_size_gb = self.total_dim * M * 4 / (1024**3)
            compact_size_mb = self.projection_coefficients.numel() * 4 / (1024**2)
            
            torch.save({
                "projection_coefficients": self.projection_coefficients,  # (M, T)
                "seed": self.seed,
                "M": M,
                "total_dim": self.total_dim,
                "keys_order": self.keys_order,
                "param_shapes": self.param_shapes,
                "dataset_names": dataset_names,
                # Storage info for reference
                "storage_info": {
                    "compact_size_mb": compact_size_mb,
                    "full_matrix_size_gb": full_size_gb,
                    "compression_ratio": full_size_gb * 1024 / compact_size_mb
                }
            }, basis_path)
            print(f"[Random basis coefficients saved] {basis_path}")
            print(f"Storage: {compact_size_mb:.1f} MB (vs {full_size_gb:.1f} GB full matrix)")
        
        return run_id, run_dir
    
    def _get_method_specific_weights(self, dataset_names: List[str]) -> torch.Tensor:
        """
        Get basis weights for RandomBasisMethod. Uses the stored projection coefficients
        which represent how each task projects onto the random orthogonal basis.
        """
        if self.projection_coefficients is None:
            raise ValueError("Must call build_basis_vectors first to get projection coefficients")
        
        M, T_learned = self.projection_coefficients.shape
        T_requested = len(dataset_names)
        
        # Handle dimension mismatch
        if T_learned != T_requested:
            print(f"Warning: Random basis coefficients have {T_learned} tasks, but {T_requested} datasets requested")
            if T_learned > T_requested:
                # Truncate to match requested datasets
                weights = self.projection_coefficients[:, :T_requested]  # (M, T_requested)
            else:
                # Pad with zeros for extra datasets
                padding = torch.zeros(M, T_requested - T_learned, device=self.projection_coefficients.device, dtype=self.projection_coefficients.dtype)
                weights = torch.cat([self.projection_coefficients, padding], dim=1)  # (M, T_requested)
        else:
            weights = self.projection_coefficients
        
        return weights  # Shape: (M, T)


class RandomSampleMethod(BasisMethod):
    """Simple random selection method that picks M random task vectors as basis."""
    
    def __init__(self, M: int = 8, seed: int = 0):
        self.M = M
        self.seed = seed
        self.selected_indices = None  # Which task vectors were selected
    
    def build_basis_vectors(self, 
                           task_vecs: List[Dict[str, torch.Tensor]], 
                           keys: List[str],
                           **kwargs) -> List[Dict[str, torch.Tensor]]:
        """Build basis by randomly selecting M task vectors from the T available."""
        import torch
        import random
        
        T = len(task_vecs)
        
        if self.M >= T:
            print(f"Warning: M={self.M} >= T={T}, using all {T} task vectors as basis")
            self.selected_indices = list(range(T))
            basis = [task_vec.copy() for task_vec in task_vecs]
        else:
            # Set random seed for reproducibility
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            
            # Randomly select M indices from T task vectors
            self.selected_indices = sorted(random.sample(range(T), self.M))
            
            # Create basis from selected task vectors
            basis = []
            for idx in self.selected_indices:
                # Deep copy the selected task vector
                basis_dict = {}
                for k in keys:
                    basis_dict[k] = task_vecs[idx][k].clone()
                basis.append(basis_dict)
        
        print(f"Random selection completed: selected {len(basis)} task vectors as basis")
        print(f"Selected indices: {self.selected_indices}")
        return basis
    
    def recover_task_vectors_from_basis(self,
                                       basis: List[Dict[str, torch.Tensor]],
                                       keys: List[str],
                                       **kwargs) -> List[Dict[str, torch.Tensor]]:
        """Recovery not implemented for RandomMethod - basis vectors are the selected task vectors."""
        raise NotImplementedError(
            "RandomMethod does not implement recovery. The basis vectors ARE the selected task vectors. "
            "Use the basis vectors directly for your application."
        )
    
    def save_training_artifacts(self, 
                               checkpoints_root: str, 
                               model: str, 
                               dataset_names: List[str],
                               M: int,
                               seed: int = 0,
                               basis_vectors: Optional[List[Dict[str, torch.Tensor]]] = None,
                               **kwargs) -> Tuple[str, str]:
        """Save RandomMethod artifacts - just seed, selected indices and basis vectors."""
        run_id, run_dir = super().save_training_artifacts(
            checkpoints_root, model, dataset_names, M, seed, basis_vectors=basis_vectors, **kwargs
        )
        
        # Save RandomMethod-specific info
        if self.selected_indices is not None:
            random_path = os.path.join(run_dir, "random_selection.json")
            
            selection_info = {
                "seed": self.seed,
                "M": M,
                "selected_indices": self.selected_indices,
                "selected_datasets": [dataset_names[i] for i in self.selected_indices] if len(dataset_names) > max(self.selected_indices) else None,
                "total_tasks": len(dataset_names)
            }
            
            with open(random_path, "w") as f:
                json.dump(selection_info, f, indent=2)
            
            print(f"[Random selection info saved] {random_path}")
            print(f"Selected {len(self.selected_indices)} out of {len(dataset_names)} task vectors")
        
        return run_id, run_dir
    
    def _get_method_specific_weights(self, dataset_names: List[str]) -> torch.Tensor:
        """
        Get basis weights for RandomSampleMethod. Each selected task vector gets weight 1.0 
        for its corresponding task, and 0.0 for all others.
        """
        if self.selected_indices is None:
            raise ValueError("Must call build_basis_vectors first to get selected indices")
        
        T = len(dataset_names)
        M = len(self.selected_indices)
        
        # Create weight matrix: (M, T)
        weights = torch.zeros(M, T)
        
        # Each basis vector (selected task) gets weight 1.0 for its original task
        for basis_idx, task_idx in enumerate(self.selected_indices):
            if task_idx < T:  # Make sure task index is valid
                weights[basis_idx, task_idx] = 1.0
        
        return weights
