"""StocBench 2D Stochastic Incompressible Navier-Stokes (Kolmogorov Flow) Dataset.

Provides contracts, dataset loading, and batch adapters for StocBench:
- STOCBENCH_STATE_SPEC: Single-channel vorticity field (omega).
- StocBenchTrainDataset: Memory-mapped reader for training trajectories (traj_seed_*.npy).
- StocBenchReferenceEnsemble: Reader and validator for one-step bifurcation ensembles (step_seed_*.npz).
- stocbench_batch_adapter: Adapter binding batch tensors to WorldModelBatch with STOCBENCH_STATE_SPEC.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.contracts.batch import WorldModelBatch
from src.contracts.context import Context
from src.contracts.state_spec import StateSpec


# Canonical StateSpec for StocBench: 2D vorticity state (1 channel)
STOCBENCH_STATE_SPEC = StateSpec(
    variables=("vorticity",),
    num_channels=1,
    spatial_dim=2,
)

# Physical simulation parameters according to StocBench solver configuration
STOCBENCH_SAMPLE_DT = 0.5       # Frame interval between consecutive snapshots (dt_sample)
STOCBENCH_SOLVER_DT = 0.0001    # Internal numerical integration timestep (dt_solver)
STOCBENCH_SCALING_MEAN = 0.0    # Solver normalization mean
STOCBENCH_SCALING_STD = 3.0     # Solver normalization standard deviation (w_phys = w_stored * 3.0)


class StocBenchTrainDataset(Dataset):
    """Dataset for training trajectories from StocBench (traj_seed_*.npy).

    Safe, memory-mapped reader yielding single-frame or multi-frame windows
    without falsifying unobserved physics or channels.

    Args:
        file_path: Path to traj_seed_*.npy file.
        history_length: Historical sequence length L (default: 1 for single-frame current state).
        horizon: Prediction horizon H (default: 1 for next-step future).
        stride: Temporal window stride (default: 1).
        dataset_id: Dataset identifier for provenance tracking.
        dataset_revision: Dataset git commit / revision.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        history_length: int = 1,
        horizon: int = 1,
        stride: int = 1,
        dataset_id: str = "stocbench",
        dataset_revision: Optional[str] = None,
    ):
        super().__init__()
        self.file_path = str(Path(file_path).resolve())
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"StocBench trajectory file not found: {self.file_path}")

        if history_length <= 0:
            raise ValueError(f"history_length must be positive, got {history_length}")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self.history_length = history_length
        self.horizon = horizon
        self.stride = stride
        self.dataset_id = dataset_id
        self.dataset_revision = dataset_revision

        # Safe mmap load without unpickling
        loaded = np.load(self.file_path, mmap_mode="r", allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError(
                f"Invalid trajectory shape in {self.file_path}: expected 5D .npy array, "
                f"got NPZ archive (step_seed_*.npz files must not be used for training)."
            )
        self._mmap_data = loaded

        # Validate array structure: expected (N_sim, T_frames, 1, Ny, Nx)
        if getattr(self._mmap_data, "ndim", None) != 5:
            raise ValueError(
                f"Invalid trajectory shape in {self.file_path}: expected 5D (N, T, C, Ny, Nx), "
                f"got ndim={getattr(self._mmap_data, 'ndim', None)} shape={getattr(self._mmap_data, 'shape', None)}"
            )

        n_sims, t_frames, channels, ny, nx = self._mmap_data.shape
        if channels != 1:
            raise ValueError(
                f"Expected 1 vorticity channel, got {channels} in {self.file_path}"
            )
        if ny <= 0 or nx <= 0:
            raise ValueError(f"Invalid spatial resolution ({ny}, {nx}) in {self.file_path}")

        window_size = self.history_length + self.horizon
        if t_frames < window_size:
            raise ValueError(
                f"Trajectory frames T={t_frames} is smaller than window size L+H={window_size}"
            )

        self.n_sims = n_sims
        self.t_frames = t_frames
        self.ny = ny
        self.nx = nx

        # Build index: (sim_idx, start_t)
        self.samples: List[Tuple[int, int]] = []
        for s_idx in range(n_sims):
            for t_start in range(0, t_frames - window_size + 1, self.stride):
                self.samples.append((s_idx, t_start))

        if len(self.samples) == 0:
            raise ValueError("No valid sample windows found in trajectory.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sim_idx, start_t = self.samples[idx]
        split_t = start_t + self.history_length
        end_t = split_t + self.horizon

        # Slice mmap and convert to torch float32
        window_raw = np.array(self._mmap_data[sim_idx, start_t:end_t], dtype=np.float32)

        # Fail-fast check for NaN / Inf
        if not np.all(np.isfinite(window_raw)):
            raise ValueError(f"Non-finite (NaN or Inf) detected in sample {idx} from {self.file_path}")

        hist_raw = window_raw[: self.history_length]  # (L, 1, Ny, Nx)
        fut_raw = window_raw[self.history_length :]   # (H, 1, Ny, Nx)

        return {
            "history": torch.from_numpy(hist_raw),
            "future": torch.from_numpy(fut_raw),
            "dt": torch.tensor(STOCBENCH_SAMPLE_DT, dtype=torch.float32),
            "source_file": os.path.basename(self.file_path),
            "traj_idx": sim_idx,
            "start_t": start_t,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision or "unknown",
            "state_variables": ("vorticity",),
        }


class StocBenchReferenceEnsemble:
    """Reader and numerical auditor for StocBench one-step bifurcation test files (step_seed_*.npz).

    Enforces:
    - Safe unpickle-free loading (allow_pickle=False).
    - Preserves reference ensemble member axis K_ref without confounding with time horizon H.
    - Consistency verification between member sample statistics (mean/std) and stored arrays.
    - Non-degeneracy verification: asserts futures are not constant or identical.
    """

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = str(Path(file_path).resolve())
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"StocBench reference file not found: {self.file_path}")

        with np.load(self.file_path, allow_pickle=False) as data:
            keys = list(data.keys())
            if "init" not in keys:
                raise KeyError(f"Missing required key 'init' in {self.file_path}; available: {keys}")
            if "raw" not in keys:
                raise KeyError(f"Missing required key 'raw' in {self.file_path}; available: {keys}")

            self.init_np = np.array(data["init"], dtype=np.float32)
            self.raw_np = np.array(data["raw"], dtype=np.float32)
            self.stored_mean_np = np.array(data["mean"], dtype=np.float32) if "mean" in data else None
            self.stored_std_np = np.array(data["std"], dtype=np.float32) if "std" in data else None

        self._validate_shapes_and_values()

    def _validate_shapes_and_values(self) -> None:
        """Validate structure, finiteness, and non-degeneracy."""
        # Validate init: expected shape (1, Ny, Nx) or (1, 1, Ny, Nx)
        if self.init_np.ndim not in (3, 4):
            raise ValueError(f"Invalid init shape: expected 3D or 4D, got {self.init_np.shape}")

        if not np.all(np.isfinite(self.init_np)):
            raise ValueError(f"Non-finite values (NaN/Inf) found in 'init' of {self.file_path}")

        # Validate raw: expected (K_ref, 1, 1, Ny, Nx) or (K_ref, 1, Ny, Nx)
        if self.raw_np.ndim not in (4, 5):
            raise ValueError(f"Invalid raw shape: expected 4D or 5D, got {self.raw_np.shape}")

        if self.raw_np.shape[0] < 2:
            raise ValueError(
                f"Reference ensemble must have at least 2 members for bifurcation, got {self.raw_np.shape[0]}"
            )

        if not np.all(np.isfinite(self.raw_np)):
            raise ValueError(f"Non-finite values (NaN/Inf) found in 'raw' of {self.file_path}")

        # Non-degeneracy check: all futures must not be identical
        # Compute standard deviation across reference ensemble members (axis 0)
        # Note: in float32 arithmetic, identical values have roundoff std ~ 1e-6.
        # Genuine physical bifurcation standard deviation is on the order of 0.1 ~ 1.0.
        sample_std = np.std(self.raw_np, axis=0, ddof=0)
        max_std = float(np.max(sample_std))
        if max_std < 1e-4:
            raise ValueError(
                f"Degenerate reference ensemble in {self.file_path}: all members are virtually identical (max_std={max_std:.2e})."
            )

    @property
    def num_members(self) -> int:
        """Number of stochastic future realizations K_ref."""
        return self.raw_np.shape[0]

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        """Spatial resolution (Ny, Nx)."""
        return self.init_np.shape[-2], self.init_np.shape[-1]

    def verify_statistical_consistency(self, atol: float = 1e-4, rtol: float = 1e-4) -> Dict[str, float]:
        """Verify that member sample statistics match stored mean and std.

        Note: StocBench computes std with ddof=0 (population standard deviation).
        """
        if self.stored_mean_np is None or self.stored_std_np is None:
            raise ValueError("Cannot verify consistency: file does not contain stored 'mean' and 'std'.")

        # In StocBench raw has shape (K, 1, 1, Ny, Nx) and mean/std have (1, Ny, Nx)
        # Compute mean across member axis 0
        computed_mean = np.mean(self.raw_np, axis=0)
        computed_std = np.std(self.raw_np, axis=0, ddof=0)

        # Match dimensions for comparison
        target_mean = self.stored_mean_np
        target_std = self.stored_std_np

        while computed_mean.ndim > target_mean.ndim:
            computed_mean = computed_mean.squeeze(0)
        while target_mean.ndim > computed_mean.ndim:
            target_mean = target_mean.squeeze(0)

        while computed_std.ndim > target_std.ndim:
            computed_std = computed_std.squeeze(0)
        while target_std.ndim > computed_std.ndim:
            target_std = target_std.squeeze(0)

        diff_mean = np.max(np.abs(computed_mean - target_mean))
        diff_std = np.max(np.abs(computed_std - target_std))

        if diff_mean > atol:
            raise ValueError(
                f"Statistical mean mismatch in {self.file_path}: max_diff={diff_mean:.6e} > atol={atol}"
            )
        if diff_std > atol:
            raise ValueError(
                f"Statistical std mismatch in {self.file_path}: max_diff={diff_std:.6e} > atol={atol}"
            )

        return {
            "max_mean_discrepancy": float(diff_mean),
            "max_std_discrepancy": float(diff_std),
            "num_members": self.num_members,
        }

    def get_canonical_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return standardized PyTorch tensors for condition history and reference futures.

        Returns:
            condition_history: (1, 1, 1, Ny, Nx) representing (B=1, L=1, C=1, Ny, Nx).
            reference_futures: (1, K_ref, 1, 1, Ny, Nx) representing (B=1, K_ref, H=1, C=1, Ny, Nx).
        """
        ny, nx = self.spatial_shape
        # Ensure init has shape (1, 1, 1, Ny, Nx)
        init_t = torch.from_numpy(self.init_np.reshape(1, 1, 1, ny, nx))

        # Ensure raw has shape (1, K_ref, 1, 1, Ny, Nx)
        k_ref = self.num_members
        raw_t = torch.from_numpy(self.raw_np.reshape(1, k_ref, 1, 1, ny, nx))

        return init_t, raw_t


def stocbench_batch_adapter(
    batch_dict: Dict[str, Any],
    boundary: Optional[Any] = "periodic",
    geometry: Optional[Any] = None,
) -> WorldModelBatch:
    """Domain adapter for StocBench 2D vorticity dataset.

    Injects STOCBENCH_STATE_SPEC (single-channel vorticity), default 'periodic' boundary condition,
    and preserves all provenance metadata (dataset_id, dataset_revision, source_file, etc.).

    Args:
        batch_dict: Dictionary returned by PyTorch default_collate or dataset sample.
        boundary: Boundary condition (default: "periodic").
        geometry: Domain geometry (default: None for torus [0, 2*pi)^2).

    Returns:
        Structured WorldModelBatch conforming strictly to STOCBENCH_STATE_SPEC.
    """
    batch = WorldModelBatch.from_batch_dict(
        batch_dict=batch_dict,
        state_spec=STOCBENCH_STATE_SPEC,
        boundary=boundary,
        geometry=geometry,
    )

    # Preserve provenance and tracking metadata
    for k in ("dataset_id", "dataset_revision", "source_file", "traj_idx", "start_t", "state_variables"):
        if k in batch_dict and k not in batch.metadata:
            batch.metadata[k] = batch_dict[k]

    return batch


def collate_stocbench_batch(
    batch_list: List[Dict[str, Any]],
    boundary: Optional[Any] = "periodic",
    geometry: Optional[Any] = None,
) -> WorldModelBatch:
    """Collate function for PyTorch DataLoader returning a WorldModelBatch with STOCBENCH_STATE_SPEC."""
    from torch.utils.data.dataloader import default_collate
    collated = default_collate(batch_list)
    return stocbench_batch_adapter(collated, boundary=boundary, geometry=geometry)
