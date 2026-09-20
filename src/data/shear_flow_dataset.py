"""PyTorch Dataset implementation for The Well periodic shear_flow data."""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.normalization import FieldNormalizer
from src.data.splits import parse_shear_flow_filename
from src.data.windows import generate_window_indices


class ShearFlowDataset(Dataset):
    """Dataset loader for 2D periodic incompressible shear flow trajectories.

    Loads and slices physical states q = [u, v, p, s] (4 channels).

    Args:
        file_paths: List of paths to .hdf5 files.
        history_length: Historical frame count L (default: 4).
        horizon: Prediction frame count H (default: 1).
        stride: Stride between window starts within a trajectory (default: 1).
        normalizer: Optional FieldNormalizer instance.
        preload_to_memory: If True, load all data into RAM for faster training.
        downsample_factor: Optional spatial downsampling factor (e.g. 2 for 256x512 -> 128x256).
    """

    def __init__(
        self,
        file_paths: List[str],
        history_length: int = 4,
        horizon: int = 1,
        stride: int = 1,
        normalizer: Optional[FieldNormalizer] = None,
        preload_to_memory: bool = False,
        downsample_factor: int = 1,
    ):
        super().__init__()
        self.file_paths = sorted(file_paths)
        self.history_length = history_length
        self.horizon = horizon
        self.stride = stride
        self.normalizer = normalizer
        self.preload_to_memory = preload_to_memory
        self.downsample_factor = downsample_factor

        # Index all valid windows across files and simulations
        # Window entry: (file_idx, sim_idx, start_t, split_t, end_t, re, sc)
        self.samples: List[Tuple[int, int, int, int, int, float, float]] = []
        self._cached_data: Dict[int, np.ndarray] = {}

        self._build_index()

    @staticmethod
    def _find_dset(h5, candidates: List[str]):
        for c in candidates:
            if c in h5:
                return h5[c]
        return None

    def _build_index(self):
        for f_idx, path in enumerate(self.file_paths):
            if not os.path.exists(path):
                continue
            params = parse_shear_flow_filename(os.path.basename(path))
            re_val = params["re"]
            sc_val = params["sc"]

            with h5py.File(path, "r") as h5:
                dset = self._find_dset(h5, ["t1_fields/velocity", "velocity", "t0_fields/tracer", "tracer", "u"])
                if dset is None:
                    raise KeyError(f"No valid field dataset found in {path}")
                shape = dset.shape

                # If shape is (N_sim, T, ...)
                if len(shape) >= 4:
                    n_sims = shape[0]
                    t_steps = shape[1]
                else:
                    n_sims = 1
                    t_steps = shape[0]

                windows = generate_window_indices(
                    total_timesteps=t_steps,
                    history_length=self.history_length,
                    horizon=self.horizon,
                    stride=self.stride,
                )

                for sim_idx in range(n_sims):
                    for start_t, split_t, end_t in windows:
                        self.samples.append((f_idx, sim_idx, start_t, split_t, end_t, re_val, sc_val))

            if self.preload_to_memory:
                self._cached_data[f_idx] = self._load_full_file(self.file_paths[f_idx])

    def _load_full_file(self, path: str) -> np.ndarray:
        """Load full file fields [u, v, p, s] into memory as float32 array (N_sim, T, 4, Ny, Nx)."""
        with h5py.File(path, "r") as h5:
            vel_ds = self._find_dset(h5, ["t1_fields/velocity", "velocity"])
            if vel_ds is not None:
                vel = np.asarray(vel_ds, dtype=np.float32)
                if vel.shape[-1] == 2:
                    u = vel[..., 0]
                    v = vel[..., 1]
                elif vel.shape[2] == 2:
                    u = vel[:, :, 0]
                    v = vel[:, :, 1]
                else:
                    raise ValueError(f"Unexpected velocity shape: {vel.shape}")
            elif "u" in h5 and "v" in h5:
                u = np.asarray(h5["u"], dtype=np.float32)
                v = np.asarray(h5["v"], dtype=np.float32)
            else:
                raise KeyError(f"Cannot find velocity fields in {path}")

            # Pressure
            p_ds = self._find_dset(h5, ["t0_fields/pressure", "pressure"])
            if p_ds is not None:
                p = np.asarray(p_ds, dtype=np.float32)
                if p.ndim == u.ndim + 1 and p.shape[-1] == 1:
                    p = p.squeeze(-1)
            else:
                p = np.zeros_like(u)

            # Tracer
            s_ds = self._find_dset(h5, ["t0_fields/tracer", "tracer"])
            if s_ds is not None:
                s = np.asarray(s_ds, dtype=np.float32)
                if s.ndim == u.ndim + 1 and s.shape[-1] == 1:
                    s = s.squeeze(-1)
            elif "s" in h5:
                s = np.asarray(h5["s"], dtype=np.float32)
            else:
                s = np.zeros_like(u)

            return np.stack([u, v, p, s], axis=2)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, float]]:
        f_idx, sim_idx, start_t, split_t, end_t, re_val, sc_val = self.samples[idx]

        if self.preload_to_memory and f_idx in self._cached_data:
            full_data = self._cached_data[f_idx]
            traj_slice = full_data[sim_idx, start_t:end_t]  # (L + H, 4, Ny, Nx)
        else:
            path = self.file_paths[f_idx]
            with h5py.File(path, "r") as h5:
                vel_ds = self._find_dset(h5, ["t1_fields/velocity", "velocity"])
                if vel_ds is not None:
                    if vel_ds.shape[-1] == 2:
                        u = np.asarray(vel_ds[sim_idx, start_t:end_t, ..., 0], dtype=np.float32)
                        v = np.asarray(vel_ds[sim_idx, start_t:end_t, ..., 1], dtype=np.float32)
                    else:
                        u = np.asarray(vel_ds[sim_idx, start_t:end_t, 0], dtype=np.float32)
                        v = np.asarray(vel_ds[sim_idx, start_t:end_t, 1], dtype=np.float32)
                else:
                    u = np.asarray(h5["u"][sim_idx, start_t:end_t], dtype=np.float32)
                    v = np.asarray(h5["v"][sim_idx, start_t:end_t], dtype=np.float32)

                p_ds = self._find_dset(h5, ["t0_fields/pressure", "pressure"])
                if p_ds is not None:
                    p = np.asarray(p_ds[sim_idx, start_t:end_t], dtype=np.float32)
                    if p.ndim == 4 and p.shape[-1] == 1:
                        p = p.squeeze(-1)
                else:
                    p = np.zeros_like(u)

                s_ds = self._find_dset(h5, ["t0_fields/tracer", "tracer"])
                if s_ds is not None:
                    s = np.asarray(s_ds[sim_idx, start_t:end_t], dtype=np.float32)
                    if s.ndim == 4 and s.shape[-1] == 1:
                        s = s.squeeze(-1)
                else:
                    s = np.zeros_like(u)

                traj_slice = np.stack([u, v, p, s], axis=1)

        tensor_slice = torch.from_numpy(traj_slice)

        if self.downsample_factor > 1:
            # Downsample spatial dimensions by factor
            tensor_slice = tensor_slice[..., :: self.downsample_factor, :: self.downsample_factor]

        if self.normalizer is not None:
            tensor_slice = self.normalizer.normalize(tensor_slice)

        history = tensor_slice[: self.history_length]
        future = tensor_slice[self.history_length :]

        return {
            "history": history,  # (L, 4, Ny, Nx)
            "future": future,  # (H, 4, Ny, Nx)
            "re": torch.tensor(re_val, dtype=torch.float32),
            "sc": torch.tensor(sc_val, dtype=torch.float32),
        }
