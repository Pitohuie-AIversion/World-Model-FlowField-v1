"""PyTorch Dataset implementation for The Well periodic shear_flow data."""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
        file_paths: Optional[List[str]] = None,
        trajectories: Optional[List[Dict]] = None,
        data_root: Optional[str] = None,
        history_length: int = 4,
        horizon: int = 1,
        stride: int = 1,
        normalizer: Optional[FieldNormalizer] = None,
        preload_to_memory: bool = False,
        downsample_factor: int = 1,
        require_pressure: bool = False,
        require_tracer: bool = False,
    ):
        super().__init__()
        if file_paths is None and trajectories is None:
            raise ValueError("Either file_paths or trajectories must be provided.")

        self.data_root = data_root or os.environ.get("SHEAR_FLOW_DATA_DIR")

        def _resolve_path(p: str) -> str:
            if self.data_root and not os.path.isabs(p):
                return os.path.join(self.data_root, p)
            return p

        if trajectories is not None:
            resolved_trajs = []
            for t in trajectories:
                item = dict(t)
                item["file_path"] = _resolve_path(item["file_path"])
                resolved_trajs.append(item)
            self.trajectories = resolved_trajs
            self.allowed_set = {(t["file_path"], t["traj_idx"]) for t in resolved_trajs}
            self.traj_metadata = {(t["file_path"], t["traj_idx"]): t for t in resolved_trajs}
            file_paths = sorted(list(set(t["file_path"] for t in resolved_trajs)))
        else:
            self.trajectories = None
            self.allowed_set = None
            self.traj_metadata = {}
            file_paths = [_resolve_path(p) for p in file_paths]

        self.file_paths = sorted(file_paths)
        self.history_length = history_length
        self.horizon = horizon
        self.stride = stride
        self.normalizer = normalizer
        self.preload_to_memory = preload_to_memory
        self.downsample_factor = downsample_factor
        self.require_pressure = require_pressure
        self.require_tracer = require_tracer

        # Index all valid windows across files and simulations
        # Window entry: (file_idx, sim_idx, start_t, split_t, end_t, re, sc)
        self.samples: List[Tuple[int, int, int, int, int, float, float]] = []
        self._cached_data: Dict[int, np.ndarray] = {}
        self._time_data: Dict[int, np.ndarray] = {}
        self._worker_pid: Optional[int] = None
        self._file_handles: Dict[str, h5py.File] = {}

        self._build_index()

    def _get_h5_handle(self, path: str) -> h5py.File:
        """Get or create cached HDF5 file handle for the current process."""
        pid = os.getpid()
        if self._worker_pid != pid:
            self._worker_pid = pid
            self._file_handles = {}
        if path not in self._file_handles or not getattr(self._file_handles[path].id, "valid", False):
            self._file_handles[path] = h5py.File(path, "r")
        return self._file_handles[path]

    def close(self) -> None:
        """Close all cached HDF5 file handles in this process."""
        if hasattr(self, "_file_handles") and self._file_handles:
            for handle in list(self._file_handles.values()):
                try:
                    if getattr(handle.id, "valid", False):
                        handle.close()
                except Exception:
                    pass
            self._file_handles.clear()

    def __del__(self) -> None:
        self.close()

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
            try:
                params = parse_shear_flow_filename(os.path.basename(path))
                re_val = params["re"]
                sc_val = params["sc"]
            except ValueError:
                with h5py.File(path, "r") as h5_check:
                    if "scalars/Reynolds" in h5_check and "scalars/Schmidt" in h5_check:
                        re_val = float(h5_check["scalars/Reynolds"][()])
                        sc_val = float(h5_check["scalars/Schmidt"][()])
                    else:
                        raise

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

                # Extract time coordinates
                has_time = ("dimensions/time" in h5) or ("time" in h5)
                if has_time:
                    raw_time = h5["dimensions/time"] if "dimensions/time" in h5 else h5["time"]
                    time_arr = np.asarray(raw_time, dtype=np.float32)
                else:
                    if self.require_pressure or self.require_tracer:
                        raise KeyError(
                            f"require_pressure={self.require_pressure} or require_tracer={self.require_tracer} "
                            f"requires explicit ground-truth time dataset in {path}, but no time dataset was found. "
                            f"Aborting to prevent fabricating synthetic time intervals."
                        )
                    time_arr = np.arange(t_steps, dtype=np.float32) * 0.1

                if self.require_pressure or self.require_tracer:
                    if len(time_arr) < t_steps:
                        raise ValueError(
                            f"Time coordinates in {path} have length {len(time_arr)}, but trajectory has {t_steps} timesteps."
                        )
                    if not np.all(np.isfinite(time_arr)):
                        raise ValueError(f"Time dataset in {path} contains non-finite values (NaN/Inf).")
                    diffs = np.diff(time_arr)
                    if not np.all(diffs > 0):
                        raise ValueError(f"Time dataset in {path} is not strictly monotonically increasing.")
                    dt0 = float(diffs[0])
                    if not np.all(np.abs(diffs - dt0) <= 1e-4 * dt0 + 1e-6):
                        raise ValueError(
                            f"Time dataset in {path} has irregular intervals (min={diffs.min()}, max={diffs.max()}). "
                            f"PDE residual formulation requires a uniform time grid."
                        )

                self._time_data[f_idx] = time_arr

                # Strict validation for physical fields when requested
                has_p = self._find_dset(h5, ["t0_fields/pressure", "pressure"]) is not None
                has_s = self._find_dset(h5, ["t0_fields/tracer", "tracer", "s"]) is not None
                if self.require_pressure and not has_p:
                    raise KeyError(f"require_pressure=True but pressure field is missing in {path}")
                if self.require_tracer and not has_s:
                    raise KeyError(f"require_tracer=True but tracer field is missing in {path}")

                windows = generate_window_indices(
                    total_timesteps=t_steps,
                    history_length=self.history_length,
                    horizon=self.horizon,
                    stride=self.stride,
                )

                for sim_idx in range(n_sims):
                    if self.allowed_set is not None and (path, sim_idx) not in self.allowed_set:
                        continue

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

    def get_window_metadata(self, idx: int) -> Dict[str, Any]:
        f_idx, sim_idx, start_t, split_t, end_t, re_val, sc_val = self.samples[idx]
        path = self.file_paths[f_idx]
        meta = self.traj_metadata.get((path, sim_idx), {}) if hasattr(self, "traj_metadata") and self.traj_metadata else {}
        cluster_id = int(meta.get("cluster_id", -1))
        return {
            "source_file": path,
            "traj_idx": sim_idx,
            "start_t": start_t,
            "split_t": split_t,
            "end_t": end_t,
            "re": re_val,
            "sc": sc_val,
            "cluster_id": cluster_id,
        }

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, float, str, int]]:
        f_idx, sim_idx, start_t, split_t, end_t, re_val, sc_val = self.samples[idx]
        path = self.file_paths[f_idx]
        meta = self.traj_metadata.get((path, sim_idx), {}) if hasattr(self, "traj_metadata") and self.traj_metadata else {}
        cluster_id = int(meta.get("cluster_id", -1))

        if self.preload_to_memory and f_idx in self._cached_data:
            full_data = self._cached_data[f_idx]
            traj_slice = full_data[sim_idx, start_t:end_t]  # (L + H, 4, Ny, Nx)
        else:
            h5 = self._get_h5_handle(path)
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

        # Temporal coordinates
        time_arr = self._time_data.get(f_idx)
        if time_arr is not None and len(time_arr) > 0:
            time_slice = time_arr[start_t:end_t]
            if len(time_slice) >= 2:
                dt_val = float(time_slice[1] - time_slice[0])
            elif len(time_arr) >= 2:
                dt_val = float(time_arr[1] - time_arr[0])
            else:
                if self.require_pressure or self.require_tracer:
                    raise ValueError(f"Cannot determine dt from time array in {path}")
                dt_val = 0.1
        else:
            if self.require_pressure or self.require_tracer:
                raise KeyError(f"Missing time coordinates for sample in {path}")
            time_slice = np.arange(start_t, end_t, dtype=np.float32) * 0.1
            dt_val = 0.1

        if self.require_pressure or self.require_tracer:
            if not (np.isfinite(dt_val) and dt_val > 0):
                raise ValueError(f"Invalid non-positive or non-finite dt={dt_val} for sample from {path}")
            if not (np.isfinite(re_val) and re_val > 0):
                raise ValueError(f"Invalid non-positive or non-finite Re={re_val} for sample from {path}")
            if self.require_tracer and not (np.isfinite(sc_val) and sc_val > 0):
                raise ValueError(f"Invalid non-positive or non-finite Sc={sc_val} for sample from {path}")

        return {
            "history": history,  # (L, 4, Ny, Nx)
            "future": future,  # (H, 4, Ny, Nx)
            "re": torch.tensor(re_val, dtype=torch.float32),
            "sc": torch.tensor(sc_val, dtype=torch.float32),
            "dt": torch.tensor(dt_val, dtype=torch.float32),
            "time": torch.from_numpy(time_slice).float(),
            "source_file": path,
            "traj_idx": torch.tensor(sim_idx, dtype=torch.long),
            "start_t": torch.tensor(start_t, dtype=torch.long),
            "cluster_id": torch.tensor(cluster_id, dtype=torch.long),
        }
