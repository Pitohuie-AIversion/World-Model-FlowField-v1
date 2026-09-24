"""Unit tests for Phase 1 performance optimizations:
1. Wavenumber grid caching for spectral derivatives.
2. Per-process HDF5 file handle caching and lifecycle in ShearFlowDataset.
3. DataLoader persistent_workers and prefetch_factor configuration.
"""

import os
import tempfile
import h5py
import numpy as np
import pytest
import torch

from src.data.shear_flow_dataset import ShearFlowDataset
from src.data.pipeline import create_flow_dataloaders
from src.utils.fft_derivatives import (
    clear_wavenumber_cache,
    get_wavenumbers,
    spectral_grad_2d,
    compute_laplacian_2d,
)


def create_mock_shear_flow_hdf5(file_path: str, n_sims: int = 2, t_steps: int = 10, ny: int = 16, nx: int = 32):
    """Creates a synthetic HDF5 file conforming to The Well's shear_flow schema."""
    with h5py.File(file_path, "w") as h5:
        vel = np.random.randn(n_sims, t_steps, ny, nx, 2).astype(np.float32)
        h5.create_dataset("velocity", data=vel)
        p = np.random.randn(n_sims, t_steps, ny, nx).astype(np.float32)
        h5.create_dataset("pressure", data=p)
        s = np.random.uniform(0.0, 1.0, size=(n_sims, t_steps, ny, nx)).astype(np.float32)
        h5.create_dataset("tracer", data=s)


def test_wavenumber_cache_hits_and_clearing():
    """Verify get_wavenumbers caches and reuses tensor objects across calls."""
    clear_wavenumber_cache()
    device = torch.device("cpu")
    dtype = torch.float32

    kx1, ky1 = get_wavenumbers(64, 128, 1.0, 2.0, device, dtype)
    kx2, ky2 = get_wavenumbers(64, 128, 1.0, 2.0, device, dtype)

    assert kx1 is kx2, "kx tensor should be cached and identical in memory"
    assert ky1 is ky2, "ky tensor should be cached and identical in memory"

    # Different resolution or domain should produce different cached items
    kx3, ky3 = get_wavenumbers(32, 64, 1.0, 2.0, device, dtype)
    assert kx3 is not kx1
    assert ky3 is not ky1

    # Clearing cache should invalidate references
    clear_wavenumber_cache()
    kx4, ky4 = get_wavenumbers(64, 128, 1.0, 2.0, device, dtype)
    assert kx4 is not kx1


def test_wavenumber_cache_preserves_numerical_exactness():
    """Verify spectral_grad_2d and compute_laplacian_2d produce identical results with cache."""
    clear_wavenumber_cache()
    f = torch.randn(2, 4, 32, 64, dtype=torch.float32)

    df_dx_1, df_dy_1 = spectral_grad_2d(f, domain_size=(1.0, 2.0))
    df_dx_2, df_dy_2 = spectral_grad_2d(f, domain_size=(1.0, 2.0))
    assert torch.allclose(df_dx_1, df_dx_2)
    assert torch.allclose(df_dy_1, df_dy_2)

    lap_1 = compute_laplacian_2d(f, domain_size=(1.0, 2.0))
    lap_2 = compute_laplacian_2d(f, domain_size=(1.0, 2.0))
    assert torch.allclose(lap_1, lap_2)


def test_shear_flow_dataset_handle_caching_and_close():
    """Verify ShearFlowDataset caches open HDF5 handles and closes them cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_file = os.path.join(tmpdir, "shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5")
        create_mock_shear_flow_hdf5(mock_file, n_sims=2, t_steps=10, ny=16, nx=32)

        dataset = ShearFlowDataset(
            file_paths=[mock_file],
            history_length=2,
            horizon=1,
            stride=1,
            preload_to_memory=False,
        )

        assert len(dataset._file_handles) == 0

        # First sample access opens the file handle
        _ = dataset[0]
        assert mock_file in dataset._file_handles
        handle1 = dataset._file_handles[mock_file]
        assert handle1.id.valid, "Handle should be open and valid"

        # Subsequent accesses reuse the same cached handle
        _ = dataset[1]
        handle2 = dataset._file_handles[mock_file]
        assert handle1 is handle2, "Handle must be reused without reopening"

        # Simulating worker PID divergence (as in multiprocessing fork)
        dataset._worker_pid = 99999999
        _ = dataset[0]
        assert dataset._worker_pid == os.getpid()
        handle_new = dataset._file_handles[mock_file]
        assert handle_new.id.valid

        # Calling close() cleanly closes all open handles
        dataset.close()
        assert len(dataset._file_handles) == 0
        assert not handle_new.id.valid, "Underlying HDF5 handle must be closed after dataset.close()"


def test_create_flow_dataloaders_persistent_workers():
    """Verify DataLoader applies persistent_workers and prefetch_factor when num_workers > 0."""
    import json
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_file = os.path.join(tmpdir, "shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5")
        create_mock_shear_flow_hdf5(mock_file, n_sims=2, t_steps=10, ny=16, nx=32)

        split_content = {
            "train": [{"file_path": mock_file, "traj_idx": 0, "re": 10000.0, "sc": 0.1, "cluster_id": 0}],
            "valid": [{"file_path": mock_file, "traj_idx": 1, "re": 10000.0, "sc": 0.1, "cluster_id": 1}],
            "test": [{"file_path": mock_file, "traj_idx": 1, "re": 10000.0, "sc": 0.1, "cluster_id": 1}],
        }
        split_path = os.path.join(tmpdir, "mock_split.json")
        with open(split_path, "w") as f:
            json.dump(split_content, f)

        # 1. num_workers = 0: persistent_workers and prefetch_factor should be default (False/None)
        train_loader_0, val_loader_0, test_loader_0, _ = create_flow_dataloaders(
            split_file=split_path,
            data_root=tmpdir,
            num_workers=0,
            batch_size=2,
            preload_to_memory=False,
            stats_dir=tmpdir,
        )
        assert train_loader_0.persistent_workers is False
        assert train_loader_0.prefetch_factor is None

        # 2. num_workers = 2: persistent_workers should be True and prefetch_factor should be 2
        train_loader_2, val_loader_2, test_loader_2, _ = create_flow_dataloaders(
            split_file=split_path,
            data_root=tmpdir,
            num_workers=2,
            batch_size=2,
            preload_to_memory=False,
            stats_dir=tmpdir,
        )
        assert train_loader_2.persistent_workers is True
        assert train_loader_2.prefetch_factor == 2
        assert val_loader_2.persistent_workers is True
        assert val_loader_2.prefetch_factor == 2
        assert test_loader_2.persistent_workers is True
        assert test_loader_2.prefetch_factor == 2
