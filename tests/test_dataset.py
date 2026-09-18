"""Unit tests for ShearFlowDataset and Sliding Window mechanism."""

import os
import tempfile
import h5py
import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader
from src.data.shear_flow_dataset import ShearFlowDataset
from src.data.normalization import FieldNormalizer
from src.data.windows import generate_window_indices


def create_mock_shear_flow_hdf5(file_path: str, n_sims: int = 2, t_steps: int = 15, ny: int = 32, nx: int = 64):
    """Creates a synthetic HDF5 file conforming to The Well's shear_flow schema."""
    with h5py.File(file_path, "w") as h5:
        # velocity: (N_sim, T, Ny, Nx, 2)
        vel = np.random.randn(n_sims, t_steps, ny, nx, 2).astype(np.float32)
        h5.create_dataset("velocity", data=vel)

        # pressure: (N_sim, T, Ny, Nx)
        p = np.random.randn(n_sims, t_steps, ny, nx).astype(np.float32)
        h5.create_dataset("pressure", data=p)

        # tracer: (N_sim, T, Ny, Nx)
        s = np.random.uniform(0.0, 1.0, size=(n_sims, t_steps, ny, nx)).astype(np.float32)
        h5.create_dataset("tracer", data=s)


def test_shear_flow_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_file = os.path.join(tmpdir, "shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5")
        create_mock_shear_flow_hdf5(mock_file, n_sims=2, t_steps=15, ny=32, nx=64)

        # L=4, H=2, stride=2
        dataset = ShearFlowDataset(
            file_paths=[mock_file],
            history_length=4,
            horizon=2,
            stride=2,
            preload_to_memory=False,
        )

        # Check total sample count
        # Windows per trajectory: (15 - 6) // 2 + 1 = 5 windows
        # 2 simulations * 5 windows = 10 samples
        assert len(dataset) == 10

        sample = dataset[0]
        assert sample["history"].shape == (4, 4, 32, 64)
        assert sample["future"].shape == (2, 4, 32, 64)
        assert sample["re"].item() == pytest.approx(1e4)
        assert sample["sc"].item() == pytest.approx(0.1)

        # Test DataLoader batching
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert batch["history"].shape == (2, 4, 4, 32, 64)
        assert batch["future"].shape == (2, 2, 4, 32, 64)
        assert batch["re"].shape == (2,)
        assert batch["sc"].shape == (2,)


def test_dataset_with_normalizer():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_file = os.path.join(tmpdir, "shear_flow_Reynolds_5e4_Schmidt_1e0.hdf5")
        create_mock_shear_flow_hdf5(mock_file, n_sims=1, t_steps=10, ny=16, nx=32)

        normalizer = FieldNormalizer(
            mean=[0.0, 0.0, 1.0, 0.5],
            std=[1.0, 1.0, 2.0, 0.25],
        )

        dataset = ShearFlowDataset(
            file_paths=[mock_file],
            history_length=3,
            horizon=1,
            normalizer=normalizer,
        )

        sample = dataset[0]
        assert sample["history"].shape == (3, 4, 16, 32)
        assert sample["future"].shape == (1, 4, 16, 32)
