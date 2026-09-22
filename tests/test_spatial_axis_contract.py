"""Integration test binding ShearFlowDataset to Closure-R4 physics axes."""

import os
import tempfile
import h5py
import numpy as np
import torch

from src.data.shear_flow_dataset import ShearFlowDataset
from src.utils.fft_derivatives import compute_divergence
from src.utils.physics_contract import SPATIAL_AXIS_CONTRACT, SHEAR_FLOW_DOMAIN_SIZE_XY


def _write_mock(path, nx=16, ny=32, nt=3):
    x = np.arange(nx, dtype=np.float32) / nx
    y_label = np.arange(ny, dtype=np.float32) / ny
    x_phys = x.astype(np.float64)
    y_phys = -1.0 + np.arange(ny, dtype=np.float64) * (2.0 / ny)
    xx, yy = np.meshgrid(x_phys, y_phys, indexing="ij")
    u = -np.sin(2.0 * np.pi * xx) * np.sin(np.pi * yy)
    v = -2.0 * np.cos(2.0 * np.pi * xx) * np.cos(np.pi * yy)

    vel = np.zeros((1, nt, nx, ny, 2), dtype=np.float32)
    vel[..., 0] = u
    vel[..., 1] = v
    with h5py.File(path, "w") as f:
        dims = f.create_group("dimensions")
        dims.create_dataset("x", data=x)
        dims.create_dataset("y", data=y_label)
        t1 = f.create_group("t1_fields")
        t1.create_dataset("velocity", data=vel)
        t0 = f.create_group("t0_fields")
        t0.create_dataset("pressure", data=np.zeros((1, nt, nx, ny), dtype=np.float32))
        t0.create_dataset("tracer", data=np.zeros((1, nt, nx, ny), dtype=np.float32))


def test_dataset_nx_ny_layout_matches_physics_operator():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5")
        _write_mock(path)
        ds = ShearFlowDataset(file_paths=[path], history_length=1, horizon=1)
        q = ds[0]["history"][0]

        assert q.shape == (4, 16, 32)
        assert SPATIAL_AXIS_CONTRACT == "tensor(...,C,Nx,Ny):dim-2=x,dim-1=y"
        assert SHEAR_FLOW_DOMAIN_SIZE_XY == (1.0, 2.0)
        div = compute_divergence(q[0], q[1])
        assert torch.sqrt(torch.mean(div**2)).item() < 2e-4
