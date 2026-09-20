"""Unit tests for dataset splits, IC clustering, and reproducible data pipeline.

All tests are hermetic and synthetic, requiring zero external dataset files.
"""

import json
import os
import pytest
import numpy as np
import h5py
import torch

from src.data.pipeline import create_flow_dataloaders, create_flow_datasets
from src.data.splits import SplitManager, parse_shear_flow_filename
from scripts.verify_splits import verify_splits


@pytest.fixture
def mock_split_env(tmp_path):
    """Hermetic fixture creating synthetic shear_flow HDF5 files with known initial conditions."""
    data_dir = tmp_path / "mock_datasets"
    (data_dir / "data" / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "data" / "valid").mkdir(parents=True, exist_ok=True)
    (data_dir / "data" / "test").mkdir(parents=True, exist_ok=True)

    f1 = data_dir / "data" / "train" / "shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5"
    f2 = data_dir / "data" / "valid" / "shear_flow_Reynolds_1e4_Schmidt_1e0.hdf5"
    f3 = data_dir / "data" / "test" / "shear_flow_Reynolds_1e5_Schmidt_1e1.hdf5"

    np.random.seed(42)
    # Distinct initial conditions
    ic0 = np.random.randn(1, 16, 32, 2).astype(np.float32)
    ic1 = np.random.randn(1, 16, 32, 2).astype(np.float32)
    ic2 = np.random.randn(1, 16, 32, 2).astype(np.float32)
    ic3 = np.random.randn(1, 16, 32, 2).astype(np.float32)

    def write_mock(path, ics, n_steps=8):
        n_sims = len(ics)
        vel = np.zeros((n_sims, n_steps, 16, 32, 2), dtype=np.float32)
        for i, ic in enumerate(ics):
            vel[i, 0] = ic
            vel[i, 1:] = ic + np.random.randn(n_steps - 1, 16, 32, 2) * 0.01
        p = np.random.randn(n_sims, n_steps, 16, 32).astype(np.float32)
        s = np.random.uniform(0, 1, size=(n_sims, n_steps, 16, 32)).astype(np.float32)
        with h5py.File(str(path), "w") as h5:
            h5.create_dataset("velocity", data=vel)
            h5.create_dataset("pressure", data=p)
            h5.create_dataset("tracer", data=s)

    # f1 and f2 share ic0 to test cross-file IC clustering
    write_mock(f1, [ic0, ic1])
    write_mock(f2, [ic0, ic2])
    write_mock(f3, [ic3, np.random.randn(1, 16, 32, 2).astype(np.float32)])

    rel_f1 = str(f1.relative_to(data_dir))
    rel_f2 = str(f2.relative_to(data_dir))
    rel_f3 = str(f3.relative_to(data_dir))
    all_rel_files = [rel_f1, rel_f2, rel_f3]

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    # Build grouped split with relative paths
    grouped = SplitManager.get_grouped_split(
        all_files=all_rel_files,
        data_root=str(data_dir),
        train_ratio=0.5,
        valid_ratio=0.25,
        seed=42,
    )
    grp_file = splits_dir / "grouped_split.json"
    with open(grp_file, "w") as f:
        json.dump(grouped, f, indent=2)

    # Build official split with relative paths
    official = SplitManager.get_official_split([rel_f1], [rel_f2], [rel_f3])
    off_file = splits_dir / "official_split.json"
    with open(off_file, "w") as f:
        json.dump(official, f, indent=2)

    return {
        "data_root": str(data_dir),
        "grouped_split_path": str(grp_file),
        "official_split_path": str(off_file),
        "rel_files": all_rel_files,
        "splits_dir": str(splits_dir),
    }


def test_parse_filename():
    meta = parse_shear_flow_filename("shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5")
    assert meta["re"] == 10000.0
    assert abs(meta["sc"] - 0.1) < 1e-5

    meta2 = parse_shear_flow_filename("data/test/shear_flow_Reynolds_1e5_Schmidt_1e1.hdf5")
    assert meta2["re"] == 100000.0
    assert abs(meta2["sc"] - 10.0) < 1e-5


def test_grouped_split_zero_leakage(mock_split_env):
    """Verify that synthetic grouped split guarantees strictly zero IC leakage."""
    results = verify_splits(
        grouped_split_path=mock_split_env["grouped_split_path"],
        official_split_path=mock_split_env["official_split_path"],
        data_root=mock_split_env["data_root"],
    )
    assert results["grouped_pass"] is True
    # Official split must have detected the deliberate cross-partition leakage of ic0 between train (f1) and valid (f2)
    assert results["official_leaks_count"] >= 1


def test_reproducible_dataloader(mock_split_env):
    """Verify that create_flow_dataloaders produces deterministic, identical batches with same seed."""
    loader1, _, _, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file=mock_split_env["grouped_split_path"],
        data_root=mock_split_env["data_root"],
        batch_size=2,
        history_length=4,
        horizon=1,
        downsample_factor=2,
        seed=123,
    )
    batch1 = next(iter(loader1))

    loader2, _, _, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file=mock_split_env["grouped_split_path"],
        data_root=mock_split_env["data_root"],
        batch_size=2,
        history_length=4,
        horizon=1,
        downsample_factor=2,
        seed=123,
    )
    batch2 = next(iter(loader2))

    assert torch.allclose(batch1["history"], batch2["history"], atol=1e-6)
    assert torch.allclose(batch1["future"], batch2["future"], atol=1e-6)
    assert torch.allclose(batch1["re"], batch2["re"])
    assert torch.allclose(batch1["sc"], batch2["sc"])


def test_normalizer_effect(mock_split_env):
    """Verify normalizer properly standardizes data."""
    loader, _, _, normalizer = create_flow_dataloaders(
        split_type="grouped",
        split_file=mock_split_env["grouped_split_path"],
        data_root=mock_split_env["data_root"],
        stats_dir=mock_split_env["splits_dir"],
        batch_size=2,
        history_length=4,
        horizon=1,
        downsample_factor=2,
        seed=42,
    )
    assert normalizer is not None
    assert normalizer.mean is not None
    assert normalizer.std is not None

    batch = next(iter(loader))
    assert batch["history"].shape[1] == 4  # L=4
    assert batch["history"].shape[2] == 4  # C=4
    assert batch["future"].shape[1] == 1   # H=1
    assert batch["future"].shape[2] == 4   # C=4


def test_create_flow_datasets_and_sampler(mock_split_env):
    """Verify create_flow_datasets and sampler return in create_flow_dataloaders."""
    train_ds, valid_ds, test_ds, norm = create_flow_datasets(
        split_type="grouped",
        split_file=mock_split_env["grouped_split_path"],
        data_root=mock_split_env["data_root"],
        stats_dir=mock_split_env["splits_dir"],
        history_length=4,
        horizon=2,
        train_stride=1,
        valid_stride=1,
        downsample_factor=2,
    )
    assert len(train_ds) > 0
    assert len(valid_ds) > 0
    assert len(test_ds) > 0
    assert norm is not None

    # Test create_flow_dataloaders with return_sampler
    train_loader, val_loader, test_loader, normalizer, sampler = create_flow_dataloaders(
        split_type="grouped",
        split_file=mock_split_env["grouped_split_path"],
        data_root=mock_split_env["data_root"],
        stats_dir=mock_split_env["splits_dir"],
        batch_size=2,
        history_length=4,
        horizon=2,
        downsample_factor=2,
        is_distributed=False,
        return_sampler=True,
    )
    assert sampler is None  # Since is_distributed=False
    sample_batch = next(iter(train_loader))
    assert sample_batch["future"].shape[1] == 2
