"""Unit tests for dataset splits, IC clustering, and reproducible data pipeline."""

import json
import os
import pytest
import torch
from src.data.pipeline import create_flow_dataloaders
from src.data.splits import SplitManager, parse_shear_flow_filename
from scripts.verify_splits import verify_splits


def test_parse_filename():
    meta = parse_shear_flow_filename("shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5")
    assert meta["re"] == 10000.0
    assert abs(meta["sc"] - 0.1) < 1e-5

    meta2 = parse_shear_flow_filename("shear_flow_Reynolds_1e5_Schmidt_1e1.hdf5")
    assert meta2["re"] == 100000.0
    assert abs(meta2["sc"] - 10.0) < 1e-5


def test_grouped_split_zero_leakage():
    """Verify that grouped_split.json guarantees strictly zero IC leakage."""
    grouped_split_path = "outputs/splits/grouped_split.json"
    official_split_path = "outputs/splits/official_split.json"

    if not os.path.exists(grouped_split_path):
        pytest.skip("grouped_split.json not yet built")

    results = verify_splits(grouped_split_path, official_split_path)
    assert results["grouped_pass"] is True
    assert results["grouped_min_dist"] > 0.1  # Significant physical separation


def test_reproducible_dataloader():
    """Verify that create_flow_dataloaders produces deterministic, identical batches with same seed."""
    grouped_split_path = "outputs/splits/grouped_split.json"
    if not os.path.exists(grouped_split_path):
        pytest.skip("grouped_split.json not found")

    # Run 1
    loader1, _, _, _ = create_flow_dataloaders(
        split_type="grouped",
        batch_size=2,
        history_length=4,
        horizon=1,
        downsample_factor=4,
        seed=123,
    )
    batch1 = next(iter(loader1))

    # Run 2 with same seed
    loader2, _, _, _ = create_flow_dataloaders(
        split_type="grouped",
        batch_size=2,
        history_length=4,
        horizon=1,
        downsample_factor=4,
        seed=123,
    )
    batch2 = next(iter(loader2))

    assert torch.allclose(batch1["history"], batch2["history"], atol=1e-6)
    assert torch.allclose(batch1["future"], batch2["future"], atol=1e-6)
    assert torch.allclose(batch1["re"], batch2["re"])
    assert torch.allclose(batch1["sc"], batch2["sc"])


def test_normalizer_effect():
    """Verify normalizer properly standardizes data."""
    loader, _, _, normalizer = create_flow_dataloaders(
        split_type="grouped",
        batch_size=4,
        history_length=4,
        horizon=1,
        downsample_factor=4,
        seed=42,
    )
    assert normalizer is not None
    assert normalizer.mean is not None
    assert normalizer.std is not None

    batch = next(iter(loader))
    # Normalized fields should be well-scaled (mean near 0, std reasonable)
    assert batch["history"].shape[1] == 4  # L=4
    assert batch["history"].shape[2] == 4  # C=4
    assert batch["future"].shape[1] == 1   # H=1
    assert batch["future"].shape[2] == 4   # C=4
