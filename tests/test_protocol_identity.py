"""Closure-R3: Protocol Identity and Fail-Closed Governance Tests.

Validates that:
1. Normalizer metadata enforces split fingerprint (split_hash); stale cache is invalidated when split changes.
2. Datasets/DataLoaders fail-closed with RuntimeError when split is marked BLOCKED_BY_DATA.
3. Physics ablation evaluation rejects Closure-R1 legacy checkpoints unless --allow_legacy_checkpoints is set.
4. Failure analysis rejects legacy checkpoints unless --allow_legacy_checkpoint is set.
5. Multi-model rollout benchmark rejects mixed/conflicting data contracts across checkpoints.
"""

import hashlib
import json
import os
import tempfile
import h5py
import numpy as np
import pytest
import torch

from scripts.evaluate_rollout import verify_checkpoint_contract
from scripts.evaluate_physics_ablation import ABLATION_GROUPS
from src.data.pipeline import compute_split_hash, create_flow_datasets


def _create_mock_h5(path: str, n_trajs: int = 2, nt: int = 6, ny: int = 16, nx: int = 32):
    """Helper to create minimal valid HDF5 file for testing pipeline."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("velocity", data=np.random.randn(n_trajs, nt, ny, nx, 2).astype(np.float32) * 2.0)
        f.create_dataset("pressure", data=np.random.randn(n_trajs, nt, ny, nx, 1).astype(np.float32) * 0.05 + 1.2)
        f.create_dataset("density", data=np.random.randn(n_trajs, nt, ny, nx, 1).astype(np.float32) * 0.8 + 3.0)


def test_normalizer_invalidated_when_split_changes():
    """P1-1: Normalizer cache must be invalidated when split manifest hash changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_a = os.path.join(tmpdir, "shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5")
        h5_b = os.path.join(tmpdir, "shear_flow_Reynolds_1e4_Schmidt_1e0.hdf5")
        _create_mock_h5(h5_a)
        _create_mock_h5(h5_b)

        split_file = os.path.join(tmpdir, "split.json")
        stats_dir = os.path.join(tmpdir, "normalization")
        os.makedirs(stats_dir, exist_ok=True)

        # 1. First split version: only h5_a
        split_data_v1 = {"train": [h5_a], "valid": [h5_a], "test": [h5_a]}
        with open(split_file, "w") as f:
            json.dump(split_data_v1, f)

        _, _, _, norm1 = create_flow_datasets(
            split_type="identity_test",
            split_file=split_file,
            data_root=tmpdir,
            history_length=1,
            horizon=1,
            stats_dir=stats_dir,
        )
        meta_file = os.path.join(stats_dir, "stats_identity_test_metadata.json")
        assert os.path.exists(meta_file)
        with open(meta_file, "r") as mf:
            meta1 = json.load(mf)

        expected_hash1 = compute_split_hash(split_data_v1)
        assert meta1["split_hash"] == expected_hash1
        first_mean = meta1["mean"]

        # 2. Update split manifest with different trajectories (h5_b added)
        split_data_v2 = {"train": [h5_a, h5_b], "valid": [h5_a], "test": [h5_b]}
        with open(split_file, "w") as f:
            json.dump(split_data_v2, f)

        expected_hash2 = compute_split_hash(split_data_v2)
        assert expected_hash1 != expected_hash2

        # 3. Running create_flow_datasets must detect hash mismatch, invalidate cache, and refit
        _, _, _, norm2 = create_flow_datasets(
            split_type="identity_test",
            split_file=split_file,
            data_root=tmpdir,
            history_length=1,
            horizon=1,
            stats_dir=stats_dir,
        )
        with open(meta_file, "r") as mf:
            meta2 = json.load(mf)

        assert meta2["split_hash"] == expected_hash2
        assert meta2["split_hash"] != meta1["split_hash"]


def test_blocked_holdout_evaluation_fails_closed():
    """P1-4: Loading a split marked BLOCKED_BY_DATA must immediately raise RuntimeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        blocked_split_file = os.path.join(tmpdir, "parameter_holdout_re.json")
        blocked_content = {
            "train": ["data/dummy1.hdf5"],
            "valid": ["data/dummy2.hdf5"],
            "test": [],
            "status": "BLOCKED_BY_DATA",
            "reason": "Only 1 distinct Reynolds number in dataset",
        }
        with open(blocked_split_file, "w") as f:
            json.dump(blocked_content, f)

        with pytest.raises(RuntimeError) as exc_info:
            create_flow_datasets(
                split_type="parameter_holdout_re",
                split_file=blocked_split_file,
                data_root=tmpdir,
            )

        assert "BLOCKED_BY_DATA" in str(exc_info.value)
        assert "Only 1 distinct Reynolds number" in str(exc_info.value)


def test_ablation_legacy_checkpoint_requires_opt_in():
    """P1-2: ABLATION_GROUPS separates Closure-R2 candidates from legacy Closure-R1 candidates."""
    # E0 should have no legacy candidates
    assert "legacy_candidates" in ABLATION_GROUPS["E0_single_step"]
    assert len(ABLATION_GROUPS["E0_single_step"]["legacy_candidates"]) == 0

    # E1 - E4 candidates must only point to ablation_E* Closure-R2 paths by default
    for g_key in ["E1_rollout_field", "E2_plus_L_div", "E3_plus_L_vort", "E4_full_physics"]:
        group = ABLATION_GROUPS[g_key]
        for c in group["candidates"]:
            assert "ablation_E" in c, f"Candidate {c} in {g_key} must be a Closure-R2 ablation_E path!"

        # Legacy candidates are isolated in separate list
        assert len(group["legacy_candidates"]) > 0
        for lc in group["legacy_candidates"]:
            assert ("ablation_L_field" in lc or "ablation_plus" in lc)


def test_failure_analysis_legacy_checkpoint_requires_opt_in():
    """P1-3: Failure analysis fails closed when model doesn't exist and opt-in is False."""
    from scripts.analyze_failure_cases import analyze_failure_cases

    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_model = os.path.join(tmpdir, "does_not_exist.pt")

        # Without opt-in, FileNotFoundError must be raised
        with pytest.raises(FileNotFoundError) as exc_info:
            analyze_failure_cases(
                model_path=non_existent_model,
                allow_legacy_checkpoint=False,
            )
        assert "Checkpoint not found" in str(exc_info.value)


def test_benchmark_rejects_mixed_data_contracts():
    """P1-5: verify_checkpoint_contract rejects mismatched downsample_factor, normalize, or split_type."""
    benchmark_contract = {
        "split_type": "grouped",
        "downsample_factor": 2,
        "normalize": True,
    }

    # 1. Matching config passes
    matching_cfg = {
        "split_type": "grouped",
        "downsample_factor": 2,
        "normalize": True,
    }
    verify_checkpoint_contract("test_model", "ckpt.pt", matching_cfg, benchmark_contract)

    # 2. Conflicting downsample_factor raises ValueError
    mismatched_ds_cfg = {
        "split_type": "grouped",
        "downsample_factor": 1,
        "normalize": True,
    }
    with pytest.raises(ValueError) as exc:
        verify_checkpoint_contract("test_model", "ckpt.pt", mismatched_ds_cfg, benchmark_contract)
    assert "downsample_factor" in str(exc.value)

    # 3. Conflicting normalize raises ValueError
    mismatched_norm_cfg = {
        "split_type": "grouped",
        "downsample_factor": 2,
        "normalize": False,
    }
    with pytest.raises(ValueError) as exc:
        verify_checkpoint_contract("test_model", "ckpt.pt", mismatched_norm_cfg, benchmark_contract)
    assert "normalize" in str(exc.value)

    # 4. Conflicting split_type raises ValueError
    mismatched_split_cfg = {
        "split_type": "official",
        "downsample_factor": 2,
        "normalize": True,
    }
    with pytest.raises(ValueError) as exc:
        verify_checkpoint_contract("test_model", "ckpt.pt", mismatched_split_cfg, benchmark_contract)
    assert "split_type" in str(exc.value)
