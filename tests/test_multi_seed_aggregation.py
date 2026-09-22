"""Unit tests for multi-seed statistical aggregation and paired delta analysis."""

import json
import os
import tempfile
import pytest
import numpy as np

from scripts.aggregate_multi_seed import (
    compute_sample_statistics,
    compute_paired_deltas,
    load_and_validate_seed_metrics,
    aggregate_multi_seed_metrics,
)
from src.utils.physics_contract import PHYSICS_PROTOCOL


def _create_mock_seed_json(
    seed: int,
    split_hash: str = "split_hash_abc",
    normalizer_hash: str = "norm_hash_123",
    protocol: str = PHYSICS_PROTOCOL,
    training_git_dirty: bool = False,
    vrmse_e1: float = 1.0,
    vrmse_e4: float = 0.8,
) -> dict:
    return {
        "E1_rollout_field": {
            "__metadata__": {
                "seed": seed,
                "split_hash": split_hash,
                "normalizer_hash": normalizer_hash,
                "evaluation_protocol": protocol,
                "training_git_commit": "commit_123",
                "training_git_dirty": training_git_dirty,
            },
            "step_1": {"vrmse_mean": vrmse_e1, "div_rmse": 0.2, "vort_rmse": 0.5},
            "step_5": {"vrmse_mean": vrmse_e1 * 2, "div_rmse": 0.4, "vort_rmse": 1.0},
        },
        "E4_full_physics": {
            "__metadata__": {
                "seed": seed,
                "split_hash": split_hash,
                "normalizer_hash": normalizer_hash,
                "evaluation_protocol": protocol,
                "training_git_commit": "commit_123",
                "training_git_dirty": training_git_dirty,
            },
            "step_1": {"vrmse_mean": vrmse_e4, "div_rmse": 0.15, "vort_rmse": 0.3},
            "step_5": {"vrmse_mean": vrmse_e4 * 2, "div_rmse": 0.3, "vort_rmse": 0.6},
        },
    }


def test_compute_sample_statistics_three_seeds():
    """Verify sample mean and sample standard deviation (ddof=1) for N=3."""
    vals = [1.0, 2.0, 3.0]
    res = compute_sample_statistics(vals)
    assert res["mean"] == pytest.approx(2.0)
    assert res["std"] == pytest.approx(1.0)  # np.std([1,2,3], ddof=1) == 1.0
    assert res["n"] == 3


def test_compute_sample_statistics_n_less_than_two():
    """Verify N < 2 does not fabricate a non-zero sample standard deviation."""
    res_one = compute_sample_statistics([5.0])
    assert res_one["mean"] == 5.0
    assert res_one["std"] is None
    assert res_one["n"] == 1

    res_zero = compute_sample_statistics([])
    assert res_zero["mean"] is None
    assert res_zero["std"] is None
    assert res_zero["n"] == 0


def test_paired_delta_calculation():
    """Verify paired delta (E4 - E1) per-seed and aggregate mean/std."""
    seed_data = {
        42: {
            "E1_rollout_field": {"step_1": {"vrmse_mean": 1.0}},
            "E4_full_physics": {"step_1": {"vrmse_mean": 0.8}},
        },
        43: {
            "E1_rollout_field": {"step_1": {"vrmse_mean": 1.5}},
            "E4_full_physics": {"step_1": {"vrmse_mean": 1.1}},
        },
        44: {
            "E1_rollout_field": {"step_1": {"vrmse_mean": 0.9}},
            "E4_full_physics": {"step_1": {"vrmse_mean": 0.6}},
        },
    }
    # Deltas: (0.8 - 1.0) = -0.2, (1.1 - 1.5) = -0.4, (0.6 - 0.9) = -0.3
    # Mean: -0.3, Std: 0.1, Win rate: 3/3 = 1.0
    res = compute_paired_deltas(
        seed_metrics=seed_data,
        baseline_group="E1_rollout_field",
        compare_groups=["E4_full_physics"],
        horizons=[1],
        metric_keys=["vrmse_mean"],
    )

    p_info = res["E4_full_physics"]["step_1"]["vrmse_mean"]
    assert p_info["mean_delta"] == pytest.approx(-0.3)
    assert p_info["std_delta"] == pytest.approx(0.1)
    assert p_info["win_count"] == 3
    assert p_info["win_rate"] == 1.0
    assert p_info["per_seed_deltas"]["42"] == pytest.approx(-0.2)
    assert p_info["per_seed_deltas"]["43"] == pytest.approx(-0.4)
    assert p_info["per_seed_deltas"]["44"] == pytest.approx(-0.3)


def test_missing_seed_file_rejection():
    """Missing seed file raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent = os.path.join(tmpdir, "closure_r4_physics_ablation_seed99_v2.json")
        with pytest.raises(FileNotFoundError):
            load_and_validate_seed_metrics(non_existent, expected_seed=99)


def test_split_hash_mismatch_rejection():
    """Mismatched split hash across seeds raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p42 = os.path.join(tmpdir, "seed42.json")
        p43 = os.path.join(tmpdir, "seed43.json")

        with open(p42, "w") as f:
            json.dump(_create_mock_seed_json(42, split_hash="split_A"), f)
        with open(p43, "w") as f:
            json.dump(_create_mock_seed_json(43, split_hash="split_B"), f)

        seed_files = {42: p42, 43: p43}
        with pytest.raises(ValueError) as exc_info:
            aggregate_multi_seed_metrics(seed_files=seed_files, groups=["E1_rollout_field", "E4_full_physics"])
        assert "Split hash mismatch across seeds" in str(exc_info.value)


def test_normalizer_hash_mismatch_rejection():
    """Mismatched normalizer hash across seeds raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p42 = os.path.join(tmpdir, "seed42.json")
        p43 = os.path.join(tmpdir, "seed43.json")

        with open(p42, "w") as f:
            json.dump(_create_mock_seed_json(42, normalizer_hash="norm_A"), f)
        with open(p43, "w") as f:
            json.dump(_create_mock_seed_json(43, normalizer_hash="norm_B"), f)

        seed_files = {42: p42, 43: p43}
        with pytest.raises(ValueError) as exc_info:
            aggregate_multi_seed_metrics(seed_files=seed_files, groups=["E1_rollout_field", "E4_full_physics"])
        assert "Normalizer hash mismatch across seeds" in str(exc_info.value)


def test_dirty_training_state_rejection():
    """Evaluation output indicating dirty training git state is rejected by default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p42 = os.path.join(tmpdir, "seed42.json")
        with open(p42, "w") as f:
            json.dump(_create_mock_seed_json(42, training_git_dirty=True), f)

        with pytest.raises(ValueError) as exc_info:
            load_and_validate_seed_metrics(p42, expected_seed=42, require_clean=True)
        assert "Dirty training state detected" in str(exc_info.value)


def test_seed_identity_mismatch_rejection():
    """Mismatched seed number inside metrics JSON raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p43 = os.path.join(tmpdir, "seed43.json")
        with open(p43, "w") as f:
            # File claims seed 43 in file name but contains seed 42 in metadata
            json.dump(_create_mock_seed_json(42), f)

        with pytest.raises(ValueError) as exc_info:
            load_and_validate_seed_metrics(p43, expected_seed=43)
        assert "Seed mismatch" in str(exc_info.value)
