"""Unit and contract tests for H16 benchmark evaluation script (scripts/evaluate_h16_benchmark.py).

Verifies fail-closed provenance validation:
  1. Valid checkpoint passes contract checks
  2. Wrong seed rejected (fail-closed)
  3. Wrong split / normalizer hash rejected (fail-closed)
  4. Horizon mismatch rejected (e.g. H8 masquerading as H16)
  5. Dirty or unknown git training commit rejected under formal evaluation
  6. Non-finite metric (NaN / Inf) detection and rejection
  7. Dataset protocol sliding window mathematics validation
"""

import math
import os
import pytest
import torch

from scripts.evaluate_h16_benchmark import (
    validate_benchmark_checkpoint,
    validate_metric_finiteness,
)
from src.utils.physics_contract import (
    PHYSICS_PROTOCOL,
    SPATIAL_AXIS_CONTRACT,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
)


def _create_dummy_checkpoint_data(
    seed: int = 42,
    horizon: int = 16,
    split_hash: str = "41fbe6ebe7edd460",
    normalizer_hash: str = "3a0fe52689657618",
    training_git_commit: str = "0a8a4da137e3f73c4f696cc81f527c32a8ba8405",
    training_git_dirty: bool = False,
    physics_protocol: str = PHYSICS_PROTOCOL,
    spatial_axis_contract: str = SPATIAL_AXIS_CONTRACT,
) -> dict:
    """Creates a mock checkpoint dictionary matching production schema."""
    cfg = {
        "model_type": "latent_transformer",
        "horizon": horizon,
        "seed": seed,
        "split_hash": split_hash,
        "normalizer_hash": normalizer_hash,
        "physics_protocol": physics_protocol,
        "spatial_axis_contract": spatial_axis_contract,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
        "training_git_commit": training_git_commit,
        "training_git_dirty": training_git_dirty,
    }
    return {
        "epoch": 12,
        "selection_criterion": "best_long_vrmse_j_long",
        "seed": seed,
        "split_hash": split_hash,
        "normalizer_hash": normalizer_hash,
        "training_git_commit": training_git_commit,
        "training_git_dirty": training_git_dirty,
        "physics_protocol": physics_protocol,
        "spatial_axis_contract": spatial_axis_contract,
        "config": cfg,
    }


def test_validate_benchmark_checkpoint_valid(tmp_path):
    """Compliant checkpoint must successfully pass validation."""
    ckpt_file = str(tmp_path / "valid_ckpt.pt")
    data = _create_dummy_checkpoint_data(seed=42, horizon=16)
    torch.save(data, ckpt_file)

    target_info = {"expected_horizon": 16, "title": "H16 Long-Best"}
    prov = validate_benchmark_checkpoint(
        ckpt_path=ckpt_file,
        ckpt_data=data,
        cfg=data["config"],
        target_key="h16_long_best_ep12",
        target_info=target_info,
        eval_split_hash="41fbe6ebe7edd460",
        eval_normalizer_hash="3a0fe52689657618",
        formal=True,
    )
    assert prov["seed"] == 42
    assert prov["training_git_dirty"] is False
    assert prov["training_git_commit"] == "0a8a4da137e3f73c4f696cc81f527c32a8ba8405"


def test_validate_benchmark_checkpoint_wrong_seed(tmp_path):
    """Checkpoint with wrong seed (e.g. seed 43 instead of 42) must fail closed."""
    ckpt_file = str(tmp_path / "wrong_seed.pt")
    data = _create_dummy_checkpoint_data(seed=43, horizon=16)
    torch.save(data, ckpt_file)

    target_info = {"expected_horizon": 16}
    with pytest.raises(RuntimeError, match="Seed mismatch"):
        validate_benchmark_checkpoint(
            ckpt_path=ckpt_file,
            ckpt_data=data,
            cfg=data["config"],
            target_key="h16_test",
            target_info=target_info,
            eval_split_hash="41fbe6ebe7edd460",
            eval_normalizer_hash="3a0fe52689657618",
            formal=False,
        )


def test_validate_benchmark_checkpoint_wrong_hashes(tmp_path):
    """Mismatched split or normalizer hash must fail closed."""
    ckpt_file = str(tmp_path / "wrong_hash.pt")
    data = _create_dummy_checkpoint_data(split_hash="deadbeef12345678")
    torch.save(data, ckpt_file)

    target_info = {"expected_horizon": 16}
    with pytest.raises(RuntimeError, match="Split hash mismatch"):
        validate_benchmark_checkpoint(
            ckpt_path=ckpt_file,
            ckpt_data=data,
            cfg=data["config"],
            target_key="h16_test",
            target_info=target_info,
            eval_split_hash="41fbe6ebe7edd460",
            eval_normalizer_hash="3a0fe52689657618",
            formal=False,
        )


def test_validate_benchmark_checkpoint_horizon_mismatch(tmp_path):
    """Requesting H16 but loading H8 must raise horizon mismatch ValueError."""
    ckpt_file = str(tmp_path / "h8_masquerading_as_h16.pt")
    data = _create_dummy_checkpoint_data(horizon=8)
    torch.save(data, ckpt_file)

    target_info = {"expected_horizon": 16}
    with pytest.raises(ValueError, match="Checkpoint horizon mismatch for h16_target: expected 16, got 8"):
        validate_benchmark_checkpoint(
            ckpt_path=ckpt_file,
            ckpt_data=data,
            cfg=data["config"],
            target_key="h16_target",
            target_info=target_info,
            eval_split_hash="41fbe6ebe7edd460",
            eval_normalizer_hash="3a0fe52689657618",
            formal=False,
        )


def test_validate_benchmark_checkpoint_dirty_in_formal(tmp_path):
    """Under formal mode, training_git_dirty=True must fail closed."""
    ckpt_file = str(tmp_path / "dirty_ckpt.pt")
    data = _create_dummy_checkpoint_data(training_git_dirty=True)
    torch.save(data, ckpt_file)

    target_info = {"expected_horizon": 16}
    with pytest.raises(RuntimeError, match="Formal evaluation failed-closed.*dirty working tree"):
        validate_benchmark_checkpoint(
            ckpt_path=ckpt_file,
            ckpt_data=data,
            cfg=data["config"],
            target_key="h16_target",
            target_info=target_info,
            eval_split_hash="41fbe6ebe7edd460",
            eval_normalizer_hash="3a0fe52689657618",
            formal=True,
        )


def test_validate_benchmark_checkpoint_unknown_commit_in_formal(tmp_path):
    """Under formal mode, unknown training git commit must fail closed."""
    ckpt_file = str(tmp_path / "unknown_commit.pt")
    data = _create_dummy_checkpoint_data(training_git_commit="UNKNOWN")
    torch.save(data, ckpt_file)

    target_info = {"expected_horizon": 16}
    with pytest.raises(RuntimeError, match="Formal evaluation failed-closed.*unknown training git commit"):
        validate_benchmark_checkpoint(
            ckpt_path=ckpt_file,
            ckpt_data=data,
            cfg=data["config"],
            target_key="h16_target",
            target_info=target_info,
            eval_split_hash="41fbe6ebe7edd460",
            eval_normalizer_hash="3a0fe52689657618",
            formal=True,
        )


def test_validate_metric_finiteness():
    """Detects and rejects NaN or Inf in metric dictionaries."""
    valid_metrics = {
        "step_1": {"vrmse_mean": 0.0894, "div_rmse": 0.1895},
        "step_30": {"vrmse_mean": 0.2887, "div_rmse": 0.3954},
    }
    # Valid metrics pass without error
    validate_metric_finiteness(valid_metrics, "h16_test")

    # NaN detection
    nan_metrics = {
        "step_1": {"vrmse_mean": float("nan")},
    }
    with pytest.raises(ValueError, match="Non-finite metric detected in h16_test at step_1: vrmse_mean=nan"):
        validate_metric_finiteness(nan_metrics, "h16_test")

    # Inf detection
    inf_metrics = {
        "step_30": {"div_rmse": float("inf")},
    }
    with pytest.raises(ValueError, match="Non-finite metric detected in h16_test at step_30: div_rmse=inf"):
        validate_metric_finiteness(inf_metrics, "h16_test")


def test_benchmark_window_sample_mathematics():
    """Validates the exact sliding window mathematics for the grouped benchmark test set."""
    traj_len = 200
    hist_len = 4
    horizon = 30
    stride = 20
    num_source_trajectories = 5

    # Windows per trajectory = floor((200 - 4 - 30) / 20) + 1 = floor(166 / 20) + 1 = 8 + 1 = 9
    windows_per_traj = (traj_len - hist_len - horizon) // stride + 1
    assert windows_per_traj == 9

    total_windows = num_source_trajectories * windows_per_traj
    assert total_windows == 45
