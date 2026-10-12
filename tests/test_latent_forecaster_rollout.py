"""Tests for Stage 2 B Multi-Step Latent Forecaster Rollout Evaluation."""

import math
from pathlib import Path
import pytest
import torch
import yaml

from scripts.audit_temporal_vorticity_representation import compute_file_sha256
from scripts.evaluate_latent_forecaster_rollout import (
    compute_rel_l2_pointwise,
    run_rollout_evaluation,
)
from src.data.synthetic_advection_diffusion import compute_enstrophy_budget_residual
from src.models.latent_transformer import LatentSTTransformer
from src.models.latent_forecaster_pilot import VorticityLatentForecaster
from scripts.train_vorticity_autoencoder import VorticityAutoencoder


def create_mock_checkpoint(tmp_path: Path, latent_channels: int = 64) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ckpt_file = tmp_path / "mock_ae.pt"
    model = VorticityAutoencoder(
        in_channels=1,
        out_channels=1,
        latent_channels=latent_channels,
        base_channels=32,
        project_pressure=False,
    )
    state = {
        "model_state_dict": model.state_dict(),
        "latent_channels": latent_channels,
        "completed_epoch": 20,
    }
    torch.save(state, ckpt_file)
    return ckpt_file


def test_rollout_budget_residual_exact_physical_formula():
    """Verify that adjacent step budget residual strictly satisfies R = dZ/dt - 0.5*(D_t + D_{t+1})."""
    dt = 0.05
    # Case 1: Perfect dissipation balance (dZ/dt = D(t) <= 0)
    # If enstrophy drops by 0.004 in 0.05s -> rate is -0.08
    # If dissipation rate is -0.08, residual must be identically zero.
    z_t = 0.500
    z_next = 0.496
    diss_t = -0.080
    diss_next = -0.080

    r_bal = compute_enstrophy_budget_residual(z_t, z_next, diss_t, diss_next, dt=dt)
    assert abs(r_bal - 0.0) < 1e-7, f"Expected exact 0.0 balance, got {r_bal}"

    # Case 2: Numerical distinction when dissipation is attenuated (D is less negative, e.g. -0.04)
    # dZ/dt = -0.08, avg_diss = -0.04 -> R = -0.08 - (-0.04) = -0.04
    r_att = compute_enstrophy_budget_residual(z_t, z_next, -0.040, -0.040, dt=dt)
    assert abs(r_att - (-0.04)) < 1e-6


def test_pointwise_rel_l2_norm_properties():
    """Verify compute_rel_l2_pointwise behaves correctly across batches and spatial dims."""
    a = torch.ones(2, 3, 1, 16, 16)
    b = torch.ones(2, 3, 1, 16, 16) * 1.1

    err = compute_rel_l2_pointwise(b, a)
    assert err.shape == (2, 3, 1)
    # Relative error should be exactly 0.1
    assert torch.allclose(err, torch.tensor(0.1), atol=1e-4)


def test_rollout_evaluation_mock_smoke(tmp_path):
    """Verify run_rollout_evaluation runs end-to-end decoupled on mock weights."""
    mock_ae = create_mock_checkpoint(tmp_path / "mock_ae", latent_channels=64)
    sha_ae_orig = compute_file_sha256(mock_ae)

    # Mock transformer
    tf = LatentSTTransformer(
        latent_channels=64,
        embed_dim=64,
        cond_dim=32,
        depth=1,
        num_heads=2,
        history_length=2,
        prediction_mode="direct",
    )
    mock_tf_ckpt = tmp_path / "mock_transformer.pt"
    payload = {
        "transformer_state_dict": tf.state_dict(),
        "transformer_config": {
            "latent_channels": 64,
            "embed_dim": 64,
            "cond_dim": 32,
            "depth": 1,
            "num_heads": 2,
            "history_length": 2,
            "prediction_mode": "direct",
        },
        "ae_reference": {
            "ae_checkpoint_path": str(mock_ae),
            "ae_checkpoint_sha256": sha_ae_orig,
            "latent_channels": 64,
        },
        "completed_epoch": 5,
    }
    torch.save(payload, mock_tf_ckpt)
    sha_tf_orig = compute_file_sha256(mock_tf_ckpt)

    # Mock config
    cfg_file = tmp_path / "mock_rollout_cfg.yaml"
    cfg_data = {
        "domain": {"nx": 16, "ny": 16, "lx": 1.0, "ly": 1.0},
        "physical_params": {
            "u0": 0.5, "v0": 0.5, "nu": 0.001,
            "base_wavenumber": 1,
            "perturbation_modes": [[1, 0]],
            "perturbation_amplitude": 0.1,
        },
        "temporal_params": {"dt": 0.05, "num_steps": 5, "total_time": 0.25},
        "windowing": {"history_len": 2, "future_len": 1, "stride": 1},
    }
    with open(cfg_file, "w") as f:
        yaml.safe_dump(cfg_data, f)

    out_json = tmp_path / "mock_eval_results.json"

    res = run_rollout_evaluation(
        transformer_checkpoint_path=mock_tf_ckpt,
        ae_checkpoint_path=mock_ae,
        config_path=cfg_file,
        horizons=[2],
        output_json_path=out_json,
        device_str="cpu",
        num_test_trajectories=2,
        test_seed_base=3000,
    )

    # 1. Output file assertion
    assert out_json.is_file()
    assert "controlled_common_window_comparison" in res
    assert "sliding_window_multi_horizon" in res
    assert "H_2" in res["controlled_common_window_comparison"]
    assert "H_2" in res["sliding_window_multi_horizon"]

    h2_ctrl_stats = res["controlled_common_window_comparison"]["H_2"]["statistics"]
    assert "final_step_rel_l2" in h2_ctrl_stats
    assert "horizon_mean_rel_l2" in h2_ctrl_stats
    assert "final_step_persist_rel_l2" in h2_ctrl_stats
    assert "latent_norm_drift_ratio" in h2_ctrl_stats
    assert "final_step_latent_variance" in h2_ctrl_stats
    assert "enstrophy_budget_residual_model" in h2_ctrl_stats
    assert "enstrophy_budget_residual_delta" in h2_ctrl_stats

    # Student-t distribution assertion
    assert "Student-t" in h2_ctrl_stats["final_step_rel_l2"]["distribution"]

    # 2. Checkpoint immutability assertion
    assert compute_file_sha256(mock_tf_ckpt) == sha_tf_orig
    assert compute_file_sha256(mock_ae) == sha_ae_orig


def test_rollout_deterministic_prefix_consistency(tmp_path):
    """Verify that deterministic autoregressive rollout satisfies prefix identity: q_hat_{1:4} == q_hat_{1:16}[:4]."""
    mock_ae = create_mock_checkpoint(tmp_path / "mock_ae_prefix", latent_channels=64)
    tf = LatentSTTransformer(
        latent_channels=64,
        embed_dim=64,
        cond_dim=32,
        depth=1,
        num_heads=2,
        history_length=2,
        prediction_mode="direct",
    )
    mock_tf_ckpt = tmp_path / "mock_transformer_prefix.pt"
    payload = {
        "transformer_state_dict": tf.state_dict(),
        "transformer_config": {
            "latent_channels": 64,
            "embed_dim": 64,
            "cond_dim": 32,
            "depth": 1,
            "num_heads": 2,
            "history_length": 2,
            "prediction_mode": "direct",
        },
        "ae_reference": {
            "ae_checkpoint_path": str(mock_ae),
            "ae_checkpoint_sha256": compute_file_sha256(mock_ae),
            "latent_channels": 64,
        },
        "completed_epoch": 5,
    }
    torch.save(payload, mock_tf_ckpt)

    forecaster = VorticityLatentForecaster.load_trained(
        transformer_checkpoint_path=mock_tf_ckpt,
        ae_checkpoint_path=mock_ae,
        device=torch.device("cpu"),
    )
    forecaster.eval()

    torch.manual_seed(999)
    # History length = 2, spatial 16x16
    q_hist = torch.randn(1, 2, 1, 16, 16)
    with torch.no_grad():
        lat_hist = forecaster.encode_history(q_hist)
        lat_rollout_4 = forecaster.rollout_latent(lat_hist, horizon=4)
        lat_rollout_8 = forecaster.rollout_latent(lat_hist, horizon=8)
        lat_rollout_16 = forecaster.rollout_latent(lat_hist, horizon=16)

        # 1. Latent space prefix consistency
        assert torch.allclose(lat_rollout_4, lat_rollout_16[:, :4], atol=1e-6)
        assert torch.allclose(lat_rollout_8, lat_rollout_16[:, :8], atol=1e-6)

        # 2. Decoded physical field prefix consistency
        q_pred_4 = forecaster.decode(lat_rollout_4)
        q_pred_8 = forecaster.decode(lat_rollout_8)
        q_pred_16 = forecaster.decode(lat_rollout_16)
        assert torch.allclose(q_pred_4, q_pred_16[:, :4], atol=1e-6)
        assert torch.allclose(q_pred_8, q_pred_16[:, :8], atol=1e-6)


def test_student_t_confidence_interval_statistics():
    """Verify trajectory grouping strictly computes Student-t (df=15) 95% confidence intervals."""
    import numpy as np
    from scripts.evaluate_latent_forecaster_rollout import calc_trajectory_group_stats

    # 16 independent trajectory measurements
    errors = np.linspace(0.05, 0.15, 16)
    stats = calc_trajectory_group_stats(errors)

    assert stats["sample_size"] == 16
    assert abs(stats["mean"] - 0.10) < 1e-6
    # For df=15, t_crit is approx 2.13145 (greater than normal 1.96)
    expected_t_crit = 2.131449545559323
    expected_std = float(np.std(errors, ddof=1))
    expected_half_ci = expected_t_crit * expected_std / math.sqrt(16)

    assert abs(stats["ci95_half_width"] - expected_half_ci) < 1e-5
    assert abs(stats["ci95_low"] - (0.10 - expected_half_ci)) < 1e-5
    assert abs(stats["ci95_high"] - (0.10 + expected_half_ci)) < 1e-5
    assert "df=15" in stats["distribution"]

