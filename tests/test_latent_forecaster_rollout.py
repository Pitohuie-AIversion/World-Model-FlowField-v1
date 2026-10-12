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


def test_spatial_vs_overall_variance_counterexample():
    """Verify that overall variance cannot detect spatial collapse, proving the necessity of V_spatial.

    Mathematical proof by counterexample:
    If each latent channel collapses to a spatial constant (zero spatial variation),
    the channel-averaged spatial variance V_spatial is identically 0.
    However, if different channels have different constant offsets, the overall element-wise
    variance remains strictly positive, falsely indicating variability.
    """
    # Shape: (B=1, H=1, Cz=4, Hz=8, Wz=8)
    # Channel 0: all 1.0, Channel 1: all 2.0, Channel 2: all 3.0, Channel 3: all 4.0
    channel_constants = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 4, 1, 1)
    z_spatially_collapsed = channel_constants.expand(1, 1, 4, 8, 8).clone()

    # 1. Compute overall element variance across (Cz, Hz, Wz)
    overall_var = torch.var(z_spatially_collapsed, dim=(-3, -2, -1)).item()
    # 2. Compute channel-averaged spatial variance: (1/Cz) * sum_c Var_{x,y}(z_c)
    spatial_var_per_channel = torch.var(z_spatially_collapsed, dim=(-2, -1))
    mean_spatial_var = torch.mean(spatial_var_per_channel, dim=-1).item()

    # Crucial mathematical distinction:
    assert abs(mean_spatial_var - 0.0) < 1e-8, f"Expected 0.0 spatial variance, got {mean_spatial_var}"
    assert overall_var > 1.0, f"Expected positive overall variance, got {overall_var}"

    # Active spatial structure case: sinusoidal grid with identical mean across channels
    x = torch.linspace(0, 2 * math.pi, 8)
    grid = torch.sin(x).unsqueeze(0).expand(8, 8)  # (8, 8)
    z_structured = grid.view(1, 1, 1, 8, 8).expand(1, 1, 4, 8, 8).clone()

    spatial_var_struct = torch.mean(torch.var(z_structured, dim=(-2, -1)), dim=-1).item()
    overall_var_struct = torch.var(z_structured, dim=(-3, -2, -1)).item()
    assert spatial_var_struct > 0.1
    assert overall_var_struct > 0.1


def test_rollout_evaluation_overwrite_protection(tmp_path):
    """Verify run_rollout_evaluation refuses to overwrite existing results unless allow_overwrite=True."""
    mock_ae = create_mock_checkpoint(tmp_path / "mock_ae_ow", latent_channels=64)
    tf = LatentSTTransformer(
        latent_channels=64,
        embed_dim=64,
        cond_dim=32,
        depth=1,
        num_heads=2,
        history_length=2,
        prediction_mode="direct",
    )
    mock_tf_ckpt = tmp_path / "mock_transformer_ow.pt"
    torch.save({
        "transformer_state_dict": tf.state_dict(),
        "transformer_config": {
            "latent_channels": 64, "embed_dim": 64, "cond_dim": 32,
            "depth": 1, "num_heads": 2, "history_length": 2, "prediction_mode": "direct",
        },
        "ae_reference": {
            "ae_checkpoint_path": str(mock_ae),
            "ae_checkpoint_sha256": compute_file_sha256(mock_ae),
            "latent_channels": 64,
        },
        "completed_epoch": 5,
    }, mock_tf_ckpt)

    cfg_file = tmp_path / "mock_rollout_cfg_ow.yaml"
    with open(cfg_file, "w") as f:
        yaml.safe_dump({
            "domain": {"nx": 16, "ny": 16, "lx": 1.0, "ly": 1.0},
            "physical_params": {
                "u0": 0.5, "v0": 0.5, "nu": 0.001,
                "base_wavenumber": 1, "perturbation_modes": [[1, 0]], "perturbation_amplitude": 0.1,
            },
            "temporal_params": {"dt": 0.05, "num_steps": 5, "total_time": 0.25},
            "windowing": {"history_len": 2, "future_len": 1, "stride": 1},
        }, f)

    target_json = tmp_path / "benchmark_record.json"
    target_json.write_text('{"dummy": "old_record"}', encoding="utf-8")

    # 1. Attempting to evaluate without allow_overwrite must raise FileExistsError
    with pytest.raises(FileExistsError) as exc_info:
        run_rollout_evaluation(
            transformer_checkpoint_path=mock_tf_ckpt,
            ae_checkpoint_path=mock_ae,
            config_path=cfg_file,
            horizons=[2],
            output_json_path=target_json,
            allow_overwrite=False,
            num_test_trajectories=2,
        )
    assert "already exists" in str(exc_info.value)
    assert target_json.read_text(encoding="utf-8") == '{"dummy": "old_record"}'

    # 2. Specifying allow_overwrite=True succeeds and safely updates the file
    res = run_rollout_evaluation(
        transformer_checkpoint_path=mock_tf_ckpt,
        ae_checkpoint_path=mock_ae,
        config_path=cfg_file,
        horizons=[2],
        output_json_path=target_json,
        allow_overwrite=True,
        num_test_trajectories=2,
    )
    assert "controlled_common_window_comparison" in res
    assert target_json.read_text(encoding="utf-8") != '{"dummy": "old_record"}'


def test_budget_residual_forecast_step_interval_semantics(tmp_path):
    """Verify that physical budget residuals strictly evaluate H-1 adjacent forecast intervals."""
    mock_ae = create_mock_checkpoint(tmp_path / "mock_ae_res", latent_channels=64)
    tf = LatentSTTransformer(
        latent_channels=64,
        embed_dim=64,
        cond_dim=32,
        depth=1,
        num_heads=2,
        history_length=2,
        prediction_mode="direct",
    )
    mock_tf_ckpt = tmp_path / "mock_transformer_res.pt"
    torch.save({
        "transformer_state_dict": tf.state_dict(),
        "transformer_config": {
            "latent_channels": 64, "embed_dim": 64, "cond_dim": 32,
            "depth": 1, "num_heads": 2, "history_length": 2, "prediction_mode": "direct",
        },
        "ae_reference": {
            "ae_checkpoint_path": str(mock_ae),
            "ae_checkpoint_sha256": compute_file_sha256(mock_ae),
            "latent_channels": 64,
        },
        "completed_epoch": 5,
    }, mock_tf_ckpt)

    cfg_file = tmp_path / "mock_rollout_cfg_res.yaml"
    with open(cfg_file, "w") as f:
        yaml.safe_dump({
            "domain": {"nx": 16, "ny": 16, "lx": 1.0, "ly": 1.0},
            "physical_params": {
                "u0": 0.5, "v0": 0.5, "nu": 0.001,
                "base_wavenumber": 1, "perturbation_modes": [[1, 0]], "perturbation_amplitude": 0.1,
            },
            "temporal_params": {"dt": 0.05, "num_steps": 5, "total_time": 0.25},
            "windowing": {"history_len": 2, "future_len": 1, "stride": 1},
        }, f)

    res = run_rollout_evaluation(
        transformer_checkpoint_path=mock_tf_ckpt,
        ae_checkpoint_path=mock_ae,
        config_path=cfg_file,
        horizons=[2],
        output_json_path=tmp_path / "res_test.json",
        allow_overwrite=True,
        num_test_trajectories=2,
    )

    # For horizon=2, number of forecast-to-forecast intervals is exactly 2 - 1 = 1
    h2_ctrl = res["controlled_common_window_comparison"]["H_2"]
    assert h2_ctrl["num_budget_residual_intervals"] == 1
    meta_contract = res["metadata"]["evaluation_contract"]
    assert "forecast-to-forecast" in meta_contract["budget_residual_interval"]
    assert "git_provenance" in res["metadata"]

    # Verify that metadata.evaluation_contract.physical_interpretation is dynamically generated
    phys_interp = meta_contract["physical_interpretation"]
    assert "R_pred = " in phys_interp
    assert "R_AE = " in phys_interp
    assert "Delta R = " in phys_interp
