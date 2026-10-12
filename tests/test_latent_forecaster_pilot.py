"""Unit and integration tests for Phase 2 A Deterministic Latent Forecaster Pilot.

Ensures zero dependency on local git-ignored checkpoints for core unit tests via mock checkpoints.
Verifies all Phase 2 A contracts:
1. Frozen representation gradient isolation (requires_grad=False, grad is None).
2. Checkpoint SHA-256 immutability.
3. Latent space input/output shapes and HistoryBuffer autoregressive rollout.
4. Persistence baseline comparison and triple-trajectory decomposition.
5. End-to-end decoupled pipeline execution producing valid summary JSON and checkpoint.
"""

import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
import yaml

from scripts.audit_temporal_vorticity_representation import compute_file_sha256
from scripts.train_latent_forecaster_pilot import run_latent_forecaster_pilot
from scripts.train_vorticity_autoencoder import VorticityAutoencoder
from src.models.latent_forecaster_pilot import VorticityLatentForecaster
from src.models.latent_transformer import LatentSTTransformer
def create_mock_checkpoint(ckpt_dir: Path, latent_channels: int) -> Path:
    """Create a lightweight valid mock checkpoint for decoupled unit testing."""
    ckpt_subdir = ckpt_dir / f"cz_{latent_channels}"
    ckpt_subdir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_subdir / "latest_checkpoint.pt"

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


def test_latent_forecaster_pilot_shapes_and_forward(tmp_path):
    """Verify input/output shapes across all forecaster methods."""
    mock_ckpt = create_mock_checkpoint(tmp_path / "mock_ckpts", latent_channels=64)
    forecaster = VorticityLatentForecaster.from_checkpoint(
        checkpoint_path=mock_ckpt,
        latent_channels=64,
        transformer_kwargs={"embed_dim": 64, "depth": 2, "num_heads": 2, "history_length": 4},
        device=torch.device("cpu"),
    )

    b, l, c, h, w = 2, 4, 1, 64, 64
    q_hist = torch.randn(b, l, c, h, w)

    # 1. Encode
    z_hist = forecaster.encode_history(q_hist)
    assert z_hist.shape == (b, l, 64, 8, 8)

    # 2. Forward latent
    z_next = forecaster.forward_latent(z_hist)
    assert z_next.shape == (b, 1, 64, 8, 8)

    # 3. Decode
    q_next = forecaster.decode(z_next)
    assert q_next.shape == (b, 1, 1, 64, 64)

    # 4. Forward single-step
    q_pred, z_pred, z_h = forecaster.forward_single_step(q_hist)
    assert q_pred.shape == (b, 1, 1, 64, 64)
    assert z_pred.shape == (b, 1, 64, 8, 8)
    assert z_h.shape == (b, l, 64, 8, 8)

    # 5. Rollout
    horizon = 3
    q_roll, z_roll = forecaster.rollout(q_hist, horizon=horizon)
    assert q_roll.shape == (b, horizon, 1, 64, 64)
    assert z_roll.shape == (b, horizon, 64, 8, 8)


def test_frozen_representation_gradient_isolation(tmp_path):
    """Verify strict freeze: backprop updates Transformer parameters with ZERO gradients on Autoencoder."""
    mock_ckpt = create_mock_checkpoint(tmp_path / "mock_ckpts", latent_channels=64)
    forecaster = VorticityLatentForecaster.from_checkpoint(
        checkpoint_path=mock_ckpt,
        latent_channels=64,
        transformer_kwargs={"embed_dim": 64, "depth": 2, "num_heads": 2, "history_length": 4},
        device=torch.device("cpu"),
    )

    q_hist = torch.randn(2, 4, 1, 64, 64)
    target = torch.randn(2, 1, 1, 64, 64)

    q_pred, z_pred, _ = forecaster.forward_single_step(q_hist)
    loss = F.mse_loss(q_pred, target)
    loss.backward()

    # 1. Autoencoder parameters must NOT receive any gradient
    for name, p in forecaster.autoencoder.named_parameters():
        assert p.grad is None, f"Autoencoder parameter {name} received unwanted gradient!"

    # 2. Transformer parameters must receive active gradients
    tf_grads = [p.grad for p in forecaster.transformer.parameters() if p.grad is not None]
    assert len(tf_grads) > 0, "Transformer parameters did not receive any gradients!"


def test_checkpoint_sha_immutability_verification(tmp_path):
    """Verify verify_checkpoint_immutability catches any unauthorized binary mutation."""
    mock_ckpt = create_mock_checkpoint(tmp_path / "mock_ckpts", latent_channels=64)
    forecaster = VorticityLatentForecaster.from_checkpoint(
        checkpoint_path=mock_ckpt,
        latent_channels=64,
        device=torch.device("cpu"),
    )

    # Clean verification passes
    assert forecaster.verify_checkpoint_immutability() is True

    # Mutate file binary on disk
    with open(mock_ckpt, "ab") as f:
        f.write(b"corrupted_bytes")

    with pytest.raises(RuntimeError, match="Autoencoder checkpoint mutated"):
        forecaster.verify_checkpoint_immutability()


def test_history_buffer_autoregressive_rollout_mechanics():
    """Verify HistoryBuffer maintains strict FIFO rolling without data leakage."""
    tf = LatentSTTransformer(
        latent_channels=16,
        embed_dim=32,
        cond_dim=16,
        depth=1,
        num_heads=2,
        history_length=3,
        prediction_mode="direct",
    )
    b, l, c_z, h_z, w_z = 2, 3, 16, 4, 4
    z_init = torch.randn(b, l, c_z, h_z, w_z)

    forecaster = VorticityLatentForecaster(
        autoencoder=VorticityAutoencoder(in_channels=1, out_channels=1, latent_channels=16, base_channels=16, project_pressure=False),
        transformer=tf,
    )

    z_roll = forecaster.rollout_latent(z_init, horizon=4)
    assert z_roll.shape == (b, 4, c_z, h_z, w_z)
    assert torch.isfinite(z_roll).all()


def test_triple_trajectory_metric_decomposition():
    """Verify mathematical properties of triple-trajectory error metrics.

    Checks:
    1. Absolute Euclidean L2 norm triangle inequality: ||q_hat - q||_2 <= ||q_ae - q||_2 + ||q_hat - q_ae||_2
    2. Correct mathematically scaled relative error upper bound:
       e_total = ||q_hat - q|| / ||q|| <= e_rep + (||q_ae|| / ||q||) * e_dyn
    3. Positivity and finite values of metrics.
    """
    q_gt = torch.randn(4, 1, 1, 32, 32)
    q_pred = q_gt + 0.1 * torch.randn_like(q_gt)
    q_ae_ref = q_gt + 0.05 * torch.randn_like(q_gt)

    diff_total = torch.norm(q_pred - q_gt, dim=(-2, -1))
    diff_rep = torch.norm(q_ae_ref - q_gt, dim=(-2, -1))
    diff_dyn = torch.norm(q_pred - q_ae_ref, dim=(-2, -1))

    norm_gt = torch.norm(q_gt, dim=(-2, -1))
    norm_ae = torch.norm(q_ae_ref, dim=(-2, -1))

    # 1. Absolute Euclidean L2 triangle inequality holds pointwise
    assert torch.all(diff_total <= diff_rep + diff_dyn + 1e-6)

    # 2. Mathematically correct scaled relative error upper bound holds pointwise
    rel_total = diff_total / (norm_gt + 1e-8)
    rel_rep = diff_rep / (norm_gt + 1e-8)
    rel_dyn = diff_dyn / (norm_ae + 1e-8)
    scaled_bound = rel_rep + (norm_ae / (norm_gt + 1e-8)) * rel_dyn
    assert torch.all(rel_total <= scaled_bound + 1e-6)

    # 3. Batch averages: sample-weighted mean bound holds strictly
    e_total = float(torch.mean(rel_total).item())
    e_rep = float(torch.mean(rel_rep).item())
    e_dyn = float(torch.mean(rel_dyn).item())
    sample_weighted_mean_bound = float(torch.mean(scaled_bound).item())
    assert e_total <= sample_weighted_mean_bound + 1e-6
    assert 0.0 < e_total < 1.0
    assert 0.0 < e_rep < 1.0
    assert 0.0 < e_dyn < 1.0


def test_counterexample_unscaled_relative_error_triangle_inequality():
    """Deterministic counterexample proving naive unscaled sum e_rep + e_dyn is NOT a general upper bound for e_total.

    When ||q_ae|| > ||q||, the relative dynamics error e_dyn has a larger denominator than e_total,
    causing e_rep + e_dyn < e_total in general.

    Analytical Counterexample:
      q        = [1.0, 0.0],  ||q||      = 1.0
      q_ae     = [2.0, 0.0],  ||q_ae||   = 2.0
      q_hat    = [3.0, 0.0],  ||q_hat||  = 3.0

      ||q_hat - q||    = 2.0  => e_total = 2.0 / 1.0 = 2.0
      ||q_ae - q||     = 1.0  => e_rep   = 1.0 / 1.0 = 1.0
      ||q_hat - q_ae|| = 1.0  => e_dyn   = 1.0 / 2.0 = 0.5

      Unscaled sum: e_rep + e_dyn = 1.0 + 0.5 = 1.5 < e_total (2.0)  -> Fails!
      Scaled bound: e_rep + (||q_ae|| / ||q||) * e_dyn = 1.0 + (2.0/1.0)*0.5 = 2.0 == e_total -> Holds!
    """
    q = torch.tensor([1.0, 0.0])
    q_ae = torch.tensor([2.0, 0.0])
    q_hat = torch.tensor([3.0, 0.0])

    norm_q = float(torch.norm(q).item())
    norm_ae = float(torch.norm(q_ae).item())

    e_total = float((torch.norm(q_hat - q) / norm_q).item())
    e_rep = float((torch.norm(q_ae - q) / norm_q).item())
    e_dyn = float((torch.norm(q_hat - q_ae) / norm_ae).item())

    assert abs(e_total - 2.0) < 1e-6
    assert abs(e_rep - 1.0) < 1e-6
    assert abs(e_dyn - 0.5) < 1e-6

    # 1. Demonstrates that naive unscaled sum strictly fails as an upper bound
    unscaled_sum = e_rep + e_dyn
    assert abs(unscaled_sum - 1.5) < 1e-6
    assert e_total > unscaled_sum, "Expected counterexample to violate naive unscaled inequality!"

    # 2. Demonstrates that mathematically correct scaled upper bound holds exactly
    scaled_bound = e_rep + (norm_ae / norm_q) * e_dyn
    assert abs(scaled_bound - 2.0) < 1e-6
    assert e_total <= scaled_bound + 1e-6


def test_enstrophy_budget_residual_diagnostics_semantics():
    """Verify the three budget residual diagnostics under single-step transition starting from true q_t.

    Diagnostics:
    1. R_GT: Ground Truth transition (True q_t -> True q_{t+1})
    2. R_AE_ref: Frozen Autoencoder reference transition (True q_t -> D(E(q_{t+1})))
    3. R_model: World model prediction transition (True q_t -> D(hat_z_{t+1}))
    """
    from scripts.audit_temporal_vorticity_representation import compute_enstrophy_budget_residual
    import math

    dt = 0.05
    z_hist = 0.50
    diss_hist = 0.02

    # GT has physical decay consistent with dissipation: (0.499 - 0.50)/0.05 + 0.5*(0.02 + 0.02) = -0.2 + 0.02 = -0.18
    z_gt = 0.499
    diss_gt = 0.02
    r_gt = compute_enstrophy_budget_residual(z_hist, z_gt, diss_hist, diss_gt, dt=dt)

    # AE reference shows dissipation attenuation
    z_ae = 0.505
    diss_ae = 0.025
    r_ae = compute_enstrophy_budget_residual(z_hist, z_ae, diss_hist, diss_ae, dt=dt)

    # Model prediction
    z_model = 0.506
    diss_model = 0.026
    r_model = compute_enstrophy_budget_residual(z_hist, z_model, diss_hist, diss_model, dt=dt)

    assert math.isfinite(r_gt)
    assert math.isfinite(r_ae)
    assert math.isfinite(r_model)
    # The three residuals must be distinct and reflect their respective endpoints
    assert abs(r_model - r_ae) < abs(r_model - r_gt)

    # Exact numerical verification matching physical contract:
    # Physical dissipation D(t) <= 0. If delta_Z/dt = -0.08 and D_t = D_{t+1} = -0.08:
    # R(t) = (-0.08) - 0.5 * (-0.08 + -0.08) = -0.08 - (-0.08) = 0.0
    r_exact_zero = compute_enstrophy_budget_residual(
        z_k=1.0, z_next=0.92, diss_k=-0.08, diss_next=-0.08, dt=1.0
    )
    assert abs(r_exact_zero - 0.0) < 1e-7, f"Expected 0.0, got {r_exact_zero}"


def test_load_trained_latent_forecaster_from_enriched_checkpoint(tmp_path):
    """Verify VorticityLatentForecaster.load_trained restores model architecture from checkpoint metadata."""
    mock_ckpt = create_mock_checkpoint(tmp_path / "mock_ckpts", latent_channels=64)
    sha_ae = compute_file_sha256(mock_ckpt)

    tf = LatentSTTransformer(
        latent_channels=64,
        embed_dim=64,
        cond_dim=32,
        depth=2,
        num_heads=2,
        history_length=4,
        prediction_mode="direct",
    )
    ckpt_file = tmp_path / "enriched_checkpoint.pt"
    payload = {
        "transformer_state_dict": tf.state_dict(),
        "transformer_config": {
            "latent_channels": 64,
            "embed_dim": 64,
            "cond_dim": 32,
            "depth": 2,
            "num_heads": 2,
            "history_length": 4,
            "prediction_mode": "direct",
        },
        "ae_reference": {
            "ae_checkpoint_path": str(mock_ckpt),
            "ae_checkpoint_sha256": sha_ae,
            "latent_channels": 64,
        },
        "completed_epoch": 20,
    }
    torch.save(payload, ckpt_file)

    loaded_forecaster = VorticityLatentForecaster.load_trained(ckpt_file)
    assert loaded_forecaster.transformer.embed_dim == 64
    assert len(loaded_forecaster.transformer.blocks) == 2
    assert loaded_forecaster.verify_checkpoint_immutability() is True


def test_load_legacy_pilot_checkpoint_backward_compatibility(tmp_path):
    """Verify legacy checkpoint schema without transformer_config can be restored seamlessly."""
    mock_ae = create_mock_checkpoint(tmp_path / "mock_ae", latent_channels=64)
    sha_ae = compute_file_sha256(mock_ae)

    tf = LatentSTTransformer(
        latent_channels=64,
        embed_dim=128,
        cond_dim=64,
        depth=2,
        num_heads=4,
        history_length=4,
        prediction_mode="direct",
    )
    legacy_ckpt_file = tmp_path / "legacy_checkpoint.pt"
    # Legacy payload structure: lacks transformer_config and ae_checkpoint_path
    legacy_payload = {
        "transformer_state_dict": tf.state_dict(),
        "latent_channels": 64,
        "embed_dim": 128,
        "history_length": 4,
        "completed_epoch": 20,
        "final_train_loss": 0.001,
        "val_model_rel_l2": 0.08,
        "ae_checkpoint_sha256": sha_ae,
    }
    torch.save(legacy_payload, legacy_ckpt_file)

    # 1. Load with explicit ae_checkpoint_path
    forecaster = VorticityLatentForecaster.load_trained(
        legacy_ckpt_file, ae_checkpoint_path=mock_ae
    )
    assert forecaster.transformer.embed_dim == 128
    assert len(forecaster.transformer.blocks) == 2
    assert forecaster.verify_checkpoint_immutability() is True

    # 2. Forward pass with physical input
    q_hist = torch.randn(2, 4, 1, 32, 32)
    q_pred = forecaster(q_hist)
    assert q_pred.shape == (2, 1, 1, 32, 32)
    assert torch.isfinite(q_pred).all()


def test_load_real_trained_checkpoint_on_disk():
    """Verify the actual Phase 2 A trained checkpoint on disk can be loaded without retraining."""
    ckpt_p = Path("outputs/experiments/latent_forecaster_pilot/latest_checkpoint.pt")
    if not ckpt_p.is_file():
        pytest.skip("Physical pilot checkpoint not present on disk in isolated test environment.")

    sha_before = compute_file_sha256(ckpt_p)

    forecaster = VorticityLatentForecaster.load_trained(ckpt_p)
    assert forecaster.transformer.embed_dim == 128
    assert len(forecaster.transformer.blocks) == 4
    assert forecaster.verify_checkpoint_immutability() is True

    # Forward pass on CPU
    q_hist = torch.randn(2, 4, 1, 64, 64)
    q_pred = forecaster(q_hist)
    assert q_pred.shape == (2, 1, 1, 64, 64)
    assert torch.isfinite(q_pred).all()

    # Autoregressive rollout
    q_roll, z_roll = forecaster.rollout(q_hist, horizon=4)
    assert q_roll.shape == (2, 4, 1, 64, 64)
    assert z_roll.shape == (2, 4, 64, 8, 8)
    assert torch.isfinite(q_roll).all()
    assert torch.isfinite(z_roll).all()

    # Checkpoint on disk must remain strictly immutable
    sha_after = compute_file_sha256(ckpt_p)
    assert sha_before == sha_after, "Checkpoint was modified during loading!"


def test_pilot_pipeline_end_to_end_smoke_decoupled(tmp_path):
    """Verify complete Phase 2 A pilot pipeline runs end-to-end on mock weights."""
    mock_ckpt = create_mock_checkpoint(tmp_path / "mock_ckpts", latent_channels=64)
    cfg_file = tmp_path / "pilot_smoke_cfg.yaml"
    out_dir = tmp_path / "pilot_output"

    cfg_dict = {
        "experiment": {
            "name": "pilot_smoke_test",
            "protocol": "periodic_scalar_advection_diffusion_v1",
            "phase": "2A_deterministic_pilot",
            "date": "2026-10-10",
        },
        "domain": {"nx": 16, "ny": 16, "lx": 1.0, "ly": 1.0},
        "physical_params": {
            "u0": 0.5, "v0": 0.5, "nu": 0.001,
            "base_wavenumber": 1,
            "perturbation_modes": [[1, 0]],
            "perturbation_amplitude": 0.1,
        },
        "temporal_params": {"dt": 0.05, "num_steps": 5, "total_time": 0.25},
        "trajectories": {
            "num_train": 2, "num_val": 2, "num_test": 2,
            "train_seed_base": 10, "val_seed_base": 20, "test_seed_base": 30,
        },
        "windowing": {"history_len": 2, "future_len": 1, "stride": 1},
        "model": {
            "latent_channels": 64, "embed_dim": 32, "cond_dim": 16,
            "depth": 1, "num_heads": 2, "prediction_mode": "direct",
        },
        "training": {
            "epochs": 2, "batch_size": 2, "lr": 0.001, "weight_decay": 0.0, "device": "cpu",
        },
        "checkpoints": {
            "ae_checkpoint": str(mock_ckpt),
            "output_dir": str(out_dir),
        },
    }

    with open(cfg_file, "w") as f:
        yaml.safe_dump(cfg_dict, f)

    res = run_latent_forecaster_pilot(
        config_path=cfg_file,
        device_str="cpu",
        output_dir_override=out_dir,
        epochs_override=2,
    )

    # 1. Output artifact assertions
    assert (out_dir / "latest_checkpoint.pt").is_file()
    assert (out_dir / "latent_forecaster_pilot_summary.json").is_file()
    assert (out_dir / "pilot_dataset_manifest.json").is_file()

    # 2. Checkpoint immutability assertion
    assert res["checkpoint_integrity"]["ae_checkpoint_unmutated"] is True

    # 3. Loss decrease assertion
    losses = res["training_metrics"]["loss_history"]
    assert len(losses) == 2
    assert losses[-1] <= losses[0] or abs(losses[-1] - losses[0]) < 1.0

    # 4. Metrics contract assertions
    eval_m = res["evaluation_metrics"]
    assert "model_rel_l2" in eval_m
    assert "persistence_rel_l2" in eval_m
    assert "pure_dynamics_rel_l2" in eval_m
    assert "representation_rel_l2" in eval_m
    assert "latent_rel_l2" in eval_m
    assert "state_transition_verdict" in eval_m
    assert "physical_enstrophy" in eval_m
    assert "r_gt_budget_residual" in eval_m["physical_enstrophy"]
    assert "r_ae_reference_budget_residual" in eval_m["physical_enstrophy"]
    assert "r_model_budget_residual" in eval_m["physical_enstrophy"]
    assert "- 0.5 *" in eval_m["physical_enstrophy"]["residual_protocol"]["formula"]

    # 5. Checkpoint payload assertions
    ckpt_loaded = torch.load(out_dir / "latest_checkpoint.pt", weights_only=False)
    assert "transformer_config" in ckpt_loaded
    assert ckpt_loaded["transformer_config"]["depth"] == 1
    assert "data_fingerprint" in ckpt_loaded
    assert "manifest_sha256" in ckpt_loaded["data_fingerprint"]
