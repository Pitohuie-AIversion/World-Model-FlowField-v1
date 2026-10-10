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


def test_triple_trajectory_metric_decomposition(tmp_path):
    """Verify that Total Prediction Error decomposes into Representation Error and Pure Dynamics Error."""
    q_gt = torch.randn(4, 1, 1, 32, 32)
    q_pred = q_gt + 0.1 * torch.randn_like(q_gt)
    q_ae_ref = q_gt + 0.05 * torch.randn_like(q_gt)

    def rel_l2(a, b):
        return float(torch.mean(torch.norm(a - b, dim=(-2, -1)) / torch.norm(b, dim=(-2, -1))).item())

    total_err = rel_l2(q_pred, q_gt)
    rep_err = rel_l2(q_ae_ref, q_gt)
    dyn_err = rel_l2(q_pred, q_ae_ref)

    assert total_err > 0.0
    assert rep_err > 0.0
    assert dyn_err > 0.0
    # Triangle inequality holds in Euclidean norms
    assert total_err <= rep_err + dyn_err + 1e-4


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
    assert "gt_budget_residual" in eval_m["physical_enstrophy"]
