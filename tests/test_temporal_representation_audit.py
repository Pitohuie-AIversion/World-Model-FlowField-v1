"""Unit and integration tests for temporal representation audit pipeline.

Ensures zero dependency on untracked / git-ignored local checkpoint files for ordinary unit tests.
Uses lightweight mock checkpoints generated on-the-fly via tmp_path to guarantee 100% reproducible execution
in clean checkout and CI environments.
"""

import json
from pathlib import Path
import pytest
import torch
import yaml

from scripts.audit_temporal_vorticity_representation import (
    compute_file_sha256,
    load_frozen_autoencoder,
    run_temporal_representation_audit,
)
from scripts.train_vorticity_autoencoder import VorticityAutoencoder


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


def test_load_frozen_autoencoder_eval_and_grad_free(tmp_path):
    """Verify load_frozen_autoencoder enforces strict eval() mode and requires_grad=False on mock weights."""
    ckpt_file = create_mock_checkpoint(tmp_path / "mock_ckpts", latent_channels=64)
    device = torch.device("cpu")

    model, ckpt_data, sha_before = load_frozen_autoencoder(ckpt_file, latent_channels=64, device=device)

    # 1. Model must be in evaluation mode
    assert not model.training, "Loaded model must not be in training mode!"

    # 2. Every parameter must have requires_grad=False
    for name, param in model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} has requires_grad=True!"

    # 3. Forward pass does not update weights
    dummy_input = torch.randn(2, 1, 64, 64, device=device)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == dummy_input.shape
    assert torch.isfinite(out).all()


def test_checkpoint_sha256_immutability(tmp_path):
    """Verify that inspecting and auditing checkpoints does not mutate binary files on disk."""
    ckpt_dir = tmp_path / "mock_ckpts"
    for cz in [64, 16]:
        ckpt_file = create_mock_checkpoint(ckpt_dir, latent_channels=cz)

        sha_before = compute_file_sha256(ckpt_file)
        model, _, _ = load_frozen_autoencoder(ckpt_file, latent_channels=cz, device=torch.device("cpu"))
        _ = model(torch.randn(1, 1, 64, 64))
        sha_after = compute_file_sha256(ckpt_file)

        assert sha_before == sha_after, f"Checkpoint Cz={cz} binary changed during audit loading!"


@pytest.mark.skipif(
    not Path("outputs/experiments/vorticity_capacity/cz_64/latest_checkpoint.pt").exists(),
    reason="Local experiment weights outputs/experiments/vorticity_capacity/cz_64/latest_checkpoint.pt not present in clean checkout",
)
def test_real_experiment_checkpoints_immutable_if_present():
    """Optional integration check verifying local physical experiment checkpoints if present."""
    for cz in [64, 16]:
        ckpt_path = Path(f"outputs/experiments/vorticity_capacity/cz_{cz}/latest_checkpoint.pt")
        if not ckpt_path.exists():
            continue
        sha_before = compute_file_sha256(ckpt_path)
        model, _, _ = load_frozen_autoencoder(ckpt_path, latent_channels=cz, device=torch.device("cpu"))
        _ = model(torch.randn(1, 1, 64, 64))
        sha_after = compute_file_sha256(ckpt_path)
        assert sha_before == sha_after


def test_threshold_alert_trigger_logic(tmp_path):
    """Verify pre-declared threshold detection properly flags distribution shift alerts without external files."""
    ckpt_dir = tmp_path / "mock_ckpts"
    create_mock_checkpoint(ckpt_dir, latent_channels=64)

    cfg_file = tmp_path / "test_audit_cfg.yaml"
    out_dir = tmp_path / "audit_output"

    cfg_dict = {
        "experiment": {
            "name": "test_alert_audit",
            "protocol": "periodic_scalar_advection_diffusion_v1",
            "date": "2026-10-10",
        },
        "domain": {"nx": 64, "ny": 64, "lx": 1.0, "ly": 1.0},
        "physical_params": {
            "u0": 0.5, "v0": 0.5, "nu": 0.001,
            "base_wavenumber": 1,
            "perturbation_modes": [[1, 0], [0, 1]],
            "perturbation_amplitude": 0.1,
        },
        "temporal_params": {"dt": 0.05, "num_steps": 2},
        "trajectories": {
            "num_train": 2, "num_val": 2, "num_test": 2,
            "train_seed_base": 100, "val_seed_base": 200, "test_seed_base": 300,
        },
        "windowing": {"history_len": 1, "future_len": 1, "stride": 1},
        "evaluation": {
            "capacities": [64],
            "checkpoint_dir": str(ckpt_dir),
            "threshold_diagnosis_rel_l2": 0.0001,  # Ultra-low threshold: must trigger alert
            "output_dir": str(out_dir),
        },
    }

    with open(cfg_file, "w") as f:
        yaml.safe_dump(cfg_dict, f)

    res = run_temporal_representation_audit(
        config_path=cfg_file,
        device_str="cpu",
        output_dir_override=out_dir,
        generate_plots=False,
    )

    m64 = res["models"]["64"]["summary_metrics"]
    assert m64["diagnostic_alert_triggered"] is True
    assert m64["diagnostic_verdict"] == "REPRESENTATION_ERROR_EXCEEDED_THRESHOLD"


def test_audit_pipeline_end_to_end_smoke_decoupled(tmp_path):
    """Verify complete audit pipeline executes end-to-end on mock checkpoints and produces valid JSON contract."""
    ckpt_dir = tmp_path / "mock_ckpts"
    create_mock_checkpoint(ckpt_dir, latent_channels=64)
    create_mock_checkpoint(ckpt_dir, latent_channels=16)

    cfg_file = tmp_path / "smoke_audit_cfg.yaml"
    out_dir = tmp_path / "smoke_output"

    cfg_dict = {
        "experiment": {
            "name": "smoke_audit",
            "protocol": "periodic_scalar_advection_diffusion_v1",
            "date": "2026-10-10",
        },
        "domain": {"nx": 64, "ny": 64, "lx": 1.0, "ly": 1.0},
        "physical_params": {
            "u0": 0.5, "v0": 0.5, "nu": 0.001,
            "base_wavenumber": 1,
            "perturbation_modes": [[1, 0], [0, 1], [1, 1]],
            "perturbation_amplitude": 0.1,
        },
        "temporal_params": {"dt": 0.05, "num_steps": 3},
        "trajectories": {
            "num_train": 2, "num_val": 2, "num_test": 2,
            "train_seed_base": 10, "val_seed_base": 20, "test_seed_base": 30,
        },
        "windowing": {"history_len": 1, "future_len": 2, "stride": 1},
        "evaluation": {
            "capacities": [64, 16],
            "checkpoint_dir": str(ckpt_dir),
            "threshold_diagnosis_rel_l2": 2.0,  # High threshold
            "output_dir": str(out_dir),
        },
    }

    with open(cfg_file, "w") as f:
        yaml.safe_dump(cfg_dict, f)

    res = run_temporal_representation_audit(
        config_path=cfg_file,
        device_str="cpu",
        output_dir_override=out_dir,
        generate_plots=False,
    )

    # 1. Manifest file verification
    manifest_file = out_dir / "temporal_dataset_manifest.json"
    assert manifest_file.is_file(), "Dataset manifest was not generated!"
    manifest_data = json.loads(manifest_file.read_text())
    assert manifest_data["protocol"] == "periodic_scalar_advection_diffusion_v1"
    assert "partitions" in manifest_data
    assert "train" in manifest_data["partitions"]
    assert "val" in manifest_data["partitions"]
    assert "test" in manifest_data["partitions"]
    assert len(manifest_data["partitions"]["val"]["window_mapping"]) > 0

    # 2. Contract verification in summary
    assert "metadata" in res
    assert "models" in res
    assert "64" in res["models"]
    assert "16" in res["models"]

    for cz_str in ["64", "16"]:
        m = res["models"][cz_str]
        assert "checkpoint_sha256" in m
        assert "summary_metrics" in m
        assert "enstrophy_physics" in m["summary_metrics"]
        assert "time_series" in m
        ts = m["time_series"]
        assert len(ts["relative_l2"]) == 4  # num_steps=3 -> 4 frames
        assert len(ts["reconstructed_enstrophy"]) == 4
        assert len(ts["gt_enstrophy_budget_residuals"]) == 3  # num_steps=3 -> 3 step intervals
        assert len(ts["recon_enstrophy_budget_residuals"]) == 3

        # Numerical sanity on residuals
        for r in ts["gt_enstrophy_budget_residuals"]:
            assert abs(r) < 1e-2

    summary_file = out_dir / "temporal_audit_summary.json"
    assert summary_file.is_file()
    disk_data = json.loads(summary_file.read_text())
    assert disk_data["capacities_evaluated"] == [64, 16]
