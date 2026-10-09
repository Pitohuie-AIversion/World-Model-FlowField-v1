"""Unit and regression tests for Vorticity Autoencoder Capacity Comparison Experiment."""

from pathlib import Path
import json
import pytest
import torch

from scripts.run_vorticity_capacity_comparison import (
    run_capacity_comparison_experiment,
    run_single_capacity_evaluation,
)
from scripts.train_vorticity_autoencoder import (
    VorticityAutoencoder,
    compute_capacity_metadata,
    generate_synthetic_vorticity_dataset,
)


def test_capacity_metadata_math_consistency():
    """Verify element compression ratio and dimensions for Cz in {64, 32, 16}."""
    # Cz = 64 -> 1.0x
    meta_64 = compute_capacity_metadata(64, 64, 1, 64)
    assert meta_64["input_elements"] == 4096
    assert meta_64["latent_elements"] == 4096
    assert meta_64["element_compression_ratio"] == 1.0

    # Cz = 32 -> 2.0x
    meta_32 = compute_capacity_metadata(64, 64, 1, 32)
    assert meta_32["input_elements"] == 4096
    assert meta_32["latent_elements"] == 2048
    assert meta_32["element_compression_ratio"] == 2.0

    # Cz = 16 -> 4.0x
    meta_16 = compute_capacity_metadata(64, 64, 1, 16)
    assert meta_16["input_elements"] == 4096
    assert meta_16["latent_elements"] == 1024
    assert meta_16["element_compression_ratio"] == 4.0


def test_single_capacity_evaluation_smoke():
    """Verify run_single_capacity_evaluation produces all required metric fields on dummy model."""
    device = torch.device("cpu")
    model = VorticityAutoencoder(in_channels=1, out_channels=1, latent_channels=32, base_channels=16)
    val_data = generate_synthetic_vorticity_dataset(num_samples=4, nx=32, ny=32, seed=123)

    eval_res = run_single_capacity_evaluation(model, val_data, device=device, domain_size=(1.0, 1.0))

    required_keys = {
        "relative_l2",
        "absolute_l2",
        "variance_ratio",
        "pairwise_difference_error",
        "mean_valid_spectrum_ratio",
        "spurious_energy_in_zero_bins",
        "total_parameters",
        "latency_ms_per_sample",
        "reconstruction_tensor",
    }
    assert required_keys <= eval_res.keys()
    assert eval_res["relative_l2"] is not None and eval_res["relative_l2"] >= 0.0
    assert eval_res["total_parameters"] > 0
    assert eval_res["reconstruction_tensor"].shape == val_data.shape


def test_capacity_comparison_pipeline_end_to_end(tmp_path):
    """Run miniature 1-epoch comparison experiment for Cz in [32, 16] and verify all artifacts exist."""
    dummy_cfg = {
        "model": {"in_channels": 1, "out_channels": 1, "latent_channels": 32, "base_channels": 8},
        "domain": {"nx": 16, "ny": 16, "lx": 1.0, "ly": 1.0},
        "synthetic_data": {"num_train_samples": 8, "num_val_samples": 4, "seed": 42},
        "training": {"batch_size": 4, "epochs": 1, "lr": 1e-3, "weight_decay": 1e-4},
    }
    cfg_file = tmp_path / "mini_config.yaml"
    import yaml
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(dummy_cfg, f)

    out_root = tmp_path / "exp_output"
    summary = run_capacity_comparison_experiment(
        config_path=str(cfg_file),
        capacities=[32, 16],
        output_root=str(out_root),
        epochs=1,
        device="cpu",
    )

    assert "models" in summary
    assert "32" in summary["models"]
    assert "16" in summary["models"]

    # Verify summary JSON is readable and intact
    json_path = out_root / "capacity_comparison_summary.json"
    assert json_path.is_file()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded_summary = json.load(f)
    assert "models" in loaded_summary

    # Verify figure artifacts were generated
    assert (out_root / "capacity_comparison_metrics.png").is_file()
    assert (out_root / "capacity_comparison_spectra.png").is_file()
    assert (out_root / "capacity_comparison_fields.png").is_file()


def test_capacity_comparison_zero_real_data_leakage():
    """Ensure the experiment script contains zero reference to frozen StocBench evaluation data."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_vorticity_capacity_comparison.py"
    content = script_path.read_text(encoding="utf-8")
    assert "step_seed_100.npz" not in content
    assert "traj_seed_42.npy" not in content
    assert "outputs/data/stocbench" not in content
