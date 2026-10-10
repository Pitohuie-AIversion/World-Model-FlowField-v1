"""Unit and regression tests for Vorticity Autoencoder Capacity Comparison Experiment."""

from pathlib import Path
import json
import pytest
import torch

from scripts.run_vorticity_capacity_comparison import (
    benchmark_model_latency,
    evaluate_existing_checkpoints,
    run_capacity_comparison_experiment,
    run_single_capacity_evaluation,
    verify_dataset_separation,
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


def test_dataset_separation_fail_closed():
    """Verify fail-closed dataset separation assertion blocks identical samples."""
    # 1. Identical dataset -> must raise ValueError
    train_leak = generate_synthetic_vorticity_dataset(num_samples=16, nx=32, ny=32, seed=42)
    val_leak = generate_synthetic_vorticity_dataset(num_samples=4, nx=32, ny=32, seed=42)
    # The first 4 samples of val_leak are identical to the first 4 samples of train_leak
    with pytest.raises(ValueError, match="Data leakage detected"):
        verify_dataset_separation(train_leak, val_leak)

    # 2. Independent dataset (seed + 1000) -> must pass and return positive min distance
    val_clean = generate_synthetic_vorticity_dataset(num_samples=4, nx=32, ny=32, seed=1042)
    min_dist = verify_dataset_separation(train_leak, val_clean)
    assert min_dist > 1e-3


def test_evaluation_dataset_isolation_from_train_split(monkeypatch, tmp_path):
    """Enforce strict dataset isolation on actual invocation path and prove detection of legacy bug.

    1. Buggy version behavior:
       If val_seed == train_seed == 42, all 32 validation samples are exactly identical to the first
       32 samples of the training set. This test proves that verify_dataset_separation fails closed.
    2. Fixed canonical behavior:
       Under canonical train_seed=42 and val_seed=1042, zero samples are identical,
       and minimum pairwise L2 distance exceeds 1.0.
    3. Actual invocation interception:
       Monitors run_single_capacity_evaluation to verify that the tensor actually received
       at runtime has zero element-wise overlap with the training split.
    """
    # Part 1: Prove buggy version fails
    train_128 = generate_synthetic_vorticity_dataset(num_samples=128, nx=64, ny=64, seed=42)
    val_buggy_32 = generate_synthetic_vorticity_dataset(num_samples=32, nx=64, ny=64, seed=42)

    # In buggy version, val_buggy_32 is exactly train_128[:32]
    max_abs_diff_buggy = float(torch.max(torch.abs(val_buggy_32 - train_128[:32])).item())
    assert max_abs_diff_buggy == 0.0
    overlap_count_buggy = sum(torch.equal(val_buggy_32[i], train_128[i]) for i in range(32))
    assert overlap_count_buggy == 32

    with pytest.raises(ValueError, match="Data leakage detected"):
        verify_dataset_separation(train_128, val_buggy_32)

    # Part 2: Prove canonical independent split passes with high margin
    val_clean_32 = generate_synthetic_vorticity_dataset(num_samples=32, nx=64, ny=64, seed=1042)
    # Element-wise disjoint check across all pairs
    overlap_count_clean = 0
    for i in range(32):
        for j in range(128):
            if torch.equal(val_clean_32[i], train_128[j]):
                overlap_count_clean += 1
    assert overlap_count_clean == 0

    min_l2_clean = verify_dataset_separation(train_128, val_clean_32)
    assert min_l2_clean > 1.0  # Real distance is ~2.19

    # Part 3: Intercept actual evaluation invocation path
    captured_val_tensors = []
    original_eval = run_single_capacity_evaluation

    def spy_eval(model, val_data, device, domain_size=(1.0, 1.0)):
        captured_val_tensors.append(val_data.clone())
        return original_eval(model, val_data, device, domain_size)

    monkeypatch.setattr("scripts.run_vorticity_capacity_comparison.run_single_capacity_evaluation", spy_eval)

    # Create miniature mock checkpoint
    ckpt_root = tmp_path / "mock_ckpts"
    cz_dir = ckpt_root / "cz_16"
    cz_dir.mkdir(parents=True)
    m = VorticityAutoencoder(in_channels=1, out_channels=1, latent_channels=16, base_channels=8)
    ckpt_data = {
        "epoch": 1,
        "step": 1,
        "model_state_dict": m.state_dict(),
        "config": {
            "model": {"base_channels": 8},
            "domain": {"nx": 16, "ny": 16, "lx": 1.0, "ly": 1.0},
            "synthetic_data": {"seed": 42, "num_train_samples": 8, "num_val_samples": 4},
        },
    }
    torch.save(ckpt_data, cz_dir / "latest_checkpoint.pt")

    eval_out = tmp_path / "eval_out"
    summary = evaluate_existing_checkpoints(
        checkpoint_dir=str(ckpt_root),
        capacities=[16],
        output_dir=str(eval_out),
        device="cpu",
    )

    # Verify tensor actually intercepted
    assert len(captured_val_tensors) == 1
    actual_val_tensor = captured_val_tensors[0]
    assert actual_val_tensor.shape == (4, 1, 16, 16)

    # Reconstruct training split used by mock config
    mock_train = generate_synthetic_vorticity_dataset(num_samples=8, nx=16, ny=16, seed=42)
    min_dist_actual = verify_dataset_separation(mock_train, actual_val_tensor)
    assert min_dist_actual > 1e-3

    # Verify dataset identity metadata in summary
    ds_meta = summary["metadata"]["dataset_identity"]
    assert ds_meta["train_seed"] == 42
    assert ds_meta["val_seed"] == 1042
    assert ds_meta["train_samples"] == 8
    assert ds_meta["val_samples"] == 4
    assert ds_meta["dataset_separation"]["status"] == "PASSED_FAIL_CLOSED"
    assert ds_meta["dataset_separation"]["identical_samples_count"] == 0



def test_benchmark_model_latency_structure():
    """Verify latency benchmark separates single-request and batched throughput."""
    device = torch.device("cpu")
    model = VorticityAutoencoder(in_channels=1, out_channels=1, latent_channels=16, base_channels=8)
    bench = benchmark_model_latency(
        model, device, sample_shape=(1, 1, 16, 16), batch_shape=(4, 1, 16, 16), warmup_runs=2, repeat_runs=5
    )
    assert "single_request_latency" in bench
    assert "batched_efficiency" in bench
    assert bench["single_request_latency"]["batch_size"] == 1
    assert bench["single_request_latency"]["median_ms"] > 0
    assert bench["batched_efficiency"]["batch_size"] == 4
    assert bench["batched_efficiency"]["throughput_samples_per_sec"] > 0


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
        "total_enstrophy_retention_ratio",
        "spurious_energy_in_zero_bins",
        "total_parameters",
        "latency_benchmarks",
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


def test_evaluate_existing_checkpoints_pipeline(tmp_path):
    """Test --eval-only mode with mock checkpoint structure."""
    ckpt_root = tmp_path / "checkpoints"
    for cz in [32, 16]:
        cz_dir = ckpt_root / f"cz_{cz}"
        cz_dir.mkdir(parents=True)
        model = VorticityAutoencoder(in_channels=1, out_channels=1, latent_channels=cz, base_channels=8)
        ckpt_data = {
            "epoch": 5,
            "step": 10,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "config": {
                "model": {"base_channels": 8},
                "domain": {"nx": 16, "ny": 16, "lx": 1.0, "ly": 1.0},
                "synthetic_data": {"seed": 42, "num_train_samples": 8, "num_val_samples": 4},
            },
        }
        torch.save(ckpt_data, cz_dir / "latest_checkpoint.pt")

    eval_out = tmp_path / "independent_eval"
    summary = evaluate_existing_checkpoints(
        checkpoint_dir=str(ckpt_root),
        capacities=[32, 16],
        output_dir=str(eval_out),
        device="cpu",
    )

    assert "models" in summary
    assert "32" in summary["models"]
    assert "16" in summary["models"]
    assert (eval_out / "capacity_comparison_summary.json").is_file()
    assert (eval_out / "capacity_comparison_metrics.png").is_file()
    assert (eval_out / "capacity_comparison_spectra.png").is_file()
    assert (eval_out / "capacity_comparison_fields.png").is_file()


def test_capacity_comparison_zero_real_data_leakage():
    """Ensure the experiment script contains zero reference to frozen StocBench evaluation data."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_vorticity_capacity_comparison.py"
    content = script_path.read_text(encoding="utf-8")
    assert "step_seed_100.npz" not in content
    assert "traj_seed_42.npy" not in content
    assert "outputs/data/stocbench" not in content


def test_verify_dataset_separation_rejects_nan_inf_and_invalid_inputs():
    """Verify verify_dataset_separation fails closed on NaN, Inf, empty, or incompatible shapes."""
    clean_train = torch.randn(8, 1, 16, 16)
    clean_val = torch.randn(4, 1, 16, 16)

    # 1. NaN in val tensor must raise ValueError
    nan_val = clean_val.clone()
    nan_val[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf detected"):
        verify_dataset_separation(clean_train, nan_val)

    # 2. NaN in train tensor must raise ValueError
    nan_train = clean_train.clone()
    nan_train[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf detected"):
        verify_dataset_separation(nan_train, clean_val)

    # 3. Inf in tensor must raise ValueError
    inf_val = clean_val.clone()
    inf_val[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf detected"):
        verify_dataset_separation(clean_train, inf_val)

    # 4. Empty tensor must raise ValueError
    empty_tensor = torch.empty(0, 1, 16, 16)
    with pytest.raises(ValueError, match="Empty dataset tensor"):
        verify_dataset_separation(empty_tensor, clean_val)
    with pytest.raises(ValueError, match="Empty dataset tensor"):
        verify_dataset_separation(clean_train, empty_tensor)

    # 5. Non-4D input must raise ValueError
    with pytest.raises(ValueError, match="4D tensors"):
        verify_dataset_separation(torch.randn(8, 16, 16), clean_val)

    # 6. Mismatched spatial dimensions must raise ValueError
    mismatched_val = torch.randn(4, 1, 32, 32)
    with pytest.raises(ValueError, match="dimensions must match"):
        verify_dataset_separation(clean_train, mismatched_val)


def test_evaluate_existing_checkpoints_rejects_mismatched_data_contracts(tmp_path):
    """Enforce fail-closed termination when checkpoints have inconsistent data definitions or seeds."""
    ckpt_root = tmp_path / "checkpoints"

    # Checkpoint for Cz=32 with seed=42
    dir_32 = ckpt_root / "cz_32"
    dir_32.mkdir(parents=True)
    m32 = VorticityAutoencoder(in_channels=1, out_channels=1, latent_channels=32, base_channels=8)
    ckpt_32 = {
        "epoch": 5,
        "step": 10,
        "model_state_dict": m32.state_dict(),
        "config": {
            "model": {"base_channels": 8},
            "domain": {"nx": 16, "ny": 16, "lx": 1.0, "ly": 1.0},
            "synthetic_data": {
                "seed": 42,
                "num_train_samples": 8,
                "num_val_samples": 4,
                "perturbation_amplitude": 0.1,
            },
        },
    }
    torch.save(ckpt_32, dir_32 / "latest_checkpoint.pt")

    # Checkpoint for Cz=16 with conflicting seed=43
    dir_16 = ckpt_root / "cz_16"
    dir_16.mkdir(parents=True)
    m16 = VorticityAutoencoder(in_channels=1, out_channels=1, latent_channels=16, base_channels=8)
    ckpt_16_mismatched_seed = {
        "epoch": 5,
        "step": 10,
        "model_state_dict": m16.state_dict(),
        "config": {
            "model": {"base_channels": 8},
            "domain": {"nx": 16, "ny": 16, "lx": 1.0, "ly": 1.0},
            "synthetic_data": {
                "seed": 43,  # Inconsistent seed!
                "num_train_samples": 8,
                "num_val_samples": 4,
                "perturbation_amplitude": 0.1,
            },
        },
    }
    torch.save(ckpt_16_mismatched_seed, dir_16 / "latest_checkpoint.pt")

    eval_out = tmp_path / "eval_out"
    # Calling evaluate_existing_checkpoints must fail closed and refuse to rank models
    with pytest.raises(ValueError, match="Cross-checkpoint dataset contract mismatch detected"):
        evaluate_existing_checkpoints(
            checkpoint_dir=str(ckpt_root),
            capacities=[32, 16],
            output_dir=str(eval_out),
            device="cpu",
        )

    # Also verify rejection when checkpoint lacks required configuration
    ckpt_16_no_synth = {
        "epoch": 5,
        "step": 10,
        "model_state_dict": m16.state_dict(),
        "config": {
            "model": {"base_channels": 8},
            "domain": {"nx": 16, "ny": 16, "lx": 1.0, "ly": 1.0},
            # missing synthetic_data section!
        },
    }
    torch.save(ckpt_16_no_synth, dir_16 / "latest_checkpoint.pt")
    with pytest.raises(ValueError, match="missing 'synthetic_data' section"):
        evaluate_existing_checkpoints(
            checkpoint_dir=str(ckpt_root),
            capacities=[32, 16],
            output_dir=str(eval_out),
            device="cpu",
        )
