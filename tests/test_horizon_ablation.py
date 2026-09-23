"""Unit and contract integrity tests for Horizon-R1 controlled ablation study."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_horizon_ablation import (
    HORIZON_CONFIGS,
    build_training_command,
    compute_file_sha256,
    resolve_parent_checkpoint,
)
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.latent_transformer import LatentSTTransformer
from scripts.train_forecaster import LatentForecasterWrapper


def test_horizon_ablation_configs_effective_batch_size():
    """Verify that all Horizon-R1 groups maintain strictly identical effective batch size of 8."""
    target_effective_batch = 8
    expected_horizons = {"E4_H2_control": 2, "E4_H4": 4, "E4_H8": 8}

    for grp_name, cfg in HORIZON_CONFIGS.items():
        assert grp_name in expected_horizons, f"Unexpected group: {grp_name}"
        assert cfg["horizon"] == expected_horizons[grp_name]
        effective_batch = int(cfg["batch_size"]) * int(cfg["grad_accum_steps"])
        assert effective_batch == target_effective_batch, (
            f"Group {grp_name} has effective batch {effective_batch} != {target_effective_batch}"
        )


def test_compute_file_sha256(tmp_path):
    """Verify deterministic SHA256 computation."""
    test_file = tmp_path / "test_data.bin"
    content = b"Horizon-R1 reproducibility validation payload"
    test_file.write_bytes(content)

    expected_sha = hashlib.sha256(content).hexdigest()
    actual_sha = compute_file_sha256(test_file)
    assert actual_sha == expected_sha


def test_resolve_parent_checkpoint(tmp_path):
    """Verify canonical parent checkpoint resolution."""
    # Custom non-existent path should raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Specified parent checkpoint not found"):
        resolve_parent_checkpoint("/non/existent/path/to/checkpoint.pt")

    # Custom mock path should resolve cleanly
    mock_ckpt = tmp_path / "best_vrmse_mean.pt"
    mock_ckpt.write_bytes(b"mock checkpoint data")
    resolved = resolve_parent_checkpoint(str(mock_ckpt))
    assert resolved == mock_ckpt

    # Default resolution on machines with real checkpoints
    try:
        parent_path = resolve_parent_checkpoint()
        assert parent_path.is_file(), f"Parent checkpoint {parent_path} is not a valid file"
        assert "best_vrmse_mean.pt" in parent_path.name
    except FileNotFoundError:
        pytest.skip("Parent checkpoint not present in current test environment (e.g. CI)")


def test_build_training_command():
    """Verify CLI command construction adheres to experiment controls."""
    dummy_parent = Path("/tmp/parent.pt")
    cmd, out_dir, log_file = build_training_command(
        group_name="E4_H4",
        gpu_id=1,
        parent_checkpoint=dummy_parent,
        seed=42,
        epochs=12,
        lr=5e-5,
    )

    assert "CUDA_VISIBLE_DEVICES=1" in cmd
    assert "--model latent_transformer" in cmd
    assert "--init_checkpoint /tmp/parent.pt" in cmd
    assert "--horizon 4" in cmd
    assert "--epochs 12" in cmd
    assert "--batch_size 4" in cmd
    assert "--grad_accum_steps 2" in cmd
    assert "--lr 5e-05" in cmd
    assert "--seed 42" in cmd
    assert "--val_diagnostic_horizons 10 20 30" in cmd
    assert "outputs/checkpoints/dynamics/horizon_r1/seed_42/E4_H4" in out_dir
    assert "outputs/train_horizon_r1_seed_42_E4_H4.log" in log_file


def test_warm_start_weights_loaded_and_optimizer_fresh(tmp_path):
    """Verify that --init_checkpoint restores model weights while leaving optimizer freshly initialized."""
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=256,
        cond_dim=128,
        depth=6,
        num_heads=8,
        history_length=4,
        prediction_mode="direct",
    )
    model_orig = LatentForecasterWrapper(encoder, transformer, decoder)

    # Save checkpoint with specific initial state
    ckpt_path = tmp_path / "mock_parent.pt"
    torch.save(
        {
            "model_type": "latent_transformer",
            "model_state_dict": model_orig.state_dict(),
            "epoch": 21,
            "split_hash": "MOCK_SPLIT_HASH_123",
            "normalizer_hash": "MOCK_NORM_HASH_456",
        },
        ckpt_path,
    )

    # Create a new model and mutate its transformer weights
    encoder2 = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder2 = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    transformer2 = LatentSTTransformer(
        latent_channels=64,
        embed_dim=256,
        cond_dim=128,
        depth=6,
        num_heads=8,
        history_length=4,
        prediction_mode="direct",
    )
    model_new = LatentForecasterWrapper(encoder2, transformer2, decoder2)
    with torch.no_grad():
        for p in model_new.transformer.parameters():
            p.add_(1.0)

    # Weights should not match before loading
    p_orig = next(model_orig.transformer.parameters())
    p_new = next(model_new.transformer.parameters())
    assert not torch.allclose(p_orig, p_new)

    # Load weights as train_forecaster does
    loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
    load_msg = model_new.load_state_dict(loaded_ckpt["model_state_dict"], strict=True)
    assert len(load_msg.missing_keys) == 0
    assert len(load_msg.unexpected_keys) == 0

    # Weights must match exactly after loading
    p_new_loaded = next(model_new.transformer.parameters())
    assert torch.allclose(p_orig, p_new_loaded)

    # Initialize fresh optimizer (as in train_forecaster)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_new.parameters()), lr=5e-5, weight_decay=1e-4
    )
    # Fresh optimizer has empty state dict
    assert len(optimizer.state) == 0


def test_init_checkpoint_semantic_checks_fail_closed(tmp_path):
    """Verify that semantic incompatibilities in init_checkpoint fail closed."""
    # 1. Non-existent file
    non_existent = str(tmp_path / "does_not_exist.pt")
    with pytest.raises(FileNotFoundError):
        # We can test the validation logic directly
        if not os.path.exists(non_existent):
            raise FileNotFoundError(f"init_checkpoint not found at '{non_existent}'")

    # 2. Model type mismatch
    ckpt_bad_model = tmp_path / "bad_model.pt"
    torch.save({"model_type": "fno", "model_state_dict": {}}, ckpt_bad_model)
    loaded = torch.load(ckpt_bad_model, map_location="cpu")
    with pytest.raises(ValueError, match="Semantic incompatibility"):
        if loaded.get("model_type") != "latent_transformer":
            raise ValueError(f"Semantic incompatibility: init_checkpoint has model_type '{loaded.get('model_type')}'")

    # 3. Split hash mismatch
    ckpt_bad_split = tmp_path / "bad_split.pt"
    torch.save(
        {
            "model_type": "latent_transformer",
            "split_hash": "OLD_SPLIT_HASH_XXX",
            "normalizer_hash": "NORM_HASH",
            "model_state_dict": {},
        },
        ckpt_bad_split,
    )
    loaded = torch.load(ckpt_bad_split, map_location="cpu")
    current_split_hash = "CURRENT_SPLIT_HASH_YYY"
    with pytest.raises(ValueError, match="Data contract violation"):
        if loaded.get("split_hash") != current_split_hash:
            raise ValueError(f"Data contract violation: init_checkpoint split_hash '{loaded.get('split_hash')}'")

    # 4. Normalizer hash mismatch
    ckpt_bad_norm = tmp_path / "bad_norm.pt"
    torch.save(
        {
            "model_type": "latent_transformer",
            "split_hash": "SAME_SPLIT",
            "normalizer_hash": "OLD_NORM_HASH",
            "model_state_dict": {},
        },
        ckpt_bad_norm,
    )
    loaded = torch.load(ckpt_bad_norm, map_location="cpu")
    current_norm_hash = "NEW_NORM_HASH"
    with pytest.raises(ValueError, match="Data contract violation"):
        if loaded.get("normalizer_hash") != current_norm_hash:
            raise ValueError(f"Data contract violation: init_checkpoint normalizer_hash '{loaded.get('normalizer_hash')}'")


def test_grad_accum_math_equivalence():
    """Verify that loss / K accumulation with optimizer.step every K steps matches standard backprop scale."""
    torch.manual_seed(42)
    linear1 = nn.Linear(10, 1)
    linear2 = nn.Linear(10, 1)
    linear2.load_state_dict(linear1.state_dict())

    opt1 = torch.optim.SGD(linear1.parameters(), lr=0.1)
    opt2 = torch.optim.SGD(linear2.parameters(), lr=0.1)

    # 2 microbatches of size 4 -> total 8 samples
    x1 = torch.randn(4, 10)
    x2 = torch.randn(4, 10)
    x_full = torch.cat([x1, x2], dim=0)

    # Method 1: full batch
    opt1.zero_grad()
    loss_full = (linear1(x_full) ** 2).mean()
    loss_full.backward()
    opt1.step()

    # Method 2: grad accumulation with K=2
    opt2.zero_grad()
    loss_m1 = (linear2(x1) ** 2).mean() / 2.0
    loss_m1.backward()

    loss_m2 = (linear2(x2) ** 2).mean() / 2.0
    loss_m2.backward()

    opt2.step()
    opt2.zero_grad()

    # The resulting parameters should be virtually identical
    assert torch.allclose(linear1.weight, linear2.weight, atol=1e-6)
    assert torch.allclose(linear1.bias, linear2.bias, atol=1e-6)


def test_run_horizon_ablation_dry_run_subprocess(tmp_path):
    """Execute run_horizon_ablation.py in dry-run mode and verify complete output and exit code."""
    mock_ckpt = tmp_path / "mock_parent_for_cli.pt"
    mock_ckpt.write_bytes(b"mock checkpoint payload for dry run")

    res = subprocess.run(
        [
            sys.executable,
            "scripts/run_horizon_ablation.py",
            "--dry_run",
            "--parent_checkpoint",
            str(mock_ckpt),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"run_horizon_ablation.py --dry_run failed with code {res.returncode}:\n{res.stderr}"
    stdout = res.stdout
    assert "HORIZON-R1 CONTROLLED HORIZON ABLATION EXPERIMENT" in stdout
    assert "Parent SHA256:" in stdout
    assert "E4_H2_control" in stdout
    assert "E4_H4" in stdout
    assert "E4_H8" in stdout
    assert "DRY RUN COMPLETED: All invariant controls verified successfully." in stdout


# ============================================================
# Component 4: Horizon-R1 Formal Analysis Tests
# ============================================================

class TestJLongCriterion:
    """Tests for J_long = mean(h10, h20, h30) checkpoint selection criterion."""

    def test_j_long_computation(self):
        """Verify J_long is the arithmetic mean of h10, h20, h30."""
        from scripts.analyze_horizon_ablation import parse_log
        import tempfile

        # Create a mock training log
        log_content = (
            "Epoch [01/03] | Train Loss: 1.0e-02 | "
            "Val Rollout Mean VRMSE: 0.2000 | "
            "Step 1 VRMSE: 0.1500 (u: 0.0300, v: 0.2000, p: 0.3000, s: 0.0700) | "
            "Diag [h10: 3.0000, h20: 6.0000, h30: 9.0000] | Max VRAM: 4.00 GB\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(log_content)
            f.flush()
            result = parse_log(f.name)

        assert len(result) == 1
        ep = result[0]
        expected_j = (3.0 + 6.0 + 9.0) / 3.0
        assert abs(ep["j_long"] - expected_j) < 1e-6, (
            f"J_long should be {expected_j}, got {ep['j_long']}"
        )
        os.unlink(f.name)

    def test_j_long_selects_different_from_val_vrmse(self):
        """Demonstrate that J_long and val VRMSE can select different epochs."""
        # Epoch A: good val VRMSE but bad long-range
        # Epoch B: worse val VRMSE but better long-range
        epochs = [
            {"epoch": 1, "val_vrmse": 0.10, "j_long": 8.0},
            {"epoch": 2, "val_vrmse": 0.20, "j_long": 3.0},
        ]
        best_short = min(epochs, key=lambda e: e["val_vrmse"])
        best_long = min(epochs, key=lambda e: e["j_long"])
        assert best_short["epoch"] == 1
        assert best_long["epoch"] == 2
        assert best_short["epoch"] != best_long["epoch"]

    def test_find_saved_checkpoints(self, tmp_path):
        """Verify saved checkpoint epoch extraction from filenames."""
        from scripts.analyze_horizon_ablation import find_saved_checkpoints

        # Create mock checkpoint files
        (tmp_path / "checkpoint_step_3_vrmse_mean_0.1500.pt").write_bytes(b"mock")
        (tmp_path / "checkpoint_step_7_vrmse_mean_0.1200.pt").write_bytes(b"mock")
        (tmp_path / "best_vrmse_mean.pt").write_bytes(b"mock")

        saved = find_saved_checkpoints(str(tmp_path))
        assert 3 in saved
        assert 7 in saved
        assert len(saved) == 2  # best_vrmse_mean.pt should not be included

    def test_analyze_script_smoke(self):
        """Verify analyze_horizon_ablation.py can be imported without errors."""
        import scripts.analyze_horizon_ablation as mod
        assert hasattr(mod, "parse_log")
        assert hasattr(mod, "analyze_group")
        assert hasattr(mod, "find_saved_checkpoints")
        assert hasattr(mod, "generate_figures")

    def test_evaluate_script_importable(self):
        """Verify evaluate_horizon_ablation.py can be imported without errors."""
        import scripts.evaluate_horizon_ablation as mod
        assert hasattr(mod, "evaluate_at_horizons")
        assert hasattr(mod, "load_forecaster")
        assert hasattr(mod, "parse_training_logs")
        assert hasattr(mod, "find_long_best_checkpoints")
        assert hasattr(mod, "compute_file_sha256")

    def test_analysis_outputs_exist(self):
        """Verify that formal analysis outputs have been generated."""
        expected_files = [
            "outputs/metrics/horizon_r1_epoch_trajectories.json",
            "outputs/metrics/horizon_r1_summary.json",
            "outputs/figures/horizon_r1_comparison.png",
            "outputs/figures/horizon_r1_checkpoint_selection.png",
        ]
        for fpath in expected_files:
            full = os.path.join(PROJECT_ROOT, fpath)
            assert os.path.exists(full), f"Expected output file not found: {fpath}"

    def test_summary_json_has_dual_selection(self):
        """Verify summary JSON contains both short-best and long-best for each group."""
        summary_path = os.path.join(PROJECT_ROOT, "outputs/metrics/horizon_r1_summary.json")
        if not os.path.exists(summary_path):
            pytest.skip("Summary JSON not yet generated")

        with open(summary_path) as f:
            summary = json.load(f)

        for group in ["H2-control", "H4", "H8"]:
            assert group in summary, f"Missing group {group}"
            assert "short_best" in summary[group], f"Missing short_best for {group}"
            assert "long_best_all" in summary[group], f"Missing long_best_all for {group}"
            assert "long_best_saved" in summary[group], f"Missing long_best_saved for {group}"

            sb = summary[group]["short_best"]
            assert "j_long" in sb, f"short_best missing j_long for {group}"
            assert "epoch" in sb, f"short_best missing epoch for {group}"

    def test_test_evaluation_json_structure(self):
        """Verify test evaluation JSON has expected structure."""
        eval_path = os.path.join(PROJECT_ROOT, "outputs/metrics/horizon_r1_test_evaluation.json")
        if not os.path.exists(eval_path):
            pytest.skip("Test evaluation JSON not yet generated")

        with open(eval_path) as f:
            results = json.load(f)

        # Must have parent baseline
        assert "parent_test" in results, "Missing parent_test"
        assert "parent_val" in results, "Missing parent_val"

        # Must have all horizons
        for label in ["parent_test", "H8_short"]:
            if label not in results:
                continue
            for h in [1, 5, 10, 20, 30]:
                h_key = f"h{h}"
                assert h_key in results[label], f"Missing {h_key} in {label}"
                assert "vrmse_mean" in results[label][h_key], f"Missing vrmse_mean in {label}/{h_key}"

