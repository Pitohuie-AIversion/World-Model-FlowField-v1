"""Unit and contract tests for H12 Extension study runner and benchmark evaluation target."""

import os
import tempfile
from pathlib import Path
import pytest
import torch

from scripts.run_h12_extension import (
    H12_CONFIG,
    EXPECTED_PARENT_SHA256,
    build_training_command,
    validate_parent_for_h12,
    resolve_parent_checkpoint,
)
from scripts.evaluate_h16_benchmark import (
    BENCHMARK_TARGETS,
    validate_benchmark_checkpoint,
)
from src.utils.physics_contract import (
    PHYSICS_PROTOCOL,
    SPATIAL_AXIS_CONTRACT,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
)


class TestH12ConfigurationAndCommand:
    """Verify H12 configuration governance, dual-GPU parameters, and constant learning rate."""

    def test_h12_config_governance(self):
        """H12 configuration must adhere strictly to approved governance parameters."""
        assert H12_CONFIG["horizon"] == 12
        assert H12_CONFIG["expected_init_horizon"] == 8
        assert H12_CONFIG["effective_batch_size"] == 8
        assert H12_CONFIG["lr"] == 5e-5
        assert H12_CONFIG["epochs"] == 12
        assert H12_CONFIG["lambda_div"] == 0.01
        assert H12_CONFIG["lambda_vort"] == 0.05
        assert H12_CONFIG["val_diagnostics"] == [10, 20, 30]
        assert H12_CONFIG["seed"] == 42
        # Must not contain misleading min_lr or scheduler flags
        assert "min_lr" not in H12_CONFIG
        assert "lr_scheduler" not in H12_CONFIG

    def test_build_training_command_ddp_parameters(self):
        """Dual-GPU DDP training command must preserve Beff=8 with grad_accum=4 and constant lr."""
        cmd = build_training_command(
            parent_path="/tmp/mock_parent.pt",
            epochs=12,
            lr=5e-5,
            batch_size=1,
            grad_accum_steps=4,
            horizon=12,
            expected_init_horizon=8,
            output_dir="/tmp/output_h12",
            num_gpus=2,
            master_port=29501,
        )

        cmd_str = " ".join(cmd)
        # Distributed wrapper
        assert "torch.distributed.run" in cmd_str
        assert "--nproc_per_node=2" in cmd_str
        assert "--master_port=29501" in cmd_str
        # Key training args
        assert "--horizon 12" in cmd_str
        assert "--expected_init_horizon 8" in cmd_str
        assert "--epochs 12" in cmd_str
        assert "--batch_size 1" in cmd_str
        assert "--grad_accum_steps 4" in cmd_str
        assert "--lr 5e-05" in cmd_str
        assert "--lambda_div 0.01" in cmd_str
        assert "--lambda_vort 0.05" in cmd_str
        assert "--use_amp" in cmd_str
        assert "--lambda_spec 0.0" in cmd_str
        assert "--pushforward_steps 0" in cmd_str
        # Disabled features must not be in arguments
        assert "--curriculum_rollout" not in cmd_str
        assert "--compile_model" not in cmd_str
        assert "--min_lr" not in cmd_str

    def test_build_training_command_single_gpu(self):
        """Single-GPU fallback command must adjust grad_accum=8 to preserve Beff=8."""
        cmd = build_training_command(
            parent_path="/tmp/mock_parent.pt",
            epochs=12,
            lr=5e-5,
            batch_size=1,
            grad_accum_steps=8,
            horizon=12,
            expected_init_horizon=8,
            output_dir="/tmp/output_h12",
            num_gpus=1,
        )

        cmd_str = " ".join(cmd)
        assert "torch.distributed.run" not in cmd_str
        assert "--grad_accum_steps 8" in cmd_str
        assert "--horizon 12" in cmd_str
        assert "--lr 5e-05" in cmd_str


class TestH12ParentValidationContracts:
    """Verify fail-closed preflight checks for parent checkpoint SHA and horizon."""

    def test_validate_parent_sha256_mismatch_fails_closed(self, tmp_path):
        """Parent checkpoint with mismatched SHA-256 must be rejected."""
        fake_parent = tmp_path / "corrupted_parent.pt"
        torch.save({"dummy": 123}, fake_parent)

        with pytest.raises(ValueError, match="Parent checkpoint SHA-256 mismatch"):
            validate_parent_for_h12(
                parent_path=fake_parent,
                expected_horizon=8,
                expected_seed=42,
                expected_sha256=EXPECTED_PARENT_SHA256,
            )

    def test_validate_parent_horizon_mismatch_fails_closed(self, tmp_path):
        """Parent checkpoint with wrong horizon must be rejected."""
        # Create a mock checkpoint with horizon=4 instead of 8
        bad_ckpt = tmp_path / "bad_horizon_parent.pt"
        payload = {
            "model_type": "latent_transformer",
            "horizon": 4,  # wrong horizon
            "seed": 42,
            "lambda_div": 0.01,
            "lambda_vort": 0.05,
            "split_hash": "dummy_split",
            "normalizer_hash": "dummy_norm",
            "physics_protocol": PHYSICS_PROTOCOL,
            "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
            "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
            "prediction_mode": "direct",
            "use_condition": True,
            "training_git_dirty": False,
            "model_state_dict": {},
            "epoch": 11,
        }
        torch.save(payload, bad_ckpt)

        # Bypass SHA check to test horizon failure
        with pytest.raises(ValueError, match="Horizon contract violation"):
            validate_parent_for_h12(
                parent_path=bad_ckpt,
                expected_horizon=8,
                expected_seed=42,
                expected_sha256=None,
            )

    def test_canonical_parent_checkpoint_sha_and_contract(self):
        """Canonical parent checkpoint on disk must match the exact expected SHA-256."""
        try:
            parent_path = resolve_parent_checkpoint()
        except FileNotFoundError:
            pytest.skip("Canonical H8 Ep11 parent checkpoint not found in default locations")

        is_valid, actual_sha, ckpt_data = validate_parent_for_h12(
            parent_path=parent_path,
            expected_horizon=8,
            expected_seed=42,
            expected_sha256=EXPECTED_PARENT_SHA256,
        )
        assert is_valid
        assert actual_sha == EXPECTED_PARENT_SHA256
        assert ckpt_data.get("horizon", ckpt_data.get("config", {}).get("horizon")) == 8


class TestH12BenchmarkEvaluationIntegration:
    """Verify that evaluate_h16_benchmark supports H12 targets with expected_horizon=12."""

    def test_h12_benchmark_targets_registered(self):
        """BENCHMARK_TARGETS must register both h12_short_best and h12_long_best with horizon=12."""
        assert "h12_short_best" in BENCHMARK_TARGETS
        assert "h12_long_best" in BENCHMARK_TARGETS

        short_target = BENCHMARK_TARGETS["h12_short_best"]
        assert short_target["expected_horizon"] == 12
        assert short_target["expected_lambda_div"] == 0.01
        assert short_target["expected_lambda_vort"] == 0.05
        assert "E4_H12" in short_target["path"]

        long_target = BENCHMARK_TARGETS["h12_long_best"]
        assert long_target["expected_horizon"] == 12
        assert long_target["expected_lambda_div"] == 0.01
        assert long_target["expected_lambda_vort"] == 0.05
        assert "E4_H12" in long_target["path"]

    def test_validate_benchmark_checkpoint_rejects_wrong_horizon_for_h12(self):
        """H16 checkpoint masquerading as H12 must be rejected by validator."""
        target_info = BENCHMARK_TARGETS["h12_short_best"]
        ckpt_data = {
            "model_type": "latent_transformer",
            "horizon": 16,  # Masquerading H16 checkpoint
            "seed": 42,
            "lambda_div": 0.01,
            "lambda_vort": 0.05,
            "split_hash": "41fbe6ebe7edd460",
            "normalizer_hash": "3a0fe52689657618",
            "physics_protocol": PHYSICS_PROTOCOL,
            "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
            "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
            "training_git_commit": "0a8a4da137e3f73c4f696cc81f527c32a8ba8405",
            "training_git_dirty": False,
        }
        cfg = {"model_type": "latent_transformer", "horizon": 16, "lambda_div": 0.01, "lambda_vort": 0.05}

        with pytest.raises(ValueError, match="Checkpoint horizon mismatch for h12_short_best"):
            validate_benchmark_checkpoint(
                ckpt_path="dummy.pt",
                ckpt_data=ckpt_data,
                cfg=cfg,
                target_key="h12_short_best",
                target_info=target_info,
                eval_split_hash="41fbe6ebe7edd460",
                eval_normalizer_hash="3a0fe52689657618",
                formal=False,
            )

    def test_validate_benchmark_checkpoint_accepts_valid_h12_checkpoint(self):
        """Valid H12 checkpoint must pass benchmark validation."""
        target_info = BENCHMARK_TARGETS["h12_long_best"]
        ckpt_data = {
            "model_type": "latent_transformer",
            "horizon": 12,
            "seed": 42,
            "lambda_div": 0.01,
            "lambda_vort": 0.05,
            "split_hash": "41fbe6ebe7edd460",
            "normalizer_hash": "3a0fe52689657618",
            "physics_protocol": PHYSICS_PROTOCOL,
            "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
            "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
            "training_git_commit": "0a8a4da137e3f73c4f696cc81f527c32a8ba8405",
            "training_git_dirty": False,
        }
        cfg = {"model_type": "latent_transformer", "horizon": 12, "lambda_div": 0.01, "lambda_vort": 0.05}

        prov = validate_benchmark_checkpoint(
            ckpt_path="dummy.pt",
            ckpt_data=ckpt_data,
            cfg=cfg,
            target_key="h12_long_best",
            target_info=target_info,
            eval_split_hash="41fbe6ebe7edd460",
            eval_normalizer_hash="3a0fe52689657618",
            formal=False,
        )
        assert prov["seed"] == 42
        assert prov["split_hash"] == "41fbe6ebe7edd460"
        assert prov["normalizer_hash"] == "3a0fe52689657618"


class TestH12RunnerGovernanceAndRejection:
    """Verify H12 runner startup assertions, Beff enforcement, horizon guards, and git status checks."""

    def test_default_dual_gpu_beff8_allows_preflight(self, monkeypatch, tmp_path):
        """Default dual-GPU configuration results in Beff=8 and allows preflight in dry run."""
        from scripts.run_h12_extension import main, resolve_parent_checkpoint
        try:
            resolve_parent_checkpoint()
        except FileNotFoundError:
            mock_ckpt = tmp_path / "mock_h8.pt"
            mock_ckpt.touch()
            monkeypatch.setattr("scripts.run_h12_extension.resolve_parent_checkpoint", lambda *args, **kwargs: mock_ckpt)
            monkeypatch.setattr("scripts.run_h12_extension.validate_parent_for_h12", lambda *args, **kwargs: (True, "mock_sha", {"horizon": 8}))

        monkeypatch.setattr("sys.argv", ["run_h12_extension.py", "--dry_run", "--gpu_ids", "0,1"])
        ret = main()
        assert ret == 0

    def test_beff_mismatch_fails_closed(self, monkeypatch):
        """Configurations resulting in Beff != 8 must be rejected before preflight/training."""
        from scripts.run_h12_extension import main
        # batch_size=3 with 2 GPUs gives accum=1 -> Beff=6 != 8
        monkeypatch.setattr("sys.argv", ["run_h12_extension.py", "--dry_run", "--gpu_ids", "0,1", "--batch_size", "3"])
        with pytest.raises(ValueError, match="Effective batch size calculation mismatch"):
            main()

        # Explicit grad_accum_steps=5 with 2 GPUs gives Beff=10 != 8
        monkeypatch.setattr("sys.argv", ["run_h12_extension.py", "--dry_run", "--gpu_ids", "0,1", "--grad_accum_steps", "5"])
        with pytest.raises(ValueError, match="Effective batch size calculation mismatch"):
            main()

        # Setting effective_batch_size != 8 must also fail closed
        monkeypatch.setattr("sys.argv", ["run_h12_extension.py", "--dry_run", "--effective_batch_size", "10"])
        with pytest.raises(ValueError, match="Effective batch size contract violation"):
            main()

    def test_horizon_override_fails_closed(self, monkeypatch):
        """Attempting to run a horizon other than 12 on H12 runner must fail closed."""
        from scripts.run_h12_extension import main
        monkeypatch.setattr("sys.argv", ["run_h12_extension.py", "--dry_run", "--horizon", "16"])
        with pytest.raises(ValueError, match="Horizon contract violation"):
            main()

        monkeypatch.setattr("sys.argv", ["run_h12_extension.py", "--dry_run", "--expected_init_horizon", "12"])
        with pytest.raises(ValueError, match="Expected init horizon contract violation"):
            main()

    def test_dirty_git_rejects_formal_run_without_starting_subprocess(self, monkeypatch, tmp_path):
        """In non-dry-run mode, if git tree is dirty, runner must reject execution without spawning subprocess."""
        import subprocess
        from scripts.run_h12_extension import main, resolve_parent_checkpoint

        try:
            resolve_parent_checkpoint()
        except FileNotFoundError:
            mock_ckpt = tmp_path / "mock_h8.pt"
            mock_ckpt.touch()
            monkeypatch.setattr("scripts.run_h12_extension.resolve_parent_checkpoint", lambda *args, **kwargs: mock_ckpt)
            monkeypatch.setattr("scripts.run_h12_extension.validate_parent_for_h12", lambda *args, **kwargs: (True, "mock_sha", {"horizon": 8}))

        # Mock is_git_dirty to return True
        monkeypatch.setattr("scripts.run_h12_extension.is_git_dirty", lambda root=None: True)

        # Ensure subprocess.Popen is never called
        def fail_popen(*args, **kwargs):
            raise AssertionError("subprocess.Popen should not have been called on dirty git tree!")
        monkeypatch.setattr(subprocess, "Popen", fail_popen)

        monkeypatch.setattr("sys.argv", ["run_h12_extension.py", "--gpu_ids", "0,1"])
        with pytest.raises(RuntimeError, match="Formal training execution blocked: git working tree is dirty"):
            main()
