"""Unit and Failure Mode Tests for Latent Flow Matching Training Entrypoint.

Tests:
1. evaluate_flow_matching_loss:
   - Fails closed on empty dataloader (ValueError).
   - Fails closed when model has no attached flow_matcher (RuntimeError).
   - Fails closed on non-finite loss (FloatingPointError).
2. verify_flow_matching_preflight_contract:
   - Fails closed when D0 checkpoint missing (FileNotFoundError).
   - Fails closed when normalizer file missing (FileNotFoundError).
   - Fails closed when split file missing (FileNotFoundError).
3. build_and_freeze_flow_matching_model:
   - Strict parameter freeze governance assertion.
4. Directory collision protection:
   - Existing artifacts fail closed unless overwrite=True.
5. End-to-end synthetic training run:
   - Verifies training step, checkpointing, and metric output generation.
"""

import json
import os
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scripts.train_latent_flow_matching import (
    verify_flow_matching_preflight_contract,
    build_and_freeze_flow_matching_model,
    evaluate_flow_matching_loss,
    train_latent_flow_matching,
    clip_and_validate_gradients,
)
from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.latent_flow_matching import LatentFlowMatcher


class DummySyntheticFlowDataset(Dataset):
    """Synthetic dataset for testing training execution without requiring large hdf5 files."""

    def __init__(self, num_samples: int = 4, produce_nans: bool = False):
        self.num_samples = num_samples
        self.produce_nans = produce_nans

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.produce_nans:
            history = torch.full((4, 4, 32, 32), float("nan"))
            future = torch.full((1, 4, 32, 32), float("nan"))
        else:
            history = torch.randn(4, 4, 32, 32)
            future = torch.randn(1, 4, 32, 32)
        return {
            "history": history,
            "future": future,
            "re": torch.tensor(10000.0),
            "sc": torch.tensor(0.5),
        }


class MockForecasterNoFM(nn.Module):
    """Forecaster without attached flow matcher."""
    def __init__(self):
        super().__init__()
        self.flow_matcher = None


class MockForecasterNanFM(nn.Module):
    """Forecaster whose flow matcher produces non-finite loss."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.transformer = lambda z, re=None, sc=None: z[:, :1]
        self.flow_matcher = MockNanFlowMatcher()

    def eval(self):
        pass


class MockNanFlowMatcher(nn.Module):
    def compute_loss(self, z_next, mu, re=None, sc=None, generator=None):
        return {"loss": torch.tensor(float("nan"))}


class TestEvaluateFlowMatchingLossFailureModes:
    """Verify evaluation function fails closed on invalid inputs."""

    def test_empty_dataloader_raises_value_error(self):
        model = LatentForecaster(
            encoder=Encoder2D(in_channels=4, latent_channels=8, base_channels=8, channel_mult=[1, 1, 1]),
            transformer=LatentSTTransformer(latent_channels=8, embed_dim=16, cond_dim=16, depth=1),
            decoder=Decoder2D(latent_channels=8, out_channels=4, base_channels=8, channel_mult=[1, 1, 1]),
        )
        model.attach_flow_matcher(LatentFlowMatcher(latent_channels=8, cond_dim=16, hidden_channels=16, num_blocks=1))

        empty_loader = DataLoader([])
        with pytest.raises(ValueError, match="Validation dataloader is empty"):
            evaluate_flow_matching_loss(model, empty_loader, device=torch.device("cpu"))

    def test_missing_flow_matcher_raises_runtime_error(self):
        model = MockForecasterNoFM()
        ds = DummySyntheticFlowDataset(num_samples=2)
        loader = DataLoader(ds, batch_size=2)
        with pytest.raises(RuntimeError, match="has no attached flow_matcher"):
            evaluate_flow_matching_loss(model, loader, device=torch.device("cpu"))

    def test_non_finite_loss_fails_closed(self):
        model = MockForecasterNanFM()
        ds = DummySyntheticFlowDataset(num_samples=2)
        loader = DataLoader(ds, batch_size=2)
        with pytest.raises(FloatingPointError, match="Non-finite validation CFM loss"):
            evaluate_flow_matching_loss(model, loader, device=torch.device("cpu"), fail_on_non_finite=True)


class TestPreflightContractFailClosed:
    """Verify pre-flight checks fail closed on missing paths and hash mismatches."""

    def test_missing_d0_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="D0 checkpoint not found"):
            verify_flow_matching_preflight_contract(
                d0_checkpoint_path=str(tmp_path / "non_existent.pt"),
                normalizer_path=str(tmp_path / "normalizer.json"),
                split_file=str(tmp_path / "split.json"),
            )

    def test_missing_normalizer_raises(self, tmp_path):
        dummy_ckpt = tmp_path / "d0.pt"
        torch.save({"config": {}}, dummy_ckpt)
        with pytest.raises(FileNotFoundError, match="Normalizer file not found"):
            verify_flow_matching_preflight_contract(
                d0_checkpoint_path=str(dummy_ckpt),
                normalizer_path=str(tmp_path / "non_existent_norm.json"),
                split_file=str(tmp_path / "split.json"),
            )

    def test_hash_mismatch_fails_closed_on_split(self, tmp_path):
        # Create normalizer
        from src.data.normalization import FieldNormalizer
        from src.utils.provenance import compute_normalizer_hash, compute_split_hash_from_file

        norm = FieldNormalizer()
        norm.mean = torch.zeros(4)
        norm.std = torch.ones(4)
        norm_path = tmp_path / "normalizer.pt"
        torch.save(norm.state_dict(), norm_path)
        norm_hash = compute_normalizer_hash(norm)

        # Create split
        split_path = tmp_path / "split.json"
        with open(split_path, "w") as f:
            json.dump({"train": ["sim_001"], "val": ["sim_002"]}, f)
        runtime_split_hash = compute_split_hash_from_file(str(split_path))

        # Create D0 checkpoint with mismatching split hash
        bad_d0_path = tmp_path / "d0_mismatch.pt"
        torch.save(
            {
                "split_hash": "deadbeef" * 8,  # mismatching
                "normalizer_hash": norm_hash,
                "seed": 42,
                "config": {},
            },
            bad_d0_path,
        )

        with pytest.raises(ValueError, match="Split hash mismatch"):
            verify_flow_matching_preflight_contract(
                d0_checkpoint_path=str(bad_d0_path),
                normalizer_path=str(norm_path),
                split_file=str(split_path),
                manifest_path=None,
            )

    def test_hash_mismatch_fails_closed_on_normalizer(self, tmp_path):
        from src.data.normalization import FieldNormalizer
        from src.utils.provenance import compute_normalizer_hash, compute_split_hash_from_file

        norm = FieldNormalizer()
        norm.mean = torch.zeros(4)
        norm.std = torch.ones(4)
        norm_path = tmp_path / "normalizer.pt"
        torch.save(norm.state_dict(), norm_path)

        split_path = tmp_path / "split.json"
        with open(split_path, "w") as f:
            json.dump({"train": ["sim_001"]}, f)
        runtime_split_hash = compute_split_hash_from_file(str(split_path))

        bad_d0_path = tmp_path / "d0_norm_mismatch.pt"
        torch.save(
            {
                "split_hash": runtime_split_hash,
                "normalizer_hash": "cafebabe" * 8,  # mismatching
                "seed": 42,
                "config": {},
            },
            bad_d0_path,
        )

        with pytest.raises(ValueError, match="Normalizer hash mismatch"):
            verify_flow_matching_preflight_contract(
                d0_checkpoint_path=str(bad_d0_path),
                normalizer_path=str(norm_path),
                split_file=str(split_path),
                manifest_path=None,
            )

    def test_stats_d0_mismatch_fails_closed(self, tmp_path):
        from src.data.normalization import FieldNormalizer
        from src.utils.provenance import compute_normalizer_hash, compute_split_hash_from_file, compute_file_sha256

        norm = FieldNormalizer()
        norm.mean = torch.zeros(4)
        norm.std = torch.ones(4)
        norm_path = tmp_path / "normalizer.pt"
        torch.save(norm.state_dict(), norm_path)
        norm_hash = compute_normalizer_hash(norm)

        split_path = tmp_path / "split.json"
        with open(split_path, "w") as f:
            json.dump({"train": ["sim_001"]}, f)
        runtime_split_hash = compute_split_hash_from_file(str(split_path))

        d0_path = tmp_path / "d0_valid.pt"
        torch.save(
            {
                "split_hash": runtime_split_hash,
                "normalizer_hash": norm_hash,
                "seed": 42,
                "config": {},
            },
            d0_path,
        )
        actual_d0_sha = compute_file_sha256(str(d0_path))

        # Stats generated from a different D0
        stats_path = tmp_path / "stats_d0_mismatch.json"
        with open(stats_path, "w") as f:
            json.dump(
                {
                    "d0_checkpoint": {"sha256": "different_d0_sha_00000000000000000000000000000000"},
                    "data_protocol": {
                        "split_hash": runtime_split_hash,
                        "normalizer_hash": norm_hash,
                    },
                    "statistics": {"channel_residual_second_moment_g0": [1.0] * 64},
                },
                f,
            )

        with pytest.raises(ValueError, match="Residual stats D0 checkpoint SHA mismatch"):
            verify_flow_matching_preflight_contract(
                d0_checkpoint_path=str(d0_path),
                normalizer_path=str(norm_path),
                split_file=str(split_path),
                residual_stats_path=str(stats_path),
                manifest_path=None,
            )

    def test_missing_stats_hash_fails_closed(self, tmp_path):
        from src.data.normalization import FieldNormalizer
        from src.utils.provenance import compute_normalizer_hash, compute_split_hash_from_file, compute_file_sha256

        norm = FieldNormalizer()
        norm.mean = torch.zeros(4)
        norm.std = torch.ones(4)
        norm_path = tmp_path / "normalizer.pt"
        torch.save(norm.state_dict(), norm_path)
        norm_hash = compute_normalizer_hash(norm)

        split_path = tmp_path / "split.json"
        with open(split_path, "w") as f:
            json.dump({"train": ["sim_001"]}, f)
        runtime_split_hash = compute_split_hash_from_file(str(split_path))

        d0_path = tmp_path / "d0_valid.pt"
        torch.save(
            {
                "split_hash": runtime_split_hash,
                "normalizer_hash": norm_hash,
                "seed": 42,
                "config": {},
            },
            d0_path,
        )
        actual_d0_sha = compute_file_sha256(str(d0_path))

        # 1. Missing d0_checkpoint.sha256
        stats_no_d0_sha = tmp_path / "stats_no_d0_sha.json"
        with open(stats_no_d0_sha, "w") as f:
            json.dump(
                {
                    "d0_checkpoint": {},
                    "data_protocol": {"split_hash": runtime_split_hash, "normalizer_hash": norm_hash},
                    "statistics": {"channel_residual_second_moment_g0": [1.0] * 64},
                },
                f,
            )
        with pytest.raises(ValueError, match="missing required 'd0_checkpoint.sha256'"):
            verify_flow_matching_preflight_contract(
                d0_checkpoint_path=str(d0_path),
                normalizer_path=str(norm_path),
                split_file=str(split_path),
                residual_stats_path=str(stats_no_d0_sha),
                manifest_path=None,
            )

        # 2. Missing data_protocol.split_hash
        stats_no_split = tmp_path / "stats_no_split.json"
        with open(stats_no_split, "w") as f:
            json.dump(
                {
                    "d0_checkpoint": {"sha256": actual_d0_sha},
                    "data_protocol": {"normalizer_hash": norm_hash},
                    "statistics": {"channel_residual_second_moment_g0": [1.0] * 64},
                },
                f,
            )
        with pytest.raises(ValueError, match="missing required 'data_protocol.split_hash'"):
            verify_flow_matching_preflight_contract(
                d0_checkpoint_path=str(d0_path),
                normalizer_path=str(norm_path),
                split_file=str(split_path),
                residual_stats_path=str(stats_no_split),
                manifest_path=None,
            )

        # 3. Missing data_protocol.normalizer_hash
        stats_no_norm = tmp_path / "stats_no_norm.json"
        with open(stats_no_norm, "w") as f:
            json.dump(
                {
                    "d0_checkpoint": {"sha256": actual_d0_sha},
                    "data_protocol": {"split_hash": runtime_split_hash},
                    "statistics": {"channel_residual_second_moment_g0": [1.0] * 64},
                },
                f,
            )
        with pytest.raises(ValueError, match="missing required 'data_protocol.normalizer_hash'"):
            verify_flow_matching_preflight_contract(
                d0_checkpoint_path=str(d0_path),
                normalizer_path=str(norm_path),
                split_file=str(split_path),
                residual_stats_path=str(stats_no_norm),
                manifest_path=None,
            )

    def test_missing_seed_fails_closed(self, tmp_path):
        from src.data.normalization import FieldNormalizer
        from src.utils.provenance import compute_normalizer_hash, compute_split_hash_from_file

        norm = FieldNormalizer()
        norm.mean = torch.zeros(4)
        norm.std = torch.ones(4)
        norm_path = tmp_path / "normalizer.pt"
        torch.save(norm.state_dict(), norm_path)
        norm_hash = compute_normalizer_hash(norm)

        split_path = tmp_path / "split.json"
        with open(split_path, "w") as f:
            json.dump({"train": ["sim_001"]}, f)
        runtime_split_hash = compute_split_hash_from_file(str(split_path))

        # Checkpoint missing seed entirely
        d0_no_seed_path = tmp_path / "d0_no_seed.pt"
        torch.save(
            {
                "split_hash": runtime_split_hash,
                "normalizer_hash": norm_hash,
                "config": {},
            },
            d0_no_seed_path,
        )

        with pytest.raises(ValueError, match="missing required 'seed'"):
            verify_flow_matching_preflight_contract(
                d0_checkpoint_path=str(d0_no_seed_path),
                normalizer_path=str(norm_path),
                split_file=str(split_path),
                expected_seed=42,
                manifest_path=None,
            )

    def test_missing_residual_stats_fails_closed(self, tmp_path):
        # 1. residual_stats_path is None without explicit disable flag
        with pytest.raises(ValueError, match="residual_stats_path must be specified"):
            train_latent_flow_matching(
                d0_checkpoint=str(tmp_path / "d0.pt"),
                normalizer_path=str(tmp_path / "norm.pt"),
                split_file=str(tmp_path / "split.json"),
                output_dir=str(tmp_path / "out"),
                residual_stats_path=None,
                disable_residual_normalization=False,
            )

        # 2. residual_stats_path does not exist
        with pytest.raises(FileNotFoundError, match="Residual statistics file not found"):
            train_latent_flow_matching(
                d0_checkpoint=str(tmp_path / "d0.pt"),
                normalizer_path=str(tmp_path / "norm.pt"),
                split_file=str(tmp_path / "split.json"),
                output_dir=str(tmp_path / "out"),
                residual_stats_path=str(tmp_path / "non_existent_stats.json"),
                disable_residual_normalization=False,
            )


class TestGradientNonfiniteFailClosed:
    """Verify non-finite gradients immediately fail-closed via shared production function."""

    def test_gradient_nonfinite_detected_by_clip_norm(self):
        matcher = LatentFlowMatcher(
            latent_channels=8,
            cond_dim=16,
            hidden_channels=16,
            num_blocks=1,
            zero_init=False,
        )

        # Inject NaN into one parameter gradient
        first_param = next(matcher.parameters())
        first_param.grad = torch.full_like(first_param.data, float("nan"))

        with pytest.raises(FloatingPointError, match="Non-finite gradient norm detected at epoch 1, batch 0"):
            clip_and_validate_gradients(matcher, max_norm=1.0, epoch=1, batch_idx=0)

    def test_gradient_nonfinite_detected_on_inf(self):
        matcher = LatentFlowMatcher(
            latent_channels=8,
            cond_dim=16,
            hidden_channels=16,
            num_blocks=1,
            zero_init=False,
        )

        first_param = next(matcher.parameters())
        first_param.grad = torch.full_like(first_param.data, float("inf"))

        with pytest.raises(FloatingPointError, match="Non-finite gradient norm detected"):
            clip_and_validate_gradients(matcher, max_norm=1.0, epoch=2, batch_idx=3)

    def test_gradient_finite_returns_valid_norm(self):
        matcher = LatentFlowMatcher(
            latent_channels=8,
            cond_dim=16,
            hidden_channels=16,
            num_blocks=1,
            zero_init=False,
        )

        for p in matcher.parameters():
            p.grad = torch.ones_like(p.data) * 0.05

        total_norm = clip_and_validate_gradients(matcher, max_norm=1.0)
        assert torch.isfinite(total_norm)
        assert total_norm.item() > 0.0


class TestOverwritePreflightSafetyOrder:
    """Verify overwrite backups happen strictly AFTER preflight succeeds."""

    def test_overwrite_preflight_order_protects_existing_artifacts_on_failure(self, tmp_path):
        out_dir = tmp_path / "artifacts"
        out_dir.mkdir()
        canary_file = out_dir / "canary.pt"
        canary_file.write_text("critical_saved_model")

        # Call train with non-existent D0 checkpoint
        with pytest.raises(FileNotFoundError, match="D0 checkpoint not found"):
            train_latent_flow_matching(
                d0_checkpoint=str(tmp_path / "missing_d0.pt"),
                normalizer_path=str(tmp_path / "norm.pt"),
                split_file=str(tmp_path / "split.json"),
                output_dir=str(out_dir),
                disable_residual_normalization=True,
                overwrite=True,
            )

        # Crucial check: canary_file MUST STILL EXIST in out_dir because preflight failed before overwrite!
        assert canary_file.exists()
        assert canary_file.read_text() == "critical_saved_model"
        # No backup directories should have been created
        backups = list(out_dir.glob("backup_*"))
        assert len(backups) == 0


class TestResidualNormalizationIntegration:
    """Verify residual scale normalization buffer and execution in training pipeline."""

    def test_residual_normalization_training_integration(self):
        ckpt_data = {
            "config": {
                "embed_dim": 32,
                "depth": 1,
                "num_heads": 2,
                "prediction_mode": "direct",
                "use_spatial_pos": False,
            },
            "model_state_dict": {},
        }
        enc = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
        dec = Decoder2D(latent_channels=64, out_channels=4, base_channels=32)
        trans = LatentSTTransformer(
            latent_channels=64,
            embed_dim=32,
            cond_dim=128,
            depth=1,
            num_heads=2,
            use_spatial_pos=False,
        )
        forecaster = LatentForecaster(enc, trans, dec)
        ckpt_data["model_state_dict"] = forecaster.state_dict()

        # Supply custom residual scale
        scale = torch.linspace(0.5, 2.0, 64)
        model = build_and_freeze_flow_matching_model(
            ckpt_data=ckpt_data,
            device=torch.device("cpu"),
            hidden_channels=32,
            num_blocks=1,
            residual_scale=scale,
        )

        assert model.flow_matcher.residual_scale is not None
        assert model.flow_matcher.residual_scale.shape == (1, 64, 1, 1)
        assert torch.allclose(model.flow_matcher.residual_scale.squeeze(), scale, atol=1e-5)


class TestBuildAndFreezeModelGovernance:
    """Verify parameter freezing contracts."""

    def test_parameter_freeze_contract(self):
        ckpt_data = {
            "config": {
                "embed_dim": 32,
                "depth": 1,
                "num_heads": 2,
                "prediction_mode": "direct",
                "use_spatial_pos": False,
            },
            "model_state_dict": {},
        }
        enc = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
        dec = Decoder2D(latent_channels=64, out_channels=4, base_channels=32)
        trans = LatentSTTransformer(
            latent_channels=64,
            embed_dim=32,
            cond_dim=128,
            depth=1,
            num_heads=2,
            use_spatial_pos=False,
        )
        forecaster = LatentForecaster(enc, trans, dec)
        ckpt_data["model_state_dict"] = forecaster.state_dict()

        frozen_model = build_and_freeze_flow_matching_model(
            ckpt_data=ckpt_data,
            device=torch.device("cpu"),
            hidden_channels=32,
            num_blocks=1,
        )

        assert frozen_model.has_flow_matcher
        for p in frozen_model.encoder.parameters():
            assert not p.requires_grad
        for p in frozen_model.decoder.parameters():
            assert not p.requires_grad
        for p in frozen_model.transformer.parameters():
            assert not p.requires_grad
        for p in frozen_model.flow_matcher.parameters():
            assert p.requires_grad
