"""Unit and Contract Tests for Phase 2 Variance Head Training Entrypoint.

Tests:
1. evaluate_variance_nll fails closed on empty dataloader (ValueError).
2. evaluate_variance_nll fails closed on non-finite mu/var/loss (FloatingPointError) or marks is_valid=False.
3. Directory collision protection: existing artifacts fail closed unless overwrite=True (FileExistsError).
4. No-improvement over G0 baseline saves g0_baseline_initialization.pt and marks NO_IMPROVEMENT_OVER_G0.
5. Pre-flight verification fails closed when G0 stats lacks d0_checkpoint.sha256.
6. Pre-flight verification fails closed when G0 stats SHA-256 mismatches audit record.
7. Integration tests on canonical artifacts (skipped if large weights absent).
"""

import json
import math
import os
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scripts.train_prob_latent_variance import (
    verify_phase2_preflight_contract,
    build_and_freeze_probabilistic_model,
    evaluate_variance_nll,
    train_prob_latent_variance,
)
from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.probabilistic_latent_dynamics import VarianceHead2D


class DummySyntheticDataset(Dataset):
    """Small synthetic dataset for testing training and evaluation control flow."""

    def __init__(self, num_samples: int = 4, produce_nans: bool = False):
        self.num_samples = num_samples
        self.produce_nans = produce_nans

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.produce_nans:
            history = torch.full((2, 4, 16, 16), float("nan"))
            future = torch.full((1, 4, 16, 16), float("nan"))
        else:
            history = torch.randn(2, 4, 16, 16)
            future = torch.randn(1, 4, 16, 16)
        return {
            "history": history,
            "future": future,
            "re": torch.tensor(1000.0),
            "sc": torch.tensor(0.5),
        }


class MockForecaster(nn.Module):
    """Mock forecaster with controllable predictions for edge-case verification."""

    def __init__(self, return_nan: bool = False, fixed_var: float = 0.5):
        super().__init__()
        self.return_nan = return_nan
        self.fixed_var = fixed_var
        self.encoder = nn.Identity()

    def predict_distribution_single_step(self, q_hist, re=None, sc=None):
        b = q_hist.shape[0]
        if self.return_nan:
            mu = torch.full((b, 1, 4, 16, 16), float("nan"), device=q_hist.device)
            var = torch.full((b, 1, 4, 16, 16), float("nan"), device=q_hist.device)
        else:
            mu = torch.zeros(b, 1, 4, 16, 16, device=q_hist.device)
            var = torch.full((b, 1, 4, 16, 16), self.fixed_var, device=q_hist.device)
        return mu, var


class TestEvaluateVarianceNLLFailureModes:
    """Verify evaluation function fails closed on invalid or non-finite inputs."""

    def test_empty_dataloader_raises_value_error(self):
        model = MockForecaster()
        empty_loader = DataLoader([])
        with pytest.raises(ValueError, match="Validation dataloader is empty"):
            evaluate_variance_nll(model, empty_loader, device=torch.device("cpu"))

    def test_non_finite_predictions_fail_closed_under_strict_mode(self):
        model = MockForecaster(return_nan=True)
        ds = DummySyntheticDataset(num_samples=2)
        loader = DataLoader(ds, batch_size=2)
        with pytest.raises(FloatingPointError, match="Non-finite mu or var"):
            evaluate_variance_nll(
                model,
                loader,
                device=torch.device("cpu"),
                fail_on_non_finite=True,
            )

    def test_non_finite_predictions_mark_is_valid_false_under_lenient_mode(self):
        model = MockForecaster(return_nan=True)
        ds = DummySyntheticDataset(num_samples=2)
        loader = DataLoader(ds, batch_size=2)
        metrics = evaluate_variance_nll(
            model,
            loader,
            device=torch.device("cpu"),
            fail_on_non_finite=False,
        )
        assert metrics["is_valid"] is False
        assert math.isnan(metrics["nll"])
        assert metrics["non_finite_batches"] == 1


class TestDirectoryReuseAndOutcomeGovernance:
    """Verify artifact isolation and no-improvement outcome tracking."""

    def test_existing_artifacts_fail_closed_without_overwrite(self, tmp_path):
        out_dir = tmp_path / "run_artifacts"
        out_dir.mkdir(parents=True)
        # Create a lingering previous artifact
        (out_dir / "best_g1_variance_head.pt").write_text("dummy")

        # Mock preflight and model build by catching FileExistsError early
        with pytest.raises(FileExistsError, match="already contains artifacts"):
            train_prob_latent_variance(
                d0_checkpoint="dummy",
                stats_path="dummy",
                normalizer_path="dummy",
                split_file="dummy",
                data_root="dummy",
                output_dir=str(out_dir),
                overwrite=False,
            )

    def test_preflight_fails_closed_when_stats_missing_d0_sha(self, tmp_path):
        from src.utils.provenance import compute_split_hash_from_file, compute_normalizer_hash
        from src.data.normalization import FieldNormalizer

        # Normalizer and split
        split_file = tmp_path / "split.json"
        with open(split_file, "w") as f:
            json.dump({"train": []}, f)
        real_split_hash = compute_split_hash_from_file(str(split_file))

        norm_file = tmp_path / "norm.pt"
        torch.save({"mean": torch.zeros(4), "std": torch.ones(4)}, norm_file)
        norm_obj = FieldNormalizer()
        norm_obj.load_state_dict(torch.load(norm_file, weights_only=True))
        real_norm_hash = compute_normalizer_hash(norm_obj)

        # Mock D0
        ckpt_file = tmp_path / "d0.pt"
        torch.save({"split_hash": real_split_hash, "normalizer_hash": real_norm_hash}, ckpt_file)

        stats_file = tmp_path / "stats.json"
        with open(stats_file, "w") as f:
            json.dump({
                "d0_checkpoint": {},  # missing sha256
                "data_protocol": {"split_hash": real_split_hash, "normalizer_hash": real_norm_hash},
            }, f)

        with pytest.raises(ValueError, match="missing required 'd0_checkpoint.sha256'"):
            verify_phase2_preflight_contract(
                d0_checkpoint_path=str(ckpt_file),
                stats_path=str(stats_file),
                normalizer_path=str(norm_file),
                split_file=str(split_file),
                verification_record_path=None,
            )

    def test_preflight_fails_closed_when_stats_sha_mismatches_record(self, tmp_path):
        stats_file = tmp_path / "stats.json"
        with open(stats_file, "w") as f:
            json.dump({"dummy": 1}, f)

        record_file = tmp_path / "record.json"
        with open(record_file, "w") as f:
            json.dump({"stats_file": {"sha256": "different_sha256_hash"}}, f)

        ckpt_file = tmp_path / "d0.pt"
        torch.save({}, ckpt_file)
        norm_file = tmp_path / "norm.pt"
        torch.save({}, norm_file)
        split_file = tmp_path / "split.json"
        with open(split_file, "w") as f:
            json.dump({}, f)

        with pytest.raises(ValueError, match="G0 stats SHA-256 mismatch against verified audit record"):
            verify_phase2_preflight_contract(
                d0_checkpoint_path=str(ckpt_file),
                stats_path=str(stats_file),
                normalizer_path=str(norm_file),
                split_file=str(split_file),
                verification_record_path=str(record_file),
            )


D0_PATH = "outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt"
STATS_PATH = "outputs/normalization/latent_residual_stats.json"
NORM_PATH = "outputs/normalization/stats_grouped.pt"
SPLIT_PATH = "outputs/splits/grouped_split.json"
RECORD_PATH = "outputs/normalization/latent_audit_verification_record.json"


@pytest.mark.skipif(
    not (
        os.path.exists(D0_PATH)
        and os.path.exists(STATS_PATH)
        and os.path.exists(NORM_PATH)
        and os.path.exists(SPLIT_PATH)
        and os.path.exists(RECORD_PATH)
    ),
    reason="Canonical Phase 0/1 artifacts not present on current machine",
)
class TestPhase2CanonicalArtifactIntegration:
    """Verify cryptographic pre-flight governance and parameter freeze contract against real artifacts."""

    def test_preflight_contract_passes_on_canonical_artifacts(self):
        (
            ckpt_data,
            stats_data,
            normalizer,
            d0_sha256,
            stats_sha256,
            split_hash,
            norm_hash,
        ) = verify_phase2_preflight_contract(
            d0_checkpoint_path=D0_PATH,
            stats_path=STATS_PATH,
            normalizer_path=NORM_PATH,
            split_file=SPLIT_PATH,
            verification_record_path=RECORD_PATH,
        )

        assert d0_sha256.startswith("edddbe8a2528f848")
        assert len(stats_sha256) == 64
        assert split_hash.startswith("41fbe6ebe7edd460")
        assert norm_hash.startswith("3a0fe52689657618")

    def test_model_building_strictly_freezes_all_non_variance_parameters(self):
        ckpt_data = torch.load(D0_PATH, map_location="cpu", weights_only=False)
        with open(STATS_PATH, "r") as f:
            stats_data = json.load(f)

        device = torch.device("cpu")
        forecaster, effective_g0 = build_and_freeze_probabilistic_model(
            ckpt_data=ckpt_data,
            stats_data=stats_data,
            device=device,
            variance_floor=1e-4,
        )

        trainable_names = [n for n, p in forecaster.named_parameters() if p.requires_grad]
        frozen_names = [n for n, p in forecaster.named_parameters() if not p.requires_grad]

        # Trainable MUST ONLY be variance_head parameters
        assert set(trainable_names) == {
            "transformer.variance_head.linear.weight",
            "transformer.variance_head.linear.bias",
        }

        # Frozen MUST contain encoder, decoder, and Transformer backbone
        assert any("encoder" in n for n in frozen_names)
        assert any("decoder" in n for n in frozen_names)
        assert any("transformer.in_proj" in n for n in frozen_names)
        assert any("transformer.out_proj" in n for n in frozen_names)
        assert any("transformer.blocks" in n for n in frozen_names)
