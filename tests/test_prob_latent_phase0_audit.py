"""Rigorous Phase 0 verification suite for ProbLatent-R1.

Covers all required acceptance scenarios:
1. Unknown use_spatial_pos without provenance fails closed (raises ValueError).
2. Git ancestry check failure/missing object raises RuntimeError (no silent False fallback).
3. Field diagnostic parity: verifying custom metrics vs standard compute_vrmse / evaluate_field_metrics.
4. Mathematical correctness of fixed-mean G0 variance:
   - When residual is constant [2, 2, 2], centered variance is 0, but G0 second moment is 4.
5. Numerical boundary handling for G1 variance head initialization:
   - When v_G0 is near or below variance_floor, zero NaN/Inf, G0 and G1 align numerically within float tolerance.
6. Uneven/tail batch accumulation:
   - Streaming statistics correctly weight batches by actual token count.
7. Atomic D0 forecaster parity check and rejection of mismatched representation weights.
"""

import os
import json
from pathlib import Path
import subprocess
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.latent_transformer import LatentSTTransformer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.latent_forecaster import LatentForecaster
from src.utils.checkpoint import (
    resolve_spatial_pos_config,
    compute_g1_bias_init_from_g0,
    inverse_softplus,
)
from src.metrics.field import compute_vrmse, evaluate_field_metrics


class TestSpatialPosConfigResolutionFailClosed:
    """Verify fail-closed resolution of spatial positional encoding configuration."""

    def test_explicit_config_honored(self):
        """Explicit boolean config is honored unconditionally."""
        assert resolve_spatial_pos_config({"config": {"use_spatial_pos": True}}) is True
        assert resolve_spatial_pos_config({"config": {"use_spatial_pos": False}}) is False

    def test_legacy_known_commits_resolve_correctly(self):
        """Pre-0f2ed24 commits resolve to False, post-0f2ed24 resolve to True."""
        h8_ckpt = {"training_git_commit": "6593b65005843c3f3c2640cb262bfd56708e11ad"}
        assert resolve_spatial_pos_config(h8_ckpt) is False

        h16_ckpt = {"training_git_commit": "0a8a4da137e3f73c4f696cc81f527c32a8ba8405"}
        assert resolve_spatial_pos_config(h16_ckpt) is False

        h12_ckpt = {"training_git_commit": "f9f5815e5a8d9be37d7d0830d1b3b08b7a32866f"}
        assert resolve_spatial_pos_config(h12_ckpt) is True

    def test_unknown_config_without_provenance_fails_closed(self):
        """When use_spatial_pos is missing and commit provenance is absent, formal load must fail."""
        unknown_ckpt = {"config": {"embed_dim": 256}}
        with pytest.raises(ValueError, match="Cannot resolve 'use_spatial_pos' for checkpoint"):
            resolve_spatial_pos_config(unknown_ckpt, allow_unverified_fallback=False)

        # Fallback only works when explicitly permitted
        fallback_val = resolve_spatial_pos_config(
            unknown_ckpt, allow_unverified_fallback=True, default_if_unverified=False
        )
        assert fallback_val is False

    def test_git_missing_commit_raises_runtime_error(self, monkeypatch):
        """Git error (e.g. missing commit object, non-zero returncode > 1) must raise RuntimeError, not False."""
        ckpt_with_ghost_commit = {"training_git_commit": "deadbeef1234567890abcdef1234567890abcdef"}

        # Simulate git returning code 128 (fatal: Not a valid object name)
        def mock_subprocess_run(*args, **kwargs):
            return subprocess.CompletedProcess(
                args=args[0],
                returncode=128,
                stdout="",
                stderr="fatal: Not a valid object name deadbeef",
            )

        monkeypatch.setattr("subprocess.run", mock_subprocess_run)

        with pytest.raises(RuntimeError, match="Git ancestry check failed for commit"):
            resolve_spatial_pos_config(ckpt_with_ghost_commit)


class TestFixedMeanG0SecondMomentMath:
    """Verify that under fixed mean, G0 variance is E[r^2], NOT centered Var(r)."""

    def test_constant_bias_residual_math(self):
        """If residual r is constant 2.0, Var(r) is 0, but second moment E[r^2] is 4.0."""
        r = torch.full((10, 64, 16, 32), 2.0, dtype=torch.float64)

        mean_r = r.mean(dim=(0, 2, 3))
        var_r = r.var(dim=(0, 2, 3), unbiased=False)
        second_moment_r = (r ** 2).mean(dim=(0, 2, 3))

        assert torch.allclose(var_r, torch.zeros(64, dtype=torch.float64), atol=1e-7)
        assert torch.allclose(second_moment_r, torch.full((64,), 4.0, dtype=torch.float64), atol=1e-7)
        assert torch.allclose(second_moment_r, var_r + mean_r ** 2, atol=1e-7)

    def test_gaussian_nll_is_minimized_at_second_moment_under_fixed_mean(self):
        """Directly verify GaussianNLLLoss is minimized when var = E[r^2], not Var(r)."""
        torch.manual_seed(42)
        y = torch.randn(10000) * 1.0 + 2.0  # Var = 1.0, Mean = 2.0 -> E[y^2] = 5.0
        mu = torch.zeros_like(y)

        loss_at_centered_var = F.gaussian_nll_loss(input=mu, target=y, var=torch.full_like(y, 1.0), eps=1e-6)
        loss_at_second_moment = F.gaussian_nll_loss(input=mu, target=y, var=torch.full_like(y, 5.0), eps=1e-6)

        assert loss_at_second_moment.item() < loss_at_centered_var.item()


class TestG1VarianceHeadNumericalBoundaries:
    """Verify that G1 initialization handles boundary conditions and matches G0 numerically."""

    def test_inverse_softplus_numerical_stability(self):
        """inverse_softplus must invert softplus accurately across scales without NaN or inf."""
        x = torch.tensor([1e-6, 1e-4, 0.1, 1.0, 10.0, 50.0], dtype=torch.float64)
        sp_inv = inverse_softplus(x)
        assert not torch.isnan(sp_inv).any()
        assert not torch.isinf(sp_inv).any()

        recovered = F.softplus(sp_inv)
        torch.testing.assert_close(recovered, x, rtol=1e-5, atol=1e-6)

    def test_boundary_handling_when_g0_variance_is_below_floor(self):
        """When channel variance <= variance_floor, compute_g1_bias_init_from_g0 applies safety margin."""
        variance_floor = 1e-4
        min_margin = 1e-5

        # Channels with variance lower than floor, exactly at floor, and normal
        v_g0 = torch.tensor([0.0, 1e-5, 1e-4, 0.05, 0.2], dtype=torch.float32)
        b_init, effective_g0 = compute_g1_bias_init_from_g0(v_g0, variance_floor=variance_floor, min_margin=min_margin)

        # No NaNs or Infs
        assert not torch.isnan(b_init).any()
        assert not torch.isinf(b_init).any()

        # Forward prediction through softplus + floor
        pred_var = F.softplus(b_init) + variance_floor

        # Must numerically match effective_g0
        max_abs_err = torch.max(torch.abs(pred_var - effective_g0)).item()
        torch.testing.assert_close(pred_var, effective_g0, rtol=1e-5, atol=1e-6)
        assert max_abs_err < 1e-6

        # Check clamped channels match floor + margin
        assert torch.isclose(effective_g0[0], torch.tensor(variance_floor + min_margin))
        assert torch.isclose(effective_g0[1], torch.tensor(variance_floor + min_margin))
        assert torch.isclose(effective_g0[2], torch.tensor(variance_floor + min_margin))
        # Normal channels remain unchanged
        assert torch.isclose(effective_g0[3], torch.tensor(0.05))
        assert torch.isclose(effective_g0[4], torch.tensor(0.2))


class TestUnevenBatchTokenWeightedAccumulation:
    """Verify streaming token-weighted accumulation handles uneven/tail batches correctly."""

    def test_token_weighted_aggregation_math(self):
        num_channels = 64
        Hz, Wz = 16, 32
        tokens_per_frame = Hz * Wz

        # Batch 1: batch_size = 8 -> 8 * 512 = 4096 tokens per channel
        # Batch 2: tail batch_size = 3 -> 3 * 512 = 1536 tokens per channel
        r1 = torch.full((8, num_channels, Hz, Wz), 1.0, dtype=torch.float64)
        r2 = torch.full((3, num_channels, Hz, Wz), 3.0, dtype=torch.float64)

        # Naive batch averaging would give: (1.0 + 3.0) / 2 = 2.0
        # True token-weighted average: (8 * 1.0 + 3 * 3.0) / 11 = 17 / 11 = 1.54545...
        expected_mean = (8 * 1.0 + 3 * 3.0) / 11.0
        expected_second_moment = (8 * 1.0**2 + 3 * 3.0**2) / 11.0  # 35 / 11 = 3.1818...

        sum_r = torch.zeros(num_channels, dtype=torch.float64)
        sum_r2 = torch.zeros(num_channels, dtype=torch.float64)
        total_tokens = 0

        for r_batch in [r1, r2]:
            b = r_batch.shape[0]
            r_flat = r_batch.permute(1, 0, 2, 3).reshape(num_channels, -1)
            sum_r += r_flat.sum(dim=1)
            sum_r2 += (r_flat ** 2).sum(dim=1)
            total_tokens += b * tokens_per_frame

        computed_mean = sum_r / total_tokens
        computed_second_moment = sum_r2 / total_tokens

        assert torch.allclose(computed_mean, torch.full((num_channels,), expected_mean, dtype=torch.float64), atol=1e-7)
        assert torch.allclose(computed_second_moment, torch.full((num_channels,), expected_second_moment, dtype=torch.float64), atol=1e-7)


class TestStandardMetricConsistency:
    """Verify diagnostic functions use standard VRMSE definition from src.metrics.field."""

    def test_standard_vrmse_definition(self):
        torch.manual_seed(42)
        pred = torch.randn(4, 32, 64)
        target = torch.randn(4, 32, 64)

        vrmse_std = compute_vrmse(pred, target)

        # Manual calculation of standard VRMSE: sqrt( mean((pred-target)^2) / (Var(target) + eps) )
        mse = torch.mean((pred - target) ** 2, dim=(-2, -1))
        var = torch.var(target, dim=(-2, -1), unbiased=False)
        expected = torch.mean(torch.sqrt(mse / (var + 1e-6)))

        torch.testing.assert_close(vrmse_std, expected)


class TestFullD0AtomicParityVerification:
    """Verify atomic D0 forecaster parity check and rejection of mismatched representation weights."""

    def test_parity_verification_helper(self, tmp_path):
        from scripts.compute_latent_statistics import verify_representation_parity

        enc = Encoder2D(in_channels=4, latent_channels=16, base_channels=8)
        dec = Decoder2D(latent_channels=16, out_channels=4, base_channels=8)
        trans = LatentSTTransformer(latent_channels=16, embed_dim=32, cond_dim=16, depth=1, num_heads=2)
        forecaster = LatentForecaster(enc, trans, dec)

        ae_path = tmp_path / "matching_ae.pt"
        torch.save({
            "encoder_state_dict": enc.state_dict(),
            "decoder_state_dict": dec.state_dict(),
        }, ae_path)

        enc_match, dec_match, max_diff = verify_representation_parity(forecaster, str(ae_path))
        assert enc_match is True
        assert dec_match is True
        assert max_diff == 0.0

        corrupted_ae_path = tmp_path / "corrupted_ae.pt"
        corrupted_dec_sd = {k: v.clone() for k, v in dec.state_dict().items()}
        for k in corrupted_dec_sd:
            corrupted_dec_sd[k] = corrupted_dec_sd[k] + 0.5
            break
        torch.save({
            "encoder_state_dict": enc.state_dict(),
            "decoder_state_dict": corrupted_dec_sd,
        }, corrupted_ae_path)

        enc_match_c, dec_match_c, max_diff_c = verify_representation_parity(forecaster, str(corrupted_ae_path))
        assert enc_match_c is True
        assert dec_match_c is False
        assert max_diff_c > 0.4


class TestProductionStatisticsFunctionAudit:
    """Directly invoke production compute_latent_statistics_and_diagnostics to audit identities and tail batches."""

    def test_production_statistics_channel_identities_and_tail_batch_weighting(self):
        from scripts.compute_latent_statistics import compute_latent_statistics_and_diagnostics

        class MockEncoder(nn.Module):
            def forward(self, q):
                # q: (B, L, 4, Ny, Nx) or (B, 1, 4, Ny, Nx)
                # Map to constant latent for testing
                b = q.shape[0]
                l = q.shape[1]
                # Return z with value 3.0
                return torch.full((b, l, 64, 4, 4), 3.0, dtype=torch.float32)

        class MockTransformer(nn.Module):
            def forward(self, z_hist, re=None, sc=None):
                # z_hist: (B, L, 64, 4, 4)
                # Predict mu with value 1.0 (so residual r = 3.0 - 1.0 = 2.0)
                b = z_hist.shape[0]
                return torch.full((b, 1, 64, 4, 4), 1.0, dtype=torch.float32)

        class MockDecoder(nn.Module):
            def forward(self, z):
                b = z.shape[0]
                return torch.zeros((b, 1, 4, 8, 8), dtype=torch.float32)

        class MockForecaster(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = MockEncoder()
                self.transformer = MockTransformer()
                self.decoder = MockDecoder()

        class MockNormalizer:
            def denormalize(self, x):
                return x

        forecaster = MockForecaster()
        normalizer = MockNormalizer()

        # Create two batches with different batch sizes: B1=6, B2=2 (tail batch)
        # In batch 2, we perturb the latent so residual is 4.0 instead of 2.0
        class VaryingMockTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def forward(self, z_hist, re=None, sc=None):
                b = z_hist.shape[0]
                self.call_count += 1
                # 1st batch: mu = 1.0 -> r = 3.0 - 1.0 = 2.0
                # 2nd batch: mu = -1.0 -> r = 3.0 - (-1.0) = 4.0
                val = 1.0 if self.call_count % 2 == 1 else -1.0
                return torch.full((b, 1, 64, 4, 4), val, dtype=torch.float32)

        forecaster.transformer = VaryingMockTransformer()

        batch1 = {
            "history": torch.zeros((6, 4, 4, 8, 8)),
            "future": torch.zeros((6, 1, 4, 8, 8)),
            "re": torch.zeros((6, 1)),
            "sc": torch.zeros((6, 1)),
        }
        batch2 = {
            "history": torch.zeros((2, 4, 4, 8, 8)),
            "future": torch.zeros((2, 1, 4, 8, 8)),
            "re": torch.zeros((2, 1)),
            "sc": torch.zeros((2, 1)),
        }

        mock_dataloader = [batch1, batch2]

        results = compute_latent_statistics_and_diagnostics(
            forecaster=forecaster,
            dataloader=mock_dataloader,
            normalizer=normalizer,
            device=torch.device("cpu"),
        )

        # Expected token-weighted statistics:
        # B1=6, r=2.0; B2=2, r=4.0
        # Tokens per batch: 6 * 16 = 96, 2 * 16 = 32. Total = 128 tokens per channel.
        # Expected mean = (6 * 2.0 + 2 * 4.0) / 8 = (12 + 8) / 8 = 2.5
        # Expected second moment = (6 * 2.0^2 + 2 * 4.0^2) / 8 = (24 + 32) / 8 = 7.0
        # Expected centered variance = 7.0 - 2.5^2 = 7.0 - 6.25 = 0.75
        expected_mean = 2.5
        expected_m2 = 7.0
        expected_var = 0.75

        means = results["channel_residual_mean"]
        second_moments = results["channel_residual_second_moment_g0"]
        vars_c = results["channel_residual_centered_variance"]
        summary = results["summary"]

        assert results["sample_count"] == 8
        assert results["token_count_per_channel"] == 128
        assert len(means) == 64

        # 1. Channel statistical identity: max_c |m_2,c - v_c - m_c^2| < 1e-14
        for m, m2, v in zip(means, second_moments, vars_c):
            assert abs(m - expected_mean) < 1e-7
            assert abs(m2 - expected_m2) < 1e-7
            assert abs(v - expected_var) < 1e-7
            assert abs(m2 - v - m ** 2) < 1e-14

        # 2. Summary identities
        assert abs(summary["mean_channel_centered_variance"] + summary["mean_channel_squared_bias"] - summary["mean_channel_second_moment"]) < 1e-14
        assert abs(summary["pooled_residual_variance"] - (summary["mean_channel_second_moment"] - summary["pooled_residual_mean"] ** 2)) < 1e-14
        assert abs(summary["mean_channel_centered_variance"] - expected_var) < 1e-7
        assert abs(summary["pooled_residual_variance"] - expected_var) < 1e-7


class TestDataProtocolFingerprintVerificationFailClosed:
    """Verify that compute_latent_statistics fails closed when runtime split/normalizer mismatches checkpoint."""

    def test_split_hash_mismatch_fails_closed(self, tmp_path):
        import json
        from scripts.compute_latent_statistics import verify_data_protocol_against_checkpoint

        split_file = tmp_path / "custom_split.json"
        with open(split_file, "w") as f:
            json.dump({"train": [], "valid": []}, f)

        # Checkpoint requires a specific different split hash
        ckpt_data = {
            "split_hash": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "normalizer_hash": "NONE",
        }

        with pytest.raises(ValueError, match="Split hash contract violation"):
            verify_data_protocol_against_checkpoint(
                ckpt_data=ckpt_data,
                split_file=str(split_file),
                normalizer=None,
            )

    def test_normalizer_hash_mismatch_fails_closed(self, tmp_path):
        import json
        from scripts.compute_latent_statistics import verify_data_protocol_against_checkpoint
        from src.utils.provenance import compute_split_hash_from_file

        split_file = tmp_path / "valid_split.json"
        with open(split_file, "w") as f:
            json.dump({"train": [{"traj_idx": 0}], "valid": []}, f)

        real_split_hash = compute_split_hash_from_file(str(split_file))

        # Split matches, but normalizer hash mismatches
        ckpt_data = {
            "split_hash": real_split_hash,
            "normalizer_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        }

        with pytest.raises(ValueError, match="Normalizer hash contract violation"):
            verify_data_protocol_against_checkpoint(
                ckpt_data=ckpt_data,
                split_file=str(split_file),
                normalizer=None,  # normalizer=None produces 'NONE'
            )

    def test_matching_protocol_passes_verification(self, tmp_path):
        import json
        from scripts.compute_latent_statistics import verify_data_protocol_against_checkpoint
        from src.utils.provenance import compute_split_hash_from_file

        split_file = tmp_path / "matching_split.json"
        with open(split_file, "w") as f:
            json.dump({"train": [{"traj_idx": 0}], "valid": []}, f)

        real_split_hash = compute_split_hash_from_file(str(split_file))

        ckpt_data = {
            "split_hash": real_split_hash,
            "normalizer_hash": "NONE",
        }

        s_hash, n_hash = verify_data_protocol_against_checkpoint(
            ckpt_data=ckpt_data,
            split_file=str(split_file),
            normalizer=None,
        )
        assert s_hash == real_split_hash
        assert n_hash == "NONE"

    def test_missing_split_hash_fails_closed(self, tmp_path):
        import json
        from scripts.compute_latent_statistics import verify_data_protocol_against_checkpoint

        split_file = tmp_path / "valid_split.json"
        with open(split_file, "w") as f:
            json.dump({"train": []}, f)

        # Checkpoint missing split_hash
        ckpt_data = {
            "normalizer_hash": "NONE",
        }
        with pytest.raises(ValueError, match="Missing required 'split_hash'"):
            verify_data_protocol_against_checkpoint(
                ckpt_data=ckpt_data,
                split_file=str(split_file),
                normalizer=None,
            )

    def test_missing_normalizer_hash_fails_closed(self, tmp_path):
        import json
        from scripts.compute_latent_statistics import verify_data_protocol_against_checkpoint
        from src.utils.provenance import compute_split_hash_from_file

        split_file = tmp_path / "valid_split.json"
        with open(split_file, "w") as f:
            json.dump({"train": []}, f)

        real_split_hash = compute_split_hash_from_file(str(split_file))

        # Checkpoint missing normalizer_hash
        ckpt_data = {
            "split_hash": real_split_hash,
        }
        with pytest.raises(ValueError, match="Missing required 'normalizer_hash'"):
            verify_data_protocol_against_checkpoint(
                ckpt_data=ckpt_data,
                split_file=str(split_file),
                normalizer=None,
            )

    def test_explicit_nonexistent_split_file_raises_filenotfound(self):
        from scripts.compute_latent_statistics import resolve_split_file

        with pytest.raises(FileNotFoundError, match="Explicitly specified split file not found"):
            resolve_split_file(split_type="grouped", split_file="/path/to/nonexistent/split.json")


class TestArchivedLatentResidualStatsIntegrity:
    """Verify the archived latent_residual_stats.json satisfies all mathematical and metadata contracts."""

    def test_archived_stats_satisfy_all_invariants(self):
        import json
        from pathlib import Path

        stats_path = Path("outputs/normalization/latent_residual_stats.json")
        if not stats_path.exists():
            pytest.skip("latent_residual_stats.json not yet generated on current machine")

        with open(stats_path, "r") as f:
            data = json.load(f)

        stats = data["statistics"]
        means = stats["channel_residual_mean"]
        vars_c = stats["channel_residual_centered_variance"]
        second_moments = stats["channel_residual_second_moment_g0"]
        summary = stats["summary"]

        assert len(means) == 64
        assert len(vars_c) == 64
        assert len(second_moments) == 64

        # 1. 64-channel identity: |m2,c - vc - mc^2| < 1e-14
        max_diff = max(abs(m2 - v - m ** 2) for m, m2, v in zip(means, second_moments, vars_c))
        assert max_diff < 1e-14

        # 2. Summary variance identity: mean_channel_centered_variance + mean_channel_squared_bias == mean_channel_second_moment
        assert abs(summary["mean_channel_centered_variance"] + summary["mean_channel_squared_bias"] - summary["mean_channel_second_moment"]) < 1e-14

        # 3. Pooled variance: pooled_residual_variance == mean_channel_second_moment - pooled_residual_mean^2
        assert abs(summary["pooled_residual_variance"] - (summary["mean_channel_second_moment"] - summary["pooled_residual_mean"] ** 2)) < 1e-14

        # 4. Numerical values match established benchmarks
        assert summary["mean_channel_centered_variance"] == pytest.approx(0.21243567587842088, rel=1e-6)
        assert summary["mean_channel_squared_bias"] == pytest.approx(0.29603029999300406, rel=1e-6)
        assert summary["mean_channel_second_moment"] == pytest.approx(0.508465975871425, rel=1e-6)
        assert summary["pooled_residual_mean"] == pytest.approx(-0.09137061342520868, rel=1e-6)
        assert summary["pooled_residual_variance"] == pytest.approx(0.500117386873726, rel=1e-6)

        # 5. Dataset coverage
        coverage = data["data_protocol"]["dataset_coverage"]
        assert coverage["num_trajectories"] == 33
        assert coverage["num_initial_condition_clusters"] == 27
        assert coverage["num_windows"] == 825
        assert coverage["stride"] == 8


class TestVerifyLatentAuditContract:
    """Verify runtime verification contract with real objects and fail-closed security on tampering."""

    def test_verify_contract_passes_on_canonical_state(self):
        from scripts.verify_latent_audit_contract import verify_latent_audit_contract

        stats_path = "outputs/normalization/latent_residual_stats.json"
        norm_path = "outputs/normalization/stats_grouped.pt"
        if not os.path.exists(stats_path) or not os.path.exists(norm_path):
            pytest.skip("Required audit files not available on machine.")

        record = verify_latent_audit_contract(record_output_path="")
        assert record["status"] == "AUDIT_VERIFIED_PASS"
        assert "stats_file" in record
        assert len(record["stats_file"]["sha256"]) == 64
        assert record["data_protocol"]["normalizer_hash"].startswith("3a0fe52689657618")
        assert record["d0_checkpoint"]["sha256"].startswith("edddbe8a2528f848")
        assert record["data_protocol"]["dataset_coverage_provenance"] == "sourced_from_frozen_stats_metadata"

    def test_mutated_normalizer_fails_closed(self, tmp_path):
        from scripts.verify_latent_audit_contract import verify_latent_audit_contract

        norm_path = "outputs/normalization/stats_grouped.pt"
        if not os.path.exists(norm_path):
            pytest.skip("Required stats_grouped.pt not available on machine.")

        orig_norm = torch.load(norm_path, weights_only=True, map_location="cpu")
        tampered_norm = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in orig_norm.items()}
        tampered_norm["mean"] = tampered_norm["mean"] + 0.1
        tampered_path = tmp_path / "tampered_stats.pt"
        torch.save(tampered_norm, tampered_path)

        with pytest.raises(ValueError, match="Normalizer hash mismatch"):
            verify_latent_audit_contract(normalizer_path=str(tampered_path), record_output_path="")

    def test_tampered_stats_file_fails_closed(self, tmp_path):
        from scripts.verify_latent_audit_contract import verify_latent_audit_contract

        stats_path = "outputs/normalization/latent_residual_stats.json"
        if not os.path.exists(stats_path):
            pytest.skip("Required latent_residual_stats.json not available on machine.")

        with open(stats_path, "r") as f:
            stats_data = json.load(f)

        # Alter mean to violate channel identity m2 - v - m^2 = 0
        stats_data["statistics"]["channel_residual_mean"][0] += 1.0
        tampered_stats = tmp_path / "tampered_stats.json"
        with open(tampered_stats, "w") as f:
            json.dump(stats_data, f)

        with pytest.raises(AssertionError, match="Channel statistical identity violated"):
            verify_latent_audit_contract(stats_path=str(tampered_stats), record_output_path="")
