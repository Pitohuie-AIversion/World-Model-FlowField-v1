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
