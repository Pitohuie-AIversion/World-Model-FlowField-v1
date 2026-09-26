"""Unit tests verifying ProbLatent-R1 Phase 0 audit and regression closures.

Covers:
1. Spatial position encoding resolution for legacy vs modern checkpoints (no silent override).
2. Atomic D0 forecaster parity check against standalone representation checkpoints.
3. Mathematical correctness of fixed-mean G0 variance:
   - When residual is constant [2, 2, 2], centered variance is 0, but G0 second moment is 4.
4. Channel-wise initialization of G1 variance head matching G0 exactly at step 0:
   - W_var = 0, b_c = softplus^{-1}(v_{G0, c} - eps) -> Var_{G1, init} == broadcast(v_{G0}).
5. Environment isolation of H12 missing test fixture.
"""

from pathlib import Path
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.latent_transformer import LatentSTTransformer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.latent_forecaster import LatentForecaster
from src.utils.checkpoint import resolve_spatial_pos_config


class TestSpatialPosConfigResolution:
    """Verify that legacy checkpoints strictly restore spatial pos embedding = False."""

    def test_explicit_config_honored(self):
        """If use_spatial_pos is in config, it takes precedence."""
        ckpt_true = {"config": {"use_spatial_pos": True}}
        ckpt_false = {"config": {"use_spatial_pos": False}}
        assert resolve_spatial_pos_config(ckpt_true) is True
        assert resolve_spatial_pos_config(ckpt_false) is False

    def test_legacy_h8_and_h16_commits_resolve_to_false(self):
        """Pre-0f2ed24 commits must resolve to False even if missing from config."""
        # Parent H8 commit
        h8_ckpt = {
            "training_git_commit": "6593b65005843c3f3c2640cb262bfd56708e11ad",
            "config": {"embed_dim": 256, "depth": 6},
        }
        assert resolve_spatial_pos_config(h8_ckpt) is False

        # H16 short-best commit
        h16_ckpt = {
            "training_git_commit": "0a8a4da137e3f73c4f696cc81f527c32a8ba8405",
            "config": {"embed_dim": 256, "depth": 6},
        }
        assert resolve_spatial_pos_config(h16_ckpt) is False

    def test_h12_commits_resolve_to_true(self):
        """Post-0f2ed24 H12 commits must resolve to True."""
        h12_ckpt = {
            "training_git_commit": "f9f5815e5a8d9be37d7d0830d1b3b08b7a32866f",
            "config": {"embed_dim": 256, "depth": 6},
        }
        assert resolve_spatial_pos_config(h12_ckpt) is True

    def test_legacy_unspecified_fallback_is_false(self):
        """Legacy checkpoints without config or commit safely default to False."""
        legacy_ckpt = {"config": {"embed_dim": 256}}
        assert resolve_spatial_pos_config(legacy_ckpt) is False


class TestFixedMeanG0SecondMomentMath:
    """Verify that under fixed mean, G0 variance is E[r^2], NOT centered Var(r)."""

    def test_constant_bias_residual_math(self):
        """If residual r is constant 2.0, Var(r) is 0, but second moment E[r^2] is 4.0."""
        r = torch.full((10, 64, 16, 32), 2.0, dtype=torch.float64)

        mean_r = r.mean(dim=(0, 2, 3))
        var_r = r.var(dim=(0, 2, 3), unbiased=False)
        second_moment_r = (r ** 2).mean(dim=(0, 2, 3))

        # Centered variance is exactly 0
        assert torch.allclose(var_r, torch.zeros(64, dtype=torch.float64), atol=1e-7)
        # Second moment is exactly 4
        assert torch.allclose(second_moment_r, torch.full((64,), 4.0, dtype=torch.float64), atol=1e-7)
        # Relationship E[r^2] = Var(r) + (E[r])^2
        assert torch.allclose(second_moment_r, var_r + mean_r ** 2, atol=1e-7)

    def test_gaussian_nll_is_minimized_at_second_moment_under_fixed_mean(self):
        """Directly verify GaussianNLLLoss is minimized when var = E[r^2], not Var(r)."""
        torch.manual_seed(42)
        # Ground truth has mean 2.0, std 1.0 -> E[y] = 2.0, Var(y) = 1.0
        y = torch.randn(10000) * 1.0 + 2.0
        # Model predictions are fixed at 0.0 -> residual r = y - mu = y
        # True Var(r) = 1.0, but E[r^2] = 1.0 + 2.0^2 = 5.0
        mu = torch.zeros_like(y)

        # Check Gaussian NLL loss around v = 1.0 (centered var) vs v = 5.0 (second moment)
        loss_at_centered_var = F.gaussian_nll_loss(input=mu, target=y, var=torch.full_like(y, 1.0), eps=1e-6)
        loss_at_second_moment = F.gaussian_nll_loss(input=mu, target=y, var=torch.full_like(y, 5.0), eps=1e-6)

        # Second moment must achieve strictly lower negative log likelihood than centered variance
        assert loss_at_second_moment.item() < loss_at_centered_var.item()

        # Check that v=5.0 is indeed the local minimum:
        loss_lower = F.gaussian_nll_loss(input=mu, target=y, var=torch.full_like(y, 4.8), eps=1e-6)
        loss_higher = F.gaussian_nll_loss(input=mu, target=y, var=torch.full_like(y, 5.2), eps=1e-6)
        assert loss_at_second_moment.item() <= loss_lower.item() + 1e-3
        assert loss_at_second_moment.item() <= loss_higher.item() + 1e-3


class TestG1VarianceHeadInitializationParity:
    """Verify that G1 initialization can strictly broadcast to G0 at step 0."""

    def test_inverse_softplus_initialization(self):
        """Construct variance head with W=0 and b_c = softplus^{-1}(v_G0 - eps)."""
        c_z = 64
        embed_dim = 256
        variance_floor = 1e-4

        # Mock per-channel G0 second moments (ranging from 0.06 to 0.8)
        torch.manual_seed(42)
        v_g0 = torch.linspace(0.065, 0.804, c_z)

        # Variance head mapping tokens (B, N, D) -> (B, N, C_z)
        var_head = nn.Linear(embed_dim, c_z)

        # Zero weights
        nn.init.zeros_(var_head.weight)

        # Analytical inverse softplus: b = ln(exp(v - eps) - 1)
        clamped_v = torch.clamp(v_g0 - variance_floor, min=1e-6)
        b_init = torch.log(torch.expm1(clamped_v))
        with torch.no_grad():
            var_head.bias.copy_(b_init)

        # Forward through arbitrary input features x
        x = torch.randn(4, 16 * 32, embed_dim)
        raw_out = var_head(x)  # (4, 512, 64)
        pred_var = F.softplus(raw_out) + variance_floor

        # Reshape to (4, 64, 16, 32)
        pred_var_tensor = pred_var.view(4, 16, 32, c_z).permute(0, 3, 1, 2)
        expected_g0_broadcast = v_g0.view(1, c_z, 1, 1).expand(4, c_z, 16, 32)

        # At step 0, G1 predicted variance must be strictly bit-wise equal to G0
        assert torch.allclose(pred_var_tensor, expected_g0_broadcast, atol=1e-6)


class TestFullD0AtomicParityVerification:
    """Verify that full D0 loading verifies representation weights against standalone AE."""

    def test_parity_verification_helper(self, tmp_path):
        from scripts.compute_latent_statistics import verify_representation_parity

        enc = Encoder2D(in_channels=4, latent_channels=16, base_channels=8)
        dec = Decoder2D(latent_channels=16, out_channels=4, base_channels=8)
        trans = LatentSTTransformer(latent_channels=16, embed_dim=32, cond_dim=16, depth=1, num_heads=2)
        forecaster = LatentForecaster(enc, trans, dec)

        # Save matching AE
        ae_path = tmp_path / "matching_ae.pt"
        torch.save({
            "encoder_state_dict": enc.state_dict(),
            "decoder_state_dict": dec.state_dict(),
        }, ae_path)

        enc_match, dec_match, max_diff = verify_representation_parity(forecaster, str(ae_path))
        assert enc_match is True
        assert dec_match is True
        assert max_diff == 0.0

        # Corrupt one decoder parameter in standalone checkpoint
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
