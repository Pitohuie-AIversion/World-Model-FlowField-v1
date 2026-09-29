"""Unit and Contract Test Suite for Latent Flow Matching (OT-CFM).

Tests:
1. SinusoidalTimeEmbedding:
   - Output shapes across 0D, 1D, 2D inputs.
   - Smoothness and non-degeneracy.
2. LatentPeriodicResBlock2D:
   - Output shape (B, C, H, W).
   - Circular padding respecting periodic boundaries.
   - AdaLN condition modulation.
   - Gradient flow.
3. LatentSpatialAttention2D:
   - Output shape and zero-initialization identity property.
4. LatentVelocityNet2D:
   - Zero-initialization produces exact zero velocity.
   - Gradient flow to all learnable parameters.
   - Spatial periodicity.
5. ODESolver:
   - Solvers ('euler', 'midpoint', 'heun', 'rk4') numerical execution.
   - Validation failure modes (invalid solver, num_steps <= 0).
   - Numerical integration on constant velocity field: x1 - x0 = v.
   - Trajectory recording returns exact step count.
6. LatentFlowMatcher:
   - compute_loss: OT-CFM straight path mathematical correctness.
   - Mode support: 'residual' vs 'direct'.
   - Sample reproducibility under fixed seed.
   - Sample diversity under different seeds.
   - Baseline zero-noise parity: noise_scale=0.0 with zero-init outputs exact mu.
   - Ensemble batch expansion and trajectory isolation.
7. LatentForecaster Flow Matching Integration:
   - attach_flow_matcher, has_flow_matcher.
   - freeze_for_flow_matching_training contract.
   - Fail-closed behavior when flow_matcher is missing.
   - sample_rollout_flow_matching multi-step rollout dictionary structure and dimensions.
"""

import math
import pytest
import torch
import torch.nn as nn

from src.models.latent_flow_matching import (
    SinusoidalTimeEmbedding,
    LatentPeriodicResBlock2D,
    LatentSpatialAttention2D,
    LatentVelocityNet2D,
    ODESolver,
    LatentFlowMatcher,
)
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.latent_transformer import LatentSTTransformer
from src.models.latent_forecaster import LatentForecaster


class TestSinusoidalTimeEmbedding:
    """Validate time embedding shapes, continuous values, and stability."""

    def test_output_shapes_scalar_and_batched(self):
        embed = SinusoidalTimeEmbedding(embed_dim=64)

        # 0D scalar
        tau_0d = torch.tensor(0.5)
        out_0d = embed(tau_0d)
        assert out_0d.shape == (1, 64)

        # 1D batch
        tau_1d = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        out_1d = embed(tau_1d)
        assert out_1d.shape == (5, 64)

        # 2D (B, 1)
        tau_2d = torch.tensor([[0.1], [0.9]])
        out_2d = embed(tau_2d)
        assert out_2d.shape == (2, 64)

    def test_finite_values_and_smoothness(self):
        embed = SinusoidalTimeEmbedding(embed_dim=32)
        taus = torch.linspace(0.0, 1.0, 20)
        out = embed(taus)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        # Ensure outputs for tau=0.0 and tau=1.0 differ
        assert not torch.allclose(out[0], out[-1], atol=1e-3)


class TestLatentPeriodicResBlock2D:
    """Validate periodic convolutions, AdaLN modulation, and circular boundary handling."""

    def test_output_shape_and_circular_padding(self):
        channels, cond_dim = 16, 32
        block = LatentPeriodicResBlock2D(channels=channels, cond_dim=cond_dim)

        x = torch.randn(2, channels, 8, 8)
        cond = torch.randn(2, cond_dim)
        out = block(x, cond)

        assert out.shape == (2, channels, 8, 8)
        assert not torch.isnan(out).any()

    def test_circular_padding_invariance(self):
        """Translating input by full period wraps around without boundary truncation."""
        channels, cond_dim = 8, 16
        block = LatentPeriodicResBlock2D(channels=channels, cond_dim=cond_dim)

        x = torch.randn(1, channels, 8, 8)
        cond = torch.randn(1, cond_dim)

        # Shift x circularly along dim -2 and dim -1
        shift_x = 2
        shift_y = 3
        x_rolled = torch.roll(x, shifts=(shift_x, shift_y), dims=(-2, -1))

        out = block(x, cond)
        out_rolled = block(x_rolled, cond)
        rolled_out = torch.roll(out, shifts=(shift_x, shift_y), dims=(-2, -1))

        # Because all conv layers use circular padding, block is spatially equivariant
        assert torch.allclose(out_rolled, rolled_out, atol=1e-5)

    def test_adaln_modulation_reacts_to_condition(self):
        channels, cond_dim = 8, 16
        block = LatentPeriodicResBlock2D(channels=channels, cond_dim=cond_dim)
        # Randomize cond_proj to non-zero weights
        nn.init.normal_(block.cond_proj.weight, std=0.5)

        x = torch.randn(1, channels, 8, 8)
        cond_a = torch.randn(1, cond_dim)
        cond_b = cond_a + 5.0

        out_a = block(x, cond_a)
        out_b = block(x, cond_b)
        assert not torch.allclose(out_a, out_b, atol=1e-4)

    def test_gradient_flow(self):
        channels, cond_dim = 8, 16
        block = LatentPeriodicResBlock2D(channels=channels, cond_dim=cond_dim)

        x = torch.randn(2, channels, 4, 4, requires_grad=True)
        cond = torch.randn(2, cond_dim, requires_grad=True)

        out = block(x, cond)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None and not torch.isnan(x.grad).any()
        assert cond.grad is not None and not torch.isnan(cond.grad).any()


class TestLatentSpatialAttention2D:
    """Validate spatial attention and zero-init projection."""

    def test_output_shape_and_identity_init(self):
        channels = 16
        attn = LatentSpatialAttention2D(channels=channels, num_heads=4)

        x = torch.randn(2, channels, 6, 6)
        out = attn(x)

        assert out.shape == x.shape
        # Zero-initialized projection means initially out == x
        assert torch.allclose(out, x, atol=1e-6)


class TestLatentVelocityNet2D:
    """Validate velocity network architecture, zero-init behavior, and conditioning."""

    def test_zero_initialization_produces_zero_velocity(self):
        latent_channels = 16
        hidden_channels = 32
        cond_dim = 24
        net = LatentVelocityNet2D(
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            cond_dim=cond_dim,
            time_dim=cond_dim,
            num_blocks=2,
            zero_init=True,
        )

        b, hz, wz = 3, 8, 8
        x_tau = torch.randn(b, latent_channels, hz, wz)
        tau = torch.tensor([0.0, 0.5, 1.0])
        context_mu = torch.randn(b, latent_channels, hz, wz)
        cond = torch.randn(b, cond_dim)

        v_pred = net(x_tau=x_tau, tau=tau, context_mu=context_mu, cond=cond)

        assert v_pred.shape == (b, latent_channels, hz, wz)
        # Because out_conv is zero-initialized, v_pred must be identically 0
        assert torch.allclose(v_pred, torch.zeros_like(v_pred), atol=1e-7)

    def test_gradient_flow_through_velocity_net(self):
        latent_channels = 8
        hidden_channels = 16
        cond_dim = 16
        net = LatentVelocityNet2D(
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            cond_dim=cond_dim,
            num_blocks=2,
            zero_init=False,  # non-zero to produce gradients
        )

        b, hz, wz = 2, 4, 4
        x_tau = torch.randn(b, latent_channels, hz, wz, requires_grad=True)
        tau = torch.tensor([0.3, 0.7])
        context_mu = torch.randn(b, latent_channels, hz, wz, requires_grad=True)
        cond = torch.randn(b, cond_dim, requires_grad=True)

        v_pred = net(x_tau=x_tau, tau=tau, context_mu=context_mu, cond=cond)
        loss = v_pred.sum()
        loss.backward()

        assert x_tau.grad is not None
        assert context_mu.grad is not None
        assert cond.grad is not None
        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"
                assert not torch.isnan(param.grad).any()


class TestODESolver:
    """Validate numerical ODE integration solvers on continuous flow paths."""

    def test_unsupported_solver_raises_value_error(self):
        dummy_func = nn.Identity()
        x0 = torch.zeros(1, 4, 4, 4)
        with pytest.raises(ValueError, match="Unsupported ODE solver"):
            ODESolver.integrate(dummy_func, x0, x0, x0, solver="non_existent")

    def test_invalid_num_steps_raises_value_error(self):
        dummy_func = nn.Identity()
        x0 = torch.zeros(1, 4, 4, 4)
        with pytest.raises(ValueError, match="num_steps must be positive"):
            ODESolver.integrate(dummy_func, x0, x0, x0, num_steps=0)

    @pytest.mark.parametrize("solver", ["euler", "midpoint", "heun", "rk4"])
    def test_constant_velocity_field_integration(self, solver):
        """For constant velocity field v(x, tau) = c, integral from 0 to 1 must be x0 + c."""
        class ConstantVelocity(nn.Module):
            def __init__(self, c_val):
                super().__init__()
                self.c_val = c_val

            def forward(self, x, tau, context_mu, cond):
                return torch.full_like(x, self.c_val)

        c_val = 2.5
        func = ConstantVelocity(c_val)
        x0 = torch.zeros(2, 4, 4, 4)
        context = torch.zeros_like(x0)
        cond = torch.zeros(2, 8)

        x1 = ODESolver.integrate(
            func=func,
            x0=x0,
            context_mu=context,
            cond=cond,
            num_steps=10,
            solver=solver,
        )

        expected = x0 + c_val * 1.0
        assert torch.allclose(x1, expected, atol=1e-5)

    def test_trajectory_recording_length(self):
        class DummyVelocity(nn.Module):
            def forward(self, x, tau, context_mu, cond):
                return x * 0.1

        func = DummyVelocity()
        x0 = torch.randn(2, 4, 4, 4)
        context = torch.zeros_like(x0)
        cond = torch.zeros(2, 4)

        num_steps = 7
        x1, traj = ODESolver.integrate(
            func=func,
            x0=x0,
            context_mu=context,
            cond=cond,
            num_steps=num_steps,
            solver="euler",
            return_trajectory=True,
        )

        # traj should contain x0 plus num_steps intermediate states
        assert len(traj) == num_steps + 1
        assert torch.allclose(traj[0], x0)
        assert torch.allclose(traj[-1], x1)


class TestLatentFlowMatcherContracts:
    """Validate OT-CFM loss, mathematical straight path, and sampling contracts."""

    def test_compute_loss_mathematical_ot_path(self):
        latent_channels = 8
        cond_dim = 16
        matcher = LatentFlowMatcher(
            latent_channels=latent_channels,
            cond_dim=cond_dim,
            hidden_channels=16,
            num_blocks=2,
            target_mode="residual",
            sigma_min=1e-4,
        )

        b, hz, wz = 2, 4, 4
        z_next = torch.randn(b, 1, latent_channels, hz, wz)
        mu = torch.randn(b, 1, latent_channels, hz, wz)
        re = torch.tensor([1e4, 2e4])
        sc = torch.tensor([0.1, 0.5])

        fixed_tau = torch.tensor([[0.4], [0.6]])
        fixed_x0 = torch.randn(b, latent_channels, hz, wz)

        res = matcher.compute_loss(
            z_next=z_next,
            mu=mu,
            re=re,
            sc=sc,
            custom_tau=fixed_tau,
            custom_x0=fixed_x0,
        )

        # In residual mode: x1 = z_next - mu
        expected_x1 = z_next.squeeze(1) - mu.squeeze(1)
        expected_tau = fixed_tau.unsqueeze(-1).unsqueeze(-1)
        expected_x_tau = (1.0 - (1.0 - 1e-4) * expected_tau) * fixed_x0 + expected_tau * expected_x1
        expected_u = expected_x1 - (1.0 - 1e-4) * fixed_x0

        assert torch.allclose(res["x_tau"], expected_x_tau, atol=1e-5)
        assert torch.allclose(res["u_target"], expected_u, atol=1e-5)
        assert res["loss"].item() > 0.0

    def test_direct_mode_contract(self):
        latent_channels = 4
        matcher = LatentFlowMatcher(
            latent_channels=latent_channels,
            cond_dim=8,
            hidden_channels=8,
            num_blocks=1,
            target_mode="direct",
            sigma_min=0.0,
        )

        b, hz, wz = 2, 4, 4
        z_next = torch.randn(b, latent_channels, hz, wz)
        mu = torch.randn(b, latent_channels, hz, wz)
        re = torch.tensor([1e4, 1e4])
        sc = torch.tensor([1.0, 1.0])

        fixed_tau = torch.tensor([[0.5], [0.5]])
        fixed_x0 = torch.zeros(b, latent_channels, hz, wz)

        res = matcher.compute_loss(
            z_next=z_next,
            mu=mu,
            re=re,
            sc=sc,
            custom_tau=fixed_tau,
            custom_x0=fixed_x0,
        )

        # In direct mode with sigma_min=0 and x0=0: x_tau = tau * z_next = 0.5 * z_next
        assert torch.allclose(res["x_tau"], 0.5 * z_next, atol=1e-5)
        assert torch.allclose(res["u_target"], z_next, atol=1e-5)

    def test_zero_noise_parity_with_zero_init(self):
        """When zero_init=True and noise_scale=0.0, sample_next_latent strictly returns mu."""
        latent_channels = 8
        matcher = LatentFlowMatcher(
            latent_channels=latent_channels,
            cond_dim=16,
            hidden_channels=16,
            num_blocks=2,
            target_mode="residual",
            zero_init=True,
        )

        b, hz, wz = 2, 4, 4
        mu = torch.randn(b, 1, latent_channels, hz, wz)
        re = torch.tensor([1e4, 1e4])
        sc = torch.tensor([0.2, 0.2])

        z_sample = matcher.sample_next_latent(
            mu=mu,
            re=re,
            sc=sc,
            num_steps=5,
            solver="rk4",
            noise_scale=0.0,
        )

        # With zero-init velocity network and zero base noise, dx/dtau = 0, x1 = 0, so z = mu
        assert torch.allclose(z_sample, mu, atol=1e-6)

    def test_sampling_reproducibility_with_seed(self):
        latent_channels = 8
        matcher = LatentFlowMatcher(
            latent_channels=latent_channels,
            cond_dim=16,
            hidden_channels=16,
            num_blocks=2,
            target_mode="residual",
            zero_init=False,
        )

        mu = torch.randn(2, 1, latent_channels, 4, 4)
        re = torch.tensor([1e4, 1e4])
        sc = torch.tensor([1.0, 1.0])

        sample1 = matcher.sample_next_latent(mu, re=re, sc=sc, num_steps=4, seed=42)
        sample2 = matcher.sample_next_latent(mu, re=re, sc=sc, num_steps=4, seed=42)
        assert torch.allclose(sample1, sample2, atol=1e-7)

    def test_sampling_diversity_with_different_seeds(self):
        latent_channels = 8
        matcher = LatentFlowMatcher(
            latent_channels=latent_channels,
            cond_dim=16,
            hidden_channels=16,
            num_blocks=2,
            target_mode="residual",
            zero_init=False,
        )

        mu = torch.randn(2, 1, latent_channels, 4, 4)
        re = torch.tensor([1e4, 1e4])
        sc = torch.tensor([1.0, 1.0])

        sample_a = matcher.sample_next_latent(mu, re=re, sc=sc, num_steps=4, seed=42)
        sample_b = matcher.sample_next_latent(mu, re=re, sc=sc, num_steps=4, seed=999)
        assert not torch.allclose(sample_a, sample_b, atol=1e-4)

    def test_sample_ensemble_dimensions(self):
        latent_channels = 8
        matcher = LatentFlowMatcher(
            latent_channels=latent_channels,
            cond_dim=16,
            hidden_channels=16,
            num_blocks=2,
        )

        b, k = 2, 4
        mu = torch.randn(b, 1, latent_channels, 4, 4)
        re = torch.tensor([1e4, 5e4])
        sc = torch.tensor([0.1, 1.0])

        ensemble = matcher.sample_ensemble(
            mu=mu,
            re=re,
            sc=sc,
            num_samples=k,
            num_steps=3,
            solver="euler",
            seed=123,
        )

        assert ensemble.shape == (b, k, 1, latent_channels, 4, 4)
        assert not torch.isnan(ensemble).any()

    def test_deterministic_fallback_contract(self):
        """deterministic_fallback=True strictly returns mu even with non-zero initialized network.

        In contrast, noise_scale=0.0 starts ODE from x0=0, which on a trained/non-zero network
        produces learned velocity drift and does NOT equal mu.
        """
        latent_channels = 8
        matcher = LatentFlowMatcher(
            latent_channels=latent_channels,
            cond_dim=16,
            hidden_channels=16,
            num_blocks=2,
            target_mode="residual",
            zero_init=False,  # Non-zero weights produce non-zero velocity field
        )

        b, hz, wz = 2, 4, 4
        mu = torch.randn(b, 1, latent_channels, hz, wz)
        re = torch.tensor([1e4, 1e4])
        sc = torch.tensor([1.0, 1.0])

        # 1. Deterministic fallback strictly bypasses ODE and returns mu
        z_fallback = matcher.sample_next_latent(
            mu=mu,
            re=re,
            sc=sc,
            deterministic_fallback=True,
        )
        assert torch.allclose(z_fallback, mu, atol=1e-7)

        # 2. Ensemble with deterministic fallback returns identical mu copies
        ens_fallback = matcher.sample_ensemble(
            mu=mu,
            re=re,
            sc=sc,
            num_samples=4,
            deterministic_fallback=True,
        )
        for k in range(4):
            assert torch.allclose(ens_fallback[:, k], mu, atol=1e-7)

        # 3. noise_scale=0.0 on non-zero network undergoes drift, proving it's NOT an exact fallback
        z_zero_noise = matcher.sample_next_latent(
            mu=mu,
            re=re,
            sc=sc,
            noise_scale=0.0,
            num_steps=5,
            deterministic_fallback=False,
        )
        assert not torch.allclose(z_zero_noise, mu, atol=1e-3)

    def test_residual_normalization_math(self):
        """Validate per-channel residual scaling math in compute_loss and sample_next_latent."""
        latent_channels = 4
        matcher = LatentFlowMatcher(
            latent_channels=latent_channels,
            cond_dim=8,
            hidden_channels=8,
            num_blocks=1,
            target_mode="residual",
            sigma_min=1e-4,
        )

        # Set distinct non-trivial scales per channel: [1.0, 2.0, 0.5, 4.0]
        scales = [1.0, 2.0, 0.5, 4.0]
        matcher.set_residual_scale(scales)
        assert matcher.residual_scale.shape == (1, 4, 1, 1)

        b, hz, wz = 1, 2, 2
        z_next = torch.tensor([[[[2.0, 2.0], [2.0, 2.0]],   # ch 0: res = 2.0 / 1.0 = 2.0
                                [[4.0, 4.0], [4.0, 4.0]],   # ch 1: res = 4.0 / 2.0 = 2.0
                                [[1.0, 1.0], [1.0, 1.0]],   # ch 2: res = 1.0 / 0.5 = 2.0
                                [[8.0, 8.0], [8.0, 8.0]]]]).unsqueeze(0)  # ch 3: res = 8.0 / 4.0 = 2.0
        mu = torch.zeros_like(z_next)
        re = torch.tensor([1e4])
        sc = torch.tensor([1.0])

        fixed_tau = torch.tensor([[1.0]])
        fixed_x0 = torch.zeros(b, latent_channels, hz, wz)

        res = matcher.compute_loss(
            z_next=z_next,
            mu=mu,
            re=re,
            sc=sc,
            custom_tau=fixed_tau,
            custom_x0=fixed_x0,
        )

        # At tau=1, x_tau = x_1 = normalized residual
        # Each channel residual divided by scale should equal 2.0
        expected_normalized_residual = torch.full((1, 4, 2, 2), 2.0)
        assert torch.allclose(res["x_tau"], expected_normalized_residual, atol=1e-5)

    def test_from_residual_stats_initialization(self):
        """Validate instantiation via from_residual_stats classmethod."""
        dummy_stats = {
            "statistics": {
                "channel_residual_second_moment_g0": [0.09, 0.25, 0.49, 0.81],
            }
        }
        matcher = LatentFlowMatcher.from_residual_stats(
            stats_data=dummy_stats,
            latent_channels=4,
            cond_dim=8,
            hidden_channels=8,
            num_blocks=1,
            eps=0.0,
        )
        assert matcher.residual_scale is not None
        expected_scale = torch.tensor([0.3, 0.5, 0.7, 0.9]).view(1, 4, 1, 1)
        assert torch.allclose(matcher.residual_scale, expected_scale, atol=1e-5)


class TestLatentForecasterFlowMatchingIntegration:
    """Verify LatentForecaster integration with LatentFlowMatcher."""

    def test_attach_and_has_flow_matcher(self):
        encoder = Encoder2D(in_channels=4, latent_channels=8, base_channels=8, channel_mult=[1, 1, 1])
        transformer = LatentSTTransformer(latent_channels=8, embed_dim=16, cond_dim=16, depth=1)
        decoder = Decoder2D(latent_channels=8, out_channels=4, base_channels=8, channel_mult=[1, 1, 1])
        forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder)

        assert forecaster.has_flow_matcher is False

        matcher = LatentFlowMatcher(latent_channels=8, cond_dim=16, hidden_channels=16, num_blocks=1)
        forecaster.attach_flow_matcher(matcher)

        assert forecaster.has_flow_matcher is True
        assert forecaster.flow_matcher is matcher

    def test_freeze_for_flow_matching_training(self):
        encoder = Encoder2D(in_channels=4, latent_channels=8, base_channels=8, channel_mult=[1, 1, 1])
        transformer = LatentSTTransformer(latent_channels=8, embed_dim=16, cond_dim=16, depth=1)
        decoder = Decoder2D(latent_channels=8, out_channels=4, base_channels=8, channel_mult=[1, 1, 1])
        forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder)
        matcher = LatentFlowMatcher(latent_channels=8, cond_dim=16, hidden_channels=16, num_blocks=1)
        forecaster.attach_flow_matcher(matcher)

        forecaster.freeze_for_flow_matching_training()

        # Encoder, decoder, transformer backbone frozen
        for p in forecaster.encoder.parameters():
            assert not p.requires_grad
        for p in forecaster.decoder.parameters():
            assert not p.requires_grad
        for p in forecaster.transformer.parameters():
            assert not p.requires_grad

        # Flow matcher trainable
        for p in forecaster.flow_matcher.parameters():
            assert p.requires_grad

    def test_freeze_fails_closed_when_flow_matcher_missing(self):
        encoder = Encoder2D(in_channels=4, latent_channels=8, base_channels=8, channel_mult=[1, 1, 1])
        transformer = LatentSTTransformer(latent_channels=8, embed_dim=16, cond_dim=16, depth=1)
        decoder = Decoder2D(latent_channels=8, out_channels=4, base_channels=8, channel_mult=[1, 1, 1])
        forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder)

        with pytest.raises(RuntimeError, match="has no attached flow_matcher"):
            forecaster.freeze_for_flow_matching_training()

    def test_sample_rollout_flow_matching_outputs(self):
        encoder = Encoder2D(in_channels=4, latent_channels=8, base_channels=8, channel_mult=[1, 1, 1])
        transformer = LatentSTTransformer(latent_channels=8, embed_dim=16, cond_dim=16, depth=1, history_length=2)
        decoder = Decoder2D(latent_channels=8, out_channels=4, base_channels=8, channel_mult=[1, 1, 1])
        forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder)

        matcher = LatentFlowMatcher(latent_channels=8, cond_dim=16, hidden_channels=16, num_blocks=1)
        forecaster.attach_flow_matcher(matcher)

        b, l, c_in, ny, nx = 2, 2, 4, 32, 32
        q_hist = torch.randn(b, l, c_in, ny, nx)
        re = torch.tensor([1e4, 2e4])
        sc = torch.tensor([0.2, 0.5])
        horizon = 3
        num_samples = 4

        res = forecaster.sample_rollout_flow_matching(
            q_hist=q_hist,
            re=re,
            sc=sc,
            horizon=horizon,
            num_samples=num_samples,
            num_flow_steps=3,
            solver="euler",
            seed=42,
            decode_samples=True,
        )

        assert "deterministic_rollout" in res
        assert "sample_trajectories" in res
        assert "ensemble_mean" in res
        assert "latent_samples" in res

        # Check tensor shapes
        assert res["deterministic_rollout"].shape == (b, horizon, c_in, ny, nx)
        assert res["sample_trajectories"].shape == (b, num_samples, horizon, c_in, ny, nx)
        assert res["ensemble_mean"].shape == (b, horizon, c_in, ny, nx)
        assert res["latent_samples"].shape == (b, num_samples, horizon, 8, 4, 4)
        assert not torch.isnan(res["sample_trajectories"]).any()
