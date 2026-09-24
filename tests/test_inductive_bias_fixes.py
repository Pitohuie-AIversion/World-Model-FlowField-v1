"""Tests for inductive bias fixes: 2D Sinusoidal Positional Embeddings and Leray-Helmholtz Spectral Projection."""

import pytest
import torch
from src.models.positional_embedding import (
    build_2d_sincos_position_embedding,
    get_2d_sincos_position_embedding,
    clear_pos_embed_cache,
)
from src.models.latent_transformer import LatentSTTransformer
from src.models.direct_transformer import DirectSTTransformer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.latent_forecaster import LatentForecaster
from src.utils.fft_derivatives import (
    compute_divergence,
    compute_vorticity,
    project_divergence_free_2d,
    project_incompressible_state,
)


class TestSpatialPositionalEmbedding:
    """Verification suite for 2D Sinusoidal Positional Embedding."""

    def test_pos_embed_shape_and_cache(self):
        clear_pos_embed_cache()
        embed_dim = 128
        h, w = 16, 8
        pe1 = get_2d_sincos_position_embedding(embed_dim, h, w)
        assert pe1.shape == (1, 1, h * w, embed_dim)

        # Cache hit returns same object
        pe2 = get_2d_sincos_position_embedding(embed_dim, h, w)
        assert pe1 is pe2

        clear_pos_embed_cache()
        pe3 = get_2d_sincos_position_embedding(embed_dim, h, w)
        assert pe1 is not pe3
        assert torch.allclose(pe1, pe3)

    def test_spatial_distinction(self):
        """Verify different grid coordinates produce non-identical embedding vectors."""
        embed_dim = 64
        h, w = 8, 8
        pe = build_2d_sincos_position_embedding(embed_dim, h, w).squeeze(0).squeeze(0)  # (64, 64)
        
        # Point (0, 0) vs (0, 1) - horizontal neighbor
        assert not torch.allclose(pe[0], pe[1], atol=1e-3)
        # Point (0, 0) vs (1, 0) - vertical neighbor
        assert not torch.allclose(pe[0], pe[w], atol=1e-3)
        # Point (0, 0) vs (h-1, w-1) - opposite diagonal
        assert not torch.allclose(pe[0], pe[-1], atol=1e-3)

    def test_latent_transformer_state_dict_compatibility(self):
        """Verify zero parameters are added, maintaining 100% state_dict backward compatibility."""
        model_with_pos = LatentSTTransformer(
            latent_channels=16,
            embed_dim=64,
            cond_dim=32,
            depth=2,
            num_heads=4,
            history_length=3,
            use_spatial_pos=True,
        )
        model_no_pos = LatentSTTransformer(
            latent_channels=16,
            embed_dim=64,
            cond_dim=32,
            depth=2,
            num_heads=4,
            history_length=3,
            use_spatial_pos=False,
        )

        keys_with = set(model_with_pos.state_dict().keys())
        keys_no = set(model_no_pos.state_dict().keys())
        assert keys_with == keys_no

        # Cross load state_dict without error
        missing, unexpected = model_with_pos.load_state_dict(model_no_pos.state_dict(), strict=True)
        assert len(missing) == 0 and len(unexpected) == 0

    def test_direct_transformer_forward_with_spatial_pos(self):
        """Verify DirectSTTransformer forward pass with and without spatial pos embed."""
        model = DirectSTTransformer(
            in_channels=4,
            patch_size=(8, 8),
            embed_dim=64,
            cond_dim=32,
            depth=2,
            num_heads=4,
            history_length=2,
            use_spatial_pos=True,
        )
        q = torch.randn(2, 2, 4, 32, 16)
        re = torch.tensor([1000.0, 1200.0])
        sc = torch.tensor([1.0, 1.0])

        out = model(q, re=re, sc=sc)
        assert out.shape == (2, 1, 4, 32, 16)

    def test_latent_forecaster_end_to_end_with_pos(self):
        """Verify full LatentForecaster functions properly with spatial positional embedding."""
        encoder = Encoder2D(in_channels=4, latent_channels=8, base_channels=16)
        decoder = Decoder2D(latent_channels=8, out_channels=4, base_channels=16)
        transformer = LatentSTTransformer(
            latent_channels=8,
            embed_dim=32,
            cond_dim=16,
            depth=2,
            num_heads=2,
            history_length=2,
            use_spatial_pos=True,
        )
        forecaster = LatentForecaster(encoder, transformer, decoder)

        q_hist = torch.randn(2, 2, 4, 32, 16)
        re = torch.tensor([1000.0, 1100.0])
        sc = torch.tensor([1.0, 1.2])

        out_step = forecaster.forward_single_step(q_hist, re, sc)
        assert out_step.shape == (2, 1, 4, 32, 16)

        out_rollout = forecaster.forward_rollout(q_hist, re, sc, horizon=3)
        assert out_rollout.shape == (2, 3, 4, 32, 16)


class TestLerayHelmholtzSpectralProjection:
    """Verification suite for differentiable Leray-Helmholtz Spectral Projection."""

    def test_strict_divergence_free_float64(self):
        """In float64, projection must achieve machine precision div(u_sol) < 1e-12."""
        torch.manual_seed(42)
        nx, ny = 32, 64
        domain_size = (1.0, 2.0)
        u = torch.randn(nx, ny, dtype=torch.float64)
        v = torch.randn(nx, ny, dtype=torch.float64)

        div_orig = compute_divergence(u, v, domain_size=domain_size)
        assert div_orig.abs().max() > 1.0  # Synthetic field has massive divergence

        u_sol, v_sol = project_divergence_free_2d(u, v, domain_size=domain_size)

        div_sol = compute_divergence(u_sol, v_sol, domain_size=domain_size)
        assert div_sol.abs().max().item() < 1e-12
        assert torch.sqrt(torch.mean(div_sol**2)).item() < 1e-13

    def test_strict_divergence_free_float32(self):
        """In float32, projection must achieve single-precision divergence < 1e-4."""
        torch.manual_seed(42)
        nx, ny = 32, 64
        domain_size = (1.0, 2.0)
        u = torch.randn(2, 4, nx, ny, dtype=torch.float32)
        v = torch.randn(2, 4, nx, ny, dtype=torch.float32)

        u_sol, v_sol = project_divergence_free_2d(u, v, domain_size=domain_size)

        div_sol = compute_divergence(u_sol, v_sol, domain_size=domain_size)
        assert div_sol.abs().max().item() < 1e-4
        assert torch.sqrt(torch.mean(div_sol**2)).item() < 2e-5

    def test_idempotence(self):
        """Projection operator must be idempotent: P(P(u)) == P(u)."""
        torch.manual_seed(42)
        nx, ny = 16, 32
        u = torch.randn(nx, ny, dtype=torch.float64)
        v = torch.randn(nx, ny, dtype=torch.float64)

        u_sol1, v_sol1 = project_divergence_free_2d(u, v)
        u_sol2, v_sol2 = project_divergence_free_2d(u_sol1, v_sol1)

        assert torch.allclose(u_sol1, u_sol2, atol=1e-12, rtol=1e-12)
        assert torch.allclose(v_sol1, v_sol2, atol=1e-12, rtol=1e-12)

    def test_vorticity_preservation(self):
        """Helmholtz decomposition theorem: curl(grad(phi)) = 0, so curl(u_sol) == curl(u)."""
        torch.manual_seed(42)
        nx, ny = 32, 64
        domain_size = (1.0, 2.0)
        u = torch.randn(nx, ny, dtype=torch.float64)
        v = torch.randn(nx, ny, dtype=torch.float64)

        omega_orig = compute_vorticity(u, v, domain_size=domain_size)
        u_sol, v_sol = project_divergence_free_2d(u, v, domain_size=domain_size)
        omega_sol = compute_vorticity(u_sol, v_sol, domain_size=domain_size)

        assert torch.allclose(omega_orig, omega_sol, atol=1e-11, rtol=1e-11)

    def test_mean_momentum_preservation(self):
        """Zero wavenumber background mean velocity should remain unchanged."""
        torch.manual_seed(42)
        nx, ny = 32, 32
        u = torch.randn(nx, ny) + 3.5
        v = torch.randn(nx, ny) - 2.1

        u_sol, v_sol = project_divergence_free_2d(u, v)
        assert torch.isclose(u.mean(), u_sol.mean(), atol=1e-5)
        assert torch.isclose(v.mean(), v_sol.mean(), atol=1e-5)

    def test_orthogonality_of_decomposition(self):
        """Inner product <u_sol, u_irrot> must be 0."""
        torch.manual_seed(42)
        nx, ny = 32, 64
        u = torch.randn(nx, ny, dtype=torch.float64)
        v = torch.randn(nx, ny, dtype=torch.float64)

        u_sol, v_sol = project_divergence_free_2d(u, v)
        u_irrot = u - u_sol
        v_irrot = v - v_sol

        # L2 spatial inner product
        inner_prod = torch.sum(u_sol * u_irrot + v_sol * v_irrot).item()
        total_energy = torch.sum(u**2 + v**2).item()
        rel_inner_prod = abs(inner_prod) / total_energy
        assert rel_inner_prod < 1e-12

    def test_gradient_backprop(self):
        """Verify Leray projection is end-to-end differentiable with PyTorch autograd."""
        nx, ny = 16, 32
        u = torch.randn(2, nx, ny, requires_grad=True)
        v = torch.randn(2, nx, ny, requires_grad=True)

        u_sol, v_sol = project_divergence_free_2d(u, v)
        loss = torch.sum(u_sol**2 + v_sol**2)
        loss.backward()

        assert u.grad is not None and v.grad is not None
        assert not torch.isnan(u.grad).any() and not torch.isnan(v.grad).any()
        assert u.grad.shape == u.shape and v.grad.shape == v.shape

    def test_project_incompressible_state(self):
        """Verify 4-channel physical state projection preserves scalar and zeroes divergence."""
        torch.manual_seed(42)
        b, c, nx, ny = 2, 4, 32, 64
        q = torch.randn(b, c, nx, ny, dtype=torch.float64)
        # Shift pressure to have non-zero mean
        q[:, 2] += 10.0

        q_proj = project_incompressible_state(q)
        assert q_proj.shape == q.shape

        # Velocity is divergence-free
        div = compute_divergence(q_proj[:, 0], q_proj[:, 1])
        assert div.abs().max().item() < 1e-12

        # Pressure is zero-mean
        p_mean = q_proj[:, 2].mean(dim=(-2, -1))
        assert p_mean.abs().max().item() < 1e-12

        # Tracer c is preserved exactly
        assert torch.allclose(q_proj[:, 3], q[:, 3], atol=1e-14)
