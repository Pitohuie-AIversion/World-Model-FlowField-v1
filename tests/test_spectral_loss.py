"""Unit and regression tests for multi-scale EnergySpectrumLoss and batched spectrum utilities."""

import pytest
import torch
from src.losses.spectral import EnergySpectrumLoss
from src.metrics.spectral import compute_radial_energy_spectrum
from src.utils.fft_derivatives import compute_batched_radial_energy_spectrum, clear_wavenumber_cache


class TestBatchedRadialEnergySpectrum:
    """Tests for compute_batched_radial_energy_spectrum consistency and robustness."""

    def setup_method(self):
        clear_wavenumber_cache()

    def test_matches_reference_metric(self):
        """Batched implementation must strictly match the single-sample reference implementation."""
        nx, ny = 32, 64
        lx, ly = 1.0, 2.0
        torch.manual_seed(42)
        u = torch.randn(2, 3, nx, ny)
        v = torch.randn(2, 3, nx, ny)

        k_bins_batch, e_k_batch = compute_batched_radial_energy_spectrum(u, v, domain_size=(lx, ly))

        # Check each sample against reference
        for b in range(2):
            for t in range(3):
                k_ref, e_ref = compute_radial_energy_spectrum(u[b, t], v[b, t], domain_size=(lx, ly))
                assert torch.allclose(k_bins_batch, k_ref, atol=1e-6)
                assert torch.allclose(e_k_batch[b, t], e_ref, atol=1e-6)

    def test_differentiation_backward(self):
        """Verify autograd backpropagation through batched spectrum computation."""
        nx, ny = 32, 64
        u = torch.randn(2, nx, ny, requires_grad=True)
        v = torch.randn(2, nx, ny, requires_grad=True)

        _, e_k = compute_batched_radial_energy_spectrum(u, v)
        loss = torch.sum(e_k)
        loss.backward()

        assert u.grad is not None and v.grad is not None
        assert not torch.isnan(u.grad).any()
        assert not torch.isnan(v.grad).any()
        assert torch.isfinite(u.grad).all()


class TestEnergySpectrumLoss:
    """Tests for EnergySpectrumLoss functionality, numerical stability, and fail-closed checks."""

    def setup_method(self):
        clear_wavenumber_cache()

    def test_zero_loss_on_identical_fields(self):
        """Identical predictions and targets must yield zero spectrum loss."""
        loss_fn = EnergySpectrumLoss(domain_size=(1.0, 2.0), loss_type="log_l1")
        q = torch.randn(2, 4, 32, 64)

        loss = loss_fn(q, q)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)

    def test_all_loss_types(self):
        """All supported loss formulations compute valid non-negative scalar values."""
        q_pred = torch.randn(2, 4, 32, 64)
        q_target = torch.randn(2, 4, 32, 64)

        for ltype in ("log_l1", "log_l2", "relative"):
            loss_fn = EnergySpectrumLoss(loss_type=ltype)
            loss = loss_fn(q_pred, q_target)
            assert loss.ndim == 0
            assert loss.item() > 0.0
            assert torch.isfinite(loss)

    def test_gradient_flow_to_velocity_channels(self):
        """Loss gradients must flow strictly to u and v, without affecting unrelated channels."""
        loss_fn = EnergySpectrumLoss()
        pred_q = torch.randn(2, 4, 32, 64, requires_grad=True)
        target_q = torch.randn(2, 4, 32, 64)

        loss = loss_fn(pred_q, target_q)
        loss.backward()

        assert pred_q.grad is not None
        # Velocity channels (0, 1) should receive gradients
        assert torch.norm(pred_q.grad[:, 0]) > 0.0
        assert torch.norm(pred_q.grad[:, 1]) > 0.0
        # Passive channels (2: p, 3: s) should receive zero gradient
        assert torch.all(pred_q.grad[:, 2:] == 0.0)

    def test_high_freq_weighting_sensitivity(self):
        """Increasing high_freq_weight must increase the relative penalty on high-frequency perturbations."""
        nx, ny = 32, 64
        lx, ly = 1.0, 2.0
        base_q = torch.zeros(1, 4, nx, ny)
        y = torch.linspace(0, ly, ny).view(1, 1, 1, ny)

        # Low-frequency perturbation (mode 1) vs high-frequency perturbation (mode 28)
        q_low = base_q.clone()
        q_low[:, 0:1] = torch.sin(2.0 * torch.pi * y / ly)

        q_high = base_q.clone()
        q_high[:, 0:1] = torch.sin(28.0 * torch.pi * y / ly)

        loss_uniform = EnergySpectrumLoss(high_freq_weight=0.0)
        loss_weighted = EnergySpectrumLoss(high_freq_weight=5.0)

        ratio_uniform = (loss_uniform(q_high, base_q) / loss_uniform(q_low, base_q)).item()
        ratio_weighted = (loss_weighted(q_high, base_q) / loss_weighted(q_low, base_q)).item()

        # With high_freq_weight=5.0, the relative penalty for high vs low frequencies must increase
        assert ratio_weighted > ratio_uniform

    def test_multidimensional_rollout_shapes(self):
        """EnergySpectrumLoss handles both (B, C, Nx, Ny) and (B, T, C, Nx, Ny)."""
        loss_fn = EnergySpectrumLoss()
        # Single step (B, C, Nx, Ny)
        loss_single = loss_fn(torch.randn(2, 4, 32, 64), torch.randn(2, 4, 32, 64))
        assert loss_single.ndim == 0

        # Rollout trajectory (B, T, C, Nx, Ny)
        loss_rollout = loss_fn(torch.randn(2, 5, 4, 32, 64), torch.randn(2, 5, 4, 32, 64))
        assert loss_rollout.ndim == 0
        assert torch.isfinite(loss_rollout)

    def test_compute_loss_and_bands(self):
        """compute_loss_and_bands returns scalar loss and structured band metrics."""
        loss_fn = EnergySpectrumLoss()
        pred_q = torch.randn(2, 4, 32, 64, requires_grad=True)
        target_q = torch.randn(2, 4, 32, 64)

        loss, metrics = loss_fn.compute_loss_and_bands(pred_q, target_q)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert "spec_loss_total" in metrics
        assert "spec_loss_low" in metrics
        assert "spec_loss_mid" in metrics
        assert "spec_loss_high" in metrics
        for k, v in metrics.items():
            assert isinstance(v, float)
            assert v >= 0.0

    def test_fail_closed_validation(self):
        """Fail-closed assertion and value errors for illegal inputs."""
        with pytest.raises(ValueError, match="Unknown loss_type"):
            EnergySpectrumLoss(loss_type="invalid_type")

        with pytest.raises(ValueError, match="high_freq_weight must be >= 0.0"):
            EnergySpectrumLoss(high_freq_weight=-1.0)

        with pytest.raises(ValueError, match="eps must be > 0.0"):
            EnergySpectrumLoss(eps=0.0)

        loss_fn = EnergySpectrumLoss()
        # Only 1 channel provided -> should assert failure
        with pytest.raises(AssertionError, match="must have >= 2 channels"):
            loss_fn(torch.randn(2, 1, 32, 64), torch.randn(2, 1, 32, 64))
