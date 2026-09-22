"""Unit tests for directional spectral dissipation analysis."""

import numpy as np
import pytest
import torch

from scripts.analyze_spectral_dissipation import (
    compute_directional_energy_spectra,
    diagnose_dissipation_mode,
)


def test_directional_spectra_shapes_and_parseval():
    """Verify directional Fourier spectra obey Parseval energy conservation."""
    nx, ny = 32, 64
    lx, ly = 1.0, 2.0
    u = torch.randn(2, nx, ny)
    v = torch.randn(2, nx, ny)

    kx, e_kx, ky, e_ky = compute_directional_energy_spectra(u, v, domain_size=(lx, ly))

    assert len(kx) == nx // 2 + 1
    assert len(e_kx) == nx // 2 + 1
    assert len(ky) == ny // 2 + 1
    assert len(e_ky) == ny // 2 + 1

    assert (e_kx >= 0.0).all()
    assert (e_ky >= 0.0).all()

    # Parseval energy equality
    total_physical = 0.5 * (u**2 + v**2).mean(dim=(-2, -1)).mean().item()
    sum_kx = e_kx.sum().item()
    sum_ky = e_ky.sum().item()

    assert sum_kx == pytest.approx(total_physical, rel=1e-4)
    assert sum_ky == pytest.approx(total_physical, rel=1e-4)
    assert sum_kx == pytest.approx(sum_ky, rel=1e-5)


def test_directional_spectra_anisotropic_sinusoid():
    """Pure x-mode sinusoid should concentrate energy entirely at the fundamental kx."""
    nx, ny = 32, 64
    lx, ly = 1.0, 2.0
    x = torch.arange(nx, dtype=torch.float32) * (lx / nx)
    y = torch.arange(ny, dtype=torch.float32) * (ly / ny)
    grid_x, _ = torch.meshgrid(x, y, indexing="ij")

    # u = sin(2*pi*x / lx), v = 0
    u = torch.sin(2.0 * torch.pi * grid_x / lx)
    v = torch.zeros_like(u)

    kx, e_kx, ky, e_ky = compute_directional_energy_spectra(u, v, domain_size=(lx, ly))

    # Fundamental mode is index 1
    total_e = e_kx.sum().item()
    assert e_kx[1].item() / total_e > 0.999
    # High kx should be zero
    assert e_kx[2:].sum().item() / total_e < 1e-4


def test_diagnose_dissipation_mode():
    """Verify dissipation diagnostic correctly classifies noise vs damping."""
    k_bins = np.linspace(0, 100, 50)

    # 1. Spurious noise: ratio is 2.5 at high k
    ratio_noisy = np.ones(50)
    ratio_noisy[25:] = 2.5
    d_noisy = diagnose_dissipation_mode(k_bins, ratio_noisy, cutoff_ratio=0.5)
    assert d_noisy["diagnosis"] == "spurious_high_frequency_accumulation"
    assert d_noisy["mean_high_k_ratio"] == pytest.approx(2.5)

    # 2. Over-dissipation: ratio is 0.3 at high k
    ratio_damped = np.ones(50)
    ratio_damped[25:] = 0.3
    d_damped = diagnose_dissipation_mode(k_bins, ratio_damped, cutoff_ratio=0.5)
    assert d_damped["diagnosis"] == "numerical_over_dissipation"
    assert d_damped["mean_high_k_ratio"] == pytest.approx(0.3)

    # 3. Balanced preservation: ratio is ~1.0
    ratio_balanced = np.ones(50)
    d_balanced = diagnose_dissipation_mode(k_bins, ratio_balanced, cutoff_ratio=0.5)
    assert d_balanced["diagnosis"] == "balanced_scale_preservation"
    assert d_balanced["mean_high_k_ratio"] == pytest.approx(1.0)
