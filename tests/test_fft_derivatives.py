"""Tests for the Closure-R4 (Nx, Ny) spectral-physics contract."""

import torch

from src.utils.fft_derivatives import (
    spectral_grad_xy,
    spectral_grad_2d,
    compute_vorticity,
    compute_divergence,
    project_zero_mean_pressure,
    compute_kinetic_energy,
    compute_enstrophy,
    compute_laplacian_2d,
)


def make_periodic_grid(nx: int = 128, ny: int = 256, lx: float = 1.0, ly: float = 2.0):
    x = torch.arange(nx, dtype=torch.float64) * (lx / nx)
    y = -1.0 + torch.arange(ny, dtype=torch.float64) * (ly / ny)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return xx, yy


def test_spectral_derivatives_accuracy_xy_layout():
    xx, yy = make_periodic_grid()
    f = torch.sin(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)
    exact_dx = 2.0 * torch.pi * torch.cos(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)
    exact_dy = -torch.pi * torch.sin(2.0 * torch.pi * xx) * torch.sin(torch.pi * yy)

    calc_dx, calc_dy = spectral_grad_xy(f.float())
    assert torch.norm(calc_dx.double() - exact_dx) / torch.norm(exact_dx) < 1e-4
    assert torch.norm(calc_dy.double() - exact_dy) / torch.norm(exact_dy) < 1e-4

    alias_dx, alias_dy = spectral_grad_2d(f.float())
    assert torch.allclose(alias_dx, calc_dx)
    assert torch.allclose(alias_dy, calc_dy)


def test_divergence_free_field_xy_layout():
    xx, yy = make_periodic_grid()
    u = -torch.sin(2.0 * torch.pi * xx) * torch.sin(torch.pi * yy)
    v = -2.0 * torch.cos(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)
    div = compute_divergence(u.float(), v.float())
    assert torch.max(torch.abs(div)) < 2e-4


def test_vorticity_calculation_xy_layout():
    xx, yy = make_periodic_grid()
    u = -torch.sin(2.0 * torch.pi * xx) * torch.sin(torch.pi * yy)
    v = -2.0 * torch.cos(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)
    exact = 5.0 * torch.pi * torch.sin(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)
    calc = compute_vorticity(u.float(), v.float())
    assert torch.norm(calc.double() - exact) / torch.norm(exact) < 1e-4


def test_laplacian_accuracy_xy_layout():
    xx, yy = make_periodic_grid()
    f = torch.sin(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)
    exact = -5.0 * (torch.pi**2) * f
    calc = compute_laplacian_2d(f)
    assert torch.norm(calc - exact) / torch.norm(exact) < 1e-10


def test_zero_mean_pressure():
    p = torch.randn(4, 1, 128, 256) + 15.0
    p_proj = project_zero_mean_pressure(p)
    mean = p_proj.mean(dim=(-2, -1))
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)


def test_kinetic_energy_and_enstrophy():
    u = torch.randn(2, 64, 128)
    v = torch.randn(2, 64, 128)
    ke = compute_kinetic_energy(u, v)
    assert ke.shape == (2,)
    assert (ke >= 0.0).all()
    omega = compute_vorticity(u, v)
    ens = compute_enstrophy(omega)
    assert ens.shape == (2,)
    assert (ens >= 0.0).all()
