"""Unit tests for 2D periodic spectral derivatives and physical operators."""

import torch
import pytest
from src.utils.fft_derivatives import (
    spectral_grad_2d,
    compute_vorticity,
    compute_divergence,
    project_zero_mean_pressure,
    compute_kinetic_energy,
    compute_enstrophy,
)


def make_periodic_grid(nx: int = 128, ny: int = 256, lx: float = 1.0, ly: float = 2.0):
    """Generate 2D periodic coordinates [x, y]."""
    x = torch.linspace(0.0, 1.0 - lx / nx, nx, dtype=torch.float64)
    y = torch.linspace(-1.0, 1.0 - ly / ny, ny, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return xx, yy


def test_spectral_derivatives_accuracy():
    """Test spectral derivative against analytical derivatives."""
    nx, ny = 128, 256
    lx, ly = 1.0, 2.0
    xx, yy = make_periodic_grid(nx, ny, lx, ly)

    # Test function: f(x, y) = sin(2*pi*x) * cos(pi*y)
    # df/dx = 2*pi * cos(2*pi*x) * cos(pi*y)
    # df/dy = -pi * sin(2*pi*x) * sin(pi*y)
    f = torch.sin(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)

    exact_df_dx = 2.0 * torch.pi * torch.cos(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)
    exact_df_dy = -torch.pi * torch.sin(2.0 * torch.pi * xx) * torch.sin(torch.pi * yy)

    calc_df_dx, calc_df_dy = spectral_grad_2d(f.float(), domain_size=(lx, ly))

    err_x = torch.norm(calc_df_dx.double() - exact_df_dx) / torch.norm(exact_df_dx)
    err_y = torch.norm(calc_df_dy.double() - exact_df_dy) / torch.norm(exact_df_dy)

    assert err_x < 1e-4, f"df/dx relative error too large: {err_x}"
    assert err_y < 1e-4, f"df/dy relative error too large: {err_y}"


def test_divergence_free_field():
    """Test divergence calculation on an analytically divergence-free velocity field."""
    nx, ny = 128, 256
    lx, ly = 1.0, 2.0
    xx, yy = make_periodic_grid(nx, ny, lx, ly)

    # Streamfunction psi = sin(2*pi*x) * cos(pi*y)
    # u = d(psi)/dy = -pi * sin(2*pi*x) * sin(pi*y)
    # v = -d(psi)/dx = -2*pi * cos(2*pi*x) * cos(pi*y)
    # du/dx = -2*pi^2 * cos(2*pi*x) * sin(pi*y)
    # dv/dy = +2*pi^2 * cos(2*pi*x) * sin(pi*y)
    # div = du/dx + dv/dy = 0
    u = -torch.pi * torch.sin(2.0 * torch.pi * xx) * torch.sin(torch.pi * yy)
    v = -2.0 * torch.pi * torch.cos(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)

    div = compute_divergence(u.float(), v.float(), domain_size=(lx, ly))
    max_div = torch.max(torch.abs(div))

    assert max_div < 5e-4, f"Divergence should be ~0, got max: {max_div}"


def test_vorticity_calculation():
    """Test vorticity against analytical solution."""
    nx, ny = 128, 256
    lx, ly = 1.0, 2.0
    xx, yy = make_periodic_grid(nx, ny, lx, ly)

    # u = -pi * sin(2*pi*x) * sin(pi*y)
    # v = -2*pi * cos(2*pi*x) * cos(pi*y)
    # omega = dv/dx - du/dy
    # dv/dx = 4*pi^2 * sin(2*pi*x) * cos(pi*y)
    # du/dy = -pi^2 * sin(2*pi*x) * cos(pi*y)
    # omega = 5*pi^2 * sin(2*pi*x) * cos(pi*y)
    u = -torch.pi * torch.sin(2.0 * torch.pi * xx) * torch.sin(torch.pi * yy)
    v = -2.0 * torch.pi * torch.cos(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)
    exact_omega = 5.0 * (torch.pi ** 2) * torch.sin(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)

    calc_omega = compute_vorticity(u.float(), v.float(), domain_size=(lx, ly))
    err = torch.norm(calc_omega.double() - exact_omega) / torch.norm(exact_omega)

    assert err < 1e-4, f"Vorticity relative error too large: {err}"


def test_zero_mean_pressure():
    """Test zero-mean pressure gauge projection."""
    p = torch.randn(4, 1, 128, 256) + 15.0
    p_proj = project_zero_mean_pressure(p)
    mean_after = p_proj.mean(dim=(-2, -1))
    assert torch.allclose(mean_after, torch.zeros_like(mean_after), atol=1e-5)


def test_laplacian_accuracy():
    """Test spectral Laplacian against analytical solution: lap(sin(2pi*x)*cos(pi*y)) = -5pi^2 * sin(2pi*x)*cos(pi*y)."""
    from src.utils.fft_derivatives import compute_laplacian_2d
    nx, ny = 128, 256
    lx, ly = 1.0, 2.0
    xx, yy = make_periodic_grid(nx, ny, lx, ly)

    f = torch.sin(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)
    exact_lap = -5.0 * (torch.pi ** 2) * f

    calc_lap = compute_laplacian_2d(f, domain_size=(lx, ly))
    err = torch.norm(calc_lap - exact_lap) / torch.norm(exact_lap)
    assert err < 1e-10, f"Laplacian relative error too large: {err}"



def test_kinetic_energy_and_enstrophy():
    """Test kinetic energy and enstrophy positive definiteness and calculation."""
    u = torch.randn(2, 64, 64)
    v = torch.randn(2, 64, 64)
    ke = compute_kinetic_energy(u, v)
    assert ke.shape == (2,)
    assert (ke >= 0.0).all()

    omega = compute_vorticity(u, v)
    ens = compute_enstrophy(omega)
    assert ens.shape == (2,)
    assert (ens >= 0.0).all()

