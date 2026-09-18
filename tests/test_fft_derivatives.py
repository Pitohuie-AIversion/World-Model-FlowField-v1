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


def make_periodic_grid(ny: int = 128, nx: int = 256, ly: float = 2.0, lx: float = 1.0):
    """Generate 2D periodic coordinates [y, x]."""
    y = torch.linspace(-1.0, 1.0 - ly / ny, ny, dtype=torch.float64)
    x = torch.linspace(0.0, 1.0 - lx / nx, nx, dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return yy, xx


def test_spectral_derivatives_accuracy():
    """Test spectral derivative against analytical derivatives."""
    ny, nx = 128, 256
    ly, lx = 2.0, 1.0
    yy, xx = make_periodic_grid(ny, nx, ly, lx)

    # Test function: f(x, y) = sin(2*pi*x) * cos(pi*y)
    # df/dx = 2*pi * cos(2*pi*x) * cos(pi*y)
    # df/dy = -pi * sin(2*pi*x) * sin(pi*y)
    f = torch.sin(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)

    exact_df_dx = 2.0 * torch.pi * torch.cos(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)
    exact_df_dy = -torch.pi * torch.sin(2.0 * torch.pi * xx) * torch.sin(torch.pi * yy)

    calc_df_dy, calc_df_dx = spectral_grad_2d(f.float(), domain_size=(ly, lx))

    err_x = torch.norm(calc_df_dx.double() - exact_df_dx) / torch.norm(exact_df_dx)
    err_y = torch.norm(calc_df_dy.double() - exact_df_dy) / torch.norm(exact_df_dy)

    assert err_x < 1e-4, f"df/dx relative error too large: {err_x}"
    assert err_y < 1e-4, f"df/dy relative error too large: {err_y}"


def test_divergence_free_field():
    """Test divergence calculation on an analytically divergence-free velocity field."""
    ny, nx = 128, 256
    ly, lx = 2.0, 1.0
    yy, xx = make_periodic_grid(ny, nx, ly, lx)

    # u = -sin(2*pi*x) * sin(pi*y)
    # v = -2 * cos(2*pi*x) * cos(pi*y)
    # du/dx = -2*pi * cos(2*pi*x) * sin(pi*y)
    # dv/dy = +2*pi * cos(2*pi*x) * sin(pi*y)
    # div = du/dx + dv/dy = 0
    u = -torch.sin(2.0 * torch.pi * xx) * torch.sin(torch.pi * yy)
    v = -2.0 * torch.cos(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)

    div = compute_divergence(u.float(), v.float(), domain_size=(ly, lx))
    max_div = torch.max(torch.abs(div))

    assert max_div < 1e-4, f"Divergence should be ~0, got max: {max_div}"


def test_vorticity_calculation():
    """Test vorticity against analytical solution."""
    ny, nx = 128, 256
    ly, lx = 2.0, 1.0
    yy, xx = make_periodic_grid(ny, nx, ly, lx)

    u = -torch.sin(2.0 * torch.pi * xx) * torch.sin(torch.pi * yy)
    v = -2.0 * torch.cos(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)

    # omega = dv/dx - du/dy
    # dv/dx = 4*pi * sin(2*pi*x) * cos(pi*y)
    # du/dy = -pi * sin(2*pi*x) * cos(pi*y)
    # omega = 5*pi * sin(2*pi*x) * cos(pi*y)
    exact_omega = 5.0 * torch.pi * torch.sin(2.0 * torch.pi * xx) * torch.cos(torch.pi * yy)

    calc_omega = compute_vorticity(u.float(), v.float(), domain_size=(ly, lx))
    err = torch.norm(calc_omega.double() - exact_omega) / torch.norm(exact_omega)

    assert err < 1e-4, f"Vorticity relative error too large: {err}"


def test_zero_mean_pressure():
    """Test zero-mean pressure gauge projection."""
    p = torch.randn(4, 1, 128, 256) + 15.0
    p_proj = project_zero_mean_pressure(p)
    mean_after = p_proj.mean(dim=(-2, -1))
    assert torch.allclose(mean_after, torch.zeros_like(mean_after), atol=1e-6)
