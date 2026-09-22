"""Spectral derivative utilities for the canonical shear_flow tensor layout.

Closure-R4 contract:
    field shape (..., Nx, Ny)
    dim -2 -> x, Lx = 1
    dim -1 -> y, Ly = 2
"""

from typing import Tuple
import torch
import torch.fft

from src.utils.physics_contract import SHEAR_FLOW_DOMAIN_SIZE_XY


def spectral_grad_xy(
    field: torch.Tensor,
    domain_size: Tuple[float, float] = SHEAR_FLOW_DOMAIN_SIZE_XY,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (df/dx, df/dy) for fields stored as (..., Nx, Ny)."""
    nx, ny = field.shape[-2], field.shape[-1]
    lx, ly = domain_size
    device = field.device
    orig_dtype = field.dtype
    calc_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    field_calc = field.to(dtype=calc_dtype)

    kx = 2.0 * torch.pi * torch.fft.fftfreq(
        nx, d=lx / nx, device=device, dtype=calc_dtype
    )
    kx = kx.view(*([1] * (field.ndim - 2)), nx, 1)

    ky = 2.0 * torch.pi * torch.fft.rfftfreq(
        ny, d=ly / ny, device=device, dtype=calc_dtype
    )
    ky = ky.view(*([1] * (field.ndim - 2)), 1, ny // 2 + 1)

    f_hat = torch.fft.rfft2(field_calc, dim=(-2, -1))
    f_hat_x = 1j * kx * f_hat
    f_hat_y = 1j * ky * f_hat

    if nx % 2 == 0:
        f_hat_x[..., nx // 2, :] = 0.0
    if ny % 2 == 0:
        f_hat_y[..., :, ny // 2] = 0.0

    df_dx = torch.fft.irfft2(f_hat_x, s=(nx, ny), dim=(-2, -1))
    df_dy = torch.fft.irfft2(f_hat_y, s=(nx, ny), dim=(-2, -1))
    return df_dx.to(dtype=orig_dtype), df_dy.to(dtype=orig_dtype)


def spectral_grad_2d(
    field: torch.Tensor,
    domain_size: Tuple[float, float] = SHEAR_FLOW_DOMAIN_SIZE_XY,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compatibility name using Closure-R4 semantics; returns (df/dx, df/dy)."""
    return spectral_grad_xy(field, domain_size=domain_size)


def compute_vorticity(
    u: torch.Tensor,
    v: torch.Tensor,
    domain_size: Tuple[float, float] = SHEAR_FLOW_DOMAIN_SIZE_XY,
) -> torch.Tensor:
    """Compute omega = dv/dx - du/dy."""
    _, du_dy = spectral_grad_xy(u, domain_size=domain_size)
    dv_dx, _ = spectral_grad_xy(v, domain_size=domain_size)
    return dv_dx - du_dy


def compute_divergence(
    u: torch.Tensor,
    v: torch.Tensor,
    domain_size: Tuple[float, float] = SHEAR_FLOW_DOMAIN_SIZE_XY,
) -> torch.Tensor:
    """Compute div(u) = du/dx + dv/dy."""
    du_dx, _ = spectral_grad_xy(u, domain_size=domain_size)
    _, dv_dy = spectral_grad_xy(v, domain_size=domain_size)
    return du_dx + dv_dy


def project_zero_mean_pressure(p: torch.Tensor) -> torch.Tensor:
    """Project pressure to zero spatial mean gauge."""
    return p - p.mean(dim=(-2, -1), keepdim=True)


def compute_kinetic_energy(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute mean kinetic energy density."""
    return 0.5 * torch.mean(u**2 + v**2, dim=(-2, -1))


def compute_enstrophy(omega: torch.Tensor) -> torch.Tensor:
    """Compute mean enstrophy density."""
    return 0.5 * torch.mean(omega**2, dim=(-2, -1))


def compute_laplacian_2d(
    field: torch.Tensor,
    domain_size: Tuple[float, float] = SHEAR_FLOW_DOMAIN_SIZE_XY,
) -> torch.Tensor:
    """Compute d2f/dx2 + d2f/dy2 for (..., Nx, Ny)."""
    nx, ny = field.shape[-2], field.shape[-1]
    lx, ly = domain_size
    device = field.device
    orig_dtype = field.dtype
    calc_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    field_calc = field.to(dtype=calc_dtype)

    kx = 2.0 * torch.pi * torch.fft.fftfreq(
        nx, d=lx / nx, device=device, dtype=calc_dtype
    )
    kx = kx.view(*([1] * (field.ndim - 2)), nx, 1)
    ky = 2.0 * torch.pi * torch.fft.rfftfreq(
        ny, d=ly / ny, device=device, dtype=calc_dtype
    )
    ky = ky.view(*([1] * (field.ndim - 2)), 1, ny // 2 + 1)

    f_hat = torch.fft.rfft2(field_calc, dim=(-2, -1))
    f_hat_lap = -(kx**2 + ky**2) * f_hat
    laplacian = torch.fft.irfft2(f_hat_lap, s=(nx, ny), dim=(-2, -1))
    return laplacian.to(dtype=orig_dtype)
