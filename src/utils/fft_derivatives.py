"""Spectral spatial derivative utilities on 2D periodic domains using FFT.

Domain specifications for The Well shear_flow:
    x in [0, 1] (horizontal, periodic, Nx = 256 or 512)
    y in [-1, 1] (vertical, periodic, Ny = 128 or 256)
    Lx = 1.0, Ly = 2.0
"""

from typing import Tuple
import torch
import torch.fft


def spectral_grad_2d(
    field: torch.Tensor,
    domain_size: Tuple[float, float] = (2.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 2D spatial gradients (df/dy, df/dx) via real-to-complex FFT.

    Args:
        field: Tensor of shape (..., Ny, Nx), real-valued.
        domain_size: (Ly, Lx) extent of domain. Defaults to (2.0, 1.0) for shear_flow.

    Returns:
        df_dy: Gradient along vertical dimension (same shape as field).
        df_dx: Gradient along horizontal dimension (same shape as field).
    """
    ny, nx = field.shape[-2], field.shape[-1]
    ly, lx = domain_size

    # Wavenumber grids
    # rfft along last dim (x) has size nx//2 + 1
    # fft along second to last dim (y) has size ny
    device = field.device
    orig_dtype = field.dtype
    field_f32 = field.float()

    # ky = 2 * pi * n / Ly, shape (ny, 1)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(ny, d=ly / ny, device=device)
    ky = ky.view(*([1] * (field.ndim - 2)), ny, 1)

    # kx = 2 * pi * n / Lx, shape (1, nx//2 + 1)
    kx = 2.0 * torch.pi * torch.fft.rfftfreq(nx, d=lx / nx, device=device)
    kx = kx.view(*([1] * (field.ndim - 2)), 1, nx // 2 + 1)

    # Forward 2D RFFT
    f_hat = torch.fft.rfft2(field_f32, dim=(-2, -1))

    # Differentiation in Fourier space: d/dx -> i * kx, d/dy -> i * ky
    # Using 1j * k
    f_hat_x = 1j * kx * f_hat
    f_hat_y = 1j * ky * f_hat

    # For even ny / nx, zero out the Nyquist frequency derivative to preserve reality and avoid artifacts
    if ny % 2 == 0:
        f_hat_y[..., ny // 2, :] = 0.0
    if nx % 2 == 0:
        f_hat_x[..., :, nx // 2] = 0.0

    df_dx = torch.fft.irfft2(f_hat_x, s=(ny, nx), dim=(-2, -1))
    df_dy = torch.fft.irfft2(f_hat_y, s=(ny, nx), dim=(-2, -1))

    return df_dy.to(dtype=orig_dtype), df_dx.to(dtype=orig_dtype)


def compute_vorticity(
    u: torch.Tensor,
    v: torch.Tensor,
    domain_size: Tuple[float, float] = (2.0, 1.0),
) -> torch.Tensor:
    """Compute 2D vorticity: omega = dv/dx - du/dy.

    Args:
        u: Horizontal velocity, shape (..., Ny, Nx).
        v: Vertical velocity, shape (..., Ny, Nx).
        domain_size: (Ly, Lx), defaults to (2.0, 1.0).

    Returns:
        omega: Vorticity scalar field, shape (..., Ny, Nx).
    """
    du_dy, _ = spectral_grad_2d(u, domain_size=domain_size)
    _, dv_dx = spectral_grad_2d(v, domain_size=domain_size)
    return dv_dx - du_dy


def compute_divergence(
    u: torch.Tensor,
    v: torch.Tensor,
    domain_size: Tuple[float, float] = (2.0, 1.0),
) -> torch.Tensor:
    """Compute 2D divergence: div = du/dx + dv/dy.

    Args:
        u: Horizontal velocity, shape (..., Ny, Nx).
        v: Vertical velocity, shape (..., Ny, Nx).
        domain_size: (Ly, Lx), defaults to (2.0, 1.0).

    Returns:
        div: Divergence field, shape (..., Ny, Nx). Should be ~0 for incompressible flow.
    """
    _, du_dx = spectral_grad_2d(u, domain_size=domain_size)
    dv_dy, _ = spectral_grad_2d(v, domain_size=domain_size)
    return du_dx + dv_dy


def project_zero_mean_pressure(p: torch.Tensor) -> torch.Tensor:
    """Project pressure field to zero spatial mean gauge: p <- p - mean(p).

    Args:
        p: Pressure field, shape (..., Ny, Nx).

    Returns:
        Zero-mean normalized pressure.
    """
    mean_p = p.mean(dim=(-2, -1), keepdim=True)
    return p - mean_p


def compute_kinetic_energy(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute mean kinetic energy density: E_k = 0.5 * mean(u^2 + v^2).

    Args:
        u: Horizontal velocity, shape (..., Ny, Nx).
        v: Vertical velocity, shape (..., Ny, Nx).

    Returns:
        E_k: Scalar or shape (...) tensor.
    """
    return 0.5 * torch.mean(u**2 + v**2, dim=(-2, -1))


def compute_enstrophy(omega: torch.Tensor) -> torch.Tensor:
    """Compute mean enstrophy density: Omega = 0.5 * mean(omega^2).

    Args:
        omega: Vorticity field, shape (..., Ny, Nx).

    Returns:
        Enstrophy: Scalar or shape (...) tensor.
    """
    return 0.5 * torch.mean(omega**2, dim=(-2, -1))
