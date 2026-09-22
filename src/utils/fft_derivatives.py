"""Spectral spatial derivative utilities on 2D periodic domains using FFT.

Canonical tensor contract for The Well shear_flow:
    physical tensor layout is (..., Nx, Ny)
    axis -2 is x with Lx = 1.0
    axis -1 is y with Ly = 2.0
The stored HDF5 coordinate arrays are normalized, while the physical
Dedalus domain has the above 1:2 extent.
"""

from typing import Tuple
import torch
import torch.fft


SHEAR_FLOW_DOMAIN_SIZE: Tuple[float, float] = (1.0, 2.0)
PHYSICS_OPERATOR_VERSION = "xy-nx-ny-lx1-ly2-v1"


def spectral_grad_xy(
    field: torch.Tensor,
    domain_size: Tuple[float, float] = SHEAR_FLOW_DOMAIN_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (df/dx, df/dy) for canonical (..., Nx, Ny) shear-flow tensors.

    ShearFlowDataset preserves the HDF5 layout: axis -2 is physical x and
    axis -1 is physical y. domain_size is therefore (Lx, Ly).
    """
    nx, ny = field.shape[-2], field.shape[-1]
    lx, ly = domain_size
    device = field.device
    orig_dtype = field.dtype
    calc_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    field_calc = field.to(dtype=calc_dtype)

    # rfft2 keeps the full x-frequency axis (-2) and the non-negative
    # y-frequency half-spectrum (-1).
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
    domain_size: Tuple[float, float] = (2.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Legacy helper for explicitly y/x-arranged (..., Ny, Nx) tensors.

    This preserves the historical public API and return order (df/dy, df/dx).
    It must not be used with ShearFlowDataset outputs. New project code should
    use spectral_grad_xy, whose contract is (..., Nx, Ny) and (df/dx, df/dy).
    """
    ny, nx = field.shape[-2], field.shape[-1]
    ly, lx = domain_size

    # Wavenumber grids
    # rfft along last dim (x) has size nx//2 + 1
    # fft along second to last dim (y) has size ny
    device = field.device
    orig_dtype = field.dtype
    calc_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    field_calc = field.to(dtype=calc_dtype)

    # ky = 2 * pi * n / Ly, shape (ny, 1)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(ny, d=ly / ny, device=device, dtype=calc_dtype)
    ky = ky.view(*([1] * (field.ndim - 2)), ny, 1)

    # kx = 2 * pi * n / Lx, shape (1, nx//2 + 1)
    kx = 2.0 * torch.pi * torch.fft.rfftfreq(nx, d=lx / nx, device=device, dtype=calc_dtype)
    kx = kx.view(*([1] * (field.ndim - 2)), 1, nx // 2 + 1)

    # Forward 2D RFFT
    f_hat = torch.fft.rfft2(field_calc, dim=(-2, -1))

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
    domain_size: Tuple[float, float] = SHEAR_FLOW_DOMAIN_SIZE,
) -> torch.Tensor:
    """Compute omega = dv/dx - du/dy for canonical (..., Nx, Ny) tensors."""
    _, du_dy = spectral_grad_xy(u, domain_size=domain_size)
    dv_dx, _ = spectral_grad_xy(v, domain_size=domain_size)
    return dv_dx - du_dy


def compute_divergence(
    u: torch.Tensor,
    v: torch.Tensor,
    domain_size: Tuple[float, float] = SHEAR_FLOW_DOMAIN_SIZE,
) -> torch.Tensor:
    """Compute du/dx + dv/dy for canonical (..., Nx, Ny) tensors."""
    du_dx, _ = spectral_grad_xy(u, domain_size=domain_size)
    _, dv_dy = spectral_grad_xy(v, domain_size=domain_size)
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


def compute_laplacian_2d(
    field: torch.Tensor,
    domain_size: Tuple[float, float] = SHEAR_FLOW_DOMAIN_SIZE,
) -> torch.Tensor:
    """Compute d2f/dx2 + d2f/dy2 for canonical (..., Nx, Ny) tensors."""
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

    if nx % 2 == 0:
        f_hat_lap[..., nx // 2, :] = 0.0
    if ny % 2 == 0:
        f_hat_lap[..., :, ny // 2] = 0.0

    laplacian = torch.fft.irfft2(f_hat_lap, s=(nx, ny), dim=(-2, -1))
    return laplacian.to(dtype=orig_dtype)
