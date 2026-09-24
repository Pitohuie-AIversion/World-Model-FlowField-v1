"""Spectral spatial derivative utilities on 2D periodic domains using FFT.

Domain specifications for The Well shear_flow:
    Tensor spatial layout: (..., Nx, Ny)
    dim -2 corresponds to x in [0, 1] (horizontal, periodic, Nx = 256 or 128)
    dim -1 corresponds to y in [0, 2] (or [-1, 1], vertical, periodic, Ny = 512 or 256)
    Lx = 1.0, Ly = 2.0
"""

from typing import Dict, Tuple
import torch
import torch.fft

_WAVENUMBER_CACHE: Dict[Tuple[int, int, float, float, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}


def get_wavenumbers(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retrieve or compute cached 2D wavenumber grids (kx, ky) for RFFT2.

    Returns:
        kx: Shape (nx, 1), 1D frequency grid along dim -2.
        ky: Shape (1, ny // 2 + 1), RFFT frequency grid along dim -1.
    """
    key = (nx, ny, float(lx), float(ly), device, dtype)
    cached = _WAVENUMBER_CACHE.get(key)
    if cached is not None:
        return cached

    kx = 2.0 * torch.pi * torch.fft.fftfreq(nx, d=lx / nx, device=device, dtype=dtype).view(nx, 1)
    ky = 2.0 * torch.pi * torch.fft.rfftfreq(ny, d=ly / ny, device=device, dtype=dtype).view(1, ny // 2 + 1)
    _WAVENUMBER_CACHE[key] = (kx, ky)
    return kx, ky


def clear_wavenumber_cache() -> None:
    """Clear all cached wavenumber grids."""
    _WAVENUMBER_CACHE.clear()


def spectral_grad_2d(
    field: torch.Tensor,
    domain_size: Tuple[float, float] = (1.0, 2.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 2D spatial gradients (df/dx, df/dy) via real-to-complex FFT.

    Args:
        field: Tensor of shape (..., Nx, Ny), real-valued.
               dim -2 corresponds to x (size Nx, domain size Lx).
               dim -1 corresponds to y (size Ny, domain size Ly).
        domain_size: (Lx, Ly) extent of domain. Defaults to (1.0, 2.0) for shear_flow.

    Returns:
        df_dx: Gradient along horizontal dimension x (dim -2, same shape as field).
        df_dy: Gradient along vertical dimension y (dim -1, same shape as field).
    """
    nx, ny = field.shape[-2], field.shape[-1]
    lx, ly = domain_size

    # Wavenumber grids
    # rfft along last dim (y) has size ny//2 + 1
    # fft along second to last dim (x) has size nx
    device = field.device
    orig_dtype = field.dtype
    calc_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    field_calc = field.to(dtype=calc_dtype)

    kx_base, ky_base = get_wavenumbers(nx, ny, lx, ly, device, calc_dtype)
    if field.ndim > 2:
        kx = kx_base.view(*([1] * (field.ndim - 2)), nx, 1)
        ky = ky_base.view(*([1] * (field.ndim - 2)), 1, ny // 2 + 1)
    else:
        kx, ky = kx_base, ky_base

    # Forward 2D RFFT over (x, y)
    f_hat = torch.fft.rfft2(field_calc, dim=(-2, -1))

    # Differentiation in Fourier space: d/dx -> 1j * kx, d/dy -> 1j * ky
    f_hat_x = 1j * kx * f_hat
    f_hat_y = 1j * ky * f_hat

    # For even nx / ny, zero out the Nyquist frequency derivative to preserve reality and avoid artifacts
    if nx % 2 == 0:
        f_hat_x[..., nx // 2, :] = 0.0
    if ny % 2 == 0:
        f_hat_y[..., :, ny // 2] = 0.0

    df_dx = torch.fft.irfft2(f_hat_x, s=(nx, ny), dim=(-2, -1))
    df_dy = torch.fft.irfft2(f_hat_y, s=(nx, ny), dim=(-2, -1))

    return df_dx.to(dtype=orig_dtype), df_dy.to(dtype=orig_dtype)


# Explicit alias for semantic clarity
spectral_grad_xy = spectral_grad_2d


def compute_vorticity(
    u: torch.Tensor,
    v: torch.Tensor,
    domain_size: Tuple[float, float] = (1.0, 2.0),
) -> torch.Tensor:
    """Compute 2D vorticity: omega = dv/dx - du/dy.

    Args:
        u: Horizontal velocity along x (channel 0), shape (..., Nx, Ny).
        v: Vertical velocity along y (channel 1), shape (..., Nx, Ny).
        domain_size: (Lx, Ly), defaults to (1.0, 2.0).

    Returns:
        omega: Vorticity scalar field, shape (..., Nx, Ny).
    """
    _, du_dy = spectral_grad_2d(u, domain_size=domain_size)
    dv_dx, _ = spectral_grad_2d(v, domain_size=domain_size)
    return dv_dx - du_dy


def compute_divergence(
    u: torch.Tensor,
    v: torch.Tensor,
    domain_size: Tuple[float, float] = (1.0, 2.0),
) -> torch.Tensor:
    """Compute 2D divergence: div = du/dx + dv/dy.

    Args:
        u: Horizontal velocity along x (channel 0), shape (..., Nx, Ny).
        v: Vertical velocity along y (channel 1), shape (..., Nx, Ny).
        domain_size: (Lx, Ly), defaults to (1.0, 2.0).

    Returns:
        div: Divergence field, shape (..., Nx, Ny). Should be ~0 for incompressible flow.
    """
    du_dx, _ = spectral_grad_2d(u, domain_size=domain_size)
    _, dv_dy = spectral_grad_2d(v, domain_size=domain_size)
    return du_dx + dv_dy


def project_zero_mean_pressure(p: torch.Tensor) -> torch.Tensor:
    """Project pressure field to zero spatial mean gauge: p <- p - mean(p).

    Args:
        p: Pressure field, shape (..., Nx, Ny).

    Returns:
        Zero-mean normalized pressure.
    """
    mean_p = p.mean(dim=(-2, -1), keepdim=True)
    return p - mean_p


def compute_kinetic_energy(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute mean kinetic energy density: E_k = 0.5 * mean(u^2 + v^2).

    Args:
        u: Horizontal velocity, shape (..., Nx, Ny).
        v: Vertical velocity, shape (..., Nx, Ny).

    Returns:
        E_k: Scalar or shape (...) tensor.
    """
    return 0.5 * torch.mean(u**2 + v**2, dim=(-2, -1))


def compute_enstrophy(omega: torch.Tensor) -> torch.Tensor:
    """Compute mean enstrophy density: Omega = 0.5 * mean(omega^2).

    Args:
        omega: Vorticity field, shape (..., Nx, Ny).

    Returns:
        Enstrophy: Scalar or shape (...) tensor.
    """
    return 0.5 * torch.mean(omega**2, dim=(-2, -1))


def compute_laplacian_2d(
    field: torch.Tensor,
    domain_size: Tuple[float, float] = (1.0, 2.0),
) -> torch.Tensor:
    """Compute 2D spatial Laplacian: laplacian = d^2f/dx^2 + d^2f/dy^2 via RFFT.

    Args:
        field: Tensor of shape (..., Nx, Ny), real-valued.
        domain_size: (Lx, Ly) extent of domain. Defaults to (1.0, 2.0).

    Returns:
        laplacian: Tensor of same shape as field.
    """
    nx, ny = field.shape[-2], field.shape[-1]
    lx, ly = domain_size
    device = field.device
    orig_dtype = field.dtype
    calc_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    field_calc = field.to(dtype=calc_dtype)

    kx_base, ky_base = get_wavenumbers(nx, ny, lx, ly, device, calc_dtype)
    if field.ndim > 2:
        kx = kx_base.view(*([1] * (field.ndim - 2)), nx, 1)
        ky = ky_base.view(*([1] * (field.ndim - 2)), 1, ny // 2 + 1)
    else:
        kx, ky = kx_base, ky_base

    # -(kx^2 + ky^2)
    k_sq = kx**2 + ky**2

    f_hat = torch.fft.rfft2(field_calc, dim=(-2, -1))
    f_hat_lap = -k_sq * f_hat

    # Zero Nyquist frequencies if even
    if nx % 2 == 0:
        f_hat_lap[..., nx // 2, :] = 0.0
    if ny % 2 == 0:
        f_hat_lap[..., :, ny // 2] = 0.0

    laplacian = torch.fft.irfft2(f_hat_lap, s=(nx, ny), dim=(-2, -1))
    return laplacian.to(dtype=orig_dtype)


