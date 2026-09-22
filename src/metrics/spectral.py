"""Kinetic energy spectrum and spectral error metrics for 2D flow fields."""

from typing import Dict, Tuple
import torch
import torch.fft


def compute_radial_energy_spectrum(
    u: torch.Tensor,
    v: torch.Tensor,
    domain_size: Tuple[float, float] = (1.0, 2.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 1D radially averaged kinetic energy spectrum E(k).

    Args:
        u: Horizontal velocity along x, shape (Nx, Ny) or (..., Nx, Ny).
        v: Vertical velocity along y, shape (Nx, Ny) or (..., Nx, Ny).
        domain_size: (Lx, Ly) extent of domain. Defaults to (1.0, 2.0).

    Returns:
        k_bins: 1D wavenumber bins.
        e_k: 1D kinetic energy spectrum.
    """
    nx, ny = u.shape[-2], u.shape[-1]
    lx, ly = domain_size
    device = u.device

    u_hat = torch.fft.rfft2(u, dim=(-2, -1), norm="forward")
    v_hat = torch.fft.rfft2(v, dim=(-2, -1), norm="forward")

    # Kinetic energy density in Fourier space: 0.5 * (|u_hat|^2 + |v_hat|^2)
    # RFFT omits negative frequencies along y (dim -1), so account for double-counting positive ky != 0
    energy_2d = 0.5 * (torch.abs(u_hat) ** 2 + torch.abs(v_hat) ** 2)
    energy_2d[..., :, 1:-1] *= 2.0

    # 2D Wavenumbers: kx along dim -2 (fft), ky along dim -1 (rfft)
    kx = torch.fft.fftfreq(nx, d=lx / nx, device=device) * 2.0 * torch.pi
    ky = torch.fft.rfftfreq(ny, d=ly / ny, device=device) * 2.0 * torch.pi
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
    k_mag = torch.sqrt(kx_grid**2 + ky_grid**2)

    # Binning by integer wavenumber
    k_max = int(min(nx // 2, ny // 2))
    k_bins = torch.arange(0, k_max, dtype=torch.float32, device=device)
    e_k = torch.zeros(k_max, dtype=torch.float32, device=device)

    flat_k = k_mag.flatten()
    flat_e = energy_2d.mean(dim=tuple(range(energy_2d.ndim - 2))).flatten() if energy_2d.ndim > 2 else energy_2d.flatten()

    k_indices = torch.clamp(torch.floor(flat_k).long(), 0, k_max - 1)
    e_k.index_add_(0, k_indices, flat_e)

    return k_bins, e_k


def compute_spectral_error(
    pred_u: torch.Tensor,
    pred_v: torch.Tensor,
    target_u: torch.Tensor,
    target_v: torch.Tensor,
    domain_size: Tuple[float, float] = (1.0, 2.0),
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute relative spectrum error over low, mid, and high wavenumber bands."""
    _, e_pred = compute_radial_energy_spectrum(pred_u, pred_v, domain_size)
    _, e_target = compute_radial_energy_spectrum(target_u, target_v, domain_size)

    k_max = len(e_target)
    k_low = k_max // 3
    k_mid = 2 * (k_max // 3)

    # Relative L1 error in log energy spectrum
    log_err = torch.abs(torch.log10(e_pred + eps) - torch.log10(e_target + eps))

    return {
        "spec_err_total": float(torch.mean(log_err).item()),
        "spec_err_low": float(torch.mean(log_err[:k_low]).item()),
        "spec_err_mid": float(torch.mean(log_err[k_low:k_mid]).item()),
        "spec_err_high": float(torch.mean(log_err[k_mid:]).item()),
    }
