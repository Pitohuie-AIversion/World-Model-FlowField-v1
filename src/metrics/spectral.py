"""Kinetic-energy spectrum metrics for the canonical (Nx, Ny) layout."""

from typing import Dict, Tuple
import torch
import torch.fft

from src.utils.physics_contract import SHEAR_FLOW_DOMAIN_SIZE_XY


def compute_radial_energy_spectrum(
    u: torch.Tensor,
    v: torch.Tensor,
    domain_size: Tuple[float, float] = SHEAR_FLOW_DOMAIN_SIZE_XY,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute a radially binned kinetic-energy spectrum for (..., Nx, Ny)."""
    nx, ny = u.shape[-2], u.shape[-1]
    lx, ly = domain_size
    device = u.device

    u_hat = torch.fft.rfft2(u, dim=(-2, -1), norm="forward")
    v_hat = torch.fft.rfft2(v, dim=(-2, -1), norm="forward")
    energy_2d = 0.5 * (torch.abs(u_hat) ** 2 + torch.abs(v_hat) ** 2)

    # rfft2 truncates the last (y) frequency axis.
    if energy_2d.shape[-1] > 2:
        energy_2d[..., :, 1:-1] *= 2.0

    kx = torch.fft.fftfreq(nx, d=lx / nx, device=device) * 2.0 * torch.pi
    ky = torch.fft.rfftfreq(ny, d=ly / ny, device=device) * 2.0 * torch.pi
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
    k_mag = torch.sqrt(kx_grid**2 + ky_grid**2)

    k_max = int(min(nx // 2, ny // 2))
    k_bins = torch.arange(0, k_max, dtype=torch.float32, device=device)
    e_k = torch.zeros(k_max, dtype=torch.float32, device=device)

    flat_k = k_mag.flatten()
    if energy_2d.ndim > 2:
        batch_dims = tuple(range(energy_2d.ndim - 2))
        flat_e = energy_2d.mean(dim=batch_dims).flatten()
    else:
        flat_e = energy_2d.flatten()

    k_indices = torch.clamp(torch.floor(flat_k).long(), 0, k_max - 1)
    e_k.index_add_(0, k_indices, flat_e)
    return k_bins, e_k


def compute_spectral_error(
    pred_u: torch.Tensor,
    pred_v: torch.Tensor,
    target_u: torch.Tensor,
    target_v: torch.Tensor,
    domain_size: Tuple[float, float] = SHEAR_FLOW_DOMAIN_SIZE_XY,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute relative log-spectrum errors."""
    _, e_pred = compute_radial_energy_spectrum(pred_u, pred_v, domain_size)
    _, e_target = compute_radial_energy_spectrum(target_u, target_v, domain_size)

    k_max = len(e_target)
    k_low = k_max // 3
    k_mid = 2 * (k_max // 3)
    log_err = torch.abs(torch.log10(e_pred + eps) - torch.log10(e_target + eps))

    return {
        "spec_err_total": float(torch.mean(log_err).item()),
        "spec_err_low": float(torch.mean(log_err[:k_low]).item()),
        "spec_err_mid": float(torch.mean(log_err[k_low:k_mid]).item()),
        "spec_err_high": float(torch.mean(log_err[k_mid:]).item()),
    }
