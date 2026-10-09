"""Representation evaluation metrics for single-channel physical vorticity fields.

Provides four compact, decoupled evaluation metrics:
1. compute_relative_l2_error: Per-sample relative and absolute reconstruction error with zero-denominator handling.
2. compute_pointwise_variance_ratio: Ensemble spatial variance retention with degeneracy checks.
3. compute_pairwise_difference_error: Preservation of inter-sample differences across realizations.
4. compute_batched_radial_enstrophy_spectrum / compute_enstrophy_spectrum_ratio:
   Radial enstrophy spectrum and preservation ratio across spatial scales.

Note: This module strictly evaluates fields; it contains no synthetic data generation logic.
"""

from typing import Any, Dict, Optional, Tuple
import torch
from src.utils.fft_derivatives import get_radial_shell_indices


def compute_relative_l2_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> Dict[str, Any]:
    """Compute per-sample relative and absolute L2 error with zero-denominator protection.

    Args:
        pred: Predicted tensor of shape (..., H, W) or (..., 1, H, W).
        target: Ground-truth target tensor of shape (..., H, W) or (..., 1, H, W).
        eps: Threshold below which target norm is considered near-zero/degenerate.

    Returns:
        Dictionary containing:
            - absolute_l2: float, mean L2 error ||pred - target||_2
            - relative_l2: Optional[float], mean relative error ||pred - target||_2 / ||target||_2 (or None if degenerate)
            - valid_relative: bool, whether relative error is mathematically defined
            - reason: Optional[str], explanation if relative error is invalid
    """
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

    # Flatten spatial dims (-2, -1)
    diff = pred - target
    diff_l2 = torch.sqrt(torch.sum(diff**2, dim=(-2, -1))).flatten()
    target_l2 = torch.sqrt(torch.sum(target**2, dim=(-2, -1))).flatten()

    mean_abs_l2 = float(torch.mean(diff_l2).item())

    valid_mask = target_l2 >= eps
    num_valid = int(valid_mask.sum().item())
    total_samples = target_l2.numel()

    if num_valid == 0:
        return {
            "absolute_l2": mean_abs_l2,
            "relative_l2": None,
            "valid_relative": False,
            "reason": "near_zero_target_norm",
            "valid_samples": 0,
            "total_samples": total_samples,
        }

    valid_rel = diff_l2[valid_mask] / target_l2[valid_mask]
    mean_rel_l2 = float(torch.mean(valid_rel).item())

    return {
        "absolute_l2": mean_abs_l2,
        "relative_l2": mean_rel_l2,
        "valid_relative": True,
        "reason": None if num_valid == total_samples else "some_zero_target_norms_excluded",
        "valid_samples": num_valid,
        "total_samples": total_samples,
    }


def compute_pointwise_variance_ratio(
    pred_set: torch.Tensor,
    target_set: torch.Tensor,
    min_var_eps: float = 1e-7,
) -> Dict[str, Any]:
    """Compute spatial variance retention ratio between predicted and target ensembles.

    Evaluates whether sample-to-sample diversity is preserved or overly smoothed:
        VR = mean_spatial( Var_sample( pred ) ) / mean_spatial( Var_sample( target ) )

    Args:
        pred_set: Predicted ensemble tensor of shape (N, C, H, W) or (N, H, W), N >= 2.
        target_set: Target ensemble tensor of shape (N, C, H, W) or (N, H, W), N >= 2.
        min_var_eps: Numerical floor below which target variance is considered degenerate.

    Returns:
        Dictionary containing:
            - valid: bool, True if sample size >= 2 and target variance >= min_var_eps
            - variance_ratio: Optional[float], Var(pred) / Var(target)
            - mean_target_variance: float
            - mean_pred_variance: float
            - reason: Optional[str]
    """
    assert pred_set.shape == target_set.shape, f"Shape mismatch: {pred_set.shape} vs {target_set.shape}"
    n_samples = pred_set.shape[0]

    if n_samples < 2:
        return {
            "valid": False,
            "variance_ratio": None,
            "mean_target_variance": 0.0,
            "mean_pred_variance": 0.0,
            "reason": "insufficient_samples_less_than_2",
        }

    # Variance across ensemble/sample dimension (dim=0), unbiased estimate
    var_target = torch.var(target_set, dim=0, unbiased=True)
    var_pred = torch.var(pred_set, dim=0, unbiased=True)

    mean_target_var = float(torch.mean(var_target).item())
    mean_pred_var = float(torch.mean(var_pred).item())

    if mean_target_var < min_var_eps:
        return {
            "valid": False,
            "variance_ratio": None,
            "mean_target_variance": mean_target_var,
            "mean_pred_variance": mean_pred_var,
            "reason": "near_zero_target_variance",
        }

    ratio = mean_pred_var / mean_target_var
    return {
        "valid": True,
        "variance_ratio": float(ratio),
        "mean_target_variance": mean_target_var,
        "mean_pred_variance": mean_pred_var,
        "reason": None,
    }


def compute_pairwise_difference_error(
    pred_set: torch.Tensor,
    target_set: torch.Tensor,
    min_diff_eps: float = 1e-7,
) -> Dict[str, Any]:
    """Compute pairwise difference preservation error across distinct realization pairs.

    Directly evaluates whether fine-grained differences between different realizations
    are faithfully preserved after encoder-decoder reconstruction:
        For all i != j:
            delta_target = x_i - x_j
            delta_pred = x_hat_i - x_hat_j
            rel_err_ij = ||delta_pred - delta_target||_2 / ||delta_target||_2

    Args:
        pred_set: Predicted ensemble tensor of shape (N, C, H, W) or (N, H, W), N >= 2.
        target_set: Target ensemble tensor of shape (N, C, H, W) or (N, H, W), N >= 2.
        min_diff_eps: Threshold below which pair difference norm is considered degenerate.

    Returns:
        Dictionary containing:
            - valid: bool
            - pairwise_difference_error: Optional[float], mean relative difference error
            - mean_absolute_difference_error: float, mean ||delta_pred - delta_target||_2
            - valid_pairs: int
            - total_pairs: int, N * (N - 1)
            - reason: Optional[str]
    """
    assert pred_set.shape == target_set.shape, f"Shape mismatch: {pred_set.shape} vs {target_set.shape}"
    n_samples = pred_set.shape[0]

    if n_samples < 2:
        return {
            "valid": False,
            "pairwise_difference_error": None,
            "mean_absolute_difference_error": 0.0,
            "valid_pairs": 0,
            "total_pairs": 0,
            "reason": "insufficient_samples_less_than_2",
        }

    total_pairs = n_samples * (n_samples - 1)
    rel_errors = []
    abs_errors = []

    # Flatten spatial dims for straightforward L2 norm computation
    pred_flat = pred_set.view(n_samples, -1)
    target_flat = target_set.view(n_samples, -1)

    for i in range(n_samples):
        for j in range(n_samples):
            if i == j:
                continue  # Exclude self-pair

            delta_targ = target_flat[i] - target_flat[j]
            delta_pred = pred_flat[i] - pred_flat[j]
            diff_err = delta_pred - delta_targ

            norm_targ = float(torch.linalg.norm(delta_targ).item())
            norm_err = float(torch.linalg.norm(diff_err).item())

            abs_errors.append(norm_err)
            if norm_targ >= min_diff_eps:
                rel_errors.append(norm_err / norm_targ)

    mean_abs_err = float(sum(abs_errors) / len(abs_errors)) if abs_errors else 0.0
    valid_pairs = len(rel_errors)

    if valid_pairs == 0:
        return {
            "valid": False,
            "pairwise_difference_error": None,
            "mean_absolute_difference_error": mean_abs_err,
            "valid_pairs": 0,
            "total_pairs": total_pairs,
            "reason": "no_valid_non_degenerate_pairs",
        }

    mean_pde = float(sum(rel_errors) / valid_pairs)
    return {
        "valid": True,
        "pairwise_difference_error": mean_pde,
        "mean_absolute_difference_error": mean_abs_err,
        "valid_pairs": valid_pairs,
        "total_pairs": total_pairs,
        "reason": None if valid_pairs == total_pairs else "some_degenerate_pairs_excluded",
    }


def compute_batched_radial_enstrophy_spectrum(
    omega: torch.Tensor,
    domain_size: Tuple[float, float] = (1.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 1D shell-integrated enstrophy (vorticity square) power spectrum E_omega(k).

    Specifically for single-channel scalar vorticity field omega:
        E_omega(k) = 0.5 * shell_sum( |omega_hat(kx, ky)|^2 )
    Supporting arbitrary batch and ensemble dimensions.

    Args:
        omega: Vorticity tensor of shape (..., Nx, Ny).
        domain_size: (Lx, Ly) physical extent of domain. Defaults to (1.0, 1.0).

    Returns:
        k_bins: 1D wavenumber bins of shape (num_bins,).
        e_k: Shell-integrated enstrophy spectrum of shape (..., num_bins).
    """
    nx, ny = omega.shape[-2], omega.shape[-1]
    lx, ly = domain_size
    device = omega.device

    # Forward 2D Real-to-Complex FFT
    w_hat = torch.fft.rfft2(omega, dim=(-2, -1), norm="forward")

    # 2D spectral energy density for scalar vorticity: 0.5 * |w_hat|^2
    enstrophy_2d = 0.5 * (torch.abs(w_hat) ** 2)

    # Double non-DC/non-Nyquist columns to account for single-sided RFFT representation
    if ny > 2:
        enstrophy_2d[..., :, 1:-1] *= 2.0

    safe_indices, k_bins, num_bins = get_radial_shell_indices(nx, ny, lx, ly, device)

    flat_ens = enstrophy_2d.flatten(-2, -1)
    orig_shape = flat_ens.shape[:-1]
    b_total = flat_ens.numel() // flat_ens.shape[-1]
    flat_ens_2d = flat_ens.view(b_total, -1)

    out = torch.zeros(b_total, num_bins + 1, dtype=flat_ens.dtype, device=device)
    out.scatter_add_(1, safe_indices.unsqueeze(0).expand(b_total, -1), flat_ens_2d)
    e_k = out[:, :num_bins].view(*orig_shape, num_bins)

    return k_bins, e_k


def compute_enstrophy_spectrum_ratio(
    pred_set: torch.Tensor,
    target_set: torch.Tensor,
    domain_size: Tuple[float, float] = (1.0, 1.0),
    eps: float = 1e-7,
) -> Dict[str, Any]:
    """Compute enstrophy spectrum preservation ratio between predicted and target ensembles.

    Crucial protocol: Each member's spectrum is calculated individually before taking the
    ensemble average, preventing member-to-member phase cancellation from hiding fluctuations.

    Args:
        pred_set: Predicted ensemble tensor of shape (N, C, H, W) or (N, H, W).
        target_set: Target ensemble tensor of shape (N, C, H, W) or (N, H, W).
        domain_size: Physical extent (Lx, Ly), default (1.0, 1.0).
        eps: Threshold below which target spectral energy is considered zero.

    Returns:
        Dictionary containing:
            - k_bins: list of float wavenumbers
            - spectrum_target: list of float target energy
            - spectrum_pred: list of float pred energy
            - spectrum_ratio: list of Optional[float] (None where target energy < eps)
            - valid_bins_count: int
            - spurious_energy_in_zero_bins: float
    """
    assert pred_set.shape == target_set.shape, f"Shape mismatch: {pred_set.shape} vs {target_set.shape}"

    # Strip channel dimension if present and equal to 1
    p = pred_set.squeeze(1) if pred_set.ndim == 4 and pred_set.shape[1] == 1 else pred_set
    t = target_set.squeeze(1) if target_set.ndim == 4 and target_set.shape[1] == 1 else target_set

    # Compute spectrum for each individual ensemble member first
    k_bins, e_pred_ind = compute_batched_radial_enstrophy_spectrum(p, domain_size=domain_size)
    _, e_targ_ind = compute_batched_radial_enstrophy_spectrum(t, domain_size=domain_size)

    # Ensemble mean spectrum
    mean_e_pred = torch.mean(e_pred_ind.view(-1, k_bins.shape[0]), dim=0)
    mean_e_targ = torch.mean(e_targ_ind.view(-1, k_bins.shape[0]), dim=0)

    ratios = []
    spurious_energy = 0.0
    valid_count = 0

    for i in range(k_bins.shape[0]):
        e_t = float(mean_e_targ[i].item())
        e_p = float(mean_e_pred[i].item())
        if e_t >= eps:
            ratios.append(e_p / e_t)
            valid_count += 1
        else:
            ratios.append(None)
            spurious_energy += e_p

    return {
        "k_bins": [float(k.item()) for k in k_bins],
        "spectrum_target": [float(v.item()) for v in mean_e_targ],
        "spectrum_pred": [float(v.item()) for v in mean_e_pred],
        "spectrum_ratio": ratios,
        "valid_bins_count": valid_count,
        "total_bins": int(k_bins.shape[0]),
        "spurious_energy_in_zero_bins": float(spurious_energy),
    }
