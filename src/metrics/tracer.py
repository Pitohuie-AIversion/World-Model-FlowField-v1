"""Metrics for passive tracer scalar transport and physical conservation."""

from typing import Dict
import torch


def compute_tracer_metrics(
    pred_s: torch.Tensor,
    target_s: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Evaluates passive tracer scalar physical consistency.

    Args:
        pred_s: Predicted tracer field (..., Ny, Nx).
        target_s: Target tracer field (..., Ny, Nx).

    Returns:
        Dictionary of tracer evaluation metrics.
    """
    # 1. Variance retention ratio
    var_pred = torch.var(pred_s, dim=(-2, -1), unbiased=False)
    var_target = torch.var(target_s, dim=(-2, -1), unbiased=False)
    var_retention = torch.mean(var_pred / (var_target + eps))

    # 2. Extreme value / Maximum principle violation
    # For passive scalar diffusion without source, min(s) and max(s) should be bounded by initial conditions
    s_min = torch.amin(target_s, dim=(-2, -1), keepdim=True)
    s_max = torch.amax(target_s, dim=(-2, -1), keepdim=True)

    out_of_bounds = (pred_s < (s_min - 1e-2)) | (pred_s > (s_max + 1e-2))
    out_of_bounds_rate = out_of_bounds.float().mean()

    # 3. Total mass / integral conservation
    mass_pred = torch.sum(pred_s, dim=(-2, -1))
    mass_target = torch.sum(target_s, dim=(-2, -1))
    mass_err = torch.mean(torch.abs(mass_pred - mass_target) / (torch.abs(mass_target) + eps))

    # 4. Mean conservation error
    mean_pred = torch.mean(pred_s, dim=(-2, -1))
    mean_target = torch.mean(target_s, dim=(-2, -1))
    mean_err = torch.mean(torch.abs(mean_pred - mean_target))

    return {
        "tracer_var_retention": float(var_retention.item()),
        "tracer_out_of_bounds_rate": float(out_of_bounds_rate.item()),
        "tracer_mass_error": float(mass_err.item()),
        "tracer_mean_err": float(mean_err.item()),
    }

