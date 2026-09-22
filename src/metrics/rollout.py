"""Rollout evaluation pipeline across multiple future horizons (h in {1, 5, 10, 20, 30})."""

from typing import Dict, List, Optional
import torch
from src.metrics.field import evaluate_field_metrics
from src.metrics.tracer import compute_tracer_metrics
from src.utils.fft_derivatives import (
    compute_divergence,
    compute_enstrophy,
    compute_kinetic_energy,
    compute_vorticity,
)


from src.metrics.spectral import compute_spectral_error


def evaluate_rollout_trajectory(
    pred_trajectory: torch.Tensor,
    target_trajectory: torch.Tensor,
    evaluation_steps: Optional[List[int]] = None,
    domain_size: tuple = (1.0, 2.0),
) -> Dict[str, Dict[str, float]]:
    """Evaluates multi-step predicted trajectory against target trajectory."""
    total_steps = pred_trajectory.shape[1]
    if evaluation_steps is None:
        evaluation_steps = [s for s in [1, 5, 10, 20, 30] if s <= total_steps]

    results: Dict[str, Dict[str, float]] = {}

    for step in evaluation_steps:
        idx = step - 1
        if idx >= total_steps:
            continue

        p = pred_trajectory[:, idx]  # (B, C, Nx, Ny)
        t = target_trajectory[:, idx]

        step_res = {}
        # 1. Field errors
        field_metrics = evaluate_field_metrics(p, t)
        step_res.update(field_metrics)

        # 2. Tracer consistency
        tracer_metrics = compute_tracer_metrics(p[:, 3], t[:, 3])
        step_res.update(tracer_metrics)

        # 3. Divergence
        div_pred = compute_divergence(p[:, 0], p[:, 1], domain_size=domain_size)
        step_res["div_rmse"] = float(torch.sqrt(torch.mean(div_pred**2)).item())
        step_res["div_max"] = float(torch.max(torch.abs(div_pred)).item())

        # 4. Kinetic Energy evolution error
        ke_pred = compute_kinetic_energy(p[:, 0], p[:, 1])
        ke_targ = compute_kinetic_energy(t[:, 0], t[:, 1])
        ke_rel_err = torch.abs(ke_pred - ke_targ) / (torch.abs(ke_targ) + 1e-6)
        step_res["ke_pred"] = float(ke_pred.mean().item())
        step_res["ke_targ"] = float(ke_targ.mean().item())
        step_res["ke_rel_err"] = float(ke_rel_err.mean().item())

        # 5. Vorticity RMSE & Enstrophy
        vort_pred = compute_vorticity(p[:, 0], p[:, 1], domain_size=domain_size)
        vort_targ = compute_vorticity(t[:, 0], t[:, 1], domain_size=domain_size)
        step_res["vort_rmse"] = float(torch.sqrt(torch.mean((vort_pred - vort_targ)**2)).item())
        ens_pred = compute_enstrophy(vort_pred)
        ens_targ = compute_enstrophy(vort_targ)
        ens_rel_err = torch.abs(ens_pred - ens_targ) / (torch.abs(ens_targ) + 1e-6)
        step_res["ens_pred"] = float(ens_pred.mean().item())
        step_res["ens_targ"] = float(ens_targ.mean().item())
        step_res["enstrophy_rel_err"] = float(ens_rel_err.mean().item())

        # 6. Energy Spectrum MAE & sub-bands
        spec_res = compute_spectral_error(p[:, 0], p[:, 1], t[:, 0], t[:, 1], domain_size=domain_size)
        step_res["energy_spectrum_mae"] = float(spec_res["spec_err_total"])
        step_res["spec_err_low"] = float(spec_res["spec_err_low"])
        step_res["spec_err_mid"] = float(spec_res["spec_err_mid"])
        step_res["spec_err_high"] = float(spec_res["spec_err_high"])

        results[f"step_{step}"] = step_res

    return results

