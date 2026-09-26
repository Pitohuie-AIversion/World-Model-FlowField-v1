"""ProbLatent-R1 Phase 3: Comprehensive Probabilistic Evaluation Pipeline.

Executes a two-stage evaluation protocol comparing:
- D0: Deterministic reference baseline (mean prediction)
- G0: Homoscedastic baseline (fixed channel variance from training residual second moments)
- G1: Heteroscedastic model (input-conditional variance head trained in Phase 2)

Protocol:
1. Pre-flight Gate Verification:
   - Cryptographically verifies D0, G0 stats, G0 checkpoint, split, normalizer, and G1 variance head hashes.
   - Re-evaluates full validation set NLL on G1, verifying exact reproduction of Phase 2
     validation NLL (-0.206643 nats/latent element within 1e-4 tolerance).
   - Strict fail-closed verification: rejects NaN/Inf in all inputs, targets, model outputs, and losses.
2. Step 1: Single-Step Probabilistic Evaluation on Independent Test Set:
   - Evaluates on unseen test trajectories (grouped split, 5 trajectory entries, 4 initial condition clusters).
   - Metrics: Gaussian NLL (nats/latent element), Continuous Ranked Probability Score (CRPS),
     Prediction Interval Coverage Probability (PICP), Mean Prediction Interval Width (MPIW),
     signed calibration error (PICP - nominal), and absolute calibration error (|PICP - nominal|)
     at nominal 50%, 80%, 90%, 95% confidence levels.
   - Verifies deterministic mean parity (mu_D0 == mu_G0 == mu_G1).
   - Generates per-channel (64 channels) and per-trajectory breakdowns with cryptographic/path identity
     (source_file, traj_idx, start_t, cluster_id), failing closed if identity fields are missing.
3. Step 2: 30-Step Autonomous Autoregressive Rollout Evaluation (H=30, K=32):
   - Multi-trajectory autoregressive simulation without ground-truth feedback.
   - Complete test window coverage (105 windows under T=200, L=4, H=30, stride=8 across 5 test trajectories).
   - Applies zero-mean gauge to physical pressure channel before metric computation.
   - Non-parametric empirical quantiles in physical space from K=32 decoded samples (80% and 90% coverage and width).
   - Ensemble Mean VRMSE vs D0 vs G0.
   - Proper Spread-Skill alignment:
     RMS spread = sqrt(mean(var_k)) with Bessel's correction (ddof=1)
     RMSE = sqrt(mean((p_mean - target)^2))
     Spread-Skill Ratio = RMS spread / RMSE (raw and finite-K adjusted with sqrt((K+1)/K)).
   - Physical invariant checks: RMS divergence, vorticity RMSE, and radially integrated kinetic energy spectra E(k)
     for individual sample trajectories (averaged over all K=32 members) vs ensemble mean vs ground truth.
"""

from datetime import datetime, timezone
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import create_flow_dataloaders
from src.data.normalization import FieldNormalizer
from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.probabilistic_latent_dynamics import (
    VarianceHead2D,
    gaussian_nll_latent_loss,
    compute_g1_bias_init_from_g0,
)
from src.metrics.field import compute_vrmse, evaluate_field_metrics
from src.metrics.spectral import compute_radial_energy_spectrum
from src.utils.fft_derivatives import compute_divergence, compute_vorticity
from src.utils.checkpoint import resolve_spatial_pos_config, strip_compiled_prefix
from src.utils.provenance import (
    compute_file_sha256,
    compute_split_hash_from_file,
    compute_normalizer_hash,
    get_git_commit,
    is_git_dirty,
    hash_matches,
)


def apply_pressure_gauge(field: torch.Tensor, pressure_channel: int = 2) -> torch.Tensor:
    """Apply zero-mean pressure gauge normalization: p = p - mean(p) over spatial dimensions."""
    field = field.clone()
    p = field[..., pressure_channel, :, :]
    field[..., pressure_channel, :, :] = p - p.mean(dim=(-2, -1), keepdim=True)
    return field


def compute_gaussian_crps(
    mu: torch.Tensor,
    target: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute exact analytical Continuous Ranked Probability Score (CRPS) for Gaussian distribution.

    For Y ~ N(mu, sigma^2) and observation y:
        z = (y - mu) / sigma
        CRPS(y, mu, sigma^2) = sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]

    When sigma -> 0, CRPS smoothly converges to MAE: |y - mu|.

    Args:
        mu: Mean tensor of shape (..., ).
        target: Ground truth observation tensor of matching shape.
        var: Variance tensor of matching shape (must be non-negative).
        eps: Small floor for numerical stability.

    Returns:
        Tensor of CRPS with matching shape.
    """
    sigma = torch.sqrt(torch.clamp(var, min=eps))
    z = (target - mu) / sigma
    phi_z = torch.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    phi_cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = sigma * (z * (2.0 * phi_cdf - 1.0) + 2.0 * phi_z - 1.0 / math.sqrt(math.pi))
    return crps


def compute_prediction_intervals(
    mu: torch.Tensor,
    target: torch.Tensor,
    var: torch.Tensor,
    nominal_levels: Tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
    eps: float = 1e-8,
) -> Dict[str, Dict[str, float]]:
    """Compute empirical coverage (PICP), sharpness (MPIW), and calibration errors for central intervals."""
    sigma = torch.sqrt(torch.clamp(var, min=eps))
    res = {}
    for level in nominal_levels:
        alpha = 1.0 - level
        # Standard normal quantile for central 1 - alpha interval: z_{1 - alpha/2}
        z_crit = math.sqrt(2.0) * torch.erfinv(torch.tensor(1.0 - alpha)).item()
        lower = mu - z_crit * sigma
        upper = mu + z_crit * sigma
        inside = (target >= lower) & (target <= upper)
        picp = float(inside.float().mean().item())
        mpiw = float((upper - lower).mean().item())
        signed_cal_err = float(picp - level)
        abs_cal_err = float(abs(picp - level))
        res[f"{int(round(level * 100))}"] = {
            "nominal": float(level),
            "z_critical": float(z_crit),
            "picp": picp,
            "mpiw": mpiw,
            "signed_calibration_error": signed_cal_err,
            "absolute_calibration_error": abs_cal_err,
        }
    return res


def compute_ensemble_spread_skill(
    samps: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """Compute mathematically rigorous Ensemble Spread-Skill metrics.

    Args:
        samps: Ensemble samples tensor of shape (K, C=4, Ny, Nx).
        target: Ground truth physical state tensor of shape (C=4, Ny, Nx).
        eps: Small floor to prevent division by zero.

    Returns:
        Dict containing RMS spread, RMSE, raw Spread-Skill Ratio, and finite-K adjusted SSR.
    """
    K = samps.shape[0]
    if K < 2:
        raise ValueError(f"Ensemble size K={K} must be >= 2 for variance computation.")

    p_mean = samps.mean(dim=0)  # (4, Ny, Nx)
    # Sample variance along K with Bessel's correction (ddof=1)
    var_k = samps.var(dim=0, unbiased=True)  # (4, Ny, Nx)

    # 1. Velocity vector field (u=0, v=1)
    # Variance is sum of u and v variances
    var_vel = (var_k[0] + var_k[1]).mean()
    rms_spread_vel = float(torch.sqrt(var_vel).item())
    finite_k_factor = math.sqrt((K + 1.0) / K)
    rms_spread_adj_vel = float(rms_spread_vel * finite_k_factor)
    rmse_vel = float(torch.sqrt(((p_mean[0:2] - target[0:2]) ** 2).mean() * 2.0).item())
    vrmse = float(compute_vrmse(p_mean, target).item())
    ssr_vel = float(rms_spread_vel / (rmse_vel + eps))
    ssr_adj_vel = float(rms_spread_adj_vel / (rmse_vel + eps))

    # 2. Per-variable spread and skill (u=0, v=1, p=2, s=3)
    channel_names = ["u", "v", "p", "s"]
    per_channel = {}
    for c_idx, name in enumerate(channel_names):
        mean_var_c = var_k[c_idx].mean()
        spread_c = float(torch.sqrt(mean_var_c).item())
        spread_adj_c = float(spread_c * finite_k_factor)
        rmse_c = float(torch.sqrt(((p_mean[c_idx] - target[c_idx]) ** 2).mean()).item())
        ssr_c = float(spread_c / (rmse_c + eps))
        ssr_adj_c = float(spread_adj_c / (rmse_c + eps))
        per_channel[name] = {
            "rms_spread": spread_c,
            "finite_k_adjusted_spread": spread_adj_c,
            "rmse": rmse_c,
            "spread_skill_ratio": ssr_c,
            "finite_k_adjusted_ssr": ssr_adj_c,
        }

    return {
        "K": K,
        "degrees_of_freedom": "ddof=1 (Bessel corrected unbiased sample variance)",
        "finite_k_inflation_factor": finite_k_factor,
        "velocity": {
            "vrmse": vrmse,
            "rmse_velocity": rmse_vel,
            "rms_spread": rms_spread_vel,
            "finite_k_adjusted_spread": rms_spread_adj_vel,
            "spread_skill_ratio": ssr_vel,
            "finite_k_adjusted_ssr": ssr_adj_vel,
        },
        "per_variable": per_channel,
    }


def compute_physical_quantiles_coverage(
    samps: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, Any]:
    """Compute empirical quantiles and interval coverage in physical space from K samples.

    Args:
        samps: Ensemble samples tensor of shape (K, C=4, Ny, Nx).
        target: Ground truth physical state tensor of shape (C=4, Ny, Nx).

    Returns:
        Dict containing 80% and 90% physical space interval coverage (PICP) and width (MPIW).
    """
    quantiles = torch.quantile(
        samps,
        q=torch.tensor([0.05, 0.10, 0.50, 0.90, 0.95], device=samps.device),
        dim=0,
    )
    q05, q10, q50, q90, q95 = quantiles[0], quantiles[1], quantiles[2], quantiles[3], quantiles[4]

    # Overall (across all fields)
    in_80 = (target >= q10) & (target <= q90)
    in_90 = (target >= q05) & (target <= q95)
    picp_80 = float(in_80.float().mean().item())
    picp_90 = float(in_90.float().mean().item())
    mpiw_80 = float((q90 - q10).mean().item())
    mpiw_90 = float((q95 - q05).mean().item())

    # Velocity only (u and v)
    picp_80_vel = float(in_80[0:2].float().mean().item())
    picp_90_vel = float(in_90[0:2].float().mean().item())
    mpiw_80_vel = float((q90[0:2] - q10[0:2]).mean().item())
    mpiw_90_vel = float((q95[0:2] - q05[0:2]).mean().item())

    return {
        "overall": {
            "picp_80": picp_80,
            "picp_90": picp_90,
            "mpiw_80": mpiw_80,
            "mpiw_90": mpiw_90,
            "cal_error_80": float(abs(picp_80 - 0.80)),
            "cal_error_90": float(abs(picp_90 - 0.90)),
        },
        "velocity": {
            "picp_80": picp_80_vel,
            "picp_90": picp_90_vel,
            "mpiw_80": mpiw_80_vel,
            "mpiw_90": mpiw_90_vel,
            "cal_error_80": float(abs(picp_80_vel - 0.80)),
            "cal_error_90": float(abs(picp_90_vel - 0.90)),
        },
    }


def verify_phase3_preflight_gate(
    d0_checkpoint_path: str,
    g0_checkpoint_path: str,
    stats_path: str,
    normalizer_path: str,
    split_file: str,
    phase2_record_path: str,
    g1_variance_head_path: str,
    forecaster: LatentForecaster,
    val_loader: DataLoader,
    device: torch.device,
    tolerance: float = 1e-4,
) -> float:
    """Pre-flight verification gate validating cryptographic contracts and reproducing val NLL.

    Strict fail-closed governance:
    - Verifies cryptographic hashes of D0, G0 stats, G0 checkpoint, G1 variance head, split, and normalizer.
    - Re-evaluates validation NLL, asserting finite values on all batches, inputs, targets, variances, and losses.
    - Fails closed if any hash mismatches, if validation loader is empty, or if reproduced NLL drifts.
    """
    if not os.path.exists(phase2_record_path):
        raise FileNotFoundError(f"Phase 2 training record not found: {phase2_record_path}")
    with open(phase2_record_path, "r") as f:
        rec = json.load(f)

    # 1. Cryptographic hashes of all constituent files
    d0_sha = compute_file_sha256(d0_checkpoint_path)
    expected_d0 = rec["cryptographic_bindings"]["d0_checkpoint"]["sha256"]
    if d0_sha != expected_d0:
        raise ValueError(f"D0 SHA mismatch: expected {expected_d0}, got {d0_sha}")

    stats_sha = compute_file_sha256(stats_path)
    expected_stats = rec["cryptographic_bindings"]["stats_file"]["sha256"]
    if stats_sha != expected_stats:
        raise ValueError(f"G0 stats SHA mismatch: expected {expected_stats}, got {stats_sha}")

    g0_sha = compute_file_sha256(g0_checkpoint_path)
    expected_g0 = rec["artifacts"]["g0_baseline_initialization"]["sha256"]
    if g0_sha != expected_g0:
        raise ValueError(f"G0 baseline checkpoint SHA mismatch: expected {expected_g0}, got {g0_sha}")

    runtime_split = compute_split_hash_from_file(split_file)
    expected_split = rec["cryptographic_bindings"]["data_protocol"]["split_hash"]
    if not hash_matches(expected_split, runtime_split, min_prefix_len=16):
        raise ValueError(f"Split hash mismatch: expected {expected_split}, got {runtime_split}")

    norm_obj = FieldNormalizer()
    norm_obj.load_state_dict(torch.load(normalizer_path, weights_only=True, map_location="cpu"))
    runtime_norm = compute_normalizer_hash(norm_obj)
    expected_norm = rec["cryptographic_bindings"]["data_protocol"]["normalizer_hash"]
    if not hash_matches(expected_norm, runtime_norm, min_prefix_len=16):
        raise ValueError(f"Normalizer hash mismatch: expected {expected_norm}, got {runtime_norm}")

    g1_sha = compute_file_sha256(g1_variance_head_path)
    expected_g1 = rec["artifacts"]["best_g1_variance_head"]["sha256"]
    if g1_sha != expected_g1:
        raise ValueError(f"G1 variance head SHA mismatch: expected {expected_g1}, got {g1_sha}")

    # 2. Gate assertion: reproduce exact validation NLL on G1 with strict finiteness check
    if len(val_loader) == 0:
        raise ValueError("Validation DataLoader is empty. Pre-flight gate fails closed.")

    forecaster.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            q_hist = batch["history"].to(device)
            q_next = batch["future"][:, 0:1].to(device)
            re = batch["re"].to(device) if "re" in batch else None
            sc = batch["sc"].to(device) if "sc" in batch else None

            if not torch.isfinite(q_hist).all() or not torch.isfinite(q_next).all():
                raise ValueError(f"Validation batch {batch_idx} contains non-finite input tensors.")

            mu, var = forecaster.predict_distribution_single_step(q_hist, re=re, sc=sc)
            if not torch.isfinite(mu).all() or not torch.isfinite(var).all():
                raise ValueError(f"Validation batch {batch_idx} produced non-finite distribution (mu or var).")
            if (var <= 0.0).any():
                raise ValueError(f"Validation batch {batch_idx} contains non-positive variance: min={var.min().item()}.")

            target_z = forecaster.encoder(q_next)
            if not torch.isfinite(target_z).all():
                raise ValueError(f"Validation batch {batch_idx} produced non-finite latent target_z.")

            loss = gaussian_nll_latent_loss(mu=mu, target=target_z, variance=var)
            if not math.isfinite(loss.item()):
                raise ValueError(f"Validation batch {batch_idx} yielded non-finite loss: {loss.item()}.")

            b_tokens = target_z.numel()
            total_loss += loss.item() * b_tokens
            total_tokens += b_tokens

    if total_tokens == 0:
        raise ValueError("Total validation tokens evaluated is zero.")

    reproduced_val_nll = total_loss / total_tokens
    if not math.isfinite(reproduced_val_nll):
        raise RuntimeError(f"Recomputed validation NLL is non-finite: {reproduced_val_nll}. Fails closed.")

    expected_val_nll = rec["training_summary"]["best_val_nll"]
    if not math.isfinite(expected_val_nll):
        raise RuntimeError(f"Expected Phase 2 validation NLL is non-finite: {expected_val_nll}.")

    error = abs(reproduced_val_nll - expected_val_nll)
    print(f"\n[Pre-flight Gate] Recomputed Validation NLL: {reproduced_val_nll:.6f}")
    print(f"                 Expected Phase 2 Val NLL:  {expected_val_nll:.6f}")
    print(f"                 Absolute Difference:       {error:.8e} (Tolerance: {tolerance})")

    if not math.isfinite(error) or error > tolerance:
        raise RuntimeError(
            f"Pre-flight gate failed: recomputed val NLL ({reproduced_val_nll}) differs from "
            f"Phase 2 recorded val NLL ({expected_val_nll}) by {error} > {tolerance}. Fails closed."
        )

    print("  --> [GATE PASSED] Cryptographic provenance and validation NLL reproduction confirmed.\n")
    return float(reproduced_val_nll)


def evaluate_single_step_probability(
    forecaster: LatentForecaster,
    g0_head: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    nominal_levels: Tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
) -> Dict[str, Any]:
    """Execute Step 1: Single-step probabilistic evaluation on unseen test set with strict identity binding."""
    forecaster.eval()
    g0_head.eval()

    total_tokens = 0
    total_windows = 0
    batches_evaluated = 0
    latent_spatial_shape: Optional[List[int]] = None

    # Accumulators for overall metrics
    d0_abs_err_sum = 0.0
    d0_sq_err_sum = 0.0

    g0_nll_sum = 0.0
    g1_nll_sum = 0.0

    g0_crps_sum = 0.0
    g1_crps_sum = 0.0

    g0_var_sum = 0.0
    g1_var_sum = 0.0
    g0_min_var = float("inf")
    g0_max_var = float("-inf")
    g1_min_var = float("inf")
    g1_max_var = float("-inf")

    # Interval accumulators
    g0_interval_counts = {f"{int(round(lvl * 100))}": 0 for lvl in nominal_levels}
    g1_interval_counts = {f"{int(round(lvl * 100))}": 0 for lvl in nominal_levels}
    g0_interval_width_sum = {f"{int(round(lvl * 100))}": 0.0 for lvl in nominal_levels}
    g1_interval_width_sum = {f"{int(round(lvl * 100))}": 0.0 for lvl in nominal_levels}

    # Per-channel accumulators (64 channels)
    num_channels = 64
    ch_tokens = [0] * num_channels
    ch_g0_nll = [0.0] * num_channels
    ch_g1_nll = [0.0] * num_channels
    ch_g0_crps = [0.0] * num_channels
    ch_g1_crps = [0.0] * num_channels
    ch_g0_p90_count = [0] * num_channels
    ch_g1_p90_count = [0] * num_channels
    ch_g0_var_sum = [0.0] * num_channels
    ch_g1_var_sum = [0.0] * num_channels

    # Per-trajectory accumulators keyed by (source_file, traj_idx)
    traj_records: Dict[str, Dict[str, Any]] = {}
    max_mean_discrepancy = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batches_evaluated += 1

            # Assert required identity fields exist; fail closed if missing
            for id_field in ("source_file", "traj_idx", "start_t", "cluster_id"):
                if id_field not in batch:
                    raise ValueError(
                        f"Batch {batch_idx} missing required identity field '{id_field}'. "
                        f"Fails closed: refuse to default trajectory identity."
                    )

            q_hist = batch["history"].to(device)
            q_next = batch["future"][:, 0:1].to(device)
            re = batch["re"].to(device) if "re" in batch else None
            sc = batch["sc"].to(device) if "sc" in batch else None

            source_files: List[str] = batch["source_file"]
            traj_indices: torch.Tensor = batch["traj_idx"]
            start_times: torch.Tensor = batch["start_t"]
            cluster_ids: torch.Tensor = batch["cluster_id"]

            if not torch.isfinite(q_hist).all() or not torch.isfinite(q_next).all():
                raise ValueError(f"Test batch {batch_idx} contains non-finite input fields.")

            b = q_hist.shape[0]
            total_windows += b

            # Ground truth latent target
            target_z = forecaster.encoder(q_next)  # (B, 1, 64, Hz, Wz)
            if not torch.isfinite(target_z).all():
                raise ValueError(f"Test batch {batch_idx} target_z contains NaN/Inf.")

            if latent_spatial_shape is None:
                latent_spatial_shape = list(target_z.shape)

            # Predictions: G1 uses forecaster attached variance_head; G0 overrides with g0_head
            mu_g1, var_g1 = forecaster.predict_distribution_single_step(q_hist, re=re, sc=sc)
            mu_g0, var_g0 = forecaster.predict_distribution_single_step(q_hist, re=re, sc=sc, variance_head=g0_head)

            if not torch.isfinite(mu_g1).all() or not torch.isfinite(var_g1).all():
                raise ValueError(f"Test batch {batch_idx} G1 produced non-finite mu or var.")
            if not torch.isfinite(mu_g0).all() or not torch.isfinite(var_g0).all():
                raise ValueError(f"Test batch {batch_idx} G0 produced non-finite mu or var.")
            if (var_g1 <= 0.0).any() or (var_g0 <= 0.0).any():
                raise ValueError(f"Test batch {batch_idx} produced non-positive variance.")

            # Assert deterministic mean parity: mu_D0 == mu_G0 == mu_G1
            diff_mu = (mu_g1 - mu_g0).abs().max().item()
            if diff_mu > 1e-6:
                raise AssertionError(f"Mean parity check failed: max |mu_G1 - mu_G0| = {diff_mu} > 1e-6.")
            max_mean_discrepancy = max(max_mean_discrepancy, diff_mu)

            b_tokens = target_z.numel()
            total_tokens += b_tokens

            # D0 deterministic metrics
            d0_err = target_z - mu_g1
            d0_abs_err_sum += d0_err.abs().sum().item()
            d0_sq_err_sum += (d0_err ** 2).sum().item()

            # Gaussian NLL
            loss_g0 = gaussian_nll_latent_loss(mu=mu_g0, target=target_z, variance=var_g0)
            loss_g1 = gaussian_nll_latent_loss(mu=mu_g1, target=target_z, variance=var_g1)
            g0_nll_sum += loss_g0.item() * b_tokens
            g1_nll_sum += loss_g1.item() * b_tokens

            # CRPS
            crps_g0_elem = compute_gaussian_crps(mu=mu_g0, target=target_z, var=var_g0)
            crps_g1_elem = compute_gaussian_crps(mu=mu_g1, target=target_z, var=var_g1)
            g0_crps_sum += crps_g0_elem.sum().item()
            g1_crps_sum += crps_g1_elem.sum().item()

            # Variance stats
            g0_var_sum += var_g0.sum().item()
            g1_var_sum += var_g1.sum().item()
            g0_min_var = min(g0_min_var, var_g0.min().item())
            g0_max_var = max(g0_max_var, var_g0.max().item())
            g1_min_var = min(g1_min_var, var_g1.min().item())
            g1_max_var = max(g1_max_var, var_g1.max().item())

            # Prediction Intervals
            sigma_g0 = torch.sqrt(torch.clamp(var_g0, min=1e-8))
            sigma_g1 = torch.sqrt(torch.clamp(var_g1, min=1e-8))

            for lvl in nominal_levels:
                key = f"{int(round(lvl * 100))}"
                alpha = 1.0 - lvl
                z_crit = math.sqrt(2.0) * torch.erfinv(torch.tensor(1.0 - alpha)).item()

                in_g0 = (target_z >= mu_g0 - z_crit * sigma_g0) & (target_z <= mu_g0 + z_crit * sigma_g0)
                in_g1 = (target_z >= mu_g1 - z_crit * sigma_g1) & (target_z <= mu_g1 + z_crit * sigma_g1)

                g0_interval_counts[key] += in_g0.sum().item()
                g1_interval_counts[key] += in_g1.sum().item()

                g0_interval_width_sum[key] += (2.0 * z_crit * sigma_g0).sum().item()
                g1_interval_width_sum[key] += (2.0 * z_crit * sigma_g1).sum().item()

            # Channel-wise accumulation
            z_crit_90 = math.sqrt(2.0) * torch.erfinv(torch.tensor(0.90)).item()
            in_g0_90 = (target_z >= mu_g0 - z_crit_90 * sigma_g0) & (target_z <= mu_g0 + z_crit_90 * sigma_g0)
            in_g1_90 = (target_z >= mu_g1 - z_crit_90 * sigma_g1) & (target_z <= mu_g1 + z_crit_90 * sigma_g1)

            for c in range(num_channels):
                t_c = target_z[:, :, c]
                tokens_c = t_c.numel()
                ch_tokens[c] += tokens_c

                loss_g0_c = F.gaussian_nll_loss(mu_g0[:, :, c], t_c, var_g0[:, :, c], reduction="sum").item()
                loss_g1_c = F.gaussian_nll_loss(mu_g1[:, :, c], t_c, var_g1[:, :, c], reduction="sum").item()
                ch_g0_nll[c] += loss_g0_c
                ch_g1_nll[c] += loss_g1_c

                ch_g0_crps[c] += crps_g0_elem[:, :, c].sum().item()
                ch_g1_crps[c] += crps_g1_elem[:, :, c].sum().item()

                ch_g0_p90_count[c] += in_g0_90[:, :, c].sum().item()
                ch_g1_p90_count[c] += in_g1_90[:, :, c].sum().item()

                ch_g0_var_sum[c] += var_g0[:, :, c].sum().item()
                ch_g1_var_sum[c] += var_g1[:, :, c].sum().item()

            # Trajectory-wise accumulation: combine source_file and traj_idx to avoid collisions
            for i in range(b):
                src_path = str(source_files[i])
                sim_idx = int(traj_indices[i].item())
                c_id = int(cluster_ids[i].item())
                re_val = float(re[i].item()) if re is not None else 0.0
                sc_val = float(sc[i].item()) if sc is not None else 0.0

                traj_key = f"{Path(src_path).name}::sim_{sim_idx:02d}"
                if traj_key not in traj_records:
                    traj_records[traj_key] = {
                        "trajectory_id": traj_key,
                        "source_file": src_path,
                        "traj_idx": sim_idx,
                        "cluster_id": c_id,
                        "re": re_val,
                        "sc": sc_val,
                        "windows": 0,
                        "tokens": 0,
                        "g0_nll_sum": 0.0,
                        "g1_nll_sum": 0.0,
                        "g0_crps_sum": 0.0,
                        "g1_crps_sum": 0.0,
                        "g0_p90_count": 0,
                        "g1_p90_count": 0,
                    }

                tok_i = target_z[i].numel()
                traj_records[traj_key]["windows"] += 1
                traj_records[traj_key]["tokens"] += tok_i
                traj_records[traj_key]["g0_nll_sum"] += F.gaussian_nll_loss(
                    mu_g0[i:i+1], target_z[i:i+1], var_g0[i:i+1], reduction="sum"
                ).item()
                traj_records[traj_key]["g1_nll_sum"] += F.gaussian_nll_loss(
                    mu_g1[i:i+1], target_z[i:i+1], var_g1[i:i+1], reduction="sum"
                ).item()
                traj_records[traj_key]["g0_crps_sum"] += crps_g0_elem[i:i+1].sum().item()
                traj_records[traj_key]["g1_crps_sum"] += crps_g1_elem[i:i+1].sum().item()
                traj_records[traj_key]["g0_p90_count"] += in_g0_90[i:i+1].sum().item()
                traj_records[traj_key]["g1_p90_count"] += in_g1_90[i:i+1].sum().item()

    if total_tokens == 0:
        raise ValueError("No test tokens evaluated. Fails closed.")

    # Formulate overall results
    d0_mae = d0_abs_err_sum / total_tokens
    d0_mse = d0_sq_err_sum / total_tokens
    d0_rmse = math.sqrt(d0_mse)

    g0_mean_nll = g0_nll_sum / total_tokens
    g1_mean_nll = g1_nll_sum / total_tokens
    delta_nll = g1_mean_nll - g0_mean_nll

    g0_mean_crps = g0_crps_sum / total_tokens
    g1_mean_crps = g1_crps_sum / total_tokens
    delta_crps = g1_mean_crps - g0_mean_crps

    g0_intervals = {}
    g1_intervals = {}
    for lvl in nominal_levels:
        key = f"{int(round(lvl * 100))}"
        picp_g0 = g0_interval_counts[key] / total_tokens
        mpiw_g0 = g0_interval_width_sum[key] / total_tokens
        picp_g1 = g1_interval_counts[key] / total_tokens
        mpiw_g1 = g1_interval_width_sum[key] / total_tokens

        g0_intervals[key] = {
            "nominal": float(lvl),
            "picp": float(picp_g0),
            "mpiw": float(mpiw_g0),
            "signed_calibration_error": float(picp_g0 - lvl),
            "absolute_calibration_error": float(abs(picp_g0 - lvl)),
        }
        g1_intervals[key] = {
            "nominal": float(lvl),
            "picp": float(picp_g1),
            "mpiw": float(mpiw_g1),
            "signed_calibration_error": float(picp_g1 - lvl),
            "absolute_calibration_error": float(abs(picp_g1 - lvl)),
        }

    # Channel summary
    channel_summary = []
    for c in range(num_channels):
        tok_c = ch_tokens[c]
        g0_nll_c = ch_g0_nll[c] / tok_c
        g1_nll_c = ch_g1_nll[c] / tok_c
        picp90_g0 = ch_g0_p90_count[c] / tok_c
        picp90_g1 = ch_g1_p90_count[c] / tok_c
        channel_summary.append({
            "channel": c,
            "g0_nll": float(g0_nll_c),
            "g1_nll": float(g1_nll_c),
            "delta_nll": float(g1_nll_c - g0_nll_c),
            "g0_crps": float(ch_g0_crps[c] / tok_c),
            "g1_crps": float(ch_g1_crps[c] / tok_c),
            "delta_crps": float((ch_g1_crps[c] - ch_g0_crps[c]) / tok_c),
            "g0_picp_90": float(picp90_g0),
            "g1_picp_90": float(picp90_g1),
            "g0_abs_cal_err_90": float(abs(picp90_g0 - 0.90)),
            "g1_abs_cal_err_90": float(abs(picp90_g1 - 0.90)),
            "g0_mean_var": float(ch_g0_var_sum[c] / tok_c),
            "g1_mean_var": float(ch_g1_var_sum[c] / tok_c),
        })

    # Trajectory summary
    trajectory_summary = []
    for t_id in sorted(traj_records.keys()):
        rec = traj_records[t_id]
        tok = rec["tokens"]
        g0_nll_t = rec["g0_nll_sum"] / tok
        g1_nll_t = rec["g1_nll_sum"] / tok
        g0_crps_t = rec["g0_crps_sum"] / tok
        g1_crps_t = rec["g1_crps_sum"] / tok
        picp90_g0 = rec["g0_p90_count"] / tok
        picp90_g1 = rec["g1_p90_count"] / tok
        trajectory_summary.append({
            "trajectory_id": t_id,
            "source_file": rec["source_file"],
            "traj_idx": rec["traj_idx"],
            "cluster_id": rec["cluster_id"],
            "re": rec["re"],
            "sc": rec["sc"],
            "windows": rec["windows"],
            "tokens": tok,
            "g0_nll": float(g0_nll_t),
            "g1_nll": float(g1_nll_t),
            "delta_nll": float(g1_nll_t - g0_nll_t),
            "g0_crps": float(g0_crps_t),
            "g1_crps": float(g1_crps_t),
            "delta_crps": float(g1_crps_t - g0_crps_t),
            "g0_picp_90": float(picp90_g0),
            "g1_picp_90": float(picp90_g1),
            "g0_abs_cal_err_90": float(abs(picp90_g0 - 0.90)),
            "g1_abs_cal_err_90": float(abs(picp90_g1 - 0.90)),
        })

    g0_cal_err_90 = abs(g0_intervals["90"]["picp"] - 0.90)
    g1_cal_err_90 = abs(g1_intervals["90"]["picp"] - 0.90)

    return {
        "dataset_summary": {
            "batches_evaluated": batches_evaluated,
            "windows_evaluated": total_windows,
            "tokens_evaluated": total_tokens,
            "tokens_per_window": total_tokens // total_windows if total_windows > 0 else 0,
            "latent_spatial_shape": latent_spatial_shape,
            "unique_trajectories_evaluated": len(trajectory_summary),
        },
        "mean_parity_check": {
            "max_mean_discrepancy": float(max_mean_discrepancy),
            "status": "PASS" if max_mean_discrepancy < 1e-6 else "FAIL",
        },
        "D0_deterministic_baseline": {
            "description": "Deterministic mean point forecast reference",
            "mae": float(d0_mae),
            "mse": float(d0_mse),
            "rmse": float(d0_rmse),
        },
        "G0_homoscedastic_baseline": {
            "description": "Fixed channel variance N(mu_D0, diag(v_G0))",
            "nll": float(g0_mean_nll),
            "nll_unit": "nats/latent element",
            "crps": float(g0_mean_crps),
            "mean_variance": float(g0_var_sum / total_tokens),
            "min_variance": float(g0_min_var),
            "max_variance": float(g0_max_var),
            "intervals": g0_intervals,
        },
        "G1_heteroscedastic_model": {
            "description": "Condition-dependent variance head N(mu_D0, diag(v_theta(x)))",
            "nll": float(g1_mean_nll),
            "nll_unit": "nats/latent element",
            "crps": float(g1_mean_crps),
            "mean_variance": float(g1_var_sum / total_tokens),
            "min_variance": float(g1_min_var),
            "max_variance": float(g1_max_var),
            "intervals": g1_intervals,
        },
        "comparison_g1_vs_g0": {
            "delta_nll": float(delta_nll),
            "delta_nll_unit": "nats/latent element",
            "nll_improved": bool(delta_nll < 0.0),
            "delta_crps": float(delta_crps),
            "crps_improved": bool(delta_crps < 0.0),
            "coverage_change_p90": float(g1_intervals["90"]["picp"] - g0_intervals["90"]["picp"]),
            "sharpness_change_p90": float(g1_intervals["90"]["mpiw"] - g0_intervals["90"]["mpiw"]),
            "g0_abs_calibration_error_p90": float(g0_cal_err_90),
            "g1_abs_calibration_error_p90": float(g1_cal_err_90),
            "calibration_error_change_p90": float(g1_cal_err_90 - g0_cal_err_90),
            "better_calibrated_p90": bool(g1_cal_err_90 < g0_cal_err_90),
        },
        "channel_diagnostics": channel_summary,
        "trajectory_diagnostics": trajectory_summary,
    }


def evaluate_autoregressive_rollouts(
    forecaster: LatentForecaster,
    g0_head: nn.Module,
    test_loader: DataLoader,
    normalizer: FieldNormalizer,
    device: torch.device,
    horizon: int = 30,
    num_samples: int = 32,
    eval_horizons: Tuple[int, ...] = (1, 5, 10, 20, 30),
    seed: int = 42,
    max_rollout_windows: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute Step 2: 30-step autonomous autoregressive rollout evaluation (H=30, K=32).

    Autonomous rollout without ground-truth feedback. Evaluates:
    - Complete test window coverage with true window identity manifest
    - Non-parametric empirical quantiles in physical space from K=32 decoded samples
    - Ensemble Mean VRMSE vs D0 vs G0
    - Mathematically rigorous Spread-Skill alignment (RMS spread / RMSE, raw & finite-K adjusted)
    - Zero-mean pressure gauge normalization on physical fields
    - Physical invariant checks: RMS divergence, vorticity RMSE, and energy spectrum E(k) across all K members
    """
    forecaster.eval()
    g0_head.eval()

    # Track metrics per horizon
    horizon_results: Dict[str, Dict[str, Any]] = {
        f"h_{h}": {
            "D0": {"vrmse_list": [], "div_rms_list": [], "vort_rmse_list": []},
            "G0": {
                "vrmse_list": [],
                "spread_raw_list": [],
                "spread_adj_list": [],
                "rmse_vel_list": [],
                "ssr_raw_list": [],
                "ssr_adj_list": [],
                "div_sample_rms_list": [],
                "div_mean_rms_list": [],
                "vort_rmse_list": [],
                "vort_samp_rms_list": [],
                "picp_80_list": [],
                "picp_90_list": [],
                "mpiw_80_list": [],
                "mpiw_90_list": [],
            },
            "G1": {
                "vrmse_list": [],
                "spread_raw_list": [],
                "spread_adj_list": [],
                "rmse_vel_list": [],
                "ssr_raw_list": [],
                "ssr_adj_list": [],
                "div_sample_rms_list": [],
                "div_mean_rms_list": [],
                "vort_rmse_list": [],
                "vort_samp_rms_list": [],
                "picp_80_list": [],
                "picp_90_list": [],
                "mpiw_80_list": [],
                "mpiw_90_list": [],
            },
        }
        for h in eval_horizons
    }

    # Spectrum storage at h=30
    spectrum_data: Dict[str, Any] = {
        "k_bins": None,
        "gt_spectrum": [],
        "d0_spectrum": [],
        "g0_sample_spectrum": [],
        "g0_mean_spectrum": [],
        "g1_sample_spectrum": [],
        "g1_mean_spectrum": [],
    }

    total_windows_in_loader = len(test_loader.dataset)
    window_count = 0
    unique_trajs: Set[str] = set()
    unique_clusters: Set[int] = set()
    window_manifest: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if max_rollout_windows is not None and window_count >= max_rollout_windows:
                break

            # Assert required identity fields exist; fail closed if missing
            for id_field in ("source_file", "traj_idx", "start_t", "cluster_id"):
                if id_field not in batch:
                    raise ValueError(
                        f"Rollout batch {batch_idx} missing required identity field '{id_field}'. "
                        f"Fails closed: refuse to default trajectory identity."
                    )

            q_hist = batch["history"].to(device)  # (B, L=4, 4, Ny, Nx)
            q_future = batch["future"].to(device)  # (B, H_fut, 4, Ny, Nx)
            re = batch["re"].to(device) if "re" in batch else None
            sc = batch["sc"].to(device) if "sc" in batch else None

            source_files: List[str] = batch["source_file"]
            traj_indices: torch.Tensor = batch["traj_idx"]
            start_times: torch.Tensor = batch["start_t"]
            cluster_ids: torch.Tensor = batch["cluster_id"]

            b = q_hist.shape[0]

            for i in range(b):
                if max_rollout_windows is not None and window_count >= max_rollout_windows:
                    break
                window_count += 1

                src_file = str(source_files[i])
                sim_idx = int(traj_indices[i].item())
                s_t = int(start_times[i].item())
                c_id = int(cluster_ids[i].item())

                traj_key = f"{Path(src_file).name}::sim_{sim_idx:02d}"
                unique_trajs.add(traj_key)
                unique_clusters.add(c_id)

                window_manifest.append({
                    "window_index": window_count - 1,
                    "source_file": src_file,
                    "traj_idx": sim_idx,
                    "cluster_id": c_id,
                    "start_t": s_t,
                })

                q_h_i = q_hist[i:i+1]
                re_i = re[i:i+1] if re is not None else None
                sc_i = sc[i:i+1] if sc is not None else None
                q_fut_i = q_future[i:i+1, :horizon]  # (1, H, 4, Ny, Nx)

                # 1. D0 deterministic rollout
                rollout_d0 = forecaster.sample_rollout(
                    q_hist=q_h_i,
                    re=re_i,
                    sc=sc_i,
                    horizon=horizon,
                    num_samples=1,
                    seed=seed,
                    variance_head=None,
                    decode_samples=False,
                )
                q_d0 = rollout_d0["deterministic_rollout"]  # (1, H, 4, Ny, Nx) in normalized space

                # 2. G0 probabilistic rollout (K=32 with g0_head)
                rollout_g0 = forecaster.sample_rollout(
                    q_hist=q_h_i,
                    re=re_i,
                    sc=sc_i,
                    horizon=horizon,
                    num_samples=num_samples,
                    seed=seed,
                    variance_head=g0_head,
                    decode_samples=True,
                )
                q_g0_samples = rollout_g0["sample_trajectories"]  # (1, K, H, 4, Ny, Nx)
                q_g0_mean = rollout_g0["ensemble_mean"]  # (1, H, 4, Ny, Nx)

                # 3. G1 probabilistic rollout (K=32 with trained variance_head)
                rollout_g1 = forecaster.sample_rollout(
                    q_hist=q_h_i,
                    re=re_i,
                    sc=sc_i,
                    horizon=horizon,
                    num_samples=num_samples,
                    seed=seed,
                    variance_head=None,
                    decode_samples=True,
                )
                q_g1_samples = rollout_g1["sample_trajectories"]  # (1, K, H, 4, Ny, Nx)
                q_g1_mean = rollout_g1["ensemble_mean"]  # (1, H, 4, Ny, Nx)

                # Denormalize all predictions to physical space
                q_gt_phys = normalizer.denormalize(q_fut_i.squeeze(0))  # (H, 4, Ny, Nx)
                q_d0_phys = normalizer.denormalize(q_d0.squeeze(0))  # (H, 4, Ny, Nx)
                q_g0_mean_phys = normalizer.denormalize(q_g0_mean.squeeze(0))  # (H, 4, Ny, Nx)
                q_g1_mean_phys = normalizer.denormalize(q_g1_mean.squeeze(0))  # (H, 4, Ny, Nx)

                # Denormalize samples: (K, H, 4, Ny, Nx)
                q_g0_samp_phys = normalizer.denormalize(q_g0_samples.squeeze(0))
                q_g1_samp_phys = normalizer.denormalize(q_g1_samples.squeeze(0))

                # Apply zero-mean pressure gauge normalization (channel 2)
                q_gt_phys = apply_pressure_gauge(q_gt_phys)
                q_d0_phys = apply_pressure_gauge(q_d0_phys)
                q_g0_mean_phys = apply_pressure_gauge(q_g0_mean_phys)
                q_g1_mean_phys = apply_pressure_gauge(q_g1_mean_phys)
                q_g0_samp_phys = apply_pressure_gauge(q_g0_samp_phys)
                q_g1_samp_phys = apply_pressure_gauge(q_g1_samp_phys)

                for h in eval_horizons:
                    step_idx = h - 1
                    t_gt = q_gt_phys[step_idx]  # (4, Ny, Nx)
                    vort_gt = compute_vorticity(t_gt[0:1], t_gt[1:2])

                    # D0 metrics
                    p_d0 = q_d0_phys[step_idx]
                    vrmse_d0 = float(compute_vrmse(p_d0, t_gt).item())
                    div_d0 = compute_divergence(p_d0[0:1], p_d0[1:2])
                    div_rms_d0 = float(torch.sqrt(torch.mean(div_d0**2)).item())
                    vort_d0 = compute_vorticity(p_d0[0:1], p_d0[1:2])
                    vort_rmse_d0 = float(torch.sqrt(((vort_d0 - vort_gt) ** 2).mean()).item())

                    horizon_results[f"h_{h}"]["D0"]["vrmse_list"].append(vrmse_d0)
                    horizon_results[f"h_{h}"]["D0"]["div_rms_list"].append(div_rms_d0)
                    horizon_results[f"h_{h}"]["D0"]["vort_rmse_list"].append(vort_rmse_d0)

                    # G0 metrics
                    p_g0_mean = q_g0_mean_phys[step_idx]
                    samps_g0 = q_g0_samp_phys[:, step_idx]  # (K, 4, Ny, Nx)
                    vrmse_g0 = float(compute_vrmse(p_g0_mean, t_gt).item())
                    div_g0_mean = compute_divergence(p_g0_mean[0:1], p_g0_mean[1:2])
                    div_rms_g0_mean = float(torch.sqrt(torch.mean(div_g0_mean**2)).item())
                    vort_g0_mean = compute_vorticity(p_g0_mean[0:1], p_g0_mean[1:2])
                    vort_rmse_g0 = float(torch.sqrt(((vort_g0_mean - vort_gt) ** 2).mean()).item())

                    # G0 sample divergence and vorticity
                    div_samps_g0 = compute_divergence(samps_g0[:, 0], samps_g0[:, 1])
                    div_rms_g0_samp = float(torch.sqrt(torch.mean(div_samps_g0**2, dim=(-2, -1))).mean().item())
                    vort_samps_g0 = compute_vorticity(samps_g0[:, 0], samps_g0[:, 1])
                    vort_rms_g0_samp = float(torch.sqrt((vort_samps_g0 ** 2).mean(dim=(-2, -1))).mean().item())

                    # G0 Spread-Skill & Quantiles
                    ss_g0 = compute_ensemble_spread_skill(samps_g0, t_gt)
                    q_cov_g0 = compute_physical_quantiles_coverage(samps_g0, t_gt)

                    horizon_results[f"h_{h}"]["G0"]["vrmse_list"].append(vrmse_g0)
                    horizon_results[f"h_{h}"]["G0"]["spread_raw_list"].append(ss_g0["velocity"]["rms_spread"])
                    horizon_results[f"h_{h}"]["G0"]["spread_adj_list"].append(ss_g0["velocity"]["finite_k_adjusted_spread"])
                    horizon_results[f"h_{h}"]["G0"]["rmse_vel_list"].append(ss_g0["velocity"]["rmse_velocity"])
                    horizon_results[f"h_{h}"]["G0"]["ssr_raw_list"].append(ss_g0["velocity"]["spread_skill_ratio"])
                    horizon_results[f"h_{h}"]["G0"]["ssr_adj_list"].append(ss_g0["velocity"]["finite_k_adjusted_ssr"])
                    horizon_results[f"h_{h}"]["G0"]["div_mean_rms_list"].append(div_rms_g0_mean)
                    horizon_results[f"h_{h}"]["G0"]["div_sample_rms_list"].append(div_rms_g0_samp)
                    horizon_results[f"h_{h}"]["G0"]["vort_rmse_list"].append(vort_rmse_g0)
                    horizon_results[f"h_{h}"]["G0"]["vort_samp_rms_list"].append(vort_rms_g0_samp)
                    horizon_results[f"h_{h}"]["G0"]["picp_80_list"].append(q_cov_g0["velocity"]["picp_80"])
                    horizon_results[f"h_{h}"]["G0"]["picp_90_list"].append(q_cov_g0["velocity"]["picp_90"])
                    horizon_results[f"h_{h}"]["G0"]["mpiw_80_list"].append(q_cov_g0["velocity"]["mpiw_80"])
                    horizon_results[f"h_{h}"]["G0"]["mpiw_90_list"].append(q_cov_g0["velocity"]["mpiw_90"])

                    # G1 metrics
                    p_g1_mean = q_g1_mean_phys[step_idx]
                    samps_g1 = q_g1_samp_phys[:, step_idx]  # (K, 4, Ny, Nx)
                    vrmse_g1 = float(compute_vrmse(p_g1_mean, t_gt).item())
                    div_g1_mean = compute_divergence(p_g1_mean[0:1], p_g1_mean[1:2])
                    div_rms_g1_mean = float(torch.sqrt(torch.mean(div_g1_mean**2)).item())
                    vort_g1_mean = compute_vorticity(p_g1_mean[0:1], p_g1_mean[1:2])
                    vort_rmse_g1 = float(torch.sqrt(((vort_g1_mean - vort_gt) ** 2).mean()).item())

                    # G1 sample divergence and vorticity
                    div_samps_g1 = compute_divergence(samps_g1[:, 0], samps_g1[:, 1])
                    div_rms_g1_samp = float(torch.sqrt(torch.mean(div_samps_g1**2, dim=(-2, -1))).mean().item())
                    vort_samps_g1 = compute_vorticity(samps_g1[:, 0], samps_g1[:, 1])
                    vort_rms_g1_samp = float(torch.sqrt((vort_samps_g1 ** 2).mean(dim=(-2, -1))).mean().item())

                    # G1 Spread-Skill & Quantiles
                    ss_g1 = compute_ensemble_spread_skill(samps_g1, t_gt)
                    q_cov_g1 = compute_physical_quantiles_coverage(samps_g1, t_gt)

                    horizon_results[f"h_{h}"]["G1"]["vrmse_list"].append(vrmse_g1)
                    horizon_results[f"h_{h}"]["G1"]["spread_raw_list"].append(ss_g1["velocity"]["rms_spread"])
                    horizon_results[f"h_{h}"]["G1"]["spread_adj_list"].append(ss_g1["velocity"]["finite_k_adjusted_spread"])
                    horizon_results[f"h_{h}"]["G1"]["rmse_vel_list"].append(ss_g1["velocity"]["rmse_velocity"])
                    horizon_results[f"h_{h}"]["G1"]["ssr_raw_list"].append(ss_g1["velocity"]["spread_skill_ratio"])
                    horizon_results[f"h_{h}"]["G1"]["ssr_adj_list"].append(ss_g1["velocity"]["finite_k_adjusted_ssr"])
                    horizon_results[f"h_{h}"]["G1"]["div_mean_rms_list"].append(div_rms_g1_mean)
                    horizon_results[f"h_{h}"]["G1"]["div_sample_rms_list"].append(div_rms_g1_samp)
                    horizon_results[f"h_{h}"]["G1"]["vort_rmse_list"].append(vort_rmse_g1)
                    horizon_results[f"h_{h}"]["G1"]["vort_samp_rms_list"].append(vort_rms_g1_samp)
                    horizon_results[f"h_{h}"]["G1"]["picp_80_list"].append(q_cov_g1["velocity"]["picp_80"])
                    horizon_results[f"h_{h}"]["G1"]["picp_90_list"].append(q_cov_g1["velocity"]["picp_90"])
                    horizon_results[f"h_{h}"]["G1"]["mpiw_80_list"].append(q_cov_g1["velocity"]["mpiw_80"])
                    horizon_results[f"h_{h}"]["G1"]["mpiw_90_list"].append(q_cov_g1["velocity"]["mpiw_90"])

                # Kinetic Energy Spectrum at horizon h=30 (final step)
                final_step = horizon - 1
                u_gt, v_gt = q_gt_phys[final_step, 0], q_gt_phys[final_step, 1]
                u_d0, v_d0 = q_d0_phys[final_step, 0], q_d0_phys[final_step, 1]
                u_g0_m, v_g0_m = q_g0_mean_phys[final_step, 0], q_g0_mean_phys[final_step, 1]
                u_g1_m, v_g1_m = q_g1_mean_phys[final_step, 0], q_g1_mean_phys[final_step, 1]

                # Radially binned spectrum
                k_bins, e_gt = compute_radial_energy_spectrum(u_gt, v_gt)
                _, e_d0 = compute_radial_energy_spectrum(u_d0, v_d0)
                _, e_g0_m = compute_radial_energy_spectrum(u_g0_m, v_g0_m)
                _, e_g1_m = compute_radial_energy_spectrum(u_g1_m, v_g1_m)

                # Compute spectrum across ALL K samples (complete ensemble averaging)
                k_g0_samps = [
                    compute_radial_energy_spectrum(q_g0_samp_phys[s, final_step, 0], q_g0_samp_phys[s, final_step, 1])[1]
                    for s in range(num_samples)
                ]
                k_g1_samps = [
                    compute_radial_energy_spectrum(q_g1_samp_phys[s, final_step, 0], q_g1_samp_phys[s, final_step, 1])[1]
                    for s in range(num_samples)
                ]
                e_g0_s = torch.stack(k_g0_samps).mean(dim=0)
                e_g1_s = torch.stack(k_g1_samps).mean(dim=0)

                if spectrum_data["k_bins"] is None:
                    spectrum_data["k_bins"] = k_bins.cpu().numpy().tolist()
                spectrum_data["gt_spectrum"].append(e_gt.cpu().numpy().tolist())
                spectrum_data["d0_spectrum"].append(e_d0.cpu().numpy().tolist())
                spectrum_data["g0_sample_spectrum"].append(e_g0_s.cpu().numpy().tolist())
                spectrum_data["g0_mean_spectrum"].append(e_g0_m.cpu().numpy().tolist())
                spectrum_data["g1_sample_spectrum"].append(e_g1_s.cpu().numpy().tolist())
                spectrum_data["g1_mean_spectrum"].append(e_g1_m.cpu().numpy().tolist())

    # Formulate summary per horizon
    rollout_summary = {}
    for h in eval_horizons:
        d0_vrmse = float(np.mean(horizon_results[f"h_{h}"]["D0"]["vrmse_list"]))
        d0_div = float(np.mean(horizon_results[f"h_{h}"]["D0"]["div_rms_list"]))
        d0_vort = float(np.mean(horizon_results[f"h_{h}"]["D0"]["vort_rmse_list"]))

        # G0 metrics
        g0_vrmse = float(np.mean(horizon_results[f"h_{h}"]["G0"]["vrmse_list"]))
        g0_spread_raw = float(np.mean(horizon_results[f"h_{h}"]["G0"]["spread_raw_list"]))
        g0_spread_adj = float(np.mean(horizon_results[f"h_{h}"]["G0"]["spread_adj_list"]))
        g0_rmse_vel = float(np.mean(horizon_results[f"h_{h}"]["G0"]["rmse_vel_list"]))
        g0_ssr_raw = float(np.mean(horizon_results[f"h_{h}"]["G0"]["ssr_raw_list"]))
        g0_ssr_adj = float(np.mean(horizon_results[f"h_{h}"]["G0"]["ssr_adj_list"]))
        g0_div_samp = float(np.mean(horizon_results[f"h_{h}"]["G0"]["div_sample_rms_list"]))
        g0_div_mean = float(np.mean(horizon_results[f"h_{h}"]["G0"]["div_mean_rms_list"]))
        g0_vort_rmse = float(np.mean(horizon_results[f"h_{h}"]["G0"]["vort_rmse_list"]))
        g0_vort_samp = float(np.mean(horizon_results[f"h_{h}"]["G0"]["vort_samp_rms_list"]))
        g0_picp_80 = float(np.mean(horizon_results[f"h_{h}"]["G0"]["picp_80_list"]))
        g0_picp_90 = float(np.mean(horizon_results[f"h_{h}"]["G0"]["picp_90_list"]))
        g0_mpiw_80 = float(np.mean(horizon_results[f"h_{h}"]["G0"]["mpiw_80_list"]))
        g0_mpiw_90 = float(np.mean(horizon_results[f"h_{h}"]["G0"]["mpiw_90_list"]))

        # G1 metrics
        g1_vrmse = float(np.mean(horizon_results[f"h_{h}"]["G1"]["vrmse_list"]))
        g1_spread_raw = float(np.mean(horizon_results[f"h_{h}"]["G1"]["spread_raw_list"]))
        g1_spread_adj = float(np.mean(horizon_results[f"h_{h}"]["G1"]["spread_adj_list"]))
        g1_rmse_vel = float(np.mean(horizon_results[f"h_{h}"]["G1"]["rmse_vel_list"]))
        g1_ssr_raw = float(np.mean(horizon_results[f"h_{h}"]["G1"]["ssr_raw_list"]))
        g1_ssr_adj = float(np.mean(horizon_results[f"h_{h}"]["G1"]["ssr_adj_list"]))
        g1_div_samp = float(np.mean(horizon_results[f"h_{h}"]["G1"]["div_sample_rms_list"]))
        g1_div_mean = float(np.mean(horizon_results[f"h_{h}"]["G1"]["div_mean_rms_list"]))
        g1_vort_rmse = float(np.mean(horizon_results[f"h_{h}"]["G1"]["vort_rmse_list"]))
        g1_vort_samp = float(np.mean(horizon_results[f"h_{h}"]["G1"]["vort_samp_rms_list"]))
        g1_picp_80 = float(np.mean(horizon_results[f"h_{h}"]["G1"]["picp_80_list"]))
        g1_picp_90 = float(np.mean(horizon_results[f"h_{h}"]["G1"]["picp_90_list"]))
        g1_mpiw_80 = float(np.mean(horizon_results[f"h_{h}"]["G1"]["mpiw_80_list"]))
        g1_mpiw_90 = float(np.mean(horizon_results[f"h_{h}"]["G1"]["mpiw_90_list"]))

        rollout_summary[f"horizon_{h}"] = {
            "horizon_step": h,
            "D0": {
                "ensemble_mean_vrmse": d0_vrmse,
                "rms_divergence": d0_div,
                "vorticity_rmse": d0_vort,
            },
            "G0": {
                "ensemble_mean_vrmse": g0_vrmse,
                "velocity_rmse": g0_rmse_vel,
                "rms_spread_raw": g0_spread_raw,
                "rms_spread_adjusted": g0_spread_adj,
                "spread_skill_ratio_raw": g0_ssr_raw,
                "spread_skill_ratio_adjusted": g0_ssr_adj,
                "rms_divergence_individual_samples": g0_div_samp,
                "rms_divergence_ensemble_mean": g0_div_mean,
                "vorticity_rmse": g0_vort_rmse,
                "vorticity_rms_individual_samples": g0_vort_samp,
                "velocity_physical_quantiles": {
                    "picp_80": g0_picp_80,
                    "picp_90": g0_picp_90,
                    "mpiw_80": g0_mpiw_80,
                    "mpiw_90": g0_mpiw_90,
                    "cal_error_80": float(abs(g0_picp_80 - 0.80)),
                    "cal_error_90": float(abs(g0_picp_90 - 0.90)),
                },
            },
            "G1": {
                "ensemble_mean_vrmse": g1_vrmse,
                "velocity_rmse": g1_rmse_vel,
                "rms_spread_raw": g1_spread_raw,
                "rms_spread_adjusted": g1_spread_adj,
                "spread_skill_ratio_raw": g1_ssr_raw,
                "spread_skill_ratio_adjusted": g1_ssr_adj,
                "rms_divergence_individual_samples": g1_div_samp,
                "rms_divergence_ensemble_mean": g1_div_mean,
                "vorticity_rmse": g1_vort_rmse,
                "vorticity_rms_individual_samples": g1_vort_samp,
                "velocity_physical_quantiles": {
                    "picp_80": g1_picp_80,
                    "picp_90": g1_picp_90,
                    "mpiw_80": g1_mpiw_80,
                    "mpiw_90": g1_mpiw_90,
                    "cal_error_80": float(abs(g1_picp_80 - 0.80)),
                    "cal_error_90": float(abs(g1_picp_90 - 0.90)),
                },
            },
            "delta_vrmse_g1_vs_d0": float(g1_vrmse - d0_vrmse),
            "delta_vrmse_g1_vs_g0": float(g1_vrmse - g0_vrmse),
        }

    avg_spectra = {
        "k_bins": spectrum_data["k_bins"],
        "spectrum_samples_per_ensemble": num_samples,
        "mean_gt_spectrum": np.mean(spectrum_data["gt_spectrum"], axis=0).tolist(),
        "mean_d0_spectrum": np.mean(spectrum_data["d0_spectrum"], axis=0).tolist(),
        "mean_g0_sample_spectrum": np.mean(spectrum_data["g0_sample_spectrum"], axis=0).tolist(),
        "mean_g0_mean_spectrum": np.mean(spectrum_data["g0_mean_spectrum"], axis=0).tolist(),
        "mean_g1_sample_spectrum": np.mean(spectrum_data["g1_sample_spectrum"], axis=0).tolist(),
        "mean_g1_mean_spectrum": np.mean(spectrum_data["g1_mean_spectrum"], axis=0).tolist(),
    }

    evaluation_mode = (
        "full_test_set"
        if max_rollout_windows is None or window_count >= total_windows_in_loader
        else "diagnostic_sampling"
    )

    return {
        "configuration": {
            "evaluation_mode": evaluation_mode,
            "total_windows_in_dataset": total_windows_in_loader,
            "windows_evaluated": window_count,
            "unique_trajectories_count": len(unique_trajs),
            "unique_trajectories": sorted(list(unique_trajs)),
            "unique_clusters_count": len(unique_clusters),
            "unique_clusters": sorted(list(unique_clusters)),
            "horizon": horizon,
            "num_samples": num_samples,
            "eval_horizons": list(eval_horizons),
            "seed": seed,
            "spread_formula": "RMS spread = sqrt(mean(var_k)) with Bessel correction (ddof=1)",
            "rmse_formula": "RMSE = sqrt(mean((ensemble_mean - target)^2))",
            "finite_k_inflation_factor": float(math.sqrt((num_samples + 1.0) / num_samples)),
            "window_manifest": window_manifest,
        },
        "per_horizon_metrics": rollout_summary,
        "energy_spectrum_at_final_horizon": avg_spectra,
    }


def validate_phase3_metrics_finite(obj: Any, path: str = "root") -> None:
    """Recursively validate that all metrics are finite and free of NaN/Inf."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            validate_phase3_metrics_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            validate_phase3_metrics_finite(item, f"{path}[{idx}]")
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Metric validation failed: non-finite float at {path} = {obj}")
    elif isinstance(obj, torch.Tensor):
        if not torch.isfinite(obj).all():
            raise ValueError(f"Metric validation failed: non-finite tensor at {path}")


def main():
    parser = argparse.ArgumentParser(description="ProbLatent-R1 Phase 3 Evaluation Pipeline.")
    parser.add_argument(
        "--d0_checkpoint",
        type=str,
        default="outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt",
        help="Path to frozen D0 deterministic checkpoint.",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="outputs/normalization/latent_residual_stats.json",
        help="Path to Phase 0 latent residual statistics JSON.",
    )
    parser.add_argument(
        "--normalizer_path",
        type=str,
        default="outputs/normalization/stats_grouped.pt",
        help="Path to grouped FieldNormalizer weights.",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="outputs/splits/grouped_split.json",
        help="Path to dataset split JSON.",
    )
    parser.add_argument(
        "--phase2_record_path",
        type=str,
        default="outputs/normalization/phase2_variance_training_record.json",
        help="Path to immutable Phase 2 variance training record JSON.",
    )
    parser.add_argument(
        "--g1_variance_head",
        type=str,
        default="outputs/checkpoints/probabilistic/variance_head/formal_r1_seed42/best_g1_variance_head.pt",
        help="Path to trained G1 variance head checkpoint.",
    )
    parser.add_argument(
        "--g0_baseline_checkpoint",
        type=str,
        default="outputs/checkpoints/probabilistic/variance_head/formal_r1_seed42/g0_baseline_initialization.pt",
        help="Path to G0 baseline initialization checkpoint.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/datasets/shear_flow",
        help="Path to HDF5 shear flow dataset directory.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/metrics/phase3_probabilistic_evaluation.json",
        help="Path to write final structured evaluation report JSON.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for single-step evaluation.",
    )
    parser.add_argument(
        "--test_stride",
        type=int,
        default=8,
        help="Temporal window stride for single-step test set evaluation.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Autonomous rollout horizon steps.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=32,
        help="Number of stochastic sample trajectories per test initial condition.",
    )
    parser.add_argument(
        "--max_rollout_windows",
        type=int,
        default=None,
        help="Maximum rollout windows to evaluate (default None evaluates all windows in test set).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--skip_rollouts",
        action="store_true",
        help="If set, only runs Step 1 single-step probabilistic evaluation.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=================================================================")
    print("ProbLatent-R1 Phase 3: Comprehensive Probabilistic Evaluation")
    print("=================================================================")
    print(f"Target device:     {device}")
    print(f"Random seed:       {args.seed}")
    print(f"D0 checkpoint:     {args.d0_checkpoint}")
    print(f"G1 variance head:  {args.g1_variance_head}")
    print(f"G0 baseline head:  {args.g0_baseline_checkpoint}")
    print(f"Output report:     {args.output_file}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1. Load D0 and build LatentForecaster architecture
    d0_ckpt = torch.load(args.d0_checkpoint, map_location="cpu", weights_only=False)
    with open(args.stats_path, "r") as f:
        stats_data = json.load(f)

    cfg = d0_ckpt.get("config", {})
    pred_mode = cfg.get("prediction_mode", d0_ckpt.get("prediction_mode", "direct"))
    emb_dim = cfg.get("embed_dim", 256)
    depth = cfg.get("depth", 6)
    num_heads = cfg.get("num_heads", 8)
    use_spatial_pos = resolve_spatial_pos_config(d0_ckpt)

    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=emb_dim,
        cond_dim=128,
        depth=depth,
        num_heads=num_heads,
        history_length=4,
        prediction_mode=pred_mode,
        use_spatial_pos=use_spatial_pos,
    )
    forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder).to(device)

    sd = d0_ckpt.get("model_state_dict", d0_ckpt)
    cleaned_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    cleaned_sd = strip_compiled_prefix(cleaned_sd)
    forecaster.load_state_dict(cleaned_sd, strict=True)
    forecaster.freeze_representation = True
    for p in forecaster.parameters():
        p.requires_grad = False

    # 2. Load G0 and G1 Variance Heads
    g0_head = VarianceHead2D(embed_dim=emb_dim, latent_channels=64, variance_floor=1e-4).to(device)
    g0_ckpt = torch.load(args.g0_baseline_checkpoint, map_location="cpu", weights_only=False)
    g0_head.load_state_dict(g0_ckpt["variance_head_state_dict"])
    for p in g0_head.parameters():
        p.requires_grad = False

    g1_head = VarianceHead2D(embed_dim=emb_dim, latent_channels=64, variance_floor=1e-4).to(device)
    g1_ckpt = torch.load(args.g1_variance_head, map_location="cpu", weights_only=False)
    g1_head.load_state_dict(g1_ckpt["variance_head_state_dict"])
    for p in g1_head.parameters():
        p.requires_grad = False

    # Attach G1 variance head as default to transformer
    forecaster.transformer.attach_variance_head(g1_head)

    # 3. Create Normalizer and DataLoaders
    normalizer = FieldNormalizer()
    normalizer.load_state_dict(torch.load(args.normalizer_path, weights_only=True, map_location="cpu"))

    _, val_loader, test_loader, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file=args.split_file,
        data_root=args.data_root,
        history_length=4,
        horizon=1,
        valid_horizon=1,
        train_stride=8,
        valid_stride=8,
        test_stride=args.test_stride,
        downsample_factor=2,
        batch_size=args.batch_size,
        num_workers=2,
        normalize=True,
        normalizer=normalizer,
        seed=args.seed,
    )
    print(f"\n[DataLoaders Ready] Valid windows={len(val_loader.dataset)}, Test windows={len(test_loader.dataset)}")

    # 4. Pre-flight Gate Verification
    reproduced_val_nll = verify_phase3_preflight_gate(
        d0_checkpoint_path=args.d0_checkpoint,
        g0_checkpoint_path=args.g0_baseline_checkpoint,
        stats_path=args.stats_path,
        normalizer_path=args.normalizer_path,
        split_file=args.split_file,
        phase2_record_path=args.phase2_record_path,
        g1_variance_head_path=args.g1_variance_head,
        forecaster=forecaster,
        val_loader=val_loader,
        device=device,
        tolerance=1e-4,
    )

    # 5. Step 1: Single-Step Probabilistic Evaluation on Test Set
    print("\n=================================================================")
    print("Step 1: Single-Step Probabilistic Evaluation on Independent Test Set")
    print("=================================================================")
    step1_results = evaluate_single_step_probability(
        forecaster=forecaster,
        g0_head=g0_head,
        test_loader=test_loader,
        device=device,
        nominal_levels=(0.50, 0.80, 0.90, 0.95),
    )
    print(f"  G0 Test NLL:          {step1_results['G0_homoscedastic_baseline']['nll']:.6f} nats/latent element")
    print(f"  G1 Test NLL:          {step1_results['G1_heteroscedastic_model']['nll']:.6f} nats/latent element")
    print(f"  Delta NLL (G1 - G0):  {step1_results['comparison_g1_vs_g0']['delta_nll']:.6f} nats/latent element")
    print(f"  G0 Test CRPS:         {step1_results['G0_homoscedastic_baseline']['crps']:.6f}")
    print(f"  G1 Test CRPS:         {step1_results['G1_heteroscedastic_model']['crps']:.6f}")
    print(f"  Delta CRPS:           {step1_results['comparison_g1_vs_g0']['delta_crps']:.6f}")
    print(f"  D0 Point MAE:         {step1_results['D0_deterministic_baseline']['mae']:.6f}")
    print(f"  D0 Point RMSE:        {step1_results['D0_deterministic_baseline']['rmse']:.6f}")
    print("  Prediction Interval Coverage & Calibration Error (90% Nominal):")
    print(
        f"    G0: PICP={step1_results['G0_homoscedastic_baseline']['intervals']['90']['picp'] * 100:.2f}%, "
        f"MPIW={step1_results['G0_homoscedastic_baseline']['intervals']['90']['mpiw']:.4f}, "
        f"Abs Cal Err={step1_results['comparison_g1_vs_g0']['g0_abs_calibration_error_p90'] * 100:.2f}%"
    )
    print(
        f"    G1: PICP={step1_results['G1_heteroscedastic_model']['intervals']['90']['picp'] * 100:.2f}%, "
        f"MPIW={step1_results['G1_heteroscedastic_model']['intervals']['90']['mpiw']:.4f}, "
        f"Abs Cal Err={step1_results['comparison_g1_vs_g0']['g1_abs_calibration_error_p90'] * 100:.2f}%"
    )

    # 6. Step 2: 30-Step Autoregressive Rollouts (H=30, K=32)
    step2_results = None
    if not args.skip_rollouts:
        print("\n=================================================================")
        print("Step 2: 30-Step Autonomous Autoregressive Rollout Evaluation (H=30, K=32)")
        print("=================================================================")
        _, _, test_loader_rollout, _ = create_flow_dataloaders(
            split_type="grouped",
            split_file=args.split_file,
            data_root=args.data_root,
            history_length=4,
            horizon=args.horizon,
            valid_horizon=args.horizon,
            train_stride=8,
            valid_stride=8,
            test_stride=8,
            downsample_factor=2,
            batch_size=1,  # 1 trajectory window per batch for isolated rollouts
            num_workers=2,
            normalize=True,
            normalizer=normalizer,
            seed=args.seed,
        )

        step2_results = evaluate_autoregressive_rollouts(
            forecaster=forecaster,
            g0_head=g0_head,
            test_loader=test_loader_rollout,
            normalizer=normalizer,
            device=device,
            horizon=args.horizon,
            num_samples=args.num_samples,
            eval_horizons=(1, 5, 10, 20, 30),
            seed=args.seed,
            max_rollout_windows=args.max_rollout_windows,
        )

        print("  Autoregressive Rollout VRMSE Summary across Horizons:")
        for h in (1, 5, 10, 20, 30):
            res_h = step2_results["per_horizon_metrics"][f"horizon_{h}"]
            print(
                f"    Step {h:02d} | D0 VRMSE: {res_h['D0']['ensemble_mean_vrmse']:.4f} | "
                f"G0 VRMSE: {res_h['G0']['ensemble_mean_vrmse']:.4f} (RMS Spread: {res_h['G0']['rms_spread_raw']:.4f}, SSR: {res_h['G0']['spread_skill_ratio_raw']:.3f}) | "
                f"G1 VRMSE: {res_h['G1']['ensemble_mean_vrmse']:.4f} (RMS Spread: {res_h['G1']['rms_spread_raw']:.4f}, SSR: {res_h['G1']['spread_skill_ratio_raw']:.3f})"
            )

    # 7. Formulate Final JSON Report and Validate Finiteness
    evaluation_report = {
        "metadata": {
            "evaluation_version": "ProbLatent-R1-Phase3-Evaluation-v2",
            "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(PROJECT_ROOT),
            "git_dirty": is_git_dirty(PROJECT_ROOT),
            "device": str(device),
            "seed": args.seed,
            "d0_checkpoint": {
                "path": args.d0_checkpoint,
                "sha256": compute_file_sha256(args.d0_checkpoint),
            },
            "g1_variance_head": {
                "path": args.g1_variance_head,
                "sha256": compute_file_sha256(args.g1_variance_head),
            },
            "g0_baseline_head": {
                "path": args.g0_baseline_checkpoint,
                "sha256": compute_file_sha256(args.g0_baseline_checkpoint),
            },
            "data_protocol": {
                "split_file": args.split_file,
                "split_hash": compute_split_hash_from_file(args.split_file),
                "normalizer_file": args.normalizer_path,
                "normalizer_hash": compute_normalizer_hash(normalizer),
            },
        },
        "preflight_gate": {
            "reproduced_val_nll": float(reproduced_val_nll),
            "gate_status": "PASSED",
        },
        "step1_single_step_test_evaluation": step1_results,
        "step2_autoregressive_rollout_evaluation": step2_results,
    }

    validate_phase3_metrics_finite(evaluation_report)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(evaluation_report, f, indent=2)

    print(f"\n[Phase 3 Complete] Comprehensive evaluation report saved to: {args.output_file}")


if __name__ == "__main__":
    main()
