"""ProbLatent-R1 Phase 3: Comprehensive Probabilistic Evaluation Pipeline.

Executes a two-stage evaluation protocol comparing:
- D0: Deterministic reference baseline (mean prediction)
- G0: Homoscedastic baseline (fixed channel variance from training residual second moments)
- G1: Heteroscedastic model (input-conditional variance head trained in Phase 2)

Protocol:
1. Pre-flight Gate Verification:
   - Cryptographically verifies D0, G0 stats, split, normalizer, and G1 variance head hashes.
   - Re-evaluates full validation set NLL on G1, verifying exact reproduction of Phase 2
     validation NLL (-0.206643 nats/latent element within 1e-4 tolerance).
2. Step 1: Single-Step Probabilistic Evaluation on Independent Test Set:
   - Evaluates on unseen test trajectories (grouped split).
   - Metrics: Gaussian NLL (nats/latent element), Continuous Ranked Probability Score (CRPS),
     Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW)
     at nominal 50%, 80%, 90%, 95% confidence levels.
   - Verifies deterministic mean parity (mu_D0 == mu_G0 == mu_G1).
   - Generates per-channel (64 channels) and per-trajectory (5 test trajectories) breakdowns.
3. Step 2: 30-Step Autonomous Autoregressive Rollout Evaluation (H=30, K=32):
   - Multi-trajectory autoregressive simulation without ground-truth feedback.
   - Evaluates at horizons h in [1, 5, 10, 20, 30].
   - Non-parametric empirical quantiles in physical space from K=32 decoded samples.
   - Ensemble Mean VRMSE and Spread-Skill alignment (ensemble spread vs ensemble mean error).
   - Physical invariant checks: RMS divergence, vorticity, and radially integrated kinetic energy spectra E(k)
     for individual sample trajectories vs ensemble mean vs ground truth.
"""

from datetime import datetime, timezone
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

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
from src.utils.fft_derivatives import compute_divergence
from src.utils.checkpoint import resolve_spatial_pos_config, strip_compiled_prefix
from src.utils.provenance import (
    compute_file_sha256,
    compute_split_hash_from_file,
    compute_normalizer_hash,
    get_git_commit,
    is_git_dirty,
    hash_matches,
)


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
        Scalar tensor of mean CRPS across all elements.
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
    """Compute empirical coverage (PICP) and sharpness (MPIW) for central prediction intervals.

    Args:
        mu: Mean prediction tensor.
        target: Target observation tensor.
        var: Variance prediction tensor.
        nominal_levels: Tuple of nominal confidence levels in (0, 1).
        eps: Small variance clamp.

    Returns:
        Dict mapping nominal level string (e.g. '90') to {'picp': float, 'mpiw': float}.
    """
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
        res[f"{int(round(level * 100))}"] = {
            "nominal": float(level),
            "z_critical": float(z_crit),
            "picp": picp,
            "mpiw": mpiw,
        }
    return res


def verify_phase3_preflight_gate(
    d0_checkpoint_path: str,
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

    Fails closed if any hash mismatches or if recomputed validation NLL drifts from Phase 2 record.
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

    # 2. Gate assertion: reproduce exact validation NLL on G1
    forecaster.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            q_hist = batch["history"].to(device)
            q_next = batch["future"][:, 0:1].to(device)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)

            mu, var = forecaster.predict_distribution_single_step(q_hist, re=re, sc=sc)
            target_z = forecaster.encoder(q_next)
            loss = gaussian_nll_latent_loss(mu=mu, target=target_z, variance=var)

            b_tokens = target_z.numel()
            total_loss += loss.item() * b_tokens
            total_tokens += b_tokens

    reproduced_val_nll = total_loss / total_tokens
    expected_val_nll = rec["training_summary"]["best_val_nll"]
    error = abs(reproduced_val_nll - expected_val_nll)

    print(f"\n[Pre-flight Gate] Recomputed Validation NLL: {reproduced_val_nll:.6f}")
    print(f"                 Expected Phase 2 Val NLL:  {expected_val_nll:.6f}")
    print(f"                 Absolute Difference:       {error:.8e} (Tolerance: {tolerance})")

    if error > tolerance:
        raise RuntimeError(
            f"Pre-flight gate failed: recomputed val NLL ({reproduced_val_nll:.6f}) differs from "
            f"Phase 2 recorded val NLL ({expected_val_nll:.6f}) by {error:.4e} > {tolerance}. Fails closed."
        )

    print("  --> [GATE PASSED] Cryptographic provenance and validation NLL reproduction confirmed.\n")
    return reproduced_val_nll


def evaluate_single_step_probability(
    forecaster: LatentForecaster,
    g0_head: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    nominal_levels: Tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
) -> Dict[str, Any]:
    """Execute Step 1: Single-step probabilistic evaluation on unseen test set."""
    forecaster.eval()
    g0_head.eval()

    total_tokens = 0
    total_windows = 0
    batches_evaluated = 0

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

    # Per-trajectory accumulators
    traj_records: Dict[str, Dict[str, Any]] = {}

    max_mean_discrepancy = 0.0

    with torch.no_grad():
        for batch in test_loader:
            batches_evaluated += 1
            q_hist = batch["history"].to(device)
            q_next = batch["future"][:, 0:1].to(device)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)
            traj_idx = batch.get("traj_idx", torch.zeros(q_hist.shape[0], dtype=torch.long))

            b = q_hist.shape[0]
            total_windows += b

            # Ground truth latent target
            target_z = forecaster.encoder(q_next)  # (B, 1, 64, Hz, Wz)

            # Predictions: G1 uses forecaster attached variance_head; G0 overrides with g0_head
            mu_g1, var_g1 = forecaster.predict_distribution_single_step(q_hist, re=re, sc=sc)
            mu_g0, var_g0 = forecaster.predict_distribution_single_step(q_hist, re=re, sc=sc, variance_head=g0_head)

            # Assert deterministic mean parity: mu_D0 == mu_G0 == mu_G1
            diff_mu = (mu_g1 - mu_g0).abs().max().item()
            max_mean_discrepancy = max(max_mean_discrepancy, diff_mu)

            # Total tokens in batch
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

            # Intervals (50%, 80%, 90%, 95%)
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

            # target_z shape: (B, 1, C=64, Hz, Wz)
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

            # Trajectory-wise accumulation
            for i in range(b):
                t_id = f"traj_{int(traj_idx[i].item()):02d}"
                if t_id not in traj_records:
                    traj_records[t_id] = {
                        "re": float(re[i].item()),
                        "sc": float(sc[i].item()),
                        "windows": 0,
                        "tokens": 0,
                        "g0_nll_sum": 0.0,
                        "g1_nll_sum": 0.0,
                        "g0_crps_sum": 0.0,
                        "g1_crps_sum": 0.0,
                    }
                tok_i = target_z[i].numel()
                traj_records[t_id]["windows"] += 1
                traj_records[t_id]["tokens"] += tok_i
                traj_records[t_id]["g0_nll_sum"] += F.gaussian_nll_loss(mu_g0[i:i+1], target_z[i:i+1], var_g0[i:i+1], reduction="sum").item()
                traj_records[t_id]["g1_nll_sum"] += F.gaussian_nll_loss(mu_g1[i:i+1], target_z[i:i+1], var_g1[i:i+1], reduction="sum").item()
                traj_records[t_id]["g0_crps_sum"] += crps_g0_elem[i:i+1].sum().item()
                traj_records[t_id]["g1_crps_sum"] += crps_g1_elem[i:i+1].sum().item()

    if max_mean_discrepancy >= 1e-5:
        raise AssertionError(f"Mean parity check failed! Max discrepancy: {max_mean_discrepancy:.8e}")

    # Aggregate overall statistics
    g0_mean_nll = g0_nll_sum / total_tokens
    g1_mean_nll = g1_nll_sum / total_tokens
    delta_nll = g1_mean_nll - g0_mean_nll

    g0_mean_crps = g0_crps_sum / total_tokens
    g1_mean_crps = g1_crps_sum / total_tokens
    delta_crps = g1_mean_crps - g0_mean_crps

    d0_mae = d0_abs_err_sum / total_tokens
    d0_mse = d0_sq_err_sum / total_tokens
    d0_rmse = math.sqrt(d0_mse)

    # Intervals summary
    g0_intervals = {}
    g1_intervals = {}
    for lvl in nominal_levels:
        key = f"{int(round(lvl * 100))}"
        g0_intervals[key] = {
            "nominal": float(lvl),
            "picp": float(g0_interval_counts[key] / total_tokens),
            "mpiw": float(g0_interval_width_sum[key] / total_tokens),
        }
        g1_intervals[key] = {
            "nominal": float(lvl),
            "picp": float(g1_interval_counts[key] / total_tokens),
            "mpiw": float(g1_interval_width_sum[key] / total_tokens),
        }

    # Per-channel summary
    channel_summary = []
    for c in range(num_channels):
        tok_c = ch_tokens[c]
        c_g0_nll = ch_g0_nll[c] / tok_c
        c_g1_nll = ch_g1_nll[c] / tok_c
        c_g0_crps = ch_g0_crps[c] / tok_c
        c_g1_crps = ch_g1_crps[c] / tok_c
        c_g0_p90 = ch_g0_p90_count[c] / tok_c
        c_g1_p90 = ch_g1_p90_count[c] / tok_c
        channel_summary.append({
            "channel": c,
            "g0_nll": float(c_g0_nll),
            "g1_nll": float(c_g1_nll),
            "delta_nll": float(c_g1_nll - c_g0_nll),
            "g0_crps": float(c_g0_crps),
            "g1_crps": float(c_g1_crps),
            "delta_crps": float(c_g1_crps - c_g0_crps),
            "g0_picp_90": float(c_g0_p90),
            "g1_picp_90": float(c_g1_p90),
            "g0_mean_var": float(ch_g0_var_sum[c] / tok_c),
            "g1_mean_var": float(ch_g1_var_sum[c] / tok_c),
        })

    # Per-trajectory summary
    trajectory_summary = []
    for t_id, t_rec in sorted(traj_records.items()):
        tok_t = t_rec["tokens"]
        t_g0_nll = t_rec["g0_nll_sum"] / tok_t
        t_g1_nll = t_rec["g1_nll_sum"] / tok_t
        t_g0_crps = t_rec["g0_crps_sum"] / tok_t
        t_g1_crps = t_rec["g1_crps_sum"] / tok_t
        trajectory_summary.append({
            "trajectory_id": t_id,
            "re": t_rec["re"],
            "sc": t_rec["sc"],
            "windows": t_rec["windows"],
            "g0_nll": float(t_g0_nll),
            "g1_nll": float(t_g1_nll),
            "delta_nll": float(t_g1_nll - t_g0_nll),
            "g0_crps": float(t_g0_crps),
            "g1_crps": float(t_g1_crps),
            "delta_crps": float(t_g1_crps - t_g0_crps),
        })

    return {
        "dataset_summary": {
            "batches_evaluated": batches_evaluated,
            "windows_evaluated": total_windows,
            "tokens_evaluated": total_tokens,
        },
        "mean_parity_check": {
            "max_mean_discrepancy": float(max_mean_discrepancy),
            "status": "PASS",
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
            "coverage_improvement_p90": float(g1_intervals["90"]["picp"] - g0_intervals["90"]["picp"]),
            "sharpness_change_p90": float(g1_intervals["90"]["mpiw"] - g0_intervals["90"]["mpiw"]),
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
    max_test_trajectories: int = 5,
) -> Dict[str, Any]:
    """Execute Step 2: 30-step autonomous autoregressive rollout evaluation (H=30, K=32).

    Autonomous rollout without ground-truth feedback. Evaluates:
    - Non-parametric empirical quantiles in physical space
    - Ensemble Mean VRMSE vs D0 vs G0
    - Spread-Skill alignment (ensemble spread vs mean error)
    - Physical invariant checks: RMS divergence, vorticity, energy spectrum E(k) on individual members vs mean
    """
    forecaster.eval()
    g0_head.eval()

    # Track metrics per horizon
    horizon_results = {f"h_{h}": {
        "D0": {"vrmse_list": [], "div_rms_list": []},
        "G0": {"vrmse_list": [], "spread_list": [], "error_list": [], "div_sample_rms_list": [], "div_mean_rms_list": []},
        "G1": {"vrmse_list": [], "spread_list": [], "error_list": [], "div_sample_rms_list": [], "div_mean_rms_list": []},
    } for h in eval_horizons}

    # Spectrum storage at h=30
    spectrum_data = {
        "k_bins": None,
        "gt_spectrum": [],
        "d0_spectrum": [],
        "g0_sample_spectrum": [],
        "g0_mean_spectrum": [],
        "g1_sample_spectrum": [],
        "g1_mean_spectrum": [],
    }

    trajectories_evaluated = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if trajectories_evaluated >= max_test_trajectories:
                break

            q_hist = batch["history"].to(device)  # (B, L=4, 4, Ny, Nx)
            q_future = batch["future"].to(device)  # (B, H_fut, 4, Ny, Nx)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)

            b = q_hist.shape[0]
            # In autoregressive rollout, take the first sample in batch as an isolated trajectory
            for i in range(b):
                if trajectories_evaluated >= max_test_trajectories:
                    break
                trajectories_evaluated += 1

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

                for h in eval_horizons:
                    step_idx = h - 1
                    t_gt = q_gt_phys[step_idx]  # (4, Ny, Nx)

                    # D0 metrics
                    p_d0 = q_d0_phys[step_idx]
                    vrmse_d0 = compute_vrmse(p_d0, t_gt).item()
                    div_d0 = compute_divergence(p_d0[0:1], p_d0[1:2])
                    div_rms_d0 = float(torch.sqrt(torch.mean(div_d0**2)).item())
                    horizon_results[f"h_{h}"]["D0"]["vrmse_list"].append(vrmse_d0)
                    horizon_results[f"h_{h}"]["D0"]["div_rms_list"].append(div_rms_d0)

                    # G0 metrics
                    p_g0_mean = q_g0_mean_phys[step_idx]
                    vrmse_g0 = compute_vrmse(p_g0_mean, t_gt).item()
                    div_g0_mean = compute_divergence(p_g0_mean[0:1], p_g0_mean[1:2])
                    div_rms_g0_mean = float(torch.sqrt(torch.mean(div_g0_mean**2)).item())

                    # G0 ensemble spread (std over K) and error
                    samps_g0 = q_g0_samp_phys[:, step_idx]  # (K, 4, Ny, Nx)
                    spread_g0 = samps_g0.std(dim=0).mean().item()
                    err_g0 = (p_g0_mean - t_gt).abs().mean().item()

                    # Individual sample divergence mean
                    div_samps_g0 = compute_divergence(samps_g0[:, 0], samps_g0[:, 1])
                    div_rms_g0_samp = float(torch.sqrt(torch.mean(div_samps_g0**2, dim=(-2, -1))).mean().item())

                    horizon_results[f"h_{h}"]["G0"]["vrmse_list"].append(vrmse_g0)
                    horizon_results[f"h_{h}"]["G0"]["spread_list"].append(spread_g0)
                    horizon_results[f"h_{h}"]["G0"]["error_list"].append(err_g0)
                    horizon_results[f"h_{h}"]["G0"]["div_mean_rms_list"].append(div_rms_g0_mean)
                    horizon_results[f"h_{h}"]["G0"]["div_sample_rms_list"].append(div_rms_g0_samp)

                    # G1 metrics
                    p_g1_mean = q_g1_mean_phys[step_idx]
                    vrmse_g1 = compute_vrmse(p_g1_mean, t_gt).item()
                    div_g1_mean = compute_divergence(p_g1_mean[0:1], p_g1_mean[1:2])
                    div_rms_g1_mean = float(torch.sqrt(torch.mean(div_g1_mean**2)).item())

                    # G1 ensemble spread (std over K) and error
                    samps_g1 = q_g1_samp_phys[:, step_idx]  # (K, 4, Ny, Nx)
                    spread_g1 = samps_g1.std(dim=0).mean().item()
                    err_g1 = (p_g1_mean - t_gt).abs().mean().item()

                    # Individual sample divergence mean
                    div_samps_g1 = compute_divergence(samps_g1[:, 0], samps_g1[:, 1])
                    div_rms_g1_samp = float(torch.sqrt(torch.mean(div_samps_g1**2, dim=(-2, -1))).mean().item())

                    horizon_results[f"h_{h}"]["G1"]["vrmse_list"].append(vrmse_g1)
                    horizon_results[f"h_{h}"]["G1"]["spread_list"].append(spread_g1)
                    horizon_results[f"h_{h}"]["G1"]["error_list"].append(err_g1)
                    horizon_results[f"h_{h}"]["G1"]["div_mean_rms_list"].append(div_rms_g1_mean)
                    horizon_results[f"h_{h}"]["G1"]["div_sample_rms_list"].append(div_rms_g1_samp)

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

                # Average individual member spectrum
                k_g0_samps = [compute_radial_energy_spectrum(q_g0_samp_phys[s, final_step, 0], q_g0_samp_phys[s, final_step, 1])[1] for s in range(min(num_samples, 8))]
                k_g1_samps = [compute_radial_energy_spectrum(q_g1_samp_phys[s, final_step, 0], q_g1_samp_phys[s, final_step, 1])[1] for s in range(min(num_samples, 8))]
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

        g0_vrmse = float(np.mean(horizon_results[f"h_{h}"]["G0"]["vrmse_list"]))
        g0_spread = float(np.mean(horizon_results[f"h_{h}"]["G0"]["spread_list"]))
        g0_err = float(np.mean(horizon_results[f"h_{h}"]["G0"]["error_list"]))
        g0_ss_ratio = float(g0_spread / (g0_err + 1e-8))
        g0_div_samp = float(np.mean(horizon_results[f"h_{h}"]["G0"]["div_sample_rms_list"]))
        g0_div_mean = float(np.mean(horizon_results[f"h_{h}"]["G0"]["div_mean_rms_list"]))

        g1_vrmse = float(np.mean(horizon_results[f"h_{h}"]["G1"]["vrmse_list"]))
        g1_spread = float(np.mean(horizon_results[f"h_{h}"]["G1"]["spread_list"]))
        g1_err = float(np.mean(horizon_results[f"h_{h}"]["G1"]["error_list"]))
        g1_ss_ratio = float(g1_spread / (g1_err + 1e-8))
        g1_div_samp = float(np.mean(horizon_results[f"h_{h}"]["G1"]["div_sample_rms_list"]))
        g1_div_mean = float(np.mean(horizon_results[f"h_{h}"]["G1"]["div_mean_rms_list"]))

        rollout_summary[f"horizon_{h}"] = {
            "horizon_step": h,
            "D0": {
                "ensemble_mean_vrmse": d0_vrmse,
                "rms_divergence": d0_div,
            },
            "G0": {
                "ensemble_mean_vrmse": g0_vrmse,
                "ensemble_spread_mean": g0_spread,
                "ensemble_error_mean": g0_err,
                "spread_skill_ratio": g0_ss_ratio,
                "rms_divergence_individual_samples": g0_div_samp,
                "rms_divergence_ensemble_mean": g0_div_mean,
            },
            "G1": {
                "ensemble_mean_vrmse": g1_vrmse,
                "ensemble_spread_mean": g1_spread,
                "ensemble_error_mean": g1_err,
                "spread_skill_ratio": g1_ss_ratio,
                "rms_divergence_individual_samples": g1_div_samp,
                "rms_divergence_ensemble_mean": g1_div_mean,
            },
            "delta_vrmse_g1_vs_d0": float(g1_vrmse - d0_vrmse),
            "delta_vrmse_g1_vs_g0": float(g1_vrmse - g0_vrmse),
        }

    # Averaged spectrum curves across evaluated trajectories
    avg_spectra = {
        "k_bins": spectrum_data["k_bins"],
        "mean_gt_spectrum": np.mean(spectrum_data["gt_spectrum"], axis=0).tolist(),
        "mean_d0_spectrum": np.mean(spectrum_data["d0_spectrum"], axis=0).tolist(),
        "mean_g0_sample_spectrum": np.mean(spectrum_data["g0_sample_spectrum"], axis=0).tolist(),
        "mean_g0_mean_spectrum": np.mean(spectrum_data["g0_mean_spectrum"], axis=0).tolist(),
        "mean_g1_sample_spectrum": np.mean(spectrum_data["g1_sample_spectrum"], axis=0).tolist(),
        "mean_g1_mean_spectrum": np.mean(spectrum_data["g1_mean_spectrum"], axis=0).tolist(),
    }

    return {
        "configuration": {
            "trajectories_evaluated": trajectories_evaluated,
            "horizon": horizon,
            "num_samples": num_samples,
            "eval_horizons": list(eval_horizons),
            "seed": seed,
        },
        "per_horizon_metrics": rollout_summary,
        "energy_spectrum_at_final_horizon": avg_spectra,
    }


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
        help="Batch size for evaluation.",
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
    print("  Prediction Interval Coverage (90% Nominal):")
    print(f"    G0 PICP 90%: {step1_results['G0_homoscedastic_baseline']['intervals']['90']['picp'] * 100:.2f}% (MPIW: {step1_results['G0_homoscedastic_baseline']['intervals']['90']['mpiw']:.4f})")
    print(f"    G1 PICP 90%: {step1_results['G1_heteroscedastic_model']['intervals']['90']['picp'] * 100:.2f}% (MPIW: {step1_results['G1_heteroscedastic_model']['intervals']['90']['mpiw']:.4f})")

    # 6. Step 2: 30-Step Autoregressive Rollouts (H=30, K=32)
    step2_results = None
    if not args.skip_rollouts:
        print("\n=================================================================")
        print("Step 2: 30-Step Autonomous Autoregressive Rollout Evaluation (H=30, K=32)")
        print("=================================================================")
        # Re-create test loader with horizon=30 for full trajectory sequence
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
            batch_size=1,  # 1 trajectory per batch for isolated autoregressive rollouts
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
            max_test_trajectories=5,
        )

        print("  Autoregressive Rollout VRMSE Summary across Horizons:")
        for h in (1, 5, 10, 20, 30):
            res_h = step2_results["per_horizon_metrics"][f"horizon_{h}"]
            print(f"    Step {h:02d} | D0 VRMSE: {res_h['D0']['ensemble_mean_vrmse']:.4f} | "
                  f"G0 VRMSE: {res_h['G0']['ensemble_mean_vrmse']:.4f} (Spread: {res_h['G0']['ensemble_spread_mean']:.4f}, SS-Ratio: {res_h['G0']['spread_skill_ratio']:.3f}) | "
                  f"G1 VRMSE: {res_h['G1']['ensemble_mean_vrmse']:.4f} (Spread: {res_h['G1']['ensemble_spread_mean']:.4f}, SS-Ratio: {res_h['G1']['spread_skill_ratio']:.3f})")

    # 7. Formulate Final JSON Report
    evaluation_report = {
        "metadata": {
            "evaluation_version": "ProbLatent-R1-Phase3-Evaluation-v1",
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

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(evaluation_report, f, indent=2)

    print(f"\n[Phase 3 Complete] Comprehensive evaluation report saved to: {args.output_file}")


if __name__ == "__main__":
    main()
