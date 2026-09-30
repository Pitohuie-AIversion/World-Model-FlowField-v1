"""Comprehensive Probabilistic & Physical Evaluation: FM-R1 Pilot vs G1 / G0 / D0 Baselines.

Strictly executes identical A/B evaluation protocol on the EXACT SAME VALIDATION SPLIT:
1. One-step Probabilistic Evaluation on 144 validation windows:
   - D0 deterministic baseline (mean error)
   - G0 homoscedastic baseline (analytical Gaussian & K-sample empirical)
   - G1 heteroscedastic baseline (analytical Gaussian & K-sample empirical)
   - FM (OT-CFM Pilot with full temperature sweep: 1.0, 0.8, 0.7, 0.6, 0.5, 0.4)
   - Common Random Numbers (CRN) across sampling runs
   - Phase 3 Pooled Spread-Skill Ratio (Bessel ddof=1, finite-K adjusted)
   - Metrics: Latent CRPS, Physical CRPS (per-channel u, v, p, s and mean),
     Quantile PICP (50%, 80%, 90%, 95%) pooled AND per physical channel, MPIW,
     Pooled Spread, and Pooled SSR
2. Multi-step Autoregressive Rollout & Physical Evaluation vs GROUND TRUTH (H=5, 10):
   - Trajectory-aware evaluation manifest covering all unique validation trajectories
   - Evaluates D0, G0, G1, and FM temperature sweep (1.0, 0.7, 0.5, 0.4)
   - Ensemble Mean VRMSE vs GT (and D0 vs GT)
   - Sample Mean VRMSE vs GT across all K ensemble members
   - RMS Divergence across all K ensemble members and ensemble mean vs GT
   - Vorticity RMSE across all K ensemble members and ensemble mean vs GT
   - Radial Energy Spectrum E(k) aggregated across windows and ensemble members:
     distinguishing ensemble-mean-field spectrum error, mean-member-spectrum error,
     and individual member spectral error mean ± std vs GT.
3. Fail-closed cryptographic provenance verification across all checkpoints and data protocol:
   - Enforces D0 SHA, split_hash, normalizer_hash, and seed consistency across G0, G1, and FM.
"""

from datetime import datetime, timezone
import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, Any, Tuple, Optional, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.probabilistic_latent_dynamics import VarianceHead2D
from src.models.latent_flow_matching import LatentFlowMatcher
from src.data.pipeline import create_flow_dataloaders
from src.data.normalization import FieldNormalizer
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


def compute_sample_crps_tensor(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute empirical CRPS across K ensemble samples for target y.

    Formula: CRPS(F_K, y) = 1/K sum_{k=1}^K |x_k - y| - 1/(2 K^2) sum_{i=1}^K sum_{j=1}^K |x_i - x_j|

    Args:
        samples: Tensor of shape (K, B, ...)
        target: Tensor of shape (B, ...)

    Returns:
        crps: Tensor of shape (B, ...) with elementwise CRPS.
    """
    k = samples.shape[0]
    term1 = torch.abs(samples - target.unsqueeze(0)).mean(dim=0)
    diff_matrix = torch.abs(samples.unsqueeze(1) - samples.unsqueeze(0))
    term2 = 0.5 * diff_matrix.mean(dim=(0, 1))
    return term1 - term2


def compute_gaussian_crps_analytical(
    mu: torch.Tensor,
    target: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute analytical CRPS for Gaussian distribution N(mu, var)."""
    sigma = torch.sqrt(torch.clamp(var, min=eps))
    z = (target - mu) / sigma
    phi_z = torch.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    phi_cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = sigma * (z * (2.0 * phi_cdf - 1.0) + 2.0 * phi_z - 1.0 / math.sqrt(math.pi))
    return crps


def compute_quantile_coverage_and_width(
    samples: torch.Tensor,
    target: torch.Tensor,
    nominal_levels: List[float] = [0.5, 0.8, 0.9, 0.95],
) -> Dict[str, Dict[str, float]]:
    """Compute empirical prediction interval coverage and width from K samples."""
    results = {}
    for alpha in nominal_levels:
        lower_q = (1.0 - alpha) / 2.0
        upper_q = 1.0 - lower_q

        q_low = torch.quantile(samples, lower_q, dim=0)
        q_high = torch.quantile(samples, upper_q, dim=0)

        in_interval = (target >= q_low) & (target <= q_high)
        picp = float(in_interval.float().mean().item())
        mpiw = float((q_high - q_low).mean().item())

        results[str(int(round(alpha * 100)))] = {
            "nominal": float(alpha),
            "picp": picp,
            "mpiw": mpiw,
            "signed_calibration_error": picp - alpha,
            "absolute_calibration_error": abs(picp - alpha),
        }
    return results


def compute_gaussian_analytical_intervals(
    mu: torch.Tensor,
    target: torch.Tensor,
    var: torch.Tensor,
    nominal_levels: List[float] = [0.5, 0.8, 0.9, 0.95],
    eps: float = 1e-8,
) -> Dict[str, Dict[str, float]]:
    """Compute analytical prediction intervals for Gaussian distribution N(mu, var)."""
    sigma = torch.sqrt(torch.clamp(var, min=eps))
    results = {}
    for level in nominal_levels:
        alpha = 1.0 - level
        z_crit = math.sqrt(2.0) * torch.erfinv(torch.tensor(1.0 - alpha)).item()
        lower = mu - z_crit * sigma
        upper = mu + z_crit * sigma
        inside = (target >= lower) & (target <= upper)
        picp = float(inside.float().mean().item())
        mpiw = float((upper - lower).mean().item())
        results[str(int(round(level * 100)))] = {
            "nominal": float(level),
            "picp": picp,
            "mpiw": mpiw,
            "signed_calibration_error": picp - level,
            "absolute_calibration_error": abs(picp - level),
        }
    return results


def compute_pooled_spread_skill(
    samples: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """Compute Phase 3-compatible Pooled RMS spread and skill with Bessel correction (ddof=1).

    Args:
        samples: Decoded ensemble samples of shape (K, B, 4, Ny, Nx).
        target: Physical ground truth of shape (B, 4, Ny, Nx).

    Returns:
        Dictionary containing pooled spread, pooled RMSE, SSR, and finite-K adjusted SSR.
    """
    k = samples.shape[0]
    if k < 2:
        raise ValueError(f"Ensemble size K={k} must be >= 2 for variance computation.")

    # Bessel-corrected sample variance across ensemble K
    var_k = samples.var(dim=0, unbiased=True)  # (B, 4, Ny, Nx)
    p_mean = samples.mean(dim=0)  # (B, 4, Ny, Nx)

    finite_k_factor = math.sqrt((k + 1.0) / k)

    # 1. Total physical field pooled
    total_var = var_k.mean()
    rms_spread_total = float(torch.sqrt(total_var).item())
    rms_spread_adj_total = float(rms_spread_total * finite_k_factor)
    mse_total = float(((p_mean - target) ** 2).mean().item())
    rmse_total = float(math.sqrt(mse_total))
    ssr_total = float(rms_spread_total / (rmse_total + eps))
    ssr_adj_total = float(rms_spread_adj_total / (rmse_total + eps))

    # 2. Velocity vector field (u=0, v=1)
    var_vel = (var_k[:, 0] + var_k[:, 1]).mean()
    rms_spread_vel = float(torch.sqrt(var_vel).item())
    rms_spread_adj_vel = float(rms_spread_vel * finite_k_factor)
    mse_vel = float(((p_mean[:, 0:2] - target[:, 0:2]) ** 2).mean().item() * 2.0)
    rmse_vel = float(math.sqrt(mse_vel))
    ssr_vel = float(rms_spread_vel / (rmse_vel + eps))
    ssr_adj_vel = float(rms_spread_adj_vel / (rmse_vel + eps))

    # 3. Per physical variable
    channel_names = ["u", "v", "p", "s"]
    per_var = {}
    for c_idx, name in enumerate(channel_names):
        c_var = var_k[:, c_idx].mean()
        c_spread = float(torch.sqrt(c_var).item())
        c_spread_adj = float(c_spread * finite_k_factor)
        c_mse = float(((p_mean[:, c_idx] - target[:, c_idx]) ** 2).mean().item())
        c_rmse = float(math.sqrt(c_mse))
        per_var[name] = {
            "mean_var": float(c_var.item()),
            "rms_spread": c_spread,
            "finite_k_adjusted_spread": c_spread_adj,
            "rmse": c_rmse,
            "spread_skill_ratio": float(c_spread / (c_rmse + eps)),
            "finite_k_adjusted_ssr": float(c_spread_adj / (c_rmse + eps)),
        }

    return {
        "K": k,
        "degrees_of_freedom": "ddof=1 (Bessel corrected unbiased sample variance)",
        "finite_k_inflation_factor": finite_k_factor,
        "pooled_rms_spread": rms_spread_total,
        "finite_k_adjusted_spread": rms_spread_adj_total,
        "pooled_rmse": rmse_total,
        "spread_skill_ratio": ssr_total,
        "finite_k_adjusted_ssr": ssr_adj_total,
        "velocity": {
            "rms_spread": rms_spread_vel,
            "finite_k_adjusted_spread": rms_spread_adj_vel,
            "rmse": rmse_vel,
            "spread_skill_ratio": ssr_vel,
            "finite_k_adjusted_ssr": ssr_adj_vel,
        },
        "per_variable": per_var,
    }


def verify_checkpoint_provenance(
    d0_checkpoint_path: str,
    g0_checkpoint_path: str,
    g1_checkpoint_path: str,
    fm_checkpoint_path: str,
    split_file_path: str,
    normalizer: FieldNormalizer,
    expected_seed: int = 42,
) -> Dict[str, Any]:
    """Strict fail-closed cryptographic identity, parent lineage, and protocol check."""
    runtime_d0_sha = compute_file_sha256(d0_checkpoint_path)
    runtime_split_hash = compute_split_hash_from_file(split_file_path)
    runtime_norm_hash = compute_normalizer_hash(normalizer)

    g0_data = torch.load(g0_checkpoint_path, map_location="cpu", weights_only=False)
    g1_data = torch.load(g1_checkpoint_path, map_location="cpu", weights_only=False)
    fm_data = torch.load(fm_checkpoint_path, map_location="cpu", weights_only=False)

    errors = []

    # Check G0
    g0_prov = g0_data.get("provenance", {})
    g0_d0_sha = g0_prov.get("d0_checkpoint", {}).get("sha256")
    if not g0_d0_sha or not hash_matches(runtime_d0_sha, g0_d0_sha, min_prefix_len=16):
        errors.append(f"G0 parent D0 SHA mismatch: runtime={runtime_d0_sha[:12]}, G0={str(g0_d0_sha)[:12]}")
    g0_split = g0_prov.get("data_protocol", {}).get("split_hash")
    if not g0_split or not hash_matches(runtime_split_hash, g0_split, min_prefix_len=16):
        errors.append(f"G0 split_hash mismatch: runtime={runtime_split_hash[:12]}, G0={str(g0_split)[:12]}")
    g0_norm = g0_prov.get("data_protocol", {}).get("normalizer_hash")
    if not g0_norm or not hash_matches(runtime_norm_hash, g0_norm, min_prefix_len=16):
        errors.append(f"G0 normalizer_hash mismatch: runtime={runtime_norm_hash[:12]}, G0={str(g0_norm)[:12]}")
    g0_seed = g0_prov.get("seed")
    if g0_seed is None or g0_seed != expected_seed:
        errors.append(f"G0 seed mismatch: expected {expected_seed}, got {g0_seed}")

    # Check G1
    g1_prov = g1_data.get("provenance", {})
    g1_d0_sha = g1_prov.get("d0_checkpoint", {}).get("sha256")
    if not g1_d0_sha or not hash_matches(runtime_d0_sha, g1_d0_sha, min_prefix_len=16):
        errors.append(f"G1 parent D0 SHA mismatch: runtime={runtime_d0_sha[:12]}, G1={str(g1_d0_sha)[:12]}")
    g1_split = g1_prov.get("data_protocol", {}).get("split_hash")
    if not g1_split or not hash_matches(runtime_split_hash, g1_split, min_prefix_len=16):
        errors.append(f"G1 split_hash mismatch: runtime={runtime_split_hash[:12]}, G1={str(g1_split)[:12]}")
    g1_norm = g1_prov.get("data_protocol", {}).get("normalizer_hash")
    if not g1_norm or not hash_matches(runtime_norm_hash, g1_norm, min_prefix_len=16):
        errors.append(f"G1 normalizer_hash mismatch: runtime={runtime_norm_hash[:12]}, G1={str(g1_norm)[:12]}")
    g1_seed = g1_prov.get("seed")
    if g1_seed is None or g1_seed != expected_seed:
        errors.append(f"G1 seed mismatch: expected {expected_seed}, got {g1_seed}")

    # Check FM
    fm_prov = fm_data.get("provenance", {})
    fm_d0_sha = fm_prov.get("d0_checkpoint", {}).get("sha256")
    if not fm_d0_sha or not hash_matches(runtime_d0_sha, fm_d0_sha, min_prefix_len=16):
        errors.append(f"FM parent D0 SHA mismatch: runtime={runtime_d0_sha[:12]}, FM={str(fm_d0_sha)[:12]}")
    fm_stats_d0_sha = fm_prov.get("residual_statistics", {}).get("d0_sha256")
    if not fm_stats_d0_sha or not hash_matches(runtime_d0_sha, fm_stats_d0_sha, min_prefix_len=16):
        errors.append(f"FM stats D0 SHA mismatch: runtime={runtime_d0_sha[:12]}, FM stats={str(fm_stats_d0_sha)[:12]}")
    fm_split = fm_prov.get("data_protocol", {}).get("split_hash")
    if not fm_split or not hash_matches(runtime_split_hash, fm_split, min_prefix_len=16):
        errors.append(f"FM split_hash mismatch: runtime={runtime_split_hash[:12]}, FM={str(fm_split)[:12]}")
    fm_norm = fm_prov.get("data_protocol", {}).get("normalizer_hash")
    if not fm_norm or not hash_matches(runtime_norm_hash, fm_norm, min_prefix_len=16):
        errors.append(f"FM normalizer_hash mismatch: runtime={runtime_norm_hash[:12]}, FM={str(fm_norm)[:12]}")
    fm_seed = fm_prov.get("seed")
    if fm_seed is None or fm_seed != expected_seed:
        errors.append(f"FM seed mismatch: expected {expected_seed}, got {fm_seed}")

    if errors:
        raise ValueError("Cryptographic provenance verification FAILED with errors:\n  " + "\n  ".join(errors))

    return {
        "status": "PASSED",
        "runtime_d0_sha256": runtime_d0_sha,
        "runtime_split_hash": runtime_split_hash,
        "runtime_normalizer_hash": runtime_norm_hash,
        "expected_seed": expected_seed,
        "g0_parent_d0_sha256": g0_d0_sha,
        "g1_parent_d0_sha256": g1_d0_sha,
        "fm_parent_d0_sha256": fm_d0_sha,
        "fm_residual_stats_d0_sha256": fm_stats_d0_sha,
    }


def load_all_models_for_evaluation(
    d0_checkpoint_path: str,
    g0_checkpoint_path: str,
    g1_checkpoint_path: str,
    fm_checkpoint_path: str,
    device: torch.device,
) -> Tuple[LatentForecaster, VarianceHead2D, VarianceHead2D, LatentFlowMatcher]:
    """Load D0 Forecaster backbone, G0 head, G1 head, and FM matcher with strict state_dict verification."""
    d0_data = torch.load(d0_checkpoint_path, map_location="cpu", weights_only=False)
    cfg = d0_data.get("config", {})
    pred_mode = cfg.get("prediction_mode", d0_data.get("prediction_mode", "residual"))
    emb_dim = cfg.get("embed_dim", 256)
    depth = cfg.get("depth", 6)
    num_heads = cfg.get("num_heads", 8)
    use_spatial_pos = resolve_spatial_pos_config(d0_data)

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
    forecaster_sd = d0_data.get("forecaster_state_dict", d0_data.get("model_state_dict", d0_data.get("state_dict", d0_data)))
    forecaster.load_state_dict(strip_compiled_prefix(forecaster_sd), strict=True)
    forecaster.eval()

    # G0 Homoscedastic Head
    g0_data = torch.load(g0_checkpoint_path, map_location="cpu", weights_only=False)
    g0_head = VarianceHead2D(embed_dim=emb_dim, latent_channels=64, variance_floor=1e-4).to(device)
    g0_head.load_state_dict(strip_compiled_prefix(g0_data["variance_head_state_dict"]), strict=True)
    g0_head.eval()

    # G1 Heteroscedastic Head
    g1_data = torch.load(g1_checkpoint_path, map_location="cpu", weights_only=False)
    g1_head = VarianceHead2D(embed_dim=emb_dim, latent_channels=64, variance_floor=1e-4).to(device)
    g1_head.load_state_dict(strip_compiled_prefix(g1_data["variance_head_state_dict"]), strict=True)
    g1_head.eval()

    # FM Flow Matcher
    fm_data = torch.load(fm_checkpoint_path, map_location="cpu", weights_only=False)
    fm_cfg = fm_data.get("config", {})
    fm_sd = strip_compiled_prefix(fm_data["flow_matcher_state_dict"])
    residual_scale = fm_sd.get("residual_scale", None)
    if residual_scale is None:
        residual_scale = fm_data.get("provenance", {}).get("residual_scale", None)
    if residual_scale is None and "residual_statistics" in fm_data.get("provenance", {}):
        residual_scale = fm_data["provenance"]["residual_statistics"].get("scale_vector")
    fm = LatentFlowMatcher(
        latent_channels=64,
        cond_dim=128,
        hidden_channels=fm_cfg.get("hidden_channels", 128),
        num_blocks=fm_cfg.get("num_blocks", 4),
        residual_scale=residual_scale,
    ).to(device)
    fm.load_state_dict(fm_sd, strict=True)
    fm.eval()

    return forecaster, g0_head, g1_head, fm


def build_rollout_manifest(
    dataset: Dataset,
    windows_per_traj: Optional[int] = None,
    max_trajectories: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Build a deterministic, trajectory-aware rollout manifest across unique trajectories.

    Args:
        dataset: Validation Dataset supporting get_window_metadata(i).
        windows_per_traj: Number of windows to select per trajectory. If None or -1, selects all windows.
        max_trajectories: Maximum number of unique trajectories to include.

    Returns:
        Tuple of (manifest_records, window_indices).
    """
    total_len = len(dataset)
    traj_to_indices: Dict[Tuple[str, int], List[int]] = {}

    for idx in range(total_len):
        if hasattr(dataset, "get_window_metadata"):
            meta = dataset.get_window_metadata(idx)
            key = (meta["source_file"], meta["traj_idx"])
        else:
            # Fallback for synthetic unit test datasets
            key = ("synthetic_source", idx // max(1, total_len // 2))
        if key not in traj_to_indices:
            traj_to_indices[key] = []
        traj_to_indices[key].append(idx)

    selected_trajs = list(traj_to_indices.keys())
    if max_trajectories is not None and max_trajectories > 0:
        selected_trajs = selected_trajs[:max_trajectories]

    manifest_records = []
    selected_indices = []

    for traj_key in selected_trajs:
        indices = traj_to_indices[traj_key]
        if windows_per_traj is None or windows_per_traj <= 0 or windows_per_traj >= len(indices):
            chosen = indices
        else:
            step = (len(indices) - 1) / max(1, windows_per_traj - 1) if windows_per_traj > 1 else 0
            chosen = [indices[int(round(i * step))] for i in range(windows_per_traj)]

        for w_idx in chosen:
            selected_indices.append(w_idx)
            if hasattr(dataset, "get_window_metadata"):
                meta = dataset.get_window_metadata(w_idx)
                manifest_records.append({
                    "window_index": w_idx,
                    "source_file": meta["source_file"],
                    "traj_idx": meta["traj_idx"],
                    "start_t": meta["start_t"],
                    "cluster_id": meta.get("cluster_id", -1),
                    "re": float(meta.get("re", 0.0)),
                    "sc": float(meta.get("sc", 0.0)),
                })
            else:
                manifest_records.append({
                    "window_index": w_idx,
                    "source_file": traj_key[0],
                    "traj_idx": traj_key[1],
                    "start_t": w_idx,
                    "cluster_id": 0,
                    "re": 10000.0,
                    "sc": 0.5,
                })

    return manifest_records, selected_indices


def evaluate_one_step_comparative(
    forecaster: LatentForecaster,
    g0_head: VarianceHead2D,
    g1_head: VarianceHead2D,
    fm: LatentFlowMatcher,
    dataloader: DataLoader,
    normalizer: FieldNormalizer,
    device: torch.device,
    num_samples_K: int = 32,
    fm_temperatures: List[float] = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4],
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute rigorous one-step probabilistic comparison with Common Random Numbers."""
    generator = torch.Generator(device=device if device.type != "mps" else "cpu")
    generator.manual_seed(seed)

    channel_names = ["u", "v", "p", "s"]
    all_models = ["G0", "G1"] + [f"FM_temp_{t}" for t in fm_temperatures]

    # Cumulative accumulators
    records = {m: {
        "crps_lat_emp": [], "crps_phys": [],
        "lat_samples": [], "phys_samples": [],
    } for m in all_models}
    records["D0"] = {"mse_phys": []}
    records["G0"]["crps_lat_ana"] = []
    records["G1"]["crps_lat_ana"] = []

    # Store full pooled tensors for Phase 3 pooled SSR
    pooled_phys_samps = {m: [] for m in all_models}
    pooled_phys_targets = []

    phys_crps_by_channel = {m: {c: [] for c in channel_names} for m in all_models}
    lat_targets_sub = []
    phys_targets_sub = []

    total_windows = 0

    with torch.no_grad():
        for batch in dataloader:
            q_hist = batch["history"].to(device)  # (B, L, 4, Ny, Nx)
            q_target = batch["future"][:, :1].to(device)  # (B, 1, 4, Ny, Nx)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)
            b = q_hist.shape[0]
            total_windows += b

            # 1. Shared Backbone Forward
            z_hist = forecaster.encoder(q_hist)
            z_target = forecaster.encoder(q_target)
            c_z = z_target.shape[2]
            hz = z_target.shape[3]
            wz = z_target.shape[4]

            # G0 & G1 dynamic variance predictions
            mu, var_g0 = forecaster.transformer.predict_distribution(z_hist, re=re, sc=sc, variance_head=g0_head)
            _, var_g1 = forecaster.transformer.predict_distribution(z_hist, re=re, sc=sc, variance_head=g1_head)

            # Ground truth in physical space
            q_target_phys = apply_pressure_gauge(normalizer.denormalize(q_target))  # (B, 1, 4, Ny, Nx)
            pooled_phys_targets.append(q_target_phys.squeeze(1).cpu())

            # D0 Deterministic
            q_d0_phys = apply_pressure_gauge(normalizer.denormalize(forecaster.decoder(mu.squeeze(1)).unsqueeze(1)))
            records["D0"]["mse_phys"].append(torch.mean((q_d0_phys - q_target_phys) ** 2).item() * b)

            # -------------------------------------------------------------
            # Common Random Numbers (CRN): Base unit normal noise for batch
            # -------------------------------------------------------------
            base_eps = torch.randn((num_samples_K, b, 1, c_z, hz, wz), generator=generator, device=device)

            # 2. G0 Homoscedastic Gaussian
            crps_g0_ana = compute_gaussian_crps_analytical(mu=mu, target=z_target, var=var_g0).mean()
            records["G0"]["crps_lat_ana"].append(crps_g0_ana.item() * b)

            std_g0 = torch.sqrt(var_g0)
            z_samps_g0 = mu.unsqueeze(0) + base_eps * std_g0.unsqueeze(0)
            crps_g0_emp = compute_sample_crps_tensor(z_samps_g0.squeeze(2), z_target.squeeze(1)).mean()
            records["G0"]["crps_lat_emp"].append(crps_g0_emp.item() * b)

            z_g0_flat = z_samps_g0.view(num_samples_K * b, c_z, hz, wz)
            q_g0_phys = apply_pressure_gauge(normalizer.denormalize(
                forecaster.decoder(z_g0_flat).view(num_samples_K, b, 1, 4, q_target.shape[3], q_target.shape[4])
            ))
            crps_g0_ph = compute_sample_crps_tensor(q_g0_phys.squeeze(2), q_target_phys.squeeze(1))
            records["G0"]["crps_phys"].append(crps_g0_ph.mean().item() * b)
            for c_idx, c_name in enumerate(channel_names):
                phys_crps_by_channel["G0"][c_name].append(crps_g0_ph[:, c_idx].mean().item() * b)
            pooled_phys_samps["G0"].append(q_g0_phys.squeeze(2).cpu())

            # 3. G1 Heteroscedastic Gaussian
            crps_g1_ana = compute_gaussian_crps_analytical(mu=mu, target=z_target, var=var_g1).mean()
            records["G1"]["crps_lat_ana"].append(crps_g1_ana.item() * b)

            std_g1 = torch.sqrt(var_g1)
            z_samps_g1 = mu.unsqueeze(0) + base_eps * std_g1.unsqueeze(0)
            crps_g1_emp = compute_sample_crps_tensor(z_samps_g1.squeeze(2), z_target.squeeze(1)).mean()
            records["G1"]["crps_lat_emp"].append(crps_g1_emp.item() * b)

            z_g1_flat = z_samps_g1.view(num_samples_K * b, c_z, hz, wz)
            q_g1_phys = apply_pressure_gauge(normalizer.denormalize(
                forecaster.decoder(z_g1_flat).view(num_samples_K, b, 1, 4, q_target.shape[3], q_target.shape[4])
            ))
            crps_g1_ph = compute_sample_crps_tensor(q_g1_phys.squeeze(2), q_target_phys.squeeze(1))
            records["G1"]["crps_phys"].append(crps_g1_ph.mean().item() * b)
            for c_idx, c_name in enumerate(channel_names):
                phys_crps_by_channel["G1"][c_name].append(crps_g1_ph[:, c_idx].mean().item() * b)
            pooled_phys_samps["G1"].append(q_g1_phys.squeeze(2).cpu())

            # 4. FM Model with temperature scanning (reusing same base_eps for exact CRN)
            base_eps_flat = base_eps.permute(1, 0, 2, 3, 4, 5).contiguous().view(b * num_samples_K, c_z, hz, wz)
            for temp in fm_temperatures:
                m_key = f"FM_temp_{temp}"
                z_ens = fm.sample_ensemble(
                    mu=mu,
                    re=re,
                    sc=sc,
                    num_samples=num_samples_K,
                    num_steps=10,
                    solver="midpoint",
                    noise_scale=temp,
                    custom_x0=base_eps_flat,
                )
                z_samps_fm = z_ens.permute(1, 0, 2, 3, 4, 5)

                crps_fm_lat = compute_sample_crps_tensor(z_samps_fm.squeeze(2), z_target.squeeze(1)).mean()
                records[m_key]["crps_lat_emp"].append(crps_fm_lat.item() * b)

                z_fm_flat = z_ens.view(b * num_samples_K, c_z, hz, wz)
                q_fm_norm = forecaster.decoder(z_fm_flat).view(b, num_samples_K, 1, 4, q_target.shape[3], q_target.shape[4])
                q_fm_phys = apply_pressure_gauge(normalizer.denormalize(q_fm_norm.permute(1, 0, 2, 3, 4, 5)))

                crps_fm_ph = compute_sample_crps_tensor(q_fm_phys.squeeze(2), q_target_phys.squeeze(1))
                records[m_key]["crps_phys"].append(crps_fm_ph.mean().item() * b)
                for c_idx, c_name in enumerate(channel_names):
                    phys_crps_by_channel[m_key][c_name].append(crps_fm_ph[:, c_idx].mean().item() * b)
                pooled_phys_samps[m_key].append(q_fm_phys.squeeze(2).cpu())

                records[m_key]["lat_samples"].append(z_samps_fm[:, :, :, :, ::2, ::2].reshape(num_samples_K, -1).cpu())
                # Store unflattened channel-wise subsample for per-channel PICP
                records[m_key]["phys_samples"].append(q_fm_phys[:, :, 0, :, ::4, ::4].cpu())

            records["G0"]["lat_samples"].append(z_samps_g0[:, :, :, :, ::2, ::2].reshape(num_samples_K, -1).cpu())
            records["G0"]["phys_samples"].append(q_g0_phys[:, :, 0, :, ::4, ::4].cpu())
            records["G1"]["lat_samples"].append(z_samps_g1[:, :, :, :, ::2, ::2].reshape(num_samples_K, -1).cpu())
            records["G1"]["phys_samples"].append(q_g1_phys[:, :, 0, :, ::4, ::4].cpu())

            lat_targets_sub.append(z_target[:, :, :, ::2, ::2].reshape(-1).cpu())
            phys_targets_sub.append(q_target_phys[:, 0, :, ::4, ::4].cpu())

    # Compile Summary
    summary = {
        "validation_windows_evaluated": total_windows,
        "ensemble_size_K": num_samples_K,
        "common_random_numbers": True,
        "fm_temperatures_scanned": fm_temperatures,
        "D0": {
            "physical_rmse": math.sqrt(sum(records["D0"]["mse_phys"]) / total_windows),
        },
    }

    all_lat_targets = torch.cat(lat_targets_sub, dim=0)
    all_phys_targets = torch.cat(phys_targets_sub, dim=0)  # (N, 4, Ny_sub, Nx_sub)
    full_phys_target = torch.cat(pooled_phys_targets, dim=0)  # (N, 4, Ny, Nx)

    for m in all_models:
        all_m_lat = torch.cat(records[m]["lat_samples"], dim=1)
        all_m_phys = torch.cat(records[m]["phys_samples"], dim=1)  # (K, N, 4, Ny_sub, Nx_sub)
        full_m_samps = torch.cat(pooled_phys_samps[m], dim=1)  # (K, N, 4, Ny, Nx)

        # Pooled Spread-Skill Ratio according to Phase 3 contract
        spread_skill = compute_pooled_spread_skill(samples=full_m_samps, target=full_phys_target)

        # Per-channel physical intervals
        phys_intervals_per_channel = {}
        for c_idx, c_name in enumerate(channel_names):
            c_samps = all_m_phys[:, :, c_idx].reshape(num_samples_K, -1)
            c_target = all_phys_targets[:, c_idx].reshape(-1)
            phys_intervals_per_channel[c_name] = compute_quantile_coverage_and_width(c_samps, c_target)

        # Pooled physical intervals
        pooled_samps = all_m_phys.reshape(num_samples_K, -1)
        pooled_target = all_phys_targets.reshape(-1)

        m_dict = {
            "latent_crps_empirical": sum(records[m]["crps_lat_emp"]) / total_windows,
            "latent_intervals": compute_quantile_coverage_and_width(all_m_lat, all_lat_targets),
            "physical_crps_mean": sum(records[m]["crps_phys"]) / total_windows,
            "physical_crps_per_channel": {c: sum(phys_crps_by_channel[m][c]) / total_windows for c in channel_names},
            "phase3_pooled_spread_skill": spread_skill,
            "physical_intervals": compute_quantile_coverage_and_width(pooled_samps, pooled_target),
            "physical_intervals_per_channel": phys_intervals_per_channel,
        }
        if "crps_lat_ana" in records[m]:
            m_dict["latent_crps_analytical"] = sum(records[m]["crps_lat_ana"]) / total_windows

        summary[m] = m_dict

    return summary


def evaluate_rollout_physics_comparative(
    forecaster: LatentForecaster,
    g0_head: VarianceHead2D,
    g1_head: VarianceHead2D,
    fm: LatentFlowMatcher,
    dataloader: DataLoader,
    normalizer: FieldNormalizer,
    device: torch.device,
    horizons: List[int] = [5, 10],
    num_samples_K: int = 8,
    rollout_temperatures: List[float] = [1.0, 0.7, 0.5, 0.4],
    windows_per_traj: Optional[int] = None,
    max_trajectories: Optional[int] = None,
    seed: int = 42,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Execute multi-step autoregressive rollout comparing D0, G0, G1, and FM temperature variants against GT."""
    max_h = max(horizons)
    dataset = dataloader.dataset

    # 1. Build deterministic multi-trajectory window manifest
    manifest_records, selected_indices = build_rollout_manifest(
        dataset=dataset,
        windows_per_traj=windows_per_traj,
        max_trajectories=max_trajectories,
    )

    unique_trajs = set((r["source_file"], r["traj_idx"]) for r in manifest_records)
    unique_clusters = set(r["cluster_id"] for r in manifest_records)

    fm_model_names = [f"FM_temp_{t}" for t in rollout_temperatures]
    models = ["D0", "G0", "G1"] + fm_model_names

    # Metrics storage
    h_metrics = {m: {h: {
        "ens_vrmse": [], "samp_vrmse": [],
        "ens_div_sq": [], "samp_div_sq": [],
        "ens_vort_sq_err": [], "samp_vort_sq_err": [],
    } for h in horizons} for m in models}

    gt_div_sq_by_h = {h: [] for h in horizons}

    # Energy spectra storage at max_h (accumulating across windows)
    spectra_k_bins = None
    window_gt_spectra = []
    window_ens_spectra = {m: [] for m in models}
    window_mean_member_spectra = {m: [] for m in models}

    window_ens_rel_errors = {m: [] for m in models}
    window_mean_member_rel_errors = {m: [] for m in models}
    window_indiv_member_rel_errors = {m: [] for m in models}

    total_windows = len(selected_indices)

    with torch.no_grad():
        for b_start in range(0, total_windows, batch_size):
            b_indices = selected_indices[b_start: b_start + batch_size]
            b_len = len(b_indices)
            batch_items = [dataset[i] for i in b_indices]

            q_hist = torch.stack([item["history"] for item in batch_items]).to(device)  # (B, L, 4, Ny, Nx)
            q_gt_seq = torch.stack([item["future"][:max_h] for item in batch_items]).to(device)  # (B, H, 4, Ny, Nx)
            re = torch.tensor([float(item["re"]) for item in batch_items], device=device)
            sc = torch.tensor([float(item["sc"]) for item in batch_items], device=device)

            # Ground truth in physical space
            q_gt_phys = apply_pressure_gauge(normalizer.denormalize(q_gt_seq))  # (B, H, 4, Ny, Nx)

            # Shared seed for exact common random numbers across stochastic models
            w_seed = seed + b_start

            # 1. D0 & G0 / G1
            rollout_g0 = forecaster.sample_rollout(
                q_hist=q_hist, re=re, sc=sc, horizon=max_h,
                num_samples=num_samples_K, seed=w_seed,
                variance_head=g0_head, decode_samples=True,
            )
            rollout_g1 = forecaster.sample_rollout(
                q_hist=q_hist, re=re, sc=sc, horizon=max_h,
                num_samples=num_samples_K, seed=w_seed,
                variance_head=g1_head, decode_samples=True,
            )

            d0_phys = apply_pressure_gauge(normalizer.denormalize(rollout_g0["deterministic_rollout"]))  # (B, H, 4, Ny, Nx)
            g0_samps = apply_pressure_gauge(normalizer.denormalize(rollout_g0["sample_trajectories"].permute(1, 0, 2, 3, 4, 5)).permute(1, 0, 2, 3, 4, 5))
            g1_samps = apply_pressure_gauge(normalizer.denormalize(rollout_g1["sample_trajectories"].permute(1, 0, 2, 3, 4, 5)).permute(1, 0, 2, 3, 4, 5))

            rollout_data = {
                "D0": {"mean": d0_phys, "samps": d0_phys.unsqueeze(1)},
                "G0": {"mean": g0_samps.mean(dim=1), "samps": g0_samps},
                "G1": {"mean": g1_samps.mean(dim=1), "samps": g1_samps},
            }

            # 2. FM @ all requested rollout temperatures
            for temp in rollout_temperatures:
                m_key = f"FM_temp_{temp}"
                r_fm = forecaster.sample_rollout_flow_matching(
                    q_hist=q_hist, re=re, sc=sc, horizon=max_h,
                    num_samples=num_samples_K, seed=w_seed,
                    flow_matcher=fm, noise_scale=temp, decode_samples=True,
                )
                fm_samps = apply_pressure_gauge(normalizer.denormalize(r_fm["sample_trajectories"].permute(1, 0, 2, 3, 4, 5)).permute(1, 0, 2, 3, 4, 5))
                rollout_data[m_key] = {"mean": fm_samps.mean(dim=1), "samps": fm_samps}

            for h in horizons:
                gt_h = q_gt_phys[:, h - 1]  # (B, 4, Ny, Nx)
                u_gt = gt_h[:, 0]
                v_gt = gt_h[:, 1]
                div_gt = compute_divergence(u_gt, v_gt)
                vort_gt = compute_vorticity(u_gt, v_gt)

                gt_div_sq_by_h[h].append((div_gt ** 2).mean().item())

                for m in models:
                    pred_mean_h = rollout_data[m]["mean"][:, h - 1]  # (B, 4, Ny, Nx)
                    samps_h = rollout_data[m]["samps"][:, :, h - 1]  # (B, K, 4, Ny, Nx)
                    k_ens = samps_h.shape[1]

                    # 1. Ensemble Mean VRMSE vs GT
                    ens_vrmse = compute_vrmse(pred_mean_h, gt_h).item()
                    h_metrics[m][h]["ens_vrmse"].append(ens_vrmse)

                    # 2. Ensemble Mean Divergence & Vorticity
                    u_m = pred_mean_h[:, 0]
                    v_m = pred_mean_h[:, 1]
                    div_m = compute_divergence(u_m, v_m)
                    vort_m = compute_vorticity(u_m, v_m)
                    h_metrics[m][h]["ens_div_sq"].append((div_m ** 2).mean().item())
                    h_metrics[m][h]["ens_vort_sq_err"].append(((vort_m - vort_gt) ** 2).mean().item())

                    # 3. All K Ensemble Members Divergence, Vorticity, and VRMSE
                    b_samp_vrmses = []
                    b_samp_divs = []
                    b_samp_vorts = []
                    for k_idx in range(k_ens):
                        u_k = samps_h[:, k_idx, 0]
                        v_k = samps_h[:, k_idx, 1]
                        b_samp_vrmses.append(compute_vrmse(samps_h[:, k_idx], gt_h).item())
                        div_k = compute_divergence(u_k, v_k)
                        b_samp_divs.append((div_k ** 2).mean().item())
                        vort_k = compute_vorticity(u_k, v_k)
                        b_samp_vorts.append(((vort_k - vort_gt) ** 2).mean().item())

                    h_metrics[m][h]["samp_vrmse"].append(float(np.mean(b_samp_vrmses)))
                    h_metrics[m][h]["samp_div_sq"].append(float(np.mean(b_samp_divs)))
                    h_metrics[m][h]["samp_vort_sq_err"].append(float(np.mean(b_samp_vorts)))

            # Energy Spectrum at max horizon (h = max_h) across all windows in batch
            for b_idx in range(b_len):
                u_gt_w = q_gt_phys[b_idx, max_h - 1, 0]
                v_gt_w = q_gt_phys[b_idx, max_h - 1, 1]
                k_bins, e_gt = compute_radial_energy_spectrum(u_gt_w, v_gt_w)
                if spectra_k_bins is None:
                    spectra_k_bins = [float(x) for x in k_bins[:25].cpu().numpy()]
                e_gt_arr = e_gt[:25].cpu().numpy()
                window_gt_spectra.append(e_gt_arr)
                gt_norm = np.linalg.norm(e_gt_arr) + 1e-8

                for m in models:
                    # 1. Ensemble-mean-field spectrum
                    u_ens_w = rollout_data[m]["mean"][b_idx, max_h - 1, 0]
                    v_ens_w = rollout_data[m]["mean"][b_idx, max_h - 1, 1]
                    _, e_ens = compute_radial_energy_spectrum(u_ens_w, v_ens_w)
                    e_ens_arr = e_ens[:25].cpu().numpy()
                    window_ens_spectra[m].append(e_ens_arr)
                    rel_err_ens = float(np.linalg.norm(e_ens_arr - e_gt_arr) / gt_norm)
                    window_ens_rel_errors[m].append(rel_err_ens)

                    # 2. Member spectra & individual member errors
                    k_samps = rollout_data[m]["samps"].shape[1]
                    k_spec_list = []
                    k_indiv_rel_errs = []
                    for k_idx in range(k_samps):
                        u_k_w = rollout_data[m]["samps"][b_idx, k_idx, max_h - 1, 0]
                        v_k_w = rollout_data[m]["samps"][b_idx, k_idx, max_h - 1, 1]
                        _, e_k = compute_radial_energy_spectrum(u_k_w, v_k_w)
                        e_k_arr = e_k[:25].cpu().numpy()
                        k_spec_list.append(e_k_arr)
                        rel_err_k = float(np.linalg.norm(e_k_arr - e_gt_arr) / gt_norm)
                        k_indiv_rel_errs.append(rel_err_k)

                    e_samp_avg = np.mean(k_spec_list, axis=0)
                    window_mean_member_spectra[m].append(e_samp_avg)
                    rel_err_mean_member = float(np.linalg.norm(e_samp_avg - e_gt_arr) / gt_norm)
                    window_mean_member_rel_errors[m].append(rel_err_mean_member)
                    window_indiv_member_rel_errors[m].append(float(np.mean(k_indiv_rel_errs)))

    # Compile Rollout Summary
    avg_spectra = {
        "k_bins": spectra_k_bins,
        "GT": [float(x) for x in np.mean(window_gt_spectra, axis=0)],
    }
    for m in models:
        avg_spectra[f"{m}_ensemble_mean_field"] = [float(x) for x in np.mean(window_ens_spectra[m], axis=0)]
        avg_spectra[f"{m}_mean_member"] = [float(x) for x in np.mean(window_mean_member_spectra[m], axis=0)]

    spectral_stats = {}
    for m in models:
        spectral_stats[m] = {
            "ensemble_mean_field_spectrum_rel_error_mean": float(np.mean(window_ens_rel_errors[m])),
            "ensemble_mean_field_spectrum_rel_error_std": float(np.std(window_ens_rel_errors[m])),
            "mean_member_spectrum_rel_error_mean": float(np.mean(window_mean_member_rel_errors[m])),
            "mean_member_spectrum_rel_error_std": float(np.std(window_mean_member_rel_errors[m])),
            "individual_member_spectrum_rel_error_mean": float(np.mean(window_indiv_member_rel_errors[m])),
            "individual_member_spectrum_rel_error_std": float(np.std(window_indiv_member_rel_errors[m])),
        }

    rollout_summary = {
        "total_windows_evaluated": total_windows,
        "unique_trajectories_evaluated": len(unique_trajs),
        "unique_clusters_evaluated": len(unique_clusters),
        "rollout_temperatures_evaluated": rollout_temperatures,
        "window_manifest": manifest_records,
        "spectral_relative_error_vs_gt_h10": spectral_stats,
        "energy_spectra_first_25_modes": avg_spectra,
    }

    for h in horizons:
        h_key = f"h{h}"
        gt_rms_div = math.sqrt(float(np.mean(gt_div_sq_by_h[h])))
        h_dict = {
            "ground_truth_rms_divergence": gt_rms_div,
        }
        for m in models:
            ens_rms_div = math.sqrt(float(np.mean(h_metrics[m][h]["ens_div_sq"])))
            samp_rms_div = math.sqrt(float(np.mean(h_metrics[m][h]["samp_div_sq"])))
            ens_vort_rmse = math.sqrt(float(np.mean(h_metrics[m][h]["ens_vort_sq_err"])))
            samp_vort_rmse = math.sqrt(float(np.mean(h_metrics[m][h]["samp_vort_sq_err"])))
            h_dict[m] = {
                "ensemble_mean_vrmse_vs_gt": float(np.mean(h_metrics[m][h]["ens_vrmse"])),
                "sample_mean_vrmse_vs_gt": float(np.mean(h_metrics[m][h]["samp_vrmse"])),
                "ensemble_mean_rms_divergence": ens_rms_div,
                "sample_rms_divergence": samp_rms_div,
                "divergence_ratio_vs_gt": float(samp_rms_div / max(1e-6, gt_rms_div)),
                "ensemble_mean_vorticity_rmse_vs_gt": ens_vort_rmse,
                "sample_vorticity_rmse_vs_gt": samp_vort_rmse,
            }
        rollout_summary[h_key] = h_dict

    return rollout_summary


def main():
    parser = argparse.ArgumentParser(description="Strict Comparative Validation Benchmark: D0 vs G0 vs G1 vs FM.")
    parser.add_argument("--d0-checkpoint", type=str, default="outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt")
    parser.add_argument("--g0-checkpoint", type=str, default="outputs/checkpoints/probabilistic/variance_head/formal_r1_seed42/g0_baseline_initialization.pt")
    parser.add_argument("--g1-checkpoint", type=str, default="outputs/checkpoints/probabilistic/variance_head/formal_r1_seed42/best_g1_variance_head.pt")
    parser.add_argument("--fm-checkpoint", type=str, default="outputs/checkpoints/probabilistic/flow_matching_pilot3ep/best_latent_flow_matcher.pt")
    parser.add_argument("--normalizer-path", type=str, default="outputs/normalization/stats_grouped.pt")
    parser.add_argument("--split-file", type=str, default="outputs/splits/grouped_split.json")
    default_data_root = os.environ.get("SHEAR_FLOW_DATA_DIR", "/root/autodl-tmp/datasets/shear_flow" if os.path.exists("/root/autodl-tmp/datasets/shear_flow") else None)
    parser.add_argument("--data-root", type=str, default=default_data_root)
    parser.add_argument("--output-json", type=str, default="outputs/metrics/flow_matching_pilot3ep_evaluation.json")
    parser.add_argument("--num-samples-k", type=int, default=32, help="Ensemble K for one-step probability")
    parser.add_argument("--num-samples-rollout", type=int, default=8, help="Ensemble K for rollout trajectories")
    parser.add_argument("--fm-temperatures", type=str, default="1.0,0.8,0.7,0.6,0.5,0.4", help="Comma-separated temperature sweep for one-step")
    parser.add_argument("--rollout-temperatures", type=str, default="1.0,0.7,0.5,0.4", help="Comma-separated temperature sweep for rollout")
    parser.add_argument("--rollout-windows-per-traj", type=int, default=-1, help="Windows per trajectory for rollout (-1 for all)")
    parser.add_argument("--max-rollout-trajectories", type=int, default=6, help="Max trajectories for rollout")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-dirty", action="store_true", help="Allow dirty working tree for development runs")
    args = parser.parse_args()

    eval_git_commit = get_git_commit(str(PROJECT_ROOT))
    eval_git_dirty_at_start = is_git_dirty(str(PROJECT_ROOT))
    if eval_git_dirty_at_start and not args.allow_dirty:
        raise RuntimeError("Formal evaluation requires a clean git worktree at launch. (Fail-Closed)")

    fm_temps = [float(t.strip()) for t in args.fm_temperatures.split(",") if t.strip()]
    rollout_temps = [float(t.strip()) for t in args.rollout_temperatures.split(",") if t.strip()]

    device = torch.device(args.device)
    print("=" * 70)
    print("=== PROBABILISTIC WORLD MODEL RIGOROUS A/B BENCHMARK (ON VALIDATION SET) ===")
    print(f"Device: {device} | Seed: {args.seed} | One-step K: {args.num_samples_k} | Rollout K: {args.num_samples_rollout}")
    print(f"One-step Temperatures: {fm_temps} | Rollout Temperatures: {rollout_temps}")
    print("=" * 70)

    # 1. Load Data
    normalizer = FieldNormalizer()
    normalizer.load_state_dict(torch.load(args.normalizer_path, weights_only=True, map_location="cpu"))

    # Fail-closed cryptographic provenance check
    prov_report = verify_checkpoint_provenance(
        d0_checkpoint_path=args.d0_checkpoint,
        g0_checkpoint_path=args.g0_checkpoint,
        g1_checkpoint_path=args.g1_checkpoint,
        fm_checkpoint_path=args.fm_checkpoint,
        split_file_path=args.split_file,
        normalizer=normalizer,
        expected_seed=args.seed,
    )
    print("[Provenance Verified] Cryptographic binding confirmed: D0 SHA, split_hash, normalizer_hash, and seeds strictly match.")

    _, valid_loader, _, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file=args.split_file,
        data_root=args.data_root,
        history_length=4,
        horizon=1,
        valid_horizon=10,
        train_stride=8,
        valid_stride=8,
        downsample_factor=2,
        batch_size=8,
        num_workers=2,
        normalize=True,
        normalizer=normalizer,
        seed=args.seed,
    )
    print(f"Validation DataLoader ready: {len(valid_loader.dataset)} windows with H=10 future ground truth.")

    # 2. Load Models
    forecaster, g0_head, g1_head, fm = load_all_models_for_evaluation(
        d0_checkpoint_path=args.d0_checkpoint,
        g0_checkpoint_path=args.g0_checkpoint,
        g1_checkpoint_path=args.g1_checkpoint,
        fm_checkpoint_path=args.fm_checkpoint,
        device=device,
    )

    # 3. Run One-Step Probabilistic Benchmark
    print("\n" + "=" * 50)
    print(">>> 1. RUNNING ONE-STEP PROBABILISTIC BENCHMARK (K=32, CRN) <<<")
    print("=" * 50)
    step1_results = evaluate_one_step_comparative(
        forecaster=forecaster,
        g0_head=g0_head,
        g1_head=g1_head,
        fm=fm,
        dataloader=valid_loader,
        normalizer=normalizer,
        device=device,
        num_samples_K=args.num_samples_k,
        fm_temperatures=fm_temps,
        seed=args.seed,
    )

    print("\n--- ONE-STEP PROBABILISTIC COMPARISON (ON SAME 144 VALIDATION WINDOWS) ---")
    print(f"{'Model':<12} | {'Latent CRPS':<11} | {'Phys CRPS':<13} | {'Pooled Spread':<13} | {'Pooled RMSE':<11} | {'Pooled SSR':<10} | {'PICP 50%':<8} | {'PICP 90%':<8}")
    print("-" * 110)
    for m in ["G0", "G1"] + [f"FM_temp_{t}" for t in fm_temps]:
        m_res = step1_results[m]
        p_ss = m_res["phase3_pooled_spread_skill"]
        p_ints = m_res["physical_intervals"]
        print(f"{m:<12} | {m_res['latent_crps_empirical']:<11.4f} | {m_res['physical_crps_mean']:<13.5f} | {p_ss['pooled_rms_spread']:<13.4f} | {p_ss['pooled_rmse']:<11.4f} | {p_ss['spread_skill_ratio']:<10.3f} | {p_ints['50']['picp']*100:<7.1f}% | {p_ints['90']['picp']*100:<7.1f}%")

    # 4. Run Multi-step Rollout & Physics Evaluation
    print("\n" + "=" * 50)
    print(f">>> 2. RUNNING MULTI-STEP ROLLOUT & PHYSICS VS GROUND TRUTH (H=5, 10) <<<")
    print("=" * 50)
    step2_results = evaluate_rollout_physics_comparative(
        forecaster=forecaster,
        g0_head=g0_head,
        g1_head=g1_head,
        fm=fm,
        dataloader=valid_loader,
        normalizer=normalizer,
        device=device,
        horizons=[5, 10],
        num_samples_K=args.num_samples_rollout,
        rollout_temperatures=rollout_temps,
        windows_per_traj=args.rollout_windows_per_traj,
        max_trajectories=args.max_rollout_trajectories,
        seed=args.seed,
    )

    rollout_models = ["D0", "G0", "G1"] + [f"FM_temp_{t}" for t in rollout_temps]
    print("\n--- MULTI-STEP PHYSICAL PERFORMANCE VS GROUND TRUTH ---")
    for h in [5, 10]:
        h_res = step2_results[f"h{h}"]
        gt_div = h_res["ground_truth_rms_divergence"]
        print(f"\n[ Horizon h={h} ] Ground Truth RMS Divergence = {gt_div:.5f}")
        print(f"{'Model':<12} | {'Ens VRMSE vs GT':<16} | {'Samp VRMSE vs GT':<16} | {'Samp Divergence':<15} | {'Div / GT Ratio':<14} | {'Samp Vorticity RMSE':<18}")
        print("-" * 105)
        for m in rollout_models:
            m_res = h_res[m]
            print(f"{m:<12} | {m_res['ensemble_mean_vrmse_vs_gt']:<16.4f} | {m_res['sample_mean_vrmse_vs_gt']:<16.4f} | {m_res['sample_rms_divergence']:<15.4f} | {m_res['divergence_ratio_vs_gt']:<13.1f}x | {m_res['sample_vorticity_rmse_vs_gt']:<18.4f}")

    print("\n--- RADIAL ENERGY SPECTRUM RELATIVE L2 ERROR VS GROUND TRUTH (h=10) ---")
    spec_res = step2_results["spectral_relative_error_vs_gt_h10"]
    for m in rollout_models:
        m_s = spec_res[m]
        print(f"  {m:<12}: Ens Field L2 Err = {m_s['ensemble_mean_field_spectrum_rel_error_mean']*100:.2f}% ± {m_s['ensemble_mean_field_spectrum_rel_error_std']*100:.2f}%, "
              f"Mean Member L2 Err = {m_s['mean_member_spectrum_rel_error_mean']*100:.2f}% ± {m_s['mean_member_spectrum_rel_error_std']*100:.2f}%, "
              f"Indiv Member L2 Err = {m_s['individual_member_spectrum_rel_error_mean']*100:.2f}% ± {m_s['individual_member_spectrum_rel_error_std']*100:.2f}%")

    eval_script_sha = compute_file_sha256(__file__)
    full_report = {
        "evaluation_protocol": "STRICT_SAME_VALIDATION_SPLIT_AB_TEST_V3_TEMPERATURE_DIAGNOSTIC",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": eval_git_commit,
        "is_git_dirty": eval_git_dirty_at_start,
        "git_dirty_at_start": eval_git_dirty_at_start,
        "evaluation_script_sha256": eval_script_sha,
        "provenance_verification": prov_report,
        "step1_one_step_probabilistic": step1_results,
        "step2_rollout_physics": step2_results,
    }

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nSuccessfully written full comparative benchmark report to {out_p}")


if __name__ == "__main__":
    main()
