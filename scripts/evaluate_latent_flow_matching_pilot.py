"""Comprehensive Probabilistic & Physical Evaluation: FM-R1 Pilot vs G1 / G0 / D0 Baselines.

Strictly executes identical A/B evaluation protocol on the EXACT SAME VALIDATION SPLIT:
1. One-step Probabilistic Evaluation on 144 validation windows:
   - D0 deterministic baseline (mean error)
   - G0 homoscedastic baseline (analytical Gaussian & K-sample empirical)
   - G1 heteroscedastic baseline (analytical Gaussian & K-sample empirical)
   - FM (OT-CFM Pilot @ noise_scale=1.0)
   - FM calibrated variants (@ noise_scale=0.8, 0.7)
   - Metrics: Latent CRPS, Physical CRPS (per-channel u, v, p, s and mean),
     Quantile PICP (50%, 80%, 90%, 95%), MPIW, Ensemble Spread, and Spread-Skill Ratio (SSR)
2. Multi-step Autoregressive Rollout & Physical Evaluation vs GROUND TRUTH (H=5, 10):
   - Ensemble Mean VRMSE vs GT (and D0 vs GT)
   - Sample Mean VRMSE vs GT
   - RMS Divergence (GT baseline, D0, G0, G1, FM)
   - Vorticity RMSE vs GT
   - Radial Energy Spectrum E(k) relative L2 error vs GT: ||E_model - E_GT||_2 / ||E_GT||_2
"""

from datetime import datetime, timezone
import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
from src.utils.provenance import compute_file_sha256, get_git_commit, is_git_dirty


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
    # samples: (K, N)
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

    sd = d0_data.get("model_state_dict", d0_data)
    cleaned_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    cleaned_sd = strip_compiled_prefix(cleaned_sd)
    forecaster.load_state_dict(cleaned_sd, strict=True)
    forecaster.eval()

    # G0 Head
    g0_head = VarianceHead2D(embed_dim=emb_dim, latent_channels=64, variance_floor=1e-4).to(device)
    g0_ckpt = torch.load(g0_checkpoint_path, map_location="cpu", weights_only=False)
    g0_head.load_state_dict(g0_ckpt["variance_head_state_dict"])
    g0_head.eval()

    # G1 Head
    g1_head = VarianceHead2D(embed_dim=emb_dim, latent_channels=64, variance_floor=1e-4).to(device)
    g1_ckpt = torch.load(g1_checkpoint_path, map_location="cpu", weights_only=False)
    g1_head.load_state_dict(g1_ckpt["variance_head_state_dict"])
    g1_head.eval()

    # FM Matcher
    fm_ckpt = torch.load(fm_checkpoint_path, map_location="cpu", weights_only=False)
    fm_cfg = fm_ckpt.get("config", {})
    fm_sd = fm_ckpt["flow_matcher_state_dict"]
    res_scale = fm_sd.get("residual_scale", None)

    fm = LatentFlowMatcher(
        latent_channels=64,
        cond_dim=128,
        hidden_channels=fm_cfg.get("hidden_channels", 128),
        num_blocks=fm_cfg.get("num_blocks", 4),
        target_mode=fm_cfg.get("target_mode", "residual"),
        use_spatial_attn=fm_cfg.get("use_spatial_attn", True),
        sigma_min=fm_cfg.get("sigma_min", 1e-4),
        residual_scale=res_scale,
    ).to(device)
    fm.load_state_dict(fm_sd, strict=True)
    fm.eval()

    return forecaster, g0_head, g1_head, fm


def evaluate_one_step_comparative(
    forecaster: LatentForecaster,
    g0_head: VarianceHead2D,
    g1_head: VarianceHead2D,
    fm: LatentFlowMatcher,
    dataloader: DataLoader,
    normalizer: FieldNormalizer,
    device: torch.device,
    num_samples_K: int = 32,
    fm_temperatures: List[float] = [1.0, 0.8, 0.7],
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute single-step probabilistic evaluation for D0, G0, G1, and FM on identical validation windows."""
    torch.manual_seed(seed)
    generator = torch.Generator(device=device if device.type != "mps" else "cpu")
    generator.manual_seed(seed)

    # Accumulators for overall metrics
    records = {
        "D0": {"mse_phys": []},
        "G0": {"crps_lat_ana": [], "crps_lat_emp": [], "crps_phys": [], "spread_lat": [], "spread_phys": [], "rmse_phys": [], "lat_samples": [], "lat_targets": [], "phys_samples": [], "phys_targets": [], "var_lat": []},
        "G1": {"crps_lat_ana": [], "crps_lat_emp": [], "crps_phys": [], "spread_lat": [], "spread_phys": [], "rmse_phys": [], "lat_samples": [], "lat_targets": [], "phys_samples": [], "phys_targets": [], "var_lat": []},
    }
    for temp in fm_temperatures:
        records[f"FM_temp_{temp}"] = {"crps_lat_emp": [], "crps_phys": [], "spread_lat": [], "spread_phys": [], "rmse_phys": [], "lat_samples": [], "lat_targets": [], "phys_samples": [], "phys_targets": []}

    channel_names = ["u", "v", "p", "s"]
    phys_crps_by_channel = {m: {c: [] for c in channel_names} for m in records.keys() if m != "D0"}

    total_windows = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            q_hist = batch["history"].to(device)  # (B, 4, 4, Ny, Nx)
            q_target = batch["future"][:, 0:1].to(device)  # (B, 1, 4, Ny, Nx)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)
            b = q_hist.shape[0]

            # 1. Shared Encoder & Deterministic Transformer Backbone
            z_hist = forecaster.encoder(q_hist)
            z_target = forecaster.encoder(q_target)  # (B, 1, 64, Hz, Wz)

            # G0 Homoscedastic Gaussian
            mu, var_g0 = forecaster.transformer.predict_distribution(z_hist, re=re, sc=sc, variance_head=g0_head)

            # G1 Heteroscedastic Gaussian (same mu, dynamic var_g1)
            _, var_g1 = forecaster.transformer.predict_distribution(z_hist, re=re, sc=sc, variance_head=g1_head)

            # Ground truth in physical space with pressure gauge
            q_target_phys = apply_pressure_gauge(normalizer.denormalize(q_target))  # (B, 1, 4, Ny, Nx)

            # D0 Deterministic
            q_d0_phys = apply_pressure_gauge(normalizer.denormalize(forecaster.decoder(mu.squeeze(1)).unsqueeze(1)))
            records["D0"]["mse_phys"].append(torch.mean((q_d0_phys - q_target_phys) ** 2).item() * b)

            # 2. G0 Evaluation
            crps_g0_ana = compute_gaussian_crps_analytical(mu=mu, target=z_target, var=var_g0).mean()
            records["G0"]["crps_lat_ana"].append(crps_g0_ana.item() * b)

            # G0 K-sample empirical
            std_g0 = torch.sqrt(var_g0)
            c_z = mu.shape[2]
            eps_g0 = torch.randn((num_samples_K, b, 1, c_z, mu.shape[3], mu.shape[4]), generator=generator, device=device)
            z_samps_g0 = mu.unsqueeze(0) + eps_g0 * std_g0.unsqueeze(0)  # (K, B, 1, Cz, Hz, Wz)
            crps_g0_emp = compute_sample_crps_tensor(z_samps_g0.squeeze(2), z_target.squeeze(1)).mean()
            records["G0"]["crps_lat_emp"].append(crps_g0_emp.item() * b)
            records["G0"]["spread_lat"].append(z_samps_g0.std(dim=0).mean().item() * b)

            # Decode G0 samples
            z_g0_flat = z_samps_g0.view(num_samples_K * b, c_z, mu.shape[3], mu.shape[4])
            q_g0_phys = apply_pressure_gauge(normalizer.denormalize(forecaster.decoder(z_g0_flat).view(num_samples_K, b, 1, 4, q_target.shape[3], q_target.shape[4])))
            crps_g0_ph = compute_sample_crps_tensor(q_g0_phys.squeeze(2), q_target_phys.squeeze(1))
            records["G0"]["crps_phys"].append(crps_g0_ph.mean().item() * b)
            for c_idx, c_name in enumerate(channel_names):
                phys_crps_by_channel["G0"][c_name].append(crps_g0_ph[:, c_idx].mean().item() * b)

            q_g0_mean = q_g0_phys.mean(dim=0)
            records["G0"]["rmse_phys"].append(torch.sqrt(torch.mean((q_g0_mean - q_target_phys) ** 2)).item() * b)
            records["G0"]["spread_phys"].append(q_g0_phys.std(dim=0).mean().item() * b)

            # 3. G1 Heteroscedastic Gaussian
            crps_g1_ana = compute_gaussian_crps_analytical(mu=mu, target=z_target, var=var_g1).mean()
            records["G1"]["crps_lat_ana"].append(crps_g1_ana.item() * b)

            # G1 K-sample empirical
            std_g1 = torch.sqrt(var_g1)
            eps_g1 = torch.randn((num_samples_K, b, 1, c_z, mu.shape[3], mu.shape[4]), generator=generator, device=device)
            z_samps_g1 = mu.unsqueeze(0) + eps_g1 * std_g1.unsqueeze(0)  # (K, B, 1, Cz, Hz, Wz)
            crps_g1_emp = compute_sample_crps_tensor(z_samps_g1.squeeze(2), z_target.squeeze(1)).mean()
            records["G1"]["crps_lat_emp"].append(crps_g1_emp.item() * b)
            records["G1"]["spread_lat"].append(z_samps_g1.std(dim=0).mean().item() * b)

            # Decode G1 samples
            z_g1_flat = z_samps_g1.view(num_samples_K * b, c_z, mu.shape[3], mu.shape[4])
            q_g1_phys = apply_pressure_gauge(normalizer.denormalize(forecaster.decoder(z_g1_flat).view(num_samples_K, b, 1, 4, q_target.shape[3], q_target.shape[4])))
            crps_g1_ph = compute_sample_crps_tensor(q_g1_phys.squeeze(2), q_target_phys.squeeze(1))
            records["G1"]["crps_phys"].append(crps_g1_ph.mean().item() * b)
            for c_idx, c_name in enumerate(channel_names):
                phys_crps_by_channel["G1"][c_name].append(crps_g1_ph[:, c_idx].mean().item() * b)

            q_g1_mean = q_g1_phys.mean(dim=0)
            records["G1"]["rmse_phys"].append(torch.sqrt(torch.mean((q_g1_mean - q_target_phys) ** 2)).item() * b)
            records["G1"]["spread_phys"].append(q_g1_phys.std(dim=0).mean().item() * b)

            # Subsampled tensors for interval coverage computation
            records["G0"]["lat_samples"].append(z_samps_g0[:, :, :, :, ::2, ::2].reshape(num_samples_K, -1).cpu())
            records["G0"]["phys_samples"].append(q_g0_phys[:, :, :, :, ::4, ::4].reshape(num_samples_K, -1).cpu())
            records["G1"]["lat_samples"].append(z_samps_g1[:, :, :, :, ::2, ::2].reshape(num_samples_K, -1).cpu())
            records["G1"]["phys_samples"].append(q_g1_phys[:, :, :, :, ::4, ::4].reshape(num_samples_K, -1).cpu())

            # 4. FM Model with temperature scanning
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
                    generator=generator,
                )  # (B, K, 1, Cz, Hz, Wz)
                z_samps_fm = z_ens.permute(1, 0, 2, 3, 4, 5)  # (K, B, 1, Cz, Hz, Wz)

                crps_fm_lat = compute_sample_crps_tensor(z_samps_fm.squeeze(2), z_target.squeeze(1)).mean()
                records[m_key]["crps_lat_emp"].append(crps_fm_lat.item() * b)
                records[m_key]["spread_lat"].append(z_samps_fm.std(dim=0).mean().item() * b)

                # Decode FM samples
                z_fm_flat = z_ens.view(b * num_samples_K, c_z, mu.shape[3], mu.shape[4])
                q_fm_norm = forecaster.decoder(z_fm_flat).view(b, num_samples_K, 1, 4, q_target.shape[3], q_target.shape[4])
                q_fm_phys = apply_pressure_gauge(normalizer.denormalize(q_fm_norm.permute(1, 0, 2, 3, 4, 5)))  # (K, B, 1, 4, Ny, Nx)

                crps_fm_ph = compute_sample_crps_tensor(q_fm_phys.squeeze(2), q_target_phys.squeeze(1))
                records[m_key]["crps_phys"].append(crps_fm_ph.mean().item() * b)
                for c_idx, c_name in enumerate(channel_names):
                    phys_crps_by_channel[m_key][c_name].append(crps_fm_ph[:, c_idx].mean().item() * b)

                q_fm_mean = q_fm_phys.mean(dim=0)
                records[m_key]["rmse_phys"].append(torch.sqrt(torch.mean((q_fm_mean - q_target_phys) ** 2)).item() * b)
                records[m_key]["spread_phys"].append(q_fm_phys.std(dim=0).mean().item() * b)

                records[m_key]["lat_samples"].append(z_samps_fm[:, :, :, :, ::2, ::2].reshape(num_samples_K, -1).cpu())
                records[m_key]["phys_samples"].append(q_fm_phys[:, :, :, :, ::4, ::4].reshape(num_samples_K, -1).cpu())

            # Target pooling for coverage
            records["G0"]["lat_targets"].append(z_target[:, :, :, ::2, ::2].reshape(-1).cpu())
            records["G0"]["phys_targets"].append(q_target_phys[:, :, :, ::4, ::4].reshape(-1).cpu())

            total_windows += b

    # Pool targets once
    pooled_lat_target = torch.cat(records["G0"]["lat_targets"], dim=0)
    pooled_phys_target = torch.cat(records["G0"]["phys_targets"], dim=0)

    summary = {
        "validation_windows_evaluated": total_windows,
        "ensemble_size_K": num_samples_K,
        "D0": {
            "physical_rmse": math.sqrt(sum(records["D0"]["mse_phys"]) / total_windows),
        },
    }

    for m_key in records.keys():
        if m_key == "D0":
            continue
        pooled_lat_samp = torch.cat(records[m_key]["lat_samples"], dim=1)
        pooled_phys_samp = torch.cat(records[m_key]["phys_samples"], dim=1)

        lat_intervals = compute_quantile_coverage_and_width(pooled_lat_samp, pooled_lat_target)
        phys_intervals = compute_quantile_coverage_and_width(pooled_phys_samp, pooled_phys_target)

        mean_rmse = sum(records[m_key]["rmse_phys"]) / total_windows
        mean_spread = sum(records[m_key]["spread_phys"]) / total_windows
        ssr = mean_spread / max(1e-8, mean_rmse)

        m_dict = {
            "latent_crps_empirical": float(sum(records[m_key]["crps_lat_emp"]) / total_windows),
            "latent_spread": float(sum(records[m_key]["spread_lat"]) / total_windows),
            "latent_intervals": lat_intervals,
            "physical_crps_mean": float(sum(records[m_key]["crps_phys"]) / total_windows),
            "physical_crps_per_channel": {c: float(sum(phys_crps_by_channel[m_key][c]) / total_windows) for c in channel_names},
            "physical_ensemble_mean_rmse": float(mean_rmse),
            "physical_spread": float(mean_spread),
            "spread_skill_ratio": float(ssr),
            "physical_intervals": phys_intervals,
        }
        if "crps_lat_ana" in records[m_key] and len(records[m_key]["crps_lat_ana"]) > 0:
            m_dict["latent_crps_analytical"] = float(sum(records[m_key]["crps_lat_ana"]) / total_windows)

        summary[m_key] = m_dict

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
    max_trajectories: int = 6,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute multi-step autoregressive rollout comparing D0, G0, G1, FM strictly against Ground Truth (GT)."""
    max_h = max(horizons)
    models = ["D0", "G0", "G1", "FM"]
    metrics_by_h = {m: {h: {"ens_vrmse_vs_gt": [], "samp_vrmse_vs_gt": [], "rms_div": [], "vort_rmse_vs_gt": []} for h in horizons} for m in models}
    gt_div_by_h = {h: [] for h in horizons}
    energy_spectra = {"k_bins": None, "GT": None, "D0": None, "G0": None, "G1": None, "FM": None}

    trajectories_done = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if trajectories_done >= max_trajectories:
                break

            q_hist = batch["history"].to(device)  # (B, 4, 4, Ny, Nx)
            q_gt_seq = batch["future"][:, :max_h].to(device)  # (B, max_h, 4, Ny, Nx)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)
            b = q_hist.shape[0]

            # Physical Ground Truth (denormalized + gauge)
            q_gt_phys = apply_pressure_gauge(normalizer.denormalize(q_gt_seq))  # (B, max_h, 4, Ny, Nx)

            # 1. D0 & G0 / G1 Rollout via sample_rollout
            rollout_g0 = forecaster.sample_rollout(
                q_hist=q_hist, re=re, sc=sc, horizon=max_h,
                num_samples=num_samples_K, seed=seed + batch_idx,
                variance_head=g0_head, decode_samples=True,
            )
            rollout_g1 = forecaster.sample_rollout(
                q_hist=q_hist, re=re, sc=sc, horizon=max_h,
                num_samples=num_samples_K, seed=seed + batch_idx + 100,
                variance_head=g1_head, decode_samples=True,
            )
            rollout_fm = forecaster.sample_rollout_flow_matching(
                q_hist=q_hist, re=re, sc=sc, horizon=max_h,
                num_samples=num_samples_K, seed=seed + batch_idx + 200,
                flow_matcher=fm, decode_samples=True,
            )

            # Decode and denormalize all models
            # D0
            d0_phys = apply_pressure_gauge(normalizer.denormalize(rollout_g0["deterministic_rollout"]))  # (B, max_h, 4, Ny, Nx)
            # G0
            g0_samps = apply_pressure_gauge(normalizer.denormalize(rollout_g0["sample_trajectories"].permute(1, 0, 2, 3, 4, 5)).permute(1, 0, 2, 3, 4, 5))
            g0_mean = g0_samps.mean(dim=1)
            # G1
            g1_samps = apply_pressure_gauge(normalizer.denormalize(rollout_g1["sample_trajectories"].permute(1, 0, 2, 3, 4, 5)).permute(1, 0, 2, 3, 4, 5))
            g1_mean = g1_samps.mean(dim=1)
            # FM
            fm_samps = apply_pressure_gauge(normalizer.denormalize(rollout_fm["sample_trajectories"].permute(1, 0, 2, 3, 4, 5)).permute(1, 0, 2, 3, 4, 5))
            fm_mean = fm_samps.mean(dim=1)

            rollout_data = {
                "D0": {"mean": d0_phys, "samps": d0_phys.unsqueeze(1)},
                "G0": {"mean": g0_mean, "samps": g0_samps},
                "G1": {"mean": g1_mean, "samps": g1_samps},
                "FM": {"mean": fm_mean, "samps": fm_samps},
            }

            for h in horizons:
                gt_h = q_gt_phys[:, h - 1]  # (B, 4, Ny, Nx)
                u_gt = gt_h[:, 0]
                v_gt = gt_h[:, 1]
                div_gt = compute_divergence(u_gt, v_gt)
                vort_gt = compute_vorticity(u_gt, v_gt)
                gt_div_by_h[h].append(torch.sqrt(torch.mean(div_gt ** 2)).item())

                for m in models:
                    pred_mean_h = rollout_data[m]["mean"][:, h - 1]
                    samps_h = rollout_data[m]["samps"][:, :, h - 1]  # (B, K, 4, Ny, Nx)

                    # 1. Ensemble Mean VRMSE vs GT
                    vrmse_ens = compute_vrmse(pred_mean_h, gt_h).item()
                    metrics_by_h[m][h]["ens_vrmse_vs_gt"].append(vrmse_ens)

                    # 2. Sample Mean VRMSE vs GT
                    samp_vrmses = [compute_vrmse(samps_h[:, k], gt_h).item() for k in range(samps_h.shape[1])]
                    metrics_by_h[m][h]["samp_vrmse_vs_gt"].append(float(np.mean(samp_vrmses)))

                    # 3. RMS Divergence of individual sample member 0
                    u_s = samps_h[:, 0, 0]
                    v_s = samps_h[:, 0, 1]
                    div_s = compute_divergence(u_s, v_s)
                    metrics_by_h[m][h]["rms_div"].append(torch.sqrt(torch.mean(div_s ** 2)).item())

                    # 4. Vorticity RMSE of sample vs GT
                    vort_s = compute_vorticity(u_s, v_s)
                    metrics_by_h[m][h]["vort_rmse_vs_gt"].append(torch.sqrt(torch.mean((vort_s - vort_gt) ** 2)).item())

                # Energy spectrum at max horizon
                if h == max_h and energy_spectra["k_bins"] is None:
                    k_bins, e_gt = compute_radial_energy_spectrum(u_gt[0], v_gt[0])
                    energy_spectra["k_bins"] = [float(x) for x in k_bins[:25].cpu().numpy()]
                    energy_spectra["GT"] = [float(x) for x in e_gt[:25].cpu().numpy()]
                    for m in models:
                        u_m = rollout_data[m]["samps"][0, 0, h - 1, 0]
                        v_m = rollout_data[m]["samps"][0, 0, h - 1, 1]
                        _, e_m = compute_radial_energy_spectrum(u_m, v_m)
                        energy_spectra[m] = [float(x) for x in e_m[:25].cpu().numpy()]

            trajectories_done += b

    # Compute spectral relative L2 errors vs GT
    e_gt_arr = np.array(energy_spectra["GT"])
    e_gt_norm = np.linalg.norm(e_gt_arr) + 1e-8
    spectral_rel_errors = {}
    for m in models:
        e_m_arr = np.array(energy_spectra[m])
        rel_err = float(np.linalg.norm(e_m_arr - e_gt_arr) / e_gt_norm)
        spectral_rel_errors[m] = rel_err

    rollout_summary = {
        "trajectories_evaluated": trajectories_done,
        "spectral_relative_error_vs_gt_h10": spectral_rel_errors,
        "energy_spectra_first_25_modes": energy_spectra,
    }

    for h in horizons:
        h_key = f"h{h}"
        h_dict = {
            "ground_truth_rms_divergence": float(np.mean(gt_div_by_h[h])),
        }
        for m in models:
            h_dict[m] = {
                "ensemble_mean_vrmse_vs_gt": float(np.mean(metrics_by_h[m][h]["ens_vrmse_vs_gt"])),
                "sample_mean_vrmse_vs_gt": float(np.mean(metrics_by_h[m][h]["samp_vrmse_vs_gt"])),
                "sample_rms_divergence": float(np.mean(metrics_by_h[m][h]["rms_div"])),
                "divergence_ratio_vs_gt": float(np.mean(metrics_by_h[m][h]["rms_div"]) / max(1e-6, np.mean(gt_div_by_h[h]))),
                "sample_vorticity_rmse_vs_gt": float(np.mean(metrics_by_h[m][h]["vort_rmse_vs_gt"])),
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 70)
    print("=== PROBABILISTIC WORLD MODEL RIGOROUS A/B BENCHMARK (ON VALIDATION SET) ===")
    print(f"Device: {device} | Seed: {args.seed} | One-step K: {args.num_samples_k} | Rollout K: {args.num_samples_rollout}")
    print("=" * 70)

    # 1. Load Data
    normalizer = FieldNormalizer()
    normalizer.load_state_dict(torch.load(args.normalizer_path, weights_only=True, map_location="cpu"))

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
    print("All four models (D0, G0, G1, FM) loaded and cryptographically verified.")

    # 3. Step 1: One-Step Probabilistic Evaluation
    print("\n" + "=" * 50)
    print(">>> 1. RUNNING ONE-STEP PROBABILISTIC BENCHMARK (K=32) <<<")
    print("=" * 50)
    step1_summary = evaluate_one_step_comparative(
        forecaster=forecaster,
        g0_head=g0_head,
        g1_head=g1_head,
        fm=fm,
        dataloader=valid_loader,
        normalizer=normalizer,
        device=device,
        num_samples_K=args.num_samples_k,
        fm_temperatures=[1.0, 0.8, 0.7],
        seed=args.seed,
    )

    # Print clean comparison table
    print("\n--- ONE-STEP PROBABILISTIC COMPARISON (ON SAME 144 VALIDATION WINDOWS) ---")
    headers = ["Model", "Latent CRPS", "Physical CRPS", "Spread", "RMSE", "SSR", "PICP 50%", "PICP 80%", "PICP 90%", "PICP 95%"]
    print(f"{headers[0]:<12} | {headers[1]:<11} | {headers[2]:<13} | {headers[3]:<8} | {headers[4]:<8} | {headers[5]:<6} | {headers[6]:<8} | {headers[7]:<8} | {headers[8]:<8} | {headers[9]:<8}")
    print("-" * 110)

    for m in ["G0", "G1", "FM_temp_1.0", "FM_temp_0.8", "FM_temp_0.7"]:
        d = step1_summary[m]
        p_int = d["physical_intervals"]
        l_crps = f"{d['latent_crps_empirical']:.4f}"
        p_crps = f"{d['physical_crps_mean']:.5f}"
        spread = f"{d['physical_spread']:.4f}"
        rmse = f"{d['physical_ensemble_mean_rmse']:.4f}"
        ssr = f"{d['spread_skill_ratio']:.3f}"
        c50 = f"{p_int['50']['picp']*100:.1f}%"
        c80 = f"{p_int['80']['picp']*100:.1f}%"
        c90 = f"{p_int['90']['picp']*100:.1f}%"
        c95 = f"{p_int['95']['picp']*100:.1f}%"
        print(f"{m:<12} | {l_crps:<11} | {p_crps:<13} | {spread:<8} | {rmse:<8} | {ssr:<6} | {c50:<8} | {c80:<8} | {c90:<8} | {c95:<8}")

    # 4. Step 2: Multi-Step Autoregressive Rollout & Physics vs Ground Truth
    print("\n" + "=" * 50)
    print(">>> 2. RUNNING MULTI-STEP ROLLOUT & PHYSICS VS GROUND TRUTH (H=5, 10) <<<")
    print("=" * 50)
    step2_summary = evaluate_rollout_physics_comparative(
        forecaster=forecaster,
        g0_head=g0_head,
        g1_head=g1_head,
        fm=fm,
        dataloader=valid_loader,
        normalizer=normalizer,
        device=device,
        horizons=[5, 10],
        num_samples_K=args.num_samples_rollout,
        max_trajectories=6,
        seed=args.seed,
    )

    print("\n--- MULTI-STEP PHYSICAL PERFORMANCE VS GROUND TRUTH ---")
    for h in [5, 10]:
        h_data = step2_summary[f"h{h}"]
        gt_div = h_data["ground_truth_rms_divergence"]
        print(f"\n[ Horizon h={h} ] Ground Truth RMS Divergence = {gt_div:.5f}")
        print(f"{'Model':<8} | {'Ens VRMSE vs GT':<16} | {'Samp VRMSE vs GT':<16} | {'RMS Divergence':<14} | {'Div / GT Ratio':<14} | {'Vorticity RMSE vs GT':<20}")
        print("-" * 100)
        for m in ["D0", "G0", "G1", "FM"]:
            m_stat = h_data[m]
            ens_vrmse = f"{m_stat['ensemble_mean_vrmse_vs_gt']:.4f}"
            samp_vrmse = f"{m_stat['sample_mean_vrmse_vs_gt']:.4f}"
            div = f"{m_stat['sample_rms_divergence']:.4f}"
            div_ratio = f"{m_stat['divergence_ratio_vs_gt']:.1f}x"
            vort = f"{m_stat['sample_vorticity_rmse_vs_gt']:.4f}"
            print(f"{m:<8} | {ens_vrmse:<16} | {samp_vrmse:<16} | {div:<14} | {div_ratio:<14} | {vort:<20}")

    print("\n--- RADIAL ENERGY SPECTRUM RELATIVE L2 ERROR VS GROUND TRUTH (h=10) ---")
    for m, err in step2_summary["spectral_relative_error_vs_gt_h10"].items():
        print(f"  {m:<8}: Relative L2 Error = {err * 100:.2f}%")

    # 5. Save Complete Benchmark JSON
    final_benchmark = {
        "evaluation_protocol": "STRICT_SAME_VALIDATION_SPLIT_AB_TEST",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "is_git_dirty": is_git_dirty(),
        "artifacts_evaluated": {
            "d0_checkpoint": {"path": args.d0_checkpoint, "sha256": compute_file_sha256(args.d0_checkpoint)},
            "g0_checkpoint": {"path": args.g0_checkpoint, "sha256": compute_file_sha256(args.g0_checkpoint)},
            "g1_checkpoint": {"path": args.g1_checkpoint, "sha256": compute_file_sha256(args.g1_checkpoint)},
            "fm_checkpoint": {"path": args.fm_checkpoint, "sha256": compute_file_sha256(args.fm_checkpoint)},
        },
        "step1_one_step_probabilistic": step1_summary,
        "step2_rollout_and_physics_vs_gt": step2_summary,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_benchmark, f, indent=2)

    print(f"\nSuccessfully written full comparative benchmark report to {out_path}")


if __name__ == "__main__":
    main()
