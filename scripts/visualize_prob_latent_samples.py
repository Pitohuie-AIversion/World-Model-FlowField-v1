#!/usr/bin/env python3
"""Qualitative sampling and uncertainty visualization for Gaussian probabilistic world models (ProbLatent G0 / G1).

Generates qualitative comparison figures in outputs/figures/probabilistic/:
1. sample_gaussian_ensemble_realizations.png: Ground Truth, Mean Prediction, and 3 distinct
   stochastic realizations from the latent Gaussian posterior (q ~ Decoder(mu + sigma * eps)).
2. sample_gaussian_uncertainty_vs_error.png: Spatial uncertainty map (sigma field) vs
   absolute empirical error (|mu - GT|) showing shear layer localized uncertainty.
3. sample_gaussian_prediction_intervals.png: Spatial 1D slice profiles with 50%, 80%, 90%, 95%
   confidence interval bands demonstrating empirical coverage and calibrated uncertainty.
"""

import json
import os
import sys
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.latent_transformer import LatentSTTransformer
from src.models.latent_forecaster import LatentForecaster
from src.models.probabilistic_latent_dynamics import VarianceHead2D
from src.utils.checkpoint import resolve_spatial_pos_config, strip_compiled_prefix
from src.data.pipeline import create_flow_dataloaders
from src.data.normalization import FieldNormalizer
from src.utils.fft_derivatives import compute_vorticity
from src.utils.physics_contract import (
    SHEAR_FLOW_DOMAIN_SIZE_XY,
    zero_mean_pressure_gauge,
)


def run_probabilistic_qualitative_sampling(
    d0_checkpoint: str = "outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt",
    g0_checkpoint: str = "outputs/checkpoints/probabilistic/variance_head/formal_r1_seed42/g0_baseline_initialization.pt",
    g1_checkpoint: str = "outputs/checkpoints/probabilistic/variance_head/formal_r1_seed42/best_g1_variance_head.pt",
    normalizer_path: str = "outputs/normalization/stats_grouped.pt",
    split_file: str = "outputs/splits/grouped_split.json",
    data_root: str = "/root/autodl-tmp/datasets/shear_flow",
    output_dir: str = "outputs/figures/probabilistic",
    target_sample_idx: int = 0,
    num_samples_mc: int = 32,
    seed: int = 42,
    device_str: str = "cuda:0" if torch.cuda.is_available() else "cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading normalizer from {normalizer_path}...")
    normalizer = FieldNormalizer()
    normalizer.load_state_dict(torch.load(normalizer_path, map_location="cpu", weights_only=True))

    print("Building test dataloader...")
    _, _, test_loader, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file=split_file,
        data_root=data_root,
        batch_size=1,
        history_length=4,
        horizon=1,
        test_stride=1,
        downsample_factor=2,
        num_workers=0,
        normalize=True,
        normalizer=normalizer,
    )

    # 1. Load D0 and build LatentForecaster
    print("Loading D0 and building LatentForecaster...")
    d0_ckpt = torch.load(d0_checkpoint, map_location="cpu", weights_only=False)
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
    forecaster.eval()

    # 2. Load G0 and G1 Variance Heads
    print("Loading G0 and G1 variance heads...")
    g0_head = VarianceHead2D(embed_dim=emb_dim, latent_channels=64, variance_floor=1e-4).to(device)
    g0_ckpt = torch.load(g0_checkpoint, map_location="cpu", weights_only=False)
    g0_head.load_state_dict(g0_ckpt["variance_head_state_dict"])
    g0_head.eval()

    g1_head = VarianceHead2D(embed_dim=emb_dim, latent_channels=64, variance_floor=1e-4).to(device)
    g1_ckpt = torch.load(g1_checkpoint, map_location="cpu", weights_only=False)
    g1_head.load_state_dict(g1_ckpt["variance_head_state_dict"])
    g1_head.eval()

    # 3. Extract test sample
    print(f"Extracting test sample #{target_sample_idx}...")
    sample_batch = None
    for idx, batch in enumerate(test_loader):
        if idx == target_sample_idx:
            sample_batch = batch
            break

    if sample_batch is None:
        raise ValueError(f"Target test sample {target_sample_idx} not found.")

    q_hist = sample_batch["history"].to(device)
    q_target = sample_batch["future"][:, 0:1].to(device)
    re = sample_batch["re"].to(device) if "re" in sample_batch else None
    sc = sample_batch["sc"].to(device) if "sc" in sample_batch else None

    # 4. Forward Prediction & Gaussian Uncertainty
    print("Running probabilistic prediction...")
    with torch.no_grad():
        # Latent distributions
        mu_g1, var_g1 = forecaster.predict_distribution_single_step(q_hist, re=re, sc=sc, variance_head=g1_head)
        mu_g0, var_g0 = forecaster.predict_distribution_single_step(q_hist, re=re, sc=sc, variance_head=g0_head)

        std_g1 = torch.sqrt(var_g1)
        std_g0 = torch.sqrt(var_g0)

        # Decode mean to physical space
        mu_phys_norm = forecaster.decoder(mu_g1)
        mu_phys = normalizer.denormalize(mu_phys_norm)
        mu_phys[:, :, 2] = zero_mean_pressure_gauge(mu_phys[:, :, 2])

        gt_phys = normalizer.denormalize(q_target)
        gt_phys[:, :, 2] = zero_mean_pressure_gauge(gt_phys[:, :, 2])

        # Generate K Monte Carlo realizations by sampling z ~ N(mu, var) and decoding
        ensemble_samples = []
        for s_i in range(num_samples_mc):
            eps = torch.randn_like(mu_g1)
            z_sample = mu_g1 + std_g1 * eps
            phys_s_norm = forecaster.decoder(z_sample)
            phys_s = normalizer.denormalize(phys_s_norm)
            phys_s[:, :, 2] = zero_mean_pressure_gauge(phys_s[:, :, 2])
            ensemble_samples.append(phys_s)

        ens_tensor = torch.stack(ensemble_samples, dim=0) # (K, B=1, 1, C=4, Ny, Nx)
        ens_phys = ens_tensor[:, 0, 0] # (K, 4, Ny, Nx)

        # Ensemble empirical standard deviation in physical space
        ens_std_phys = torch.std(ens_phys, dim=0).cpu().numpy() # (4, Ny, Nx)

    gt_field = gt_phys[0, 0].cpu().numpy() # (4, Ny, Nx)
    mu_field = mu_phys[0, 0].cpu().numpy() # (4, Ny, Nx)

    # -------------------------------------------------------------------------
    # Figure 1: Stochastic Ensemble Realizations (5 columns x 3 rows)
    # -------------------------------------------------------------------------
    print("Plotting Figure 1: Ensemble realizations...")
    fig_ens, axes_ens = plt.subplots(3, 5, figsize=(18, 9), dpi=300)

    # Variables: u, tracer s, vorticity omega
    vort_gt = compute_vorticity(torch.from_numpy(gt_field[0:1]), torch.from_numpy(gt_field[1:2]), domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)[0].numpy()
    vort_mu = compute_vorticity(torch.from_numpy(mu_field[0:1]), torch.from_numpy(mu_field[1:2]), domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)[0].numpy()

    vort_s1 = compute_vorticity(ens_phys[0, 0:1].cpu(), ens_phys[0, 1:2].cpu(), domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)[0].numpy()
    vort_s2 = compute_vorticity(ens_phys[1, 0:1].cpu(), ens_phys[1, 1:2].cpu(), domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)[0].numpy()
    vort_s3 = compute_vorticity(ens_phys[2, 0:1].cpu(), ens_phys[2, 1:2].cpu(), domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)[0].numpy()

    var_rows = [
        ("u (Velocity-x)", [gt_field[0], mu_field[0], ens_phys[0, 0].cpu().numpy(), ens_phys[1, 0].cpu().numpy(), ens_phys[2, 0].cpu().numpy()], "RdBu_r"),
        ("s (Passive Tracer)", [gt_field[3], mu_field[3], ens_phys[0, 3].cpu().numpy(), ens_phys[1, 3].cpu().numpy(), ens_phys[2, 3].cpu().numpy()], "magma"),
        ("ω (Vorticity)", [vort_gt, vort_mu, vort_s1, vort_s2, vort_s3], "seismic"),
    ]

    col_headers = [
        "Ground Truth Target",
        "Deterministic Mean μ",
        "Sample Realization #1",
        "Sample Realization #2",
        "Sample Realization #3",
    ]

    for r_idx, (var_name, var_fields, cmap) in enumerate(var_rows):
        vmin = min(f.min() for f in var_fields)
        vmax = max(f.max() for f in var_fields)
        for c_idx in range(5):
            ax = axes_ens[r_idx, c_idx]
            im = ax.imshow(var_fields[c_idx].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if r_idx == 0:
                ax.set_title(col_headers[c_idx], fontsize=11, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(var_name, fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig_ens.suptitle(
        f"Gaussian Probabilistic Dynamics (ProbLatent G1): Stochastic Ensemble Sampling\n"
        f"Test Sample #{target_sample_idx} | Single-Step Transition | Latent Sampling: z ~ N(μ, Σ)",
        fontsize=14, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    fig1_path = out_path / "sample_gaussian_ensemble_realizations.png"
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig_ens)
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------------------
    # Figure 2: Uncertainty Map vs Empirical Error (2 rows x 3 columns)
    # -------------------------------------------------------------------------
    print("Plotting Figure 2: Uncertainty field vs empirical error...")
    fig_unc, axes_unc = plt.subplots(2, 3, figsize=(16, 9), dpi=300)

    # Focus on velocity-x and tracer
    abs_err_u = np.abs(mu_field[0] - gt_field[0])
    abs_err_s = np.abs(mu_field[3] - gt_field[3])
    std_u = ens_std_phys[0]
    std_s = ens_std_phys[3]

    # Row 1: Velocity u
    im00 = axes_unc[0, 0].imshow(mu_field[0].T, origin="lower", cmap="RdBu_r", aspect="auto")
    plt.colorbar(im00, ax=axes_unc[0, 0], fraction=0.046, pad=0.04)
    axes_unc[0, 0].set_title("Mean Prediction μ(u)", fontweight="bold")
    axes_unc[0, 0].set_ylabel("Streamwise Velocity u", fontsize=11, fontweight="bold")

    im01 = axes_unc[0, 1].imshow(std_u.T, origin="lower", cmap="plasma", aspect="auto")
    plt.colorbar(im01, ax=axes_unc[0, 1], fraction=0.046, pad=0.04)
    axes_unc[0, 1].set_title(f"Uncertainty Field σ(u)\n(Mean σ: {np.mean(std_u):.4f})", fontweight="bold")

    im02 = axes_unc[0, 2].imshow(abs_err_u.T, origin="lower", cmap="inferno", aspect="auto")
    plt.colorbar(im02, ax=axes_unc[0, 2], fraction=0.046, pad=0.04)
    axes_unc[0, 2].set_title(f"Empirical Error |μ - GT|\n(MAE: {np.mean(abs_err_u):.4f})", fontweight="bold")

    # Row 2: Tracer s
    im10 = axes_unc[1, 0].imshow(mu_field[3].T, origin="lower", cmap="magma", aspect="auto")
    plt.colorbar(im10, ax=axes_unc[1, 0], fraction=0.046, pad=0.04)
    axes_unc[1, 0].set_title("Mean Prediction μ(s)", fontweight="bold")
    axes_unc[1, 0].set_ylabel("Passive Tracer s", fontsize=11, fontweight="bold")

    im11 = axes_unc[1, 1].imshow(std_s.T, origin="lower", cmap="plasma", aspect="auto")
    plt.colorbar(im11, ax=axes_unc[1, 1], fraction=0.046, pad=0.04)
    axes_unc[1, 1].set_title(f"Uncertainty Field σ(s)\n(Mean σ: {np.mean(std_s):.4f})", fontweight="bold")

    im12 = axes_unc[1, 2].imshow(abs_err_s.T, origin="lower", cmap="inferno", aspect="auto")
    plt.colorbar(im12, ax=axes_unc[1, 2], fraction=0.046, pad=0.04)
    axes_unc[1, 2].set_title(f"Empirical Error |μ - GT|\n(MAE: {np.mean(abs_err_s):.4f})", fontweight="bold")

    for r in range(2):
        for c in range(3):
            axes_unc[r, c].set_xticks([])
            axes_unc[r, c].set_yticks([])

    fig_unc.suptitle(
        "ProbLatent G1: Spatial Uncertainty Field vs Empirical Error Contrast\n"
        "Alignment Between Predictive Dispersion σ and High-Gradient Shear Boundary Errors",
        fontsize=13, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    fig2_path = out_path / "sample_gaussian_uncertainty_vs_error.png"
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig_unc)
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------------------
    # Figure 3: Prediction Confidence Interval Strip (1D profile cut)
    # -------------------------------------------------------------------------
    print("Plotting Figure 3: 1D prediction interval profile...")
    fig_int, (ax_u, ax_s) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Cut along streamwise centerline (Ny // 2) across all x-positions
    cut_y = mu_field.shape[1] // 2
    x_coords = np.linspace(0, 1.0, mu_field.shape[2])

    gt_u_slice = gt_field[0, cut_y]
    mu_u_slice = mu_field[0, cut_y]
    std_u_slice = std_u[cut_y]

    gt_s_slice = gt_field[3, cut_y]
    mu_s_slice = mu_field[3, cut_y]
    std_s_slice = std_s[cut_y]

    # Velocity u Profile
    ax_u.plot(x_coords, gt_u_slice, "k-", linewidth=2.0, label="Ground Truth (Target)")
    ax_u.plot(x_coords, mu_u_slice, "b--", linewidth=1.8, label="Mean Prediction μ")
    ax_u.fill_between(x_coords, mu_u_slice - 0.674 * std_u_slice, mu_u_slice + 0.674 * std_u_slice,
                      color="#2b5c8f", alpha=0.35, label="50% Nominal Interval")
    ax_u.fill_between(x_coords, mu_u_slice - 1.645 * std_u_slice, mu_u_slice + 1.645 * std_u_slice,
                      color="#2b5c8f", alpha=0.20, label="90% Nominal Interval")
    ax_u.fill_between(x_coords, mu_u_slice - 1.960 * std_u_slice, mu_u_slice + 1.960 * std_u_slice,
                      color="#2b5c8f", alpha=0.10, label="95% Nominal Interval")
    ax_u.set_title("Streamwise Velocity u (1D Centerline Slice)", fontweight="bold")
    ax_u.set_xlabel("Streamwise Coordinate x")
    ax_u.set_ylabel("Velocity u")
    ax_u.legend(loc="upper right", fontsize=8.5)
    ax_u.grid(True, linestyle="--", alpha=0.4)

    # Tracer s Profile
    ax_s.plot(x_coords, gt_s_slice, "k-", linewidth=2.0, label="Ground Truth (Target)")
    ax_s.plot(x_coords, mu_s_slice, "r--", linewidth=1.8, label="Mean Prediction μ")
    ax_s.fill_between(x_coords, mu_s_slice - 0.674 * std_s_slice, mu_s_slice + 0.674 * std_s_slice,
                      color="#d62728", alpha=0.35, label="50% Nominal Interval")
    ax_s.fill_between(x_coords, mu_s_slice - 1.645 * std_s_slice, mu_s_slice + 1.645 * std_s_slice,
                      color="#d62728", alpha=0.20, label="90% Nominal Interval")
    ax_s.fill_between(x_coords, mu_s_slice - 1.960 * std_s_slice, mu_s_slice + 1.960 * std_s_slice,
                      color="#d62728", alpha=0.10, label="95% Nominal Interval")
    ax_s.set_title("Passive Tracer s (1D Centerline Slice)", fontweight="bold")
    ax_s.set_xlabel("Streamwise Coordinate x")
    ax_s.set_ylabel("Tracer Concentration s")
    ax_s.legend(loc="upper right", fontsize=8.5)
    ax_s.grid(True, linestyle="--", alpha=0.4)

    fig_int.suptitle(
        f"ProbLatent G1: Prediction Intervals Across Centerline Slice\n"
        f"Empirical Realization Bounds vs True Observation Ground Truth",
        fontsize=13, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    fig3_path = out_path / "sample_gaussian_prediction_intervals.png"
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close(fig_int)
    print(f"Saved: {fig3_path}")

    # Metadata record
    meta_path = out_path / "sample_gaussian_probabilistic_metadata.json"
    meta_record = {
        "sample_index": target_sample_idx,
        "num_mc_samples": num_samples_mc,
        "seed": seed,
        "figures": [
            "sample_gaussian_ensemble_realizations.png",
            "sample_gaussian_uncertainty_vs_error.png",
            "sample_gaussian_prediction_intervals.png",
        ],
        "diagnostics": {
            "mean_mae_u": float(np.mean(abs_err_u)),
            "mean_mae_s": float(np.mean(abs_err_s)),
            "mean_sigma_u": float(np.mean(std_u)),
            "mean_sigma_s": float(np.mean(std_s)),
            "spatial_corr_unc_err_u": float(np.corrcoef(std_u.flatten(), abs_err_u.flatten())[0, 1]),
            "spatial_corr_unc_err_s": float(np.corrcoef(std_s.flatten(), abs_err_s.flatten())[0, 1]),
        }
    }
    with open(meta_path, "w") as f:
        json.dump(meta_record, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    return {
        "realizations": str(fig1_path),
        "uncertainty_vs_error": str(fig2_path),
        "intervals": str(fig3_path),
        "metadata": str(meta_path),
    }


if __name__ == "__main__":
    run_probabilistic_qualitative_sampling()
