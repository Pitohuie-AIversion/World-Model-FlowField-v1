#!/usr/bin/env python3
"""Qualitative sampling and spatial field visualization for PDE-controlled models.

Generates qualitative comparison figures in outputs/figures/pde_controlled/:
1. sample_pde_flow_fields_comparison.png: 4-model (GT, D0, P0, PDE) side-by-side
   flow field predictions across velocity u, v, tracer s, and vorticity omega.
2. sample_pde_residuals_comparison.png: Spatial distribution of physical residuals
   (divergence error, momentum residuals r_u, r_v, and tracer advection-diffusion r_s)
   directly contrasting P0 (standard finetune) vs PDE (physics-supervised).
3. sample_pde_errors_comparison.png: Absolute error spatial maps (|Pred - GT|)
   showing localized error suppression.
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

from scripts.evaluate_pde_controlled_candidates import load_forecaster_model
from src.data.pipeline import create_flow_dataloaders
from src.data.normalization import FieldNormalizer
from src.losses.divergence import DivergenceLoss
from src.losses.navier_stokes import NavierStokesPDELoss
from src.utils.fft_derivatives import compute_vorticity
from src.utils.physics_contract import (
    SHEAR_FLOW_DOMAIN_SIZE_XY,
    zero_mean_pressure_gauge,
)


def run_pde_qualitative_sampling(
    d0_path: str = "outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt",
    p0_path: str = "outputs/checkpoints/dynamics/pde_controlled_experiment/p0_step_50.pt",
    pde_path: str = "outputs/checkpoints/dynamics/pde_controlled_experiment/pde_step_50.pt",
    ae_path: str = "outputs/checkpoints/representation/best_autoencoder.pt",
    norm_file: str = "outputs/normalization/stats_grouped.pt",
    split_file: str = "outputs/splits/grouped_split.json",
    data_root: str = "/root/autodl-tmp/datasets/shear_flow",
    output_dir: str = "outputs/figures/pde_controlled",
    target_sample_idx: int = 0,
    horizon_step: int = 12,
    device_str: str = "cuda:0" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading normalizer from {norm_file}...")
    normalizer = FieldNormalizer()
    normalizer.load_state_dict(torch.load(norm_file, map_location="cpu", weights_only=True))

    print("Building validation dataloader...")
    _, val_loader, _, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file=split_file,
        data_root=data_root,
        batch_size=1,
        history_length=4,
        horizon=horizon_step,
        valid_stride=1,
        downsample_factor=2,
        num_workers=0,
        normalize=True,
        normalizer=normalizer,
    )

    # 1. Load the three models
    print("Loading models (D0, P0, PDE)...")
    model_d0, _ = load_forecaster_model(d0_path, ae_path, history_length=4, device=device)
    model_p0, _ = load_forecaster_model(p0_path, ae_path, history_length=4, device=device)
    model_pde, _ = load_forecaster_model(pde_path, ae_path, history_length=4, device=device)

    model_d0.eval()
    model_p0.eval()
    model_pde.eval()

    # 2. Extract representative validation sample
    print(f"Extracting validation sample #{target_sample_idx}...")
    target_batch = None
    for idx, batch in enumerate(val_loader):
        if idx == target_sample_idx:
            target_batch = batch
            break

    if target_batch is None:
        raise ValueError(f"Could not locate sample index {target_sample_idx} in validation loader.")

    q_hist = target_batch["history"].to(device)
    q_future = target_batch["future"].to(device)
    re = target_batch["re"].to(device)
    sc = target_batch["sc"].to(device)
    dt = target_batch["dt"].to(device)

    re_val = float(re.item())
    sc_val = float(sc.item())
    dt_val = float(dt.item())

    # 3. Model Rollout
    print(f"Executing rollouts up to horizon h={horizon_step}...")
    with torch.no_grad():
        pred_d0_norm = model_d0.forward_rollout(q_hist, re=re, sc=sc, horizon=horizon_step)
        pred_p0_norm = model_p0.forward_rollout(q_hist, re=re, sc=sc, horizon=horizon_step)
        pred_pde_norm = model_pde.forward_rollout(q_hist, re=re, sc=sc, horizon=horizon_step)

        # Denormalize to physical units
        gt_phys = normalizer.denormalize(q_future)
        d0_phys = normalizer.denormalize(pred_d0_norm)
        p0_phys = normalizer.denormalize(pred_p0_norm)
        pde_phys = normalizer.denormalize(pred_pde_norm)

        # Apply zero-mean pressure gauge
        gt_phys[:, :, 2] = zero_mean_pressure_gauge(gt_phys[:, :, 2])
        d0_phys[:, :, 2] = zero_mean_pressure_gauge(d0_phys[:, :, 2])
        p0_phys[:, :, 2] = zero_mean_pressure_gauge(p0_phys[:, :, 2])
        pde_phys[:, :, 2] = zero_mean_pressure_gauge(pde_phys[:, :, 2])

    # Focus on target horizon step (e.g. step h=12)
    step_idx = horizon_step - 1
    gt_step = gt_phys[0, step_idx].cpu()
    d0_step = d0_phys[0, step_idx].cpu()
    p0_step = p0_phys[0, step_idx].cpu()
    pde_step = pde_phys[0, step_idx].cpu()

    # Compute vorticity omega for each
    vort_gt = compute_vorticity(gt_step[0:1], gt_step[1:2], domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)[0].numpy()
    vort_d0 = compute_vorticity(d0_step[0:1], d0_step[1:2], domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)[0].numpy()
    vort_p0 = compute_vorticity(p0_step[0:1], p0_step[1:2], domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)[0].numpy()
    vort_pde = compute_vorticity(pde_step[0:1], pde_step[1:2], domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)[0].numpy()

    # 4. Compute Physical Residuals
    print("Computing PDE and divergence residuals...")
    div_fn = DivergenceLoss(domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)
    pde_fn = NavierStokesPDELoss(domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)

    q0_hist_phys = normalizer.denormalize(q_hist)[:, -1]

    with torch.no_grad():
        _, p0_stats = pde_fn(p0_phys, re=re, sc=sc, dt=dt, q0_phys=q0_hist_phys)
        _, pde_stats = pde_fn(pde_phys, re=re, sc=sc, dt=dt, q0_phys=q0_hist_phys)

        # Spatial residual fields for P0 and PDE at evaluated step
        # Recompute spatial residual maps via internal helper or loss components
        res_u_p0 = p0_stats["res_momentum_u_rmse"]
        res_v_p0 = p0_stats["res_momentum_v_rmse"]
        res_s_p0 = p0_stats["res_tracer_s_rmse"]

        res_u_pde = pde_stats["res_momentum_u_rmse"]
        res_v_pde = pde_stats["res_momentum_v_rmse"]
        res_s_pde = pde_stats["res_tracer_s_rmse"]

    # -------------------------------------------------------------------------
    # Figure 1: Flow Fields Qualitative Grid (4 rows x 4 cols)
    # -------------------------------------------------------------------------
    print("Plotting Figure 1: Flow fields comparison...")
    fig, axes = plt.subplots(4, 4, figsize=(16, 13), dpi=300)
    models = [
        ("Ground Truth", gt_step, vort_gt),
        ("D0 (Frozen Baseline)", d0_step, vort_d0),
        ("P0 (Standard Finetune)", p0_step, vort_p0),
        ("PDE (Physics Supervised)", pde_step, vort_pde),
    ]

    col_names = ["Streamwise Velocity u", "Cross-stream Velocity v", "Tracer Concentration s", "Vorticity ω"]
    cmaps = ["RdBu_r", "RdBu_r", "magma", "seismic"]

    # Global shared colormap limits per channel
    u_all = [m[1][0].numpy() for m in models]
    v_all = [m[1][1].numpy() for m in models]
    s_all = [m[1][3].numpy() for m in models]
    w_all = [m[2] for m in models]

    clim_u = (min(x.min() for x in u_all), max(x.max() for x in u_all))
    clim_v = (min(x.min() for x in v_all), max(x.max() for x in v_all))
    clim_s = (min(x.min() for x in s_all), max(x.max() for x in s_all))
    clim_w = (min(x.min() for x in w_all), max(x.max() for x in w_all))

    col_clims = [clim_u, clim_v, clim_s, clim_w]

    for row_idx, (m_label, f_tensor, w_arr) in enumerate(models):
        row_fields = [f_tensor[0].numpy(), f_tensor[1].numpy(), f_tensor[3].numpy(), w_arr]
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            data = row_fields[col_idx]
            vmin, vmax = col_clims[col_idx]
            im = ax.imshow(data.T, origin="lower", cmap=cmaps[col_idx], vmin=vmin, vmax=vmax, aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if row_idx == 0:
                ax.set_title(col_names[col_idx], fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(m_label, fontsize=11, fontweight="bold")
            if row_idx == 3:
                ax.set_xlabel("x (Streamwise)", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"PDE Supervision Controlled Experiment: Flow Field Sampling\n"
        f"Sample #{target_sample_idx} | Horizon h={horizon_step} | Re={re_val:.0f}, Sc={sc_val}",
        fontsize=14, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    fig1_path = out_path / "sample_pde_flow_fields_comparison.png"
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------------------
    # Figure 2: Absolute Errors Comparison Grid (3 rows x 4 cols)
    # -------------------------------------------------------------------------
    print("Plotting Figure 2: Absolute errors comparison...")
    fig_err, axes_err = plt.subplots(3, 4, figsize=(16, 10), dpi=300)
    err_models = [
        ("D0 Error (|D0 - GT|)", d0_step, vort_d0),
        ("P0 Error (|P0 - GT|)", p0_step, vort_p0),
        ("PDE Error (|PDE - GT|)", pde_step, vort_pde),
    ]

    err_col_names = ["Error |u - u*|", "Error |v - v*|", "Error |s - s*|", "Error |ω - ω*|"]

    # Compute shared unified max errors per variable
    err_u_max = max(np.max(np.abs(m[1][0].numpy() - gt_step[0].numpy())) for m in err_models)
    err_v_max = max(np.max(np.abs(m[1][1].numpy() - gt_step[1].numpy())) for m in err_models)
    err_s_max = max(np.max(np.abs(m[1][3].numpy() - gt_step[3].numpy())) for m in err_models)
    err_w_max = max(np.max(np.abs(m[2] - vort_gt)) for m in err_models)
    err_clims = [err_u_max, err_v_max, err_s_max, err_w_max]

    for row_idx, (m_label, f_tensor, w_arr) in enumerate(err_models):
        err_fields = [
            np.abs(f_tensor[0].numpy() - gt_step[0].numpy()),
            np.abs(f_tensor[1].numpy() - gt_step[1].numpy()),
            np.abs(f_tensor[3].numpy() - gt_step[3].numpy()),
            np.abs(w_arr - vort_gt),
        ]
        for col_idx in range(4):
            ax = axes_err[row_idx, col_idx]
            data = err_fields[col_idx]
            emax = err_clims[col_idx]
            im = ax.imshow(data.T, origin="lower", cmap="inferno", vmin=0, vmax=emax, aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            mae = float(np.mean(data))
            if row_idx == 0:
                ax.set_title(f"{err_col_names[col_idx]}\n(MAE: {mae:.4f})", fontsize=10, fontweight="bold")
            else:
                ax.set_title(f"MAE: {mae:.4f}", fontsize=10)

            if col_idx == 0:
                ax.set_ylabel(m_label, fontsize=11, fontweight="bold")
            if row_idx == 2:
                ax.set_xlabel("x (Streamwise)", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig_err.suptitle(
        f"PDE Supervision Controlled Experiment: Absolute Prediction Errors\n"
        f"Sample #{target_sample_idx} | Horizon h={horizon_step} | Re={re_val:.0f}, Sc={sc_val}",
        fontsize=14, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    fig2_path = out_path / "sample_pde_errors_comparison.png"
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig_err)
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------------------
    # Figure 3: Physical Conservation & Residuals Analysis (2 rows x 3 cols)
    # -------------------------------------------------------------------------
    print("Plotting Figure 3: Physical residual damping analysis...")
    fig_res, axes_res = plt.subplots(2, 3, figsize=(15, 8), dpi=300)

    # Local divergence field
    u_p0_t = p0_step[0:1].unsqueeze(0).to(device)
    v_p0_t = p0_step[1:2].unsqueeze(0).to(device)
    u_pde_t = pde_step[0:1].unsqueeze(0).to(device)
    v_pde_t = pde_step[1:2].unsqueeze(0).to(device)

    from src.utils.fft_derivatives import compute_divergence
    div_p0 = np.abs(compute_divergence(u_p0_t, v_p0_t, domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)[0, 0].cpu().numpy())
    div_pde = np.abs(compute_divergence(u_pde_t, v_pde_t, domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)[0, 0].cpu().numpy())
    max_div = max(div_p0.max(), div_pde.max())

    # Tracer gradient magnitude |grad s|
    from src.utils.fft_derivatives import spectral_grad_2d
    ds_dx_p0, ds_dy_p0 = spectral_grad_2d(p0_step[3:4].unsqueeze(0).to(device), domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)
    ds_dx_pde, ds_dy_pde = spectral_grad_2d(pde_step[3:4].unsqueeze(0).to(device), domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)
    grad_s_p0 = torch.sqrt(ds_dx_p0**2 + ds_dy_p0**2)[0, 0].cpu().numpy()
    grad_s_pde = torch.sqrt(ds_dx_pde**2 + ds_dy_pde**2)[0, 0].cpu().numpy()
    max_grad_s = max(grad_s_p0.max(), grad_s_pde.max())

    # Row 0: P0 (Control)
    im00 = axes_res[0, 0].imshow(div_p0.T, origin="lower", cmap="magma", vmin=0, vmax=max_div, aspect="auto")
    plt.colorbar(im00, ax=axes_res[0, 0], fraction=0.046, pad=0.04)
    axes_res[0, 0].set_title(f"P0: Divergence Error |∇·u|\n(RMS: {np.sqrt(np.mean(div_p0**2)):.4f})", fontweight="bold")
    axes_res[0, 0].set_ylabel("P0 (Control)", fontsize=11, fontweight="bold")

    im01 = axes_res[0, 1].imshow(grad_s_p0.T, origin="lower", cmap="viridis", vmin=0, vmax=max_grad_s, aspect="auto")
    plt.colorbar(im01, ax=axes_res[0, 1], fraction=0.046, pad=0.04)
    axes_res[0, 1].set_title(f"P0: Tracer Gradient |∇s|\n(Mean: {np.mean(grad_s_p0):.4f})", fontweight="bold")

    axes_res[0, 2].bar(["r_u (Mom-x)", "r_v (Mom-y)", "r_s (Tracer)"], [res_u_p0, res_v_p0, res_s_p0], color="#94a3b8", edgecolor="black")
    axes_res[0, 2].set_title("P0: Rollout Residual RMS", fontweight="bold")
    axes_res[0, 2].set_ylabel("Residual RMSE")
    axes_res[0, 2].grid(True, linestyle="--", alpha=0.4)

    # Row 1: PDE (Physics Supervised)
    im10 = axes_res[1, 0].imshow(div_pde.T, origin="lower", cmap="magma", vmin=0, vmax=max_div, aspect="auto")
    plt.colorbar(im10, ax=axes_res[1, 0], fraction=0.046, pad=0.04)
    axes_res[1, 0].set_title(f"PDE: Divergence Error |∇·u|\n(RMS: {np.sqrt(np.mean(div_pde**2)):.4f})", fontweight="bold")
    axes_res[1, 0].set_ylabel("PDE (Supervised)", fontsize=11, fontweight="bold")

    im11 = axes_res[1, 1].imshow(grad_s_pde.T, origin="lower", cmap="viridis", vmin=0, vmax=max_grad_s, aspect="auto")
    plt.colorbar(im11, ax=axes_res[1, 1], fraction=0.046, pad=0.04)
    axes_res[1, 1].set_title(f"PDE: Tracer Gradient |∇s|\n(Mean: {np.mean(grad_s_pde):.4f})", fontweight="bold")

    rs_damping_pct = (res_s_pde - res_s_p0) / res_s_p0 * 100.0
    axes_res[1, 2].bar(["r_u (Mom-x)", "r_v (Mom-y)", "r_s (Tracer)"], [res_u_pde, res_v_pde, res_s_pde], color="#059669", edgecolor="black")
    axes_res[1, 2].set_title(f"PDE: Rollout Residual RMS\n(r_s Damping: {rs_damping_pct:+.2f}%)", fontweight="bold")
    axes_res[1, 2].set_ylabel("Residual RMSE")
    axes_res[1, 2].grid(True, linestyle="--", alpha=0.4)

    for r in range(2):
        for c in range(2):
            axes_res[r, c].set_xticks([])
            axes_res[r, c].set_yticks([])

    fig_res.suptitle(
        f"PDE Supervision Controlled Experiment: Physical Residual & Conservation Contrast\n"
        f"Demonstration of PDE Loss Damping Effect on Divergence and Tracer Residuals",
        fontsize=13, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    fig3_path = out_path / "sample_pde_residuals_comparison.png"
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close(fig_res)
    print(f"Saved: {fig3_path}")

    # Metadata record
    meta_path = out_path / "sample_pde_qualitative_metadata.json"
    meta_info = {
        "sample_index": target_sample_idx,
        "horizon_step": horizon_step,
        "re": re_val,
        "sc": sc_val,
        "figures": [
            "sample_pde_flow_fields_comparison.png",
            "sample_pde_errors_comparison.png",
            "sample_pde_residuals_comparison.png",
        ],
        "metrics_step": {
            "d0_mae_u": float(np.mean(np.abs(d0_step[0].numpy() - gt_step[0].numpy()))),
            "p0_mae_u": float(np.mean(np.abs(p0_step[0].numpy() - gt_step[0].numpy()))),
            "pde_mae_u": float(np.mean(np.abs(pde_step[0].numpy() - gt_step[0].numpy()))),
            "p0_res_tracer_s": float(res_s_p0),
            "pde_res_tracer_s": float(res_s_pde),
            "tracer_damping_pct": rs_damping_pct,
        }
    }
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    return {
        "flow_fields": str(fig1_path),
        "errors": str(fig2_path),
        "residuals": str(fig3_path),
        "metadata": str(meta_path),
    }


if __name__ == "__main__":
    run_pde_qualitative_sampling()
