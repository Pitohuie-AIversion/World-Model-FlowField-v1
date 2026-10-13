#!/usr/bin/env python3
"""Unified Publication-Grade Visualization for All Small-h and Large-H World Models.

Visualizes Ground Truth (GT), Prediction (prd), and Absolute Error across:
  - Large H (Training Horizons): H in {1, 2, 4, 8, 12, 16}
  - Small h (Evaluation Rollout Steps): h in {1, 2, 4, 8, 12, 16, 20, 30}

Variables:
  - Streamwise Velocity u
  - Vorticity omega = dv/dx - du/dy
  - Passive Scalar Tracer s

Generates:
  1. Per-Model Full Evolution Panels: 3 rows (GT, Pred, Error) x 8 columns (all h)
  2. Grand Horizon Ladder Master Matrix: All 6 models side-by-side (GT, Pred, Error)
  3. Quantitative Error Heatmap: Small-h vs Large-H VRMSE trajectories
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import create_flow_dataloaders
from src.data.normalization import FieldNormalizer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.latent_transformer import LatentSTTransformer
from src.models.latent_forecaster import LatentForecaster
from src.utils.fft_derivatives import compute_vorticity
from src.utils.physics_contract import zero_mean_pressure_gauge, SHEAR_FLOW_DOMAIN_SIZE_XY
from src.utils.provenance import compute_file_sha256, get_git_commit, is_git_dirty

HORIZON_MODELS = {
    "H1": {
        "title": "H=1 (E0 Single-Step)",
        "short_title": "H1 (E0)",
        "training_H": 1,
        "path": "outputs/checkpoints/dynamics/closure_r4/ablation_E0_single_step/latent_transformer/best_vrmse_mean.pt",
    },
    "H2": {
        "title": "H=2 (E4-H2 Control)",
        "short_title": "H2 (E4)",
        "training_H": 2,
        "path": "outputs/checkpoints/dynamics/horizon_r1/seed_42/E4_H2_control/latent_transformer/best_vrmse_mean.pt",
    },
    "H4": {
        "title": "H=4 (E4-H4 Curriculum)",
        "short_title": "H4 (E4)",
        "training_H": 4,
        "path": "outputs/checkpoints/dynamics/horizon_r1/seed_42/E4_H4/latent_transformer/best_vrmse_mean.pt",
    },
    "H8": {
        "title": "H=8 (E4-H8 Long-Best)",
        "short_title": "H8 (E4)",
        "training_H": 8,
        "path": "outputs/checkpoints/dynamics/horizon_r1/seed_42/E4_H8/latent_transformer/checkpoint_step_11_vrmse_mean_0.2186.pt",
    },
    "H12": {
        "title": "H=12 (E4-H12 Long-Best)",
        "short_title": "H12 (E4)",
        "training_H": 12,
        "path": "outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt",
    },
    "H16": {
        "title": "H=16 (E4-H16 Long-Best)",
        "short_title": "H16 (E4)",
        "training_H": 16,
        "path": "outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H16/latent_transformer/best_long_vrmse.pt",
    },
}

EVAL_HORIZONS = [1, 2, 4, 8, 12, 16, 20, 30]

VARIABLE_CONFIGS = {
    "u": {
        "name": "u",
        "display_name": "Streamwise Velocity $u$",
        "cmap": "RdBu_r",
        "err_cmap": "magma",
        "diverging": True,
    },
    "vorticity": {
        "name": "vorticity",
        "display_name": r"Vorticity $\omega = \partial_x v - \partial_y u$",
        "cmap": "PRGn",
        "err_cmap": "inferno",
        "diverging": True,
    },
    "tracer": {
        "name": "tracer",
        "display_name": "Passive Scalar Tracer $s$",
        "cmap": "viridis",
        "err_cmap": "plasma",
        "diverging": False,
    },
}


def load_forecaster(ckpt_path: str, device: torch.device) -> Tuple[LatentForecaster, Dict[str, Any]]:
    """Load LatentForecaster checkpoint with verified architecture."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=256,
        cond_dim=128,
        depth=6,
        num_heads=8,
        history_length=4,
        prediction_mode="direct",
        use_spatial_pos=False,
    )
    model = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder).to(device)

    state_dict = ckpt_data.get("model_state_dict", ckpt_data)
    cleaned_state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()
    return model, ckpt_data


def extract_variable_field(q_tensor: torch.Tensor, var_name: str) -> np.ndarray:
    """Extract physical 2D scalar field from q tensor (C, H, W)."""
    # q is (4, Ny, Nx) = (4, 128, 256)
    if var_name == "u":
        return q_tensor[0].cpu().numpy()
    elif var_name == "v":
        return q_tensor[1].cpu().numpy()
    elif var_name == "p":
        return q_tensor[2].cpu().numpy()
    elif var_name in ("tracer", "s"):
        return q_tensor[3].cpu().numpy()
    elif var_name == "vorticity":
        u = q_tensor[0:1]
        v = q_tensor[1:2]
        vort = compute_vorticity(u, v, domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)
        return vort.cpu().numpy()[0]
    else:
        raise ValueError(f"Unknown variable name: {var_name}")


def plot_per_model_rollout(
    model_key: str,
    model_meta: Dict[str, Any],
    gt_trajectory: torch.Tensor,
    pred_trajectory: torch.Tensor,
    var_name: str,
    horizons: List[int],
    out_dir: Path,
    sample_tag: str,
) -> Path:
    """Plot Figure Type 1: Single Model Full Rollout (3 rows: GT, Pred, Error; cols: horizons)."""
    cfg = VARIABLE_CONFIGS[var_name]
    n_cols = len(horizons)

    fig, axes = plt.subplots(3, n_cols, figsize=(3.8 * n_cols, 10.5), constrained_layout=True)
    plt.rcParams.update({"font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"]})

    gt_fields = [extract_variable_field(gt_trajectory[h - 1], var_name) for h in horizons]
    pred_fields = [extract_variable_field(pred_trajectory[h - 1], var_name) for h in horizons]
    err_fields = [np.abs(p - g) for p, g in zip(pred_fields, gt_fields)]

    # Compute colorbar norms per column (or unified)
    for col_idx, h in enumerate(horizons):
        g_f = gt_fields[col_idx]
        p_f = pred_fields[col_idx]
        e_f = err_fields[col_idx]

        if cfg["diverging"]:
            vmax = max(abs(np.percentile(g_f, 99.5)), abs(np.percentile(p_f, 99.5)), abs(np.percentile(g_f, 0.5)), abs(np.percentile(p_f, 0.5)))
            vmin = -vmax
        else:
            vmin = min(np.percentile(g_f, 0.5), np.percentile(p_f, 0.5))
            vmax = max(np.percentile(g_f, 99.5), np.percentile(p_f, 99.5))

        emax = np.percentile(e_f, 99.5) if np.percentile(e_f, 99.5) > 1e-4 else np.max(e_f)
        emax = max(emax, 1e-3)

        mae = float(np.mean(e_f))
        rmse = float(np.sqrt(np.mean(e_f**2)))

        # Row 0: Ground Truth
        im0 = axes[0, col_idx].imshow(g_f, origin="lower", cmap=cfg["cmap"], vmin=vmin, vmax=vmax, aspect="auto")
        axes[0, col_idx].set_title(f"Ground Truth ($t={h}$)", fontsize=11, fontweight="bold")
        axes[0, col_idx].set_xticks([])
        axes[0, col_idx].set_yticks([])
        cbar0 = plt.colorbar(im0, ax=axes[0, col_idx], fraction=0.046, pad=0.03)
        cbar0.ax.tick_params(labelsize=8)

        # Row 1: Prediction
        im1 = axes[1, col_idx].imshow(p_f, origin="lower", cmap=cfg["cmap"], vmin=vmin, vmax=vmax, aspect="auto")
        axes[1, col_idx].set_title(f"Pred ($t={h}$)", fontsize=11, fontweight="bold")
        axes[1, col_idx].set_xticks([])
        axes[1, col_idx].set_yticks([])
        cbar1 = plt.colorbar(im1, ax=axes[1, col_idx], fraction=0.046, pad=0.03)
        cbar1.ax.tick_params(labelsize=8)

        # Row 2: Error
        im2 = axes[2, col_idx].imshow(e_f, origin="lower", cmap=cfg["err_cmap"], vmin=0, vmax=emax, aspect="auto")
        axes[2, col_idx].set_title(f"|Error| ($t={h}$)\nRMSE: {rmse:.4f}", fontsize=10)
        axes[2, col_idx].set_xlabel("x (Streamwise)", fontsize=10)
        axes[2, col_idx].set_xticks([])
        axes[2, col_idx].set_yticks([])
        cbar2 = plt.colorbar(im2, ax=axes[2, col_idx], fraction=0.046, pad=0.03)
        cbar2.ax.tick_params(labelsize=8)

    axes[0, 0].set_ylabel("Ground Truth (GT)", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel(f"Prediction (prd)\n[{model_meta['short_title']}]", fontsize=12, fontweight="bold")
    axes[2, 0].set_ylabel("Absolute Error\n(|prd - GT|)", fontsize=12, fontweight="bold")

    fig.suptitle(
        f"Rollout Horizon Evolution: {model_meta['title']} | Variable: {cfg['display_name']} | Sample: {sample_tag}",
        fontsize=15,
        fontweight="bold",
    )

    out_path = out_dir / f"rollout_evolution_{model_key}_{var_name}_{sample_tag}.png"
    plt.savefig(out_path, dpi=250)
    plt.close()
    return out_path


def plot_grand_horizon_ladder_matrix(
    models_dict: Dict[str, Dict[str, Any]],
    gt_trajectory: torch.Tensor,
    preds_dict: Dict[str, torch.Tensor],
    var_name: str,
    key_horizons: List[int],
    out_dir: Path,
    sample_tag: str,
) -> Path:
    """Plot Figure Type 2: Grand Horizon Comparison Ladder across all models H and rollout steps h.

    Layout:
      For each key horizon h (row):
        Col 0: Ground Truth (GT)
        Cols 1..2: H1 Pred, H1 Error
        Cols 3..4: H2 Pred, H2 Error
        Cols 5..6: H4 Pred, H4 Error
        Cols 7..8: H8 Pred, H8 Error
        Cols 9..10: H12 Pred, H12 Error
        Cols 11..12: H16 Pred, H16 Error
    Total: len(key_horizons) rows x 13 columns.
    """
    cfg = VARIABLE_CONFIGS[var_name]
    model_keys = ["H1", "H2", "H4", "H8", "H12", "H16"]
    n_rows = len(key_horizons)
    n_cols = 1 + len(model_keys) * 2 # 13 columns

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.7 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r_idx, h in enumerate(key_horizons):
        gt_f = extract_variable_field(gt_trajectory[h - 1], var_name)
        model_preds = {m_k: extract_variable_field(preds_dict[m_k][h - 1], var_name) for m_k in model_keys}
        model_errs = {m_k: np.abs(model_preds[m_k] - gt_f) for m_k in model_keys}

        # Shared field scale across GT and all Predictions for this step
        all_field_vals = [gt_f] + list(model_preds.values())
        if cfg["diverging"]:
            vmax = max(abs(np.percentile(f, 99.5)) for f in all_field_vals)
            vmin = -vmax
        else:
            vmin = min(np.percentile(f, 0.5) for f in all_field_vals)
            vmax = max(np.percentile(f, 99.5) for f in all_field_vals)

        # Shared error scale across all Models for this step
        emax = max(np.percentile(e, 99.5) for e in model_errs.values())
        emax = max(emax, 1e-3)

        # Col 0: Ground Truth
        ax_gt = axes[r_idx, 0]
        im_gt = ax_gt.imshow(gt_f, origin="lower", cmap=cfg["cmap"], vmin=vmin, vmax=vmax, aspect="auto")
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        if r_idx == 0:
            ax_gt.set_title("Ground Truth\n(Target DNS)", fontsize=11, fontweight="bold", pad=8)
        ax_gt.set_ylabel(f"Step $t={h}$", fontsize=12, fontweight="bold")
        cbar_gt = plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.03)
        cbar_gt.ax.tick_params(labelsize=7)

        # Model columns: Pred and Error
        for m_idx, m_k in enumerate(model_keys):
            col_pred = 1 + m_idx * 2
            col_err = 2 + m_idx * 2

            p_f = model_preds[m_k]
            e_f = model_errs[m_k]
            rmse = float(np.sqrt(np.mean(e_f**2)))

            ax_p = axes[r_idx, col_pred]
            im_p = ax_p.imshow(p_f, origin="lower", cmap=cfg["cmap"], vmin=vmin, vmax=vmax, aspect="auto")
            ax_p.set_xticks([])
            ax_p.set_yticks([])
            if r_idx == 0:
                ax_p.set_title(f"{models_dict[m_k]['short_title']}\nPred", fontsize=10, fontweight="bold", pad=8)
            cbar_p = plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.03)
            cbar_p.ax.tick_params(labelsize=7)

            ax_e = axes[r_idx, col_err]
            im_e = ax_e.imshow(e_f, origin="lower", cmap=cfg["err_cmap"], vmin=0, vmax=emax, aspect="auto")
            ax_e.set_xticks([])
            ax_e.set_yticks([])
            if r_idx == 0:
                ax_e.set_title(f"{models_dict[m_k]['short_title']}\n|Error|", fontsize=10, fontweight="bold", pad=8)
            ax_e.set_xlabel(f"RMSE: {rmse:.4f}", fontsize=8)
            cbar_e = plt.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.03)
            cbar_e.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Grand Horizon Ladder: All Training Horizons $H \\in \\{{1, 2, 4, 8, 12, 16\\}}$ vs Rollout Steps $t=h$ | {cfg['display_name']} | {sample_tag}",
        fontsize=16,
        fontweight="bold",
    )

    out_path = out_dir / f"grand_horizon_ladder_{var_name}_{sample_tag}.png"
    plt.savefig(out_path, dpi=250)
    plt.close()
    return out_path


def plot_quantitative_horizon_error_heatmap(
    gt_trajectory: torch.Tensor,
    preds_dict: Dict[str, torch.Tensor],
    out_dir: Path,
    sample_tag: str,
) -> Path:
    """Plot Figure Type 3: VRMSE error trajectories across all 30 rollout steps for all 6 models."""
    model_keys = ["H1", "H2", "H4", "H8", "H12", "H16"]
    steps = list(range(1, 31))

    # Compute VRMSE per step per model
    vrmse_matrix = np.zeros((len(model_keys), 30))

    for m_idx, m_k in enumerate(model_keys):
        pred_traj = preds_dict[m_k]
        for t in range(30):
            g_t = gt_trajectory[t] # (4, Ny, Nx)
            p_t = pred_traj[t]
            # Velocity VRMSE across (u, v)
            u_err = (p_t[0] - g_t[0]).pow(2).mean().sqrt().item()
            v_err = (p_t[1] - g_t[1]).pow(2).mean().sqrt().item()
            u_var = g_t[0].var().sqrt().item() + 1e-6
            v_var = g_t[1].var().sqrt().item() + 1e-6
            vrmse = 0.5 * (u_err / u_var + v_err / v_var)
            vrmse_matrix[m_idx, t] = vrmse

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5), constrained_layout=True)

    # Subplot 1: VRMSE trajectories curve
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
    markers = ["x", "o", "s", "^", "v", "D"]

    for m_idx, m_k in enumerate(model_keys):
        m_info = HORIZON_MODELS[m_k]
        ax1.plot(
            steps,
            vrmse_matrix[m_idx],
            label=f"{m_info['title']} (train H={m_info['training_H']})",
            color=colors[m_idx],
            marker=markers[m_idx],
            markevery=3,
            linewidth=2.2,
        )

    ax1.set_xlabel("Rollout Step $t=h$", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Velocity VRMSE", fontsize=12, fontweight="bold")
    ax1.set_title("Rollout Degradation Curves: Small $h$ vs Large $H$", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=10)

    # Subplot 2: 2D Heatmap (Large H vs Small h)
    im = ax2.imshow(
        vrmse_matrix,
        cmap="YlOrRd",
        aspect="auto",
        origin="upper",
        extent=[0.5, 30.5, len(model_keys) - 0.5, -0.5],
    )
    ax2.set_yticks(range(len(model_keys)))
    ax2.set_yticklabels([HORIZON_MODELS[k]["short_title"] for k in model_keys], fontsize=11, fontweight="bold")
    ax2.set_xlabel("Rollout Step $h \\in [1, 30]$", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Training Horizon $H$", fontsize=12, fontweight="bold")
    ax2.set_title("Error Heatmap: Training Horizon $H$ vs Rollout Step $h$", fontsize=13, fontweight="bold")

    # Mark the training horizon H on the heatmap
    for m_idx, m_k in enumerate(model_keys):
        train_h = HORIZON_MODELS[m_k]["training_H"]
        ax2.scatter(train_h, m_idx, color="blue", marker="|", s=300, linewidths=3, zorder=5)

    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("VRMSE", fontsize=11)

    fig.suptitle(f"Quantitative Horizon Analysis: Large $H$ vs Small $h$ | {sample_tag}", fontsize=15, fontweight="bold")

    out_path = out_dir / f"horizon_quantitative_heatmap_{sample_tag}.png"
    plt.savefig(out_path, dpi=250)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize all small-h and large-H models with GT, Pred, Error.")
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--out_dir", type=str, default="outputs/figures/horizon_all_h_and_H")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_index", type=int, default=0, help="Test sample index to visualize")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Initializing All-h & Large-H GT/Pred/Error Visualization Pipeline ---")
    print(f"Device: {device}")
    print(f"Output directory: {out_dir}")

    # 1. Load Data
    print("\nLoading test dataset with horizon=30...")
    _, _, test_loader, normalizer = create_flow_dataloaders(
        data_root=args.data_root,
        split_type="grouped",
        batch_size=1,
        downsample_factor=2,
        horizon=30,
        normalize=True,
    )
    if normalizer is None:
        normalizer = FieldNormalizer.load("outputs/normalization/channel_stats.pt")

    # Retrieve target sample
    target_batch = None
    for idx, batch in enumerate(test_loader):
        if idx == args.sample_index:
            target_batch = batch
            break

    if target_batch is None:
        raise ValueError(f"Sample index {args.sample_index} not found in test dataset.")

    sample_tag = f"sample_{args.sample_index}"
    re_val = float(target_batch.get("re", torch.tensor([10000.0]))[0].item())
    sc_val = float(target_batch.get("sc", torch.tensor([0.1]))[0].item())
    print(f"Loaded sample #{args.sample_index}: Re={re_val:.0f}, Sc={sc_val:.2f}")

    q_hist = target_batch["history"].to(device) # (1, 4, 4, 128, 256)
    q_fut = target_batch["future"].to(device)   # (1, 30, 4, 128, 256)
    re_tensor = target_batch["re"].to(device)
    sc_tensor = target_batch["sc"].to(device)

    # Denormalize Ground Truth
    gt_phys = normalizer.denormalize(q_fut[0]).clone()
    gt_phys = zero_mean_pressure_gauge(gt_phys)

    # 2. Run Rollout for All Large-H Models
    print("\nExecuting 30-step rollouts for all 6 Horizon models...")
    models = {}
    preds_phys = {}

    for m_key, m_meta in HORIZON_MODELS.items():
        t0 = time.time()
        print(f"  Loading & rolling out {m_meta['title']}...")
        model, _ = load_forecaster(m_meta["path"], device=device)
        models[m_key] = model

        with torch.no_grad():
            pred_traj = model.forward_rollout(q_hist, re_tensor, sc_tensor, horizon=30)
            p_phys = normalizer.denormalize(pred_traj[0]).clone()
            p_phys = zero_mean_pressure_gauge(p_phys)
            preds_phys[m_key] = p_phys

        print(f"    Completed in {time.time()-t0:.2f}s")

    # 3. Generate Visualizations
    generated_figures = []

    print("\n--- Generating Per-Model Full Evolution Panels (GT, Pred, Error x 8 horizons) ---")
    for m_key, m_meta in HORIZON_MODELS.items():
        for var in ["u", "vorticity", "tracer"]:
            p = plot_per_model_rollout(
                model_key=m_key,
                model_meta=m_meta,
                gt_trajectory=gt_phys,
                pred_trajectory=preds_phys[m_key],
                var_name=var,
                horizons=EVAL_HORIZONS,
                out_dir=out_dir,
                sample_tag=sample_tag,
            )
            print(f"  Saved: {p.name}")
            generated_figures.append(str(p))

    print("\n--- Generating Grand Horizon Ladder Master Matrices ---")
    ladder_horizons = [1, 2, 4, 8, 12, 16, 20, 30]
    for var in ["u", "vorticity", "tracer"]:
        p = plot_grand_horizon_ladder_matrix(
            models_dict=HORIZON_MODELS,
            gt_trajectory=gt_phys,
            preds_dict=preds_phys,
            var_name=var,
            key_horizons=ladder_horizons,
            out_dir=out_dir,
            sample_tag=sample_tag,
        )
        print(f"  Saved: {p.name}")
        generated_figures.append(str(p))

    print("\n--- Generating Quantitative Horizon Error Heatmap ---")
    p_heat = plot_quantitative_horizon_error_heatmap(
        gt_trajectory=gt_phys,
        preds_dict=preds_phys,
        out_dir=out_dir,
        sample_tag=sample_tag,
    )
    print(f"  Saved: {p_heat.name}")
    generated_figures.append(str(p_heat))

    # 4. Save Metadata Manifest
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "sample_index": args.sample_index,
        "sample_tag": sample_tag,
        "re": re_val,
        "sc": sc_val,
        "eval_horizons": EVAL_HORIZONS,
        "ladder_horizons": ladder_horizons,
        "models_evaluated": {
            k: {
                "title": v["title"],
                "training_H": v["training_H"],
                "checkpoint_path": v["path"],
                "checkpoint_sha256": compute_file_sha256(v["path"]),
            }
            for k, v in HORIZON_MODELS.items()
        },
        "figures_generated": generated_figures,
        "git_commit": get_git_commit(),
        "git_dirty": is_git_dirty(),
    }

    manifest_path = out_dir / f"horizon_visualization_manifest_{sample_tag}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved metadata manifest to: {manifest_path}")

    print("\n[SUCCESS] Completed all Small-h and Large-H GT/Pred/Error visualizations!")


if __name__ == "__main__":
    main()
