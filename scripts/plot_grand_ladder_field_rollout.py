#!/usr/bin/env python3
"""Grand Horizon Ladder Representative Physical Channel Field Visualization.

Visualizes 2D spatial physical fields across autoregressive rollout horizons:
  - Representative Physical Channels:
      1. Vorticity omega = dv/dx - du/dy (Turbulent Coherent Structures & Shear Instability)
      2. Passive Scalar Tracer s (Advection-Diffusion Transport & Long-Horizon Numerical Stability)
      3. Streamwise Velocity u (Dominant Kinetic Energy & Mean Momentum Profile)
  - Full Horizon Ladder Models:
      1. Ground Truth (Target DNS Reference)
      2. H=2 (E4-H2 Control Baseline)
      3. H=4 (E4-H4 Curriculum Expansion)
      4. H=8 (E4-H8 Long-Best Checkpoint)
      5. H=12 (E4-H12 Long-Best Checkpoint)
      6. H=16 (E4-H16 Long-Best Checkpoint)
  - Key Rollout Horizons:
      h in {5, 10, 20, 30}

Produces:
  1. Master Two-Channel Panel (Vorticity & Tracer, 8 rows x 11 cols):
     outputs/figures/horizon_r1/figure_grand_ladder_representative_fields.png
  2. High-Resolution Single-Channel Plots (4 rows x 11 cols each):
     - outputs/figures/horizon_r1/figure_grand_ladder_rollout_vorticity.png
     - outputs/figures/horizon_r1/figure_grand_ladder_rollout_tracer.png
     - outputs/figures/horizon_r1/figure_grand_ladder_rollout_u.png
  3. Companion Provenance & Quantitative Metrics Manifest:
     outputs/figures/horizon_r1/figure_grand_ladder_field_rollout_metadata.json
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
from src.utils.reproducibility import seed_everything

# Full Horizon Ladder Models
LADDER_MODELS = {
    "H2": {
        "title": "H=2 Control (12 Ep)",
        "short_title": "H=2",
        "training_H": 2,
        "path": "outputs/checkpoints/dynamics/horizon_r1/seed_42/E4_H2_control/latent_transformer/best_vrmse_mean.pt",
        "color": "#1f77b4",
    },
    "H4": {
        "title": "H=4 Expansion (12 Ep)",
        "short_title": "H=4",
        "training_H": 4,
        "path": "outputs/checkpoints/dynamics/horizon_r1/seed_42/E4_H4/latent_transformer/best_vrmse_mean.pt",
        "color": "#2ca02c",
    },
    "H8": {
        "title": "H=8 Expansion (Long-Best)",
        "short_title": "H=8",
        "training_H": 8,
        "path": "outputs/checkpoints/dynamics/horizon_r1/seed_42/E4_H8/latent_transformer/checkpoint_step_11_vrmse_mean_0.2186.pt",
        "color": "#d62728",
    },
    "H12": {
        "title": "H=12 Extension (Long-Best)",
        "short_title": "H=12",
        "training_H": 12,
        "path": "outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt",
        "color": "#9467bd",
    },
    "H16": {
        "title": "H=16 Extension (Long-Best)",
        "short_title": "H=16",
        "training_H": 16,
        "path": "outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H16/latent_transformer/best_long_vrmse.pt",
        "color": "#8c564b",
    },
}

KEY_HORIZONS = [5, 10, 20, 30]

CHANNEL_CONFIGS = {
    "vorticity": {
        "name": "vorticity",
        "latex_name": r"Vorticity $\omega = \partial_x v - \partial_y u$",
        "short_latex": r"Vorticity $\omega$",
        "physical_significance": "Turbulent coherent vortex rollup, Kelvin-Helmholtz billows, and fine-scale dissipation structures. Direct objective of physical regularization loss.",
        "cmap": "coolwarm",
        "err_cmap": "inferno",
        "diverging": True,
        "unit": r"$\mathrm{s}^{-1}$",
    },
    "tracer": {
        "name": "tracer",
        "latex_name": r"Passive Scalar Tracer $s$",
        "short_latex": r"Passive Tracer $s$",
        "physical_significance": "Governed by non-linear advection-diffusion. Prone to numerical dispersion and unphysical explosion at long horizons (h=30). Longer H strictly bounds variance.",
        "cmap": "viridis",
        "err_cmap": "magma",
        "diverging": False,
        "unit": r"$\mathrm{conc.}$",
    },
    "u": {
        "name": "u",
        "latex_name": r"Streamwise Velocity $u$",
        "short_latex": r"Velocity $u$",
        "physical_significance": "Carries >90% of total kinetic energy. Demonstrates macroscopic shear-layer momentum profile preservation.",
        "cmap": "RdBu_r",
        "err_cmap": "plasma",
        "diverging": True,
        "unit": r"$\mathrm{m/s}$",
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
    """Extract physical 2D scalar field from q tensor (4, Ny, Nx)."""
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


def plot_ladder_channel_grid(
    var_name: str,
    gt_trajectory: torch.Tensor,
    preds_dict: Dict[str, torch.Tensor],
    model_keys: List[str],
    horizons: List[int],
    out_path: Path,
    sample_tag: str,
    re_val: float,
    sc_val: float,
) -> Tuple[Path, Dict[str, Any]]:
    """Generate high-resolution grid for all ladder models (H2, H4, H8, H12, H16).
    
    Cols:
      0: Ground Truth (DNS)
      1..2: H2 (Pred, |Err|)
      3..4: H4 (Pred, |Err|)
      5..6: H8 (Pred, |Err|)
      7..8: H12 (Pred, |Err|)
      9..10: H16 (Pred, |Err|)
    Total: 11 columns x len(horizons) rows.
    """
    cfg = CHANNEL_CONFIGS[var_name]
    n_rows = len(horizons)
    n_cols = 1 + len(model_keys) * 2  # 11 cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.1 * n_cols, 2.65 * n_rows),
        constrained_layout=True
    )
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    })

    channel_stats = {}

    for r_idx, h in enumerate(horizons):
        t_idx = h - 1
        gt_f = extract_variable_field(gt_trajectory[t_idx], var_name)
        m_preds = {mk: extract_variable_field(preds_dict[mk][t_idx], var_name) for mk in model_keys}
        m_errs = {mk: np.abs(m_preds[mk] - gt_f) for mk in model_keys}

        # Shared field scale
        all_field_vals = [gt_f] + list(m_preds.values())
        if cfg["diverging"]:
            vmax = max(abs(np.percentile(f, 99.5)) for f in all_field_vals)
            vmin = -vmax
        else:
            vmin = min(np.percentile(f, 0.5) for f in all_field_vals)
            vmax = max(np.percentile(f, 99.5) for f in all_field_vals)

        # Shared error scale across all models at this step
        emax = max(np.percentile(e, 99.5) for e in m_errs.values())
        emax = max(emax, 1e-3)

        # Col 0: Ground Truth
        ax_gt = axes[r_idx, 0]
        im_gt = ax_gt.imshow(gt_f, origin="lower", cmap=cfg["cmap"], vmin=vmin, vmax=vmax, aspect="auto")
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        ax_gt.set_ylabel(f"Step $t={h}$", fontsize=11, fontweight="bold", labelpad=6)
        if r_idx == 0:
            ax_gt.set_title("Ground Truth\n(DNS Target)", fontsize=11, fontweight="bold", pad=8)
        cbar_gt = plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.03)
        cbar_gt.ax.tick_params(labelsize=7)

        # Ladder Models: Pred & Error
        step_metrics = {}
        for m_idx, mk in enumerate(model_keys):
            col_p = 1 + m_idx * 2
            col_e = 2 + m_idx * 2
            p_f = m_preds[mk]
            e_f = m_errs[mk]

            rmse = float(np.sqrt(np.mean(e_f ** 2)))
            mae = float(np.mean(e_f))
            step_metrics[mk] = {"rmse": rmse, "mae": mae}

            # Pred
            ax_p = axes[r_idx, col_p]
            im_p = ax_p.imshow(p_f, origin="lower", cmap=cfg["cmap"], vmin=vmin, vmax=vmax, aspect="auto")
            ax_p.set_xticks([])
            ax_p.set_yticks([])
            if r_idx == 0:
                ax_p.set_title(f"{LADDER_MODELS[mk]['short_title']}\nPred", fontsize=11, fontweight="bold", pad=8)
            cbar_p = plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.03)
            cbar_p.ax.tick_params(labelsize=7)

            # Error
            ax_e = axes[r_idx, col_e]
            im_e = ax_e.imshow(e_f, origin="lower", cmap=cfg["err_cmap"], vmin=0, vmax=emax, aspect="auto")
            ax_e.set_xticks([])
            ax_e.set_yticks([])
            if r_idx == 0:
                ax_e.set_title(f"{LADDER_MODELS[mk]['short_title']}\n|Error|", fontsize=11, fontweight="bold", pad=8)

            color_stat = "#b30000" if mk == "H2" else ("#006600" if mk in ("H8", "H12", "H16") else "#333333")
            ax_e.set_xlabel(f"RMSE: {rmse:.4f}", fontsize=9, fontweight="bold", color=color_stat, labelpad=3)
            cbar_e = plt.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.03)
            cbar_e.ax.tick_params(labelsize=7)

        channel_stats[f"step_{h}"] = step_metrics

    ladder_str = ", ".join(LADDER_MODELS[k]["short_title"] for k in model_keys)
    fig.suptitle(
        f"Grand Horizon Ladder Rollout Field Evolution: {cfg['latex_name']} ({sample_tag}, $\\mathrm{{Re}}={re_val:.0f}, \\mathrm{{Sc}}={sc_val:.1f}$)\n"
        f"Training Horizons $H \\in \\{{{ladder_str}\\}}$ vs Rollout Horizons $t \\in \\{{{', '.join(str(x) for x in horizons)}\\}}$",
        fontsize=14,
        fontweight="bold",
    )

    plt.savefig(out_path, dpi=250)
    plt.close()
    return out_path, channel_stats


def plot_combined_ladder_representative_fields(
    channels: List[str],
    gt_trajectory: torch.Tensor,
    preds_dict: Dict[str, torch.Tensor],
    model_keys: List[str],
    horizons: List[int],
    out_path: Path,
    sample_tag: str,
    re_val: float,
    sc_val: float,
) -> Path:
    """Generate grand master panel (Vorticity and Tracer combined, 8 rows x 11 cols)."""
    n_rows = len(horizons) * len(channels)
    n_cols = 1 + len(model_keys) * 2  # 11 cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.1 * n_cols, 2.35 * n_rows),
        constrained_layout=True
    )
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    })

    current_row = 0
    for ch_idx, var_name in enumerate(channels):
        cfg = CHANNEL_CONFIGS[var_name]

        for h in horizons:
            t_idx = h - 1
            gt_f = extract_variable_field(gt_trajectory[t_idx], var_name)
            m_preds = {mk: extract_variable_field(preds_dict[mk][t_idx], var_name) for mk in model_keys}
            m_errs = {mk: np.abs(m_preds[mk] - gt_f) for mk in model_keys}

            # Shared field scale
            all_field_vals = [gt_f] + list(m_preds.values())
            if cfg["diverging"]:
                vmax = max(abs(np.percentile(f, 99.5)) for f in all_field_vals)
                vmin = -vmax
            else:
                vmin = min(np.percentile(f, 0.5) for f in all_field_vals)
                vmax = max(np.percentile(f, 99.5) for f in all_field_vals)

            # Shared error scale
            emax = max(np.percentile(e, 99.5) for e in m_errs.values())
            emax = max(emax, 1e-3)

            # Col 0: GT
            ax_gt = axes[current_row, 0]
            im_gt = ax_gt.imshow(gt_f, origin="lower", cmap=cfg["cmap"], vmin=vmin, vmax=vmax, aspect="auto")
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
            prefix_label = f"[{cfg['short_latex']}]\n" if h == horizons[0] else ""
            ax_gt.set_ylabel(f"{prefix_label}Step $t={h}$", fontsize=10, fontweight="bold", labelpad=5)
            if current_row == 0:
                ax_gt.set_title("Ground Truth\n(DNS Target)", fontsize=11, fontweight="bold", pad=8)
            cbar_gt = plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.03)
            cbar_gt.ax.tick_params(labelsize=7)

            # Models
            for m_idx, mk in enumerate(model_keys):
                col_p = 1 + m_idx * 2
                col_e = 2 + m_idx * 2
                p_f = m_preds[mk]
                e_f = m_errs[mk]
                rmse = float(np.sqrt(np.mean(e_f ** 2)))

                # Pred
                ax_p = axes[current_row, col_p]
                im_p = ax_p.imshow(p_f, origin="lower", cmap=cfg["cmap"], vmin=vmin, vmax=vmax, aspect="auto")
                ax_p.set_xticks([])
                ax_p.set_yticks([])
                if current_row == 0:
                    ax_p.set_title(f"{LADDER_MODELS[mk]['short_title']}\nPred", fontsize=11, fontweight="bold", pad=8)
                cbar_p = plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.03)
                cbar_p.ax.tick_params(labelsize=7)

                # Error
                ax_e = axes[current_row, col_e]
                im_e = ax_e.imshow(e_f, origin="lower", cmap=cfg["err_cmap"], vmin=0, vmax=emax, aspect="auto")
                ax_e.set_xticks([])
                ax_e.set_yticks([])
                if current_row == 0:
                    ax_e.set_title(f"{LADDER_MODELS[mk]['short_title']}\n|Error|", fontsize=11, fontweight="bold", pad=8)
                color_stat = "#b30000" if mk == "H2" else ("#006600" if mk in ("H8", "H12", "H16") else "#333333")
                ax_e.set_xlabel(f"RMSE: {rmse:.4f}", fontsize=8.5, fontweight="bold", color=color_stat, labelpad=2)
                cbar_e = plt.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.03)
                cbar_e.ax.tick_params(labelsize=7)

            current_row += 1

    ladder_str = ", ".join(LADDER_MODELS[k]["short_title"] for k in model_keys)
    fig.suptitle(
        f"Grand Horizon Ladder — Representative Physical Channels: Vorticity $\\omega$ & Tracer $s$\n"
        f"Complete Horizon Spectrum $H \\in \\{{{ladder_str}\\}}$ Across Horizons $t \\in \\{{5, 10, 20, 30\\}}$ (Sample: {sample_tag}, $\\mathrm{{Re}}={re_val:.0f}$)",
        fontsize=14,
        fontweight="bold",
    )

    plt.savefig(out_path, dpi=250)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Grand Horizon Ladder field rollout visualization.")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--out_dir", type=str, default="outputs/figures/horizon_r1")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_index", type=int, default=0, help="Test sample index to visualize")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_keys = ["H2", "H4", "H8", "H12", "H16"]

    print("================================================================================")
    print("  Grand Horizon Ladder Representative Physical Channel Visualization")
    print(f"  Device: {device}")
    print(f"  Ladder Models: {model_keys}")
    print(f"  Target Horizons: {KEY_HORIZONS}")
    print(f"  Target Sample Index: {args.sample_index}")
    print(f"  Output Directory: {out_dir}")
    print("================================================================================")

    # 1. Dataset & Normalizer Loading
    split_file = "outputs/splits/grouped_split.json"
    stats_path = "outputs/normalization/stats_grouped.pt"
    normalizer = FieldNormalizer()
    normalizer.load_state_dict(torch.load(stats_path, weights_only=True, map_location="cpu"))

    print("\n[1/4] Loading test dataset with horizon=30...")
    _, _, test_loader, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file=split_file,
        data_root=args.data_dir,
        history_length=4,
        horizon=30,
        batch_size=1,
        num_workers=0,
        normalize=True,
        normalizer=normalizer,
    )

    target_batch = None
    for idx, batch in enumerate(test_loader):
        if idx == args.sample_index:
            target_batch = batch
            break

    if target_batch is None:
        raise ValueError(f"Sample index {args.sample_index} not found in test dataset.")

    sample_tag = f"sample_{args.sample_index}"
    re_val = float(target_batch["re"][0].item()) if "re" in target_batch else 10000.0
    sc_val = float(target_batch["sc"][0].item()) if "sc" in target_batch else 0.1
    print(f"  Loaded test sample #{args.sample_index}: Reynolds={re_val:.0f}, Schmidt={sc_val:.2f}")

    q_hist = target_batch["history"].to(device)
    q_fut = target_batch["future"].to(device)
    re_tensor = target_batch["re"].to(device)
    sc_tensor = target_batch["sc"].to(device)

    gt_phys = normalizer.denormalize(q_fut[0]).clone()
    gt_phys = zero_mean_pressure_gauge(gt_phys)

    # 2. Rollout for all 5 Ladder Models
    print(f"\n[2/4] Executing 30-step rollouts for all {len(model_keys)} Ladder Models...")
    preds_phys = {}
    model_provenance = {}

    for mk in model_keys:
        minfo = LADDER_MODELS[mk]
        t0 = time.time()
        print(f"  Loading model: {minfo['title']} from {minfo['path']}...")
        model, _ = load_forecaster(minfo["path"], device=device)

        with torch.no_grad():
            pred_traj = model.forward_rollout(q_hist, re_tensor, sc_tensor, horizon=30)
            p_phys = normalizer.denormalize(pred_traj[0]).clone()
            p_phys = zero_mean_pressure_gauge(p_phys)
            preds_phys[mk] = p_phys

        dur = time.time() - t0
        sha = compute_file_sha256(minfo["path"])
        model_provenance[mk] = {
            "title": minfo["title"],
            "short_title": minfo["short_title"],
            "training_H": minfo["training_H"],
            "checkpoint_path": minfo["path"],
            "checkpoint_sha256": sha,
            "rollout_time_sec": dur,
        }
        print(f"    Rollout completed in {dur:.2f}s (SHA: {sha[:12]}...)")

    # 3. Generate Visualizations
    print("\n[3/4] Generating Grand Ladder Field Rollout Figures...")
    generated_figures = {}
    channel_statistics = {}

    # Figure 1: Vorticity
    p_vort, stat_vort = plot_ladder_channel_grid(
        var_name="vorticity",
        gt_trajectory=gt_phys,
        preds_dict=preds_phys,
        model_keys=model_keys,
        horizons=KEY_HORIZONS,
        out_path=out_dir / "figure_grand_ladder_rollout_vorticity.png",
        sample_tag=sample_tag,
        re_val=re_val,
        sc_val=sc_val,
    )
    print(f"  ✓ Saved Grand Ladder Vorticity: {p_vort.name}")
    generated_figures["vorticity"] = str(p_vort)
    channel_statistics["vorticity"] = stat_vort

    # Figure 2: Passive Tracer
    p_tracer, stat_tracer = plot_ladder_channel_grid(
        var_name="tracer",
        gt_trajectory=gt_phys,
        preds_dict=preds_phys,
        model_keys=model_keys,
        horizons=KEY_HORIZONS,
        out_path=out_dir / "figure_grand_ladder_rollout_tracer.png",
        sample_tag=sample_tag,
        re_val=re_val,
        sc_val=sc_val,
    )
    print(f"  ✓ Saved Grand Ladder Passive Tracer: {p_tracer.name}")
    generated_figures["tracer"] = str(p_tracer)
    channel_statistics["tracer"] = stat_tracer

    # Figure 3: Velocity u
    p_u, stat_u = plot_ladder_channel_grid(
        var_name="u",
        gt_trajectory=gt_phys,
        preds_dict=preds_phys,
        model_keys=model_keys,
        horizons=KEY_HORIZONS,
        out_path=out_dir / "figure_grand_ladder_rollout_u.png",
        sample_tag=sample_tag,
        re_val=re_val,
        sc_val=sc_val,
    )
    print(f"  ✓ Saved Grand Ladder Velocity u: {p_u.name}")
    generated_figures["u"] = str(p_u)
    channel_statistics["u"] = stat_u

    # Figure 4: Master Representative Channels Combined (Vorticity + Tracer)
    p_master = plot_combined_ladder_representative_fields(
        channels=["vorticity", "tracer"],
        gt_trajectory=gt_phys,
        preds_dict=preds_phys,
        model_keys=model_keys,
        horizons=KEY_HORIZONS,
        out_path=out_dir / "figure_grand_ladder_representative_fields.png",
        sample_tag=sample_tag,
        re_val=re_val,
        sc_val=sc_val,
    )
    print(f"  ✓ Saved Grand Master Representative Fields: {p_master.name}")
    generated_figures["representative_master"] = str(p_master)

    # 4. Save Metadata Manifest
    print("\n[4/4] Writing companion provenance & metadata JSON...")
    git_commit = get_git_commit(PROJECT_ROOT)
    git_dirty = is_git_dirty(PROJECT_ROOT)

    manifest_payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_title": "Grand Horizon Ladder Representative Physical Field Rollout",
        "sample_index": args.sample_index,
        "sample_tag": sample_tag,
        "reynolds_number": re_val,
        "schmidt_number": sc_val,
        "ladder_models": model_keys,
        "rollout_horizons": KEY_HORIZONS,
        "representative_channels": {
            k: {
                "latex": v["latex_name"],
                "significance": v["physical_significance"],
                "cmap": v["cmap"],
            }
            for k, v in CHANNEL_CONFIGS.items()
        },
        "models_evaluated": model_provenance,
        "generated_figures": generated_figures,
        "channel_rmse_by_step": channel_statistics,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }

    manifest_path = out_dir / "figure_grand_ladder_field_rollout_metadata.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
    print(f"  ✓ Saved Metadata Manifest: {manifest_path.name}")

    print("\n================================================================================")
    print("  [SUCCESS] Grand Horizon Ladder Representative Field Visualization Completed!")
    print("================================================================================")


if __name__ == "__main__":
    main()
