"""Publication-Grade Qualitative Visualization Pipeline for Flow Field World Models.

Generates Input / Prediction / Ground Truth / Absolute Error figures:
1. Single-model single-sample four-panel plots (Input, Pred, GT, Error);
2. Single-model multi-horizon rollout degradation plots (h in {1, 10, 30});
3. Multi-model comparative invariant preservation plots (e.g., E1 vs E4 vs GT).

Enforces strict fail-closed provenance validation, data contract checks,
and outputs companion metadata JSON alongside every generated figure.
"""

import argparse
import datetime
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.pipeline import create_flow_dataloaders
from src.data.normalization import FieldNormalizer
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.utils.fft_derivatives import compute_vorticity
from src.utils.physics_contract import (
    PHYSICS_PROTOCOL,
    SPATIAL_AXIS_CONTRACT,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
    validate_ablation_checkpoint_semantics,
)
from src.utils.provenance import (
    compute_file_sha256,
    compute_normalizer_hash,
    compute_split_hash_from_file,
    get_git_commit,
    is_git_dirty,
    resolve_checkpoint_provenance,
    validate_evaluation_provenance,
)
from src.utils.reproducibility import seed_everything


CHANNEL_NAMES = {
    "u": 0,
    "v": 1,
    "p": 2,
    "tracer": 3,
    "s": 3,
}

VARIABLE_DISPLAY_TITLES = {
    "u": "Streamwise Velocity $u$",
    "v": "Cross-Stream Velocity $v$",
    "p": "Gauge Pressure $p$",
    "tracer": "Passive Tracer $s$",
    "s": "Passive Tracer $s$",
    "vorticity": r"Vorticity $\omega = \partial_x v - \partial_y u$",
}


def resolve_group_checkpoint_path(grp: str, seed: int) -> str:
    """Resolve checkpoint path for a group and seed with strict fail-closed check."""
    if grp == "E0_single_step":
        candidate_paths = [
            "outputs/checkpoints/dynamics/closure_r4/ablation_E0_single_step/latent_transformer/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/closure_r4/ablation_E0_single_step/best_vrmse_mean.pt",
        ]
    else:
        candidate_paths = [
            f"outputs/checkpoints/dynamics/closure_r4/seed_{seed}/ablation_{grp}/latent_transformer/best_vrmse_mean.pt",
            f"outputs/checkpoints/dynamics/closure_r4/seed_{seed}/ablation_{grp}/best_vrmse_mean.pt",
            f"outputs/checkpoints/dynamics/closure_r4/seed_{seed}/{grp}/latent_transformer/best_vrmse_mean.pt",
            f"outputs/checkpoints/dynamics/closure_r4/seed_{seed}/{grp}/best_vrmse_mean.pt",
        ]
        if seed == 42:
            candidate_paths.extend([
                f"outputs/checkpoints/dynamics/closure_r4/ablation_{grp}/latent_transformer/best_vrmse_mean.pt",
                f"outputs/checkpoints/dynamics/closure_r4/ablation_{grp}/best_vrmse_mean.pt",
            ])

    for p in candidate_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No valid checkpoint found for group '{grp}' under seed {seed}. Looked in: {candidate_paths}"
    )


def load_and_validate_forecaster(
    grp: str,
    ckpt_path: str,
    seed: int,
    eval_split_hash: str,
    eval_normalizer_hash: str,
    manifest_path: Optional[str],
    device: torch.device,
) -> Tuple[LatentForecaster, Dict[str, Any]]:
    """Load checkpoint, execute fail-closed semantic and provenance checks, and return model."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt_data.get("config", {})

    # 1. Semantics validation: verify physics protocol and architecture contracts
    validate_ablation_checkpoint_semantics(grp, cfg, is_legacy=False)

    # 2. Checkpoint provenance resolution
    ckpt_prov = resolve_checkpoint_provenance(
        ckpt_path=ckpt_path,
        ckpt_data=ckpt_data,
        manifest_path=manifest_path,
    )

    # 3. Fail-closed evaluation provenance match
    expected_seed = 42 if grp == "E0_single_step" else seed
    validate_evaluation_provenance(
        ckpt_provenance=ckpt_prov,
        eval_split_hash=eval_split_hash,
        eval_normalizer_hash=eval_normalizer_hash,
        expected_seed=expected_seed,
        fail_closed=True,
    )

    ckpt_sha256 = compute_file_sha256(ckpt_path)
    provenance_info = {
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": ckpt_sha256,
        "seed": expected_seed,
        "training_git_commit": ckpt_prov.get("training_git_commit"),
        "training_git_dirty": ckpt_prov.get("training_git_dirty", False),
        "split_hash": ckpt_prov.get("split_hash"),
        "normalizer_hash": ckpt_prov.get("normalizer_hash"),
        "protocol": PHYSICS_PROTOCOL,
    }

    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=cfg.get("embed_dim", 256),
        cond_dim=128,
        depth=cfg.get("depth", 6),
        num_heads=cfg.get("num_heads", 8),
        history_length=4,
        prediction_mode=cfg.get("prediction_mode", "direct"),
    )
    forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder).to(device)

    if "model_state_dict" in ckpt_data:
        forecaster.load_state_dict(ckpt_data["model_state_dict"])
    elif "encoder_state_dict" in ckpt_data:
        forecaster.encoder.load_state_dict(ckpt_data["encoder_state_dict"])
        forecaster.transformer.load_state_dict(ckpt_data["transformer_state_dict"])
        forecaster.decoder.load_state_dict(ckpt_data["decoder_state_dict"])

    forecaster.eval()
    return forecaster, provenance_info


def extract_scalar_field(
    phys_tensor: torch.Tensor,
    var_name: str,
    domain_size: Tuple[float, float] = SHEAR_FLOW_DOMAIN_SIZE_XY,
) -> np.ndarray:
    """Extract 2D numpy slice for variable from tensor of shape (..., 4, Nx, Ny).
    
    Returns array of shape (Nx, Ny).
    """
    var_lower = var_name.lower()
    if var_lower in CHANNEL_NAMES:
        ch = CHANNEL_NAMES[var_lower]
        field = phys_tensor[..., ch, :, :].detach().cpu().numpy()
        while field.ndim > 2:
            field = field[0]
        return field
    elif var_lower == "vorticity":
        u = phys_tensor[..., 0, :, :]
        v = phys_tensor[..., 1, :, :]
        # Ensure at least 3D for compute_vorticity: (..., Nx, Ny)
        if u.ndim == 2:
            u_in = u.unsqueeze(0)
            v_in = v.unsqueeze(0)
            omega = compute_vorticity(u_in, v_in, domain_size=domain_size)[0]
        else:
            omega = compute_vorticity(u, v, domain_size=domain_size)
            while omega.ndim > 2:
                omega = omega[0]
        return omega.detach().cpu().numpy()
    else:
        raise ValueError(f"Unknown variable name: '{var_name}'. Supported: {list(CHANNEL_NAMES.keys()) + ['vorticity']}")


def get_colormap_and_norm(
    var_name: str,
    *fields: np.ndarray,
) -> Tuple[str, float, float]:
    """Determine colormap, vmin, and vmax for physical variable."""
    var_lower = var_name.lower()
    all_vals = np.concatenate([f.flatten() for f in fields])
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))

    if var_lower in ["vorticity", "v"]:
        cmap = "RdBu_r"
        abs_max = max(abs(vmin), abs(vmax))
        vmin = -abs_max
        vmax = abs_max
    elif var_lower == "u":
        cmap = "coolwarm"
    elif var_lower == "p":
        cmap = "viridis"
    elif var_lower in ["tracer", "s"]:
        cmap = "magma"
    else:
        cmap = "viridis"

    return cmap, vmin, vmax


def run_model_rollout_for_sample(
    model: LatentForecaster,
    sample: Dict[str, Any],
    normalizer: Optional[FieldNormalizer],
    max_horizon: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute autoregressive rollout for single sample and return denormalized physical states.
    
    Returns:
        hist_phys: (1, L, 4, Nx, Ny)
        pred_phys: (1, H, 4, Nx, Ny)
        future_phys: (1, H, 4, Nx, Ny)
    """
    model.eval()
    with torch.no_grad():
        q_hist = sample["history"].unsqueeze(0).to(device)  # (1, L, 4, Nx, Ny)
        q_future = sample["future"].unsqueeze(0).to(device)  # (1, H, 4, Nx, Ny)
        re = sample["re"].unsqueeze(0).to(device) if "re" in sample else None
        sc = sample["sc"].unsqueeze(0).to(device) if "sc" in sample else None

        pred_traj = model.forward_rollout(q_hist, re, sc, horizon=max_horizon)

        if normalizer is not None:
            pred_phys = normalizer.denormalize(pred_traj)
            future_phys = normalizer.denormalize(q_future)
            hist_phys = normalizer.denormalize(q_hist)
        else:
            pred_phys = pred_traj
            future_phys = q_future
            hist_phys = q_hist

        # Gauge freedom: zero-mean pressure projection
        pred_phys[:, :, 2:3] = pred_phys[:, :, 2:3] - pred_phys[:, :, 2:3].mean(dim=(-2, -1), keepdim=True)
        future_phys[:, :, 2:3] = future_phys[:, :, 2:3] - future_phys[:, :, 2:3].mean(dim=(-2, -1), keepdim=True)
        hist_phys[:, :, 2:3] = hist_phys[:, :, 2:3] - hist_phys[:, :, 2:3].mean(dim=(-2, -1), keepdim=True)

    return hist_phys, pred_phys, future_phys


def generate_panel_figure(
    input_field: np.ndarray,
    pred_field: np.ndarray,
    gt_field: np.ndarray,
    var_name: str,
    group: str,
    seed: int,
    horizon: int,
    sample_index: int,
    out_path: str,
    meta_info: Dict[str, Any],
) -> str:
    """Generate and save Figure A: Four-panel plot (Input, Prediction, Ground Truth, Absolute Error)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    abs_err = np.abs(pred_field - gt_field)
    mae = float(np.mean(abs_err))
    max_err = float(np.max(abs_err))

    cmap, vmin, vmax = get_colormap_and_norm(var_name, input_field, pred_field, gt_field)
    err_cmap = "inferno"

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=200)

    # 1. Input (last history frame t=0)
    im0 = axes[0].imshow(input_field.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title(f"Input ($t=0$)\n[{input_field.min():.2f}, {input_field.max():.2f}]", fontsize=12)
    axes[0].set_xlabel("x (Streamwise)")
    axes[0].set_ylabel("y (Cross-Stream)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Prediction (t=h)
    im1 = axes[1].imshow(pred_field.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title(f"Prediction ($t={horizon}$)\n{group}", fontsize=12)
    axes[1].set_xlabel("x (Streamwise)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Ground Truth (t=h)
    im2 = axes[2].imshow(gt_field.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axes[2].set_title(f"Ground Truth ($t={horizon}$)\nTarget DNS State", fontsize=12)
    axes[2].set_xlabel("x (Streamwise)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Absolute Error
    im3 = axes[3].imshow(abs_err.T, origin="lower", cmap=err_cmap, vmin=0, vmax=max_err, aspect="auto")
    axes[3].set_title(f"Absolute Error ($|\\mathrm{{Pred}} - \\mathrm{{GT}}|$) \nMAE: {mae:.4f} | Max: {max_err:.4f}", fontsize=12)
    axes[3].set_xlabel("x (Streamwise)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    var_title = VARIABLE_DISPLAY_TITLES.get(var_name, var_name)
    fig.suptitle(
        f"Qualitative Rollout Field Verification — {var_title} | Seed {seed} | Sample #{sample_index} | Horizon $h={horizon}$",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # Write companion metadata JSON
    meta_json_path = out_path.replace(".png", "_metadata.json")
    meta_record = {
        "figure_type": "panel",
        "figure_path": out_path,
        "seed": seed,
        "group": group,
        "sample_index": sample_index,
        "horizon": horizon,
        "variable": var_name,
        "error_type": "absolute_error",
        "mae": mae,
        "max_err": max_err,
        "timestamp": datetime.datetime.now().isoformat(),
        **meta_info,
    }
    with open(meta_json_path, "w") as f:
        json.dump(meta_record, f, indent=2)

    return out_path


def generate_multihorizon_figure(
    pred_fields_by_h: Dict[int, np.ndarray],
    gt_fields_by_h: Dict[int, np.ndarray],
    var_name: str,
    group: str,
    seed: int,
    horizons: List[int],
    sample_index: int,
    out_path: str,
    meta_info: Dict[str, Any],
) -> str:
    """Generate and save Figure B: Multi-horizon comparison (Rows: Pred, GT, Error; Cols: Horizons)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_phys = list(pred_fields_by_h.values()) + list(gt_fields_by_h.values())
    cmap, vmin, vmax = get_colormap_and_norm(var_name, *all_phys)
    err_cmap = "inferno"

    n_cols = len(horizons)
    fig, axes = plt.subplots(3, n_cols, figsize=(5.5 * n_cols, 11), dpi=200)
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    maes_by_h = {}
    for col_idx, h in enumerate(horizons):
        pred_h = pred_fields_by_h[h]
        gt_h = gt_fields_by_h[h]
        abs_err = np.abs(pred_h - gt_h)
        maes_by_h[h] = float(np.mean(abs_err))

        # Row 0: Prediction
        im0 = axes[0, col_idx].imshow(pred_h.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axes[0, col_idx].set_title(f"Prediction ($t={h}$)", fontsize=13)
        if col_idx == 0:
            axes[0, col_idx].set_ylabel(f"Prediction\n({group})", fontsize=12)
        plt.colorbar(im0, ax=axes[0, col_idx], fraction=0.046, pad=0.04)

        # Row 1: Ground Truth
        im1 = axes[1, col_idx].imshow(gt_h.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axes[1, col_idx].set_title(f"Ground Truth ($t={h}$)", fontsize=13)
        if col_idx == 0:
            axes[1, col_idx].set_ylabel("Ground Truth\n(Target DNS)", fontsize=12)
        plt.colorbar(im1, ax=axes[1, col_idx], fraction=0.046, pad=0.04)

        # Row 2: Error
        im2 = axes[2, col_idx].imshow(abs_err.T, origin="lower", cmap=err_cmap, vmin=0, aspect="auto")
        axes[2, col_idx].set_title(f"Error ($t={h}$)\nMAE: {maes_by_h[h]:.4f}", fontsize=13)
        axes[2, col_idx].set_xlabel("x (Streamwise)", fontsize=11)
        if col_idx == 0:
            axes[2, col_idx].set_ylabel("Absolute Error\n($|\\mathrm{Pred} - \\mathrm{GT}|$)", fontsize=12)
        plt.colorbar(im2, ax=axes[2, col_idx], fraction=0.046, pad=0.04)

    var_title = VARIABLE_DISPLAY_TITLES.get(var_name, var_name)
    fig.suptitle(
        f"Multi-Horizon Autoregressive Rollout Degradation — {var_title} | Model: {group} (Seed {seed}) | Sample #{sample_index}",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # Write companion metadata JSON
    meta_json_path = out_path.replace(".png", "_metadata.json")
    meta_record = {
        "figure_type": "multihorizon",
        "figure_path": out_path,
        "seed": seed,
        "group": group,
        "sample_index": sample_index,
        "horizons": horizons,
        "variable": var_name,
        "error_type": "absolute_error",
        "maes_by_horizon": maes_by_h,
        "timestamp": datetime.datetime.now().isoformat(),
        **meta_info,
    }
    with open(meta_json_path, "w") as f:
        json.dump(meta_record, f, indent=2)

    return out_path


def generate_compare_figure(
    pred_fields_by_grp: Dict[str, np.ndarray],
    gt_field: np.ndarray,
    var_name: str,
    groups: List[str],
    seed: int,
    horizon: int,
    sample_index: int,
    out_path: str,
    meta_info: Dict[str, Any],
) -> str:
    """Generate and save Figure C: Multi-model comparison at fixed horizon (GT + per-model Pred & Error)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_phys = [gt_field] + list(pred_fields_by_grp.values())
    cmap, vmin, vmax = get_colormap_and_norm(var_name, *all_phys)
    err_cmap = "inferno"

    # Layout: Row 0: Ground Truth (large on left) or 1 row per model
    # Elegant, clean layout: N_models rows, 3 columns: [Ground Truth, Model Pred, Model Error]
    n_models = len(groups)
    fig, axes = plt.subplots(n_models, 3, figsize=(16, 4.5 * n_models), dpi=200)
    if n_models == 1:
        axes = axes[np.newaxis, :]

    maes_by_grp = {}
    for row_idx, grp in enumerate(groups):
        pred = pred_fields_by_grp[grp]
        abs_err = np.abs(pred - gt_field)
        mae = float(np.mean(abs_err))
        maes_by_grp[grp] = mae

        # Col 0: Ground Truth
        im0 = axes[row_idx, 0].imshow(gt_field.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axes[row_idx, 0].set_title(f"Ground Truth ($t={horizon}$)", fontsize=12)
        axes[row_idx, 0].set_ylabel(f"{grp}", fontsize=12, fontweight="bold")
        plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        # Col 1: Model Prediction
        im1 = axes[row_idx, 1].imshow(pred.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axes[row_idx, 1].set_title(f"Prediction ($t={horizon}$)\n{grp}", fontsize=12)
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # Col 2: Model Absolute Error
        im2 = axes[row_idx, 2].imshow(abs_err.T, origin="lower", cmap=err_cmap, vmin=0, aspect="auto")
        axes[row_idx, 2].set_title(f"Absolute Error\nMAE: {mae:.4f} | Max: {abs_err.max():.4f}", fontsize=12)
        plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        if row_idx == n_models - 1:
            for c in range(3):
                axes[row_idx, c].set_xlabel("x (Streamwise)", fontsize=11)

    var_title = VARIABLE_DISPLAY_TITLES.get(var_name, var_name)
    fig.suptitle(
        f"Comparative Model Invariant & Rollout Verification — {var_title} | Horizon $t={horizon}$ | Seed {seed} | Sample #{sample_index}",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # Write companion metadata JSON
    meta_json_path = out_path.replace(".png", "_metadata.json")
    meta_record = {
        "figure_type": "compare",
        "figure_path": out_path,
        "seed": seed,
        "compare_groups": groups,
        "sample_index": sample_index,
        "horizon": horizon,
        "variable": var_name,
        "error_type": "absolute_error",
        "maes_by_group": maes_by_grp,
        "timestamp": datetime.datetime.now().isoformat(),
        **meta_info,
    }
    with open(meta_json_path, "w") as f:
        json.dump(meta_record, f, indent=2)

    return out_path


def select_sample_by_ranking(
    model: LatentForecaster,
    dataset,
    normalizer: Optional[FieldNormalizer],
    sample_mode: str,
    ranking_metric: str,
    eval_horizon: int,
    device: torch.device,
) -> int:
    """Compute ranking metric across all samples in dataset and return selected sample index."""
    if sample_mode == "index":
        return 0

    print(f"Ranking {len(dataset)} test samples by '{ranking_metric}' at horizon h={eval_horizon}...")
    errors = []

    for i in range(len(dataset)):
        sample = dataset[i]
        _, pred_phys, future_phys = run_model_rollout_for_sample(
            model=model,
            sample=sample,
            normalizer=normalizer,
            max_horizon=eval_horizon,
            device=device,
        )

        h_idx = eval_horizon - 1
        pred_h = pred_phys[:, h_idx]  # (1, 4, Nx, Ny)
        targ_h = future_phys[:, h_idx]

        if ranking_metric == "u_rmse":
            err = torch.sqrt(torch.mean((pred_h[:, 0] - targ_h[:, 0]) ** 2)).item()
        elif ranking_metric == "vort_rmse":
            w_pred = compute_vorticity(pred_h[:, 0], pred_h[:, 1], domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)
            w_targ = compute_vorticity(targ_h[:, 0], targ_h[:, 1], domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)
            err = torch.sqrt(torch.mean((w_pred - w_targ) ** 2)).item()
        else:  # vrmse_mean default
            vrmses = []
            for ch in range(4):
                denom = torch.var(targ_h[:, ch])
                if denom > 1e-8:
                    vrmses.append(torch.sqrt(torch.mean((pred_h[:, ch] - targ_h[:, ch]) ** 2) / denom).item())
                else:
                    vrmses.append(torch.sqrt(torch.mean((pred_h[:, ch] - targ_h[:, ch]) ** 2)).item())
            err = float(np.mean(vrmses))

        errors.append((i, err))

    errors.sort(key=lambda x: x[1])

    if sample_mode == "best":
        chosen = errors[0][0]
        print(f"Selected 'best' sample index {chosen} with error {errors[0][1]:.4f}")
    elif sample_mode == "worst":
        chosen = errors[-1][0]
        print(f"Selected 'worst' sample index {chosen} with error {errors[-1][1]:.4f}")
    elif sample_mode == "median":
        mid_idx = len(errors) // 2
        chosen = errors[mid_idx][0]
        print(f"Selected 'median' sample index {chosen} with error {errors[mid_idx][1]:.4f}")
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    return chosen


def generate_qualitative_suite(
    seed: int = 42,
    group: str = "E4_full_physics",
    compare_groups: Optional[List[str]] = None,
    sample_index: int = 0,
    sample_mode: str = "index",
    ranking_metric: str = "vrmse_mean",
    horizons: List[int] = [1, 10, 30],
    variable: str = "u",
    data_dir: str = "/root/autodl-tmp/datasets/shear_flow",
    split_file: str = "outputs/splits/grouped_split.json",
    output_dir: str = "outputs/figures/qualitative",
    manifest_path: str = "outputs/manifests/closure_r4_seed42.json",
    formal: bool = False,
    allow_dirty: bool = False,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    plot_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute complete qualitative visualization workflow with provenance guarantees."""
    seed_everything(seed)
    device = torch.device(device_str)

    # 1. Clean working tree check in formal mode
    git_commit = get_git_commit(PROJECT_ROOT)
    git_dirty = is_git_dirty(PROJECT_ROOT)
    if formal and git_dirty and not allow_dirty:
        raise RuntimeError(
            "Working tree is dirty. Formal qualitative visualization requires a clean git working tree. "
            "Please commit all changes before running or specify --allow_dirty."
        )

    if compare_groups is None:
        compare_groups = ["E1_rollout_field", "E4_full_physics"]

    if plot_types is None:
        plot_types = ["panel", "multihorizon", "compare"]
    elif "all" in plot_types:
        plot_types = ["panel", "multihorizon", "compare"]

    max_h = max(horizons)

    # 2. Data loader & split provenance
    print(f"Loading test split from {split_file} (max horizon: {max_h})...")
    _, _, test_loader, normalizer = create_flow_dataloaders(
        split_type="grouped",
        split_file=split_file,
        data_root=data_dir,
        history_length=4,
        horizon=max_h,
        stride=20,
        downsample_factor=2,
        batch_size=1,
        num_workers=0,
        normalize=True,
    )
    test_dataset = test_loader.dataset
    eval_split_hash = compute_split_hash_from_file(split_file) if os.path.exists(split_file) else "UNKNOWN"
    eval_normalizer_hash = compute_normalizer_hash(normalizer)

    print(f"Loaded {len(test_dataset)} test samples. split_hash={eval_split_hash[:12]}, normalizer_hash={eval_normalizer_hash[:12]}")

    # 3. Load primary model
    primary_ckpt_path = resolve_group_checkpoint_path(group, seed)
    primary_model, primary_prov = load_and_validate_forecaster(
        grp=group,
        ckpt_path=primary_ckpt_path,
        seed=seed,
        eval_split_hash=eval_split_hash,
        eval_normalizer_hash=eval_normalizer_hash,
        manifest_path=manifest_path,
        device=device,
    )

    # 4. Resolve sample index
    if sample_mode != "index":
        chosen_sample_idx = select_sample_by_ranking(
            model=primary_model,
            dataset=test_dataset,
            normalizer=normalizer,
            sample_mode=sample_mode,
            ranking_metric=ranking_metric,
            eval_horizon=max(horizons),
            device=device,
        )
    else:
        if not (0 <= sample_index < len(test_dataset)):
            raise IndexError(f"sample_index {sample_index} is out of bounds for test dataset of size {len(test_dataset)}")
        chosen_sample_idx = sample_index

    # Extract sample provenance info
    sample_raw = test_dataset[chosen_sample_idx]
    f_idx, sim_idx, start_t, split_t, end_t, re_val, sc_val = test_dataset.samples[chosen_sample_idx]
    source_file = test_dataset.file_paths[f_idx]

    shared_meta = {
        "split_name": "grouped",
        "split_hash": eval_split_hash,
        "normalizer_hash": eval_normalizer_hash,
        "source_file": source_file,
        "sim_idx": int(sim_idx),
        "start_t": int(start_t),
        "split_t": int(split_t),
        "end_t": int(end_t),
        "re": float(re_val),
        "sc": float(sc_val),
        "evaluation_git_commit": git_commit,
        "evaluation_git_dirty": git_dirty,
        "domain_size": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "vorticity_operator": "compute_vorticity (spectral_grad_2d)",
    }

    # 5. Execute rollout for primary model
    print(f"Rolling out primary model [{group}] for sample #{chosen_sample_idx} across h={max_h}...")
    hist_phys, pred_phys, future_phys = run_model_rollout_for_sample(
        model=primary_model,
        sample=sample_raw,
        normalizer=normalizer,
        max_horizon=max_h,
        device=device,
    )

    # Extract scalar fields across requested horizons
    input_field = extract_scalar_field(hist_phys[:, -1], variable)  # t=0
    pred_fields_by_h = {}
    gt_fields_by_h = {}

    for h in horizons:
        h_idx = h - 1
        pred_fields_by_h[h] = extract_scalar_field(pred_phys[:, h_idx], variable)
        gt_fields_by_h[h] = extract_scalar_field(future_phys[:, h_idx], variable)

    generated_figures = {}

    # Output A: Panel figure (default at h=10 or last horizon)
    if "panel" in plot_types:
        panel_h = 10 if 10 in horizons else horizons[-1]
        panel_path = os.path.join(
            output_dir,
            f"qual_case_seed{seed}_{group}_h{panel_h}_{variable}_panel.png",
        )
        panel_meta = {
            **shared_meta,
            "checkpoint_path": primary_prov["checkpoint_path"],
            "checkpoint_sha256": primary_prov["checkpoint_sha256"],
            "training_git_commit": primary_prov["training_git_commit"],
            "training_git_dirty": primary_prov["training_git_dirty"],
        }
        fig_a = generate_panel_figure(
            input_field=input_field,
            pred_field=pred_fields_by_h[panel_h],
            gt_field=gt_fields_by_h[panel_h],
            var_name=variable,
            group=group,
            seed=seed,
            horizon=panel_h,
            sample_index=chosen_sample_idx,
            out_path=panel_path,
            meta_info=panel_meta,
        )
        generated_figures["panel"] = fig_a
        print(f"Generated Panel Figure: {fig_a}")

    # Output B: Multi-horizon figure
    if "multihorizon" in plot_types:
        mh_path = os.path.join(
            output_dir,
            f"qual_case_seed{seed}_{group}_{variable}_multihorizon.png",
        )
        mh_meta = {
            **shared_meta,
            "checkpoint_path": primary_prov["checkpoint_path"],
            "checkpoint_sha256": primary_prov["checkpoint_sha256"],
            "training_git_commit": primary_prov["training_git_commit"],
            "training_git_dirty": primary_prov["training_git_dirty"],
        }
        fig_b = generate_multihorizon_figure(
            pred_fields_by_h=pred_fields_by_h,
            gt_fields_by_h=gt_fields_by_h,
            var_name=variable,
            group=group,
            seed=seed,
            horizons=horizons,
            sample_index=chosen_sample_idx,
            out_path=mh_path,
            meta_info=mh_meta,
        )
        generated_figures["multihorizon"] = fig_b
        print(f"Generated Multi-Horizon Figure: {fig_b}")

    # Output C: Comparative figure across models
    if "compare" in plot_types:
        compare_h = 30 if 30 in horizons else horizons[-1]
        compare_path = os.path.join(
            output_dir,
            f"qual_case_seed{seed}_compare_h{compare_h}_{variable}.png",
        )

        compare_preds = {}
        compare_ckpts = {}

        for c_grp in compare_groups:
            if c_grp == group:
                c_model = primary_model
                c_prov = primary_prov
            else:
                c_ckpt = resolve_group_checkpoint_path(c_grp, seed)
                c_model, c_prov = load_and_validate_forecaster(
                    grp=c_grp,
                    ckpt_path=c_ckpt,
                    seed=seed,
                    eval_split_hash=eval_split_hash,
                    eval_normalizer_hash=eval_normalizer_hash,
                    manifest_path=manifest_path,
                    device=device,
                )

            _, c_pred_phys, _ = run_model_rollout_for_sample(
                model=c_model,
                sample=sample_raw,
                normalizer=normalizer,
                max_horizon=compare_h,
                device=device,
            )
            compare_preds[c_grp] = extract_scalar_field(c_pred_phys[:, compare_h - 1], variable)
            compare_ckpts[c_grp] = {
                "checkpoint_path": c_prov["checkpoint_path"],
                "checkpoint_sha256": c_prov["checkpoint_sha256"],
                "training_git_commit": c_prov["training_git_commit"],
                "training_git_dirty": c_prov["training_git_dirty"],
            }

        compare_meta = {
            **shared_meta,
            "models_provenance": compare_ckpts,
        }
        gt_target = gt_fields_by_h[compare_h]

        fig_c = generate_compare_figure(
            pred_fields_by_grp=compare_preds,
            gt_field=gt_target,
            var_name=variable,
            groups=compare_groups,
            seed=seed,
            horizon=compare_h,
            sample_index=chosen_sample_idx,
            out_path=compare_path,
            meta_info=compare_meta,
        )
        generated_figures["compare"] = fig_c
        print(f"Generated Comparative Figure: {fig_c}")

    return {
        "status": "success",
        "sample_index": chosen_sample_idx,
        "seed": seed,
        "variable": variable,
        "figures": generated_figures,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade qualitative flow field figures.")
    parser.add_argument("--seed", type=int, default=42, help="Seed index (default: 42).")
    parser.add_argument("--group", type=str, default="E4_full_physics", help="Primary model group.")
    parser.add_argument(
        "--compare_groups",
        nargs="+",
        default=["E1_rollout_field", "E4_full_physics"],
        help="Groups for comparative figure.",
    )
    parser.add_argument("--sample_index", type=int, default=0, help="Test sample index when sample_mode=index.")
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="index",
        choices=["index", "best", "median", "worst"],
        help="Sample selection mode.",
    )
    parser.add_argument(
        "--ranking_metric",
        type=str,
        default="vrmse_mean",
        choices=["vrmse_mean", "u_rmse", "vort_rmse"],
        help="Ranking metric for best/median/worst sample selection.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 10, 30],
        help="Horizons to evaluate.",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="u",
        choices=["u", "v", "p", "tracer", "vorticity"],
        help="Physical flow variable.",
    )
    parser.add_argument(
        "--plot_types",
        nargs="+",
        default=["all"],
        choices=["panel", "multihorizon", "compare", "all"],
        help="Plot types to generate.",
    )
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--split_file", type=str, default="outputs/splits/grouped_split.json")
    parser.add_argument("--output_dir", type=str, default="outputs/figures/qualitative")
    parser.add_argument("--manifest_path", type=str, default="outputs/manifests/closure_r4_seed42.json")
    parser.add_argument("--formal", action="store_true", help="Enforce clean git tree for formal evaluation.")
    parser.add_argument("--allow_dirty", action="store_true", help="Allow dirty working tree for testing.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    res = generate_qualitative_suite(
        seed=args.seed,
        group=args.group,
        compare_groups=args.compare_groups,
        sample_index=args.sample_index,
        sample_mode=args.sample_mode,
        ranking_metric=args.ranking_metric,
        horizons=args.horizons,
        variable=args.variable,
        data_dir=args.data_dir,
        split_file=args.split_file,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        formal=args.formal,
        allow_dirty=args.allow_dirty,
        device_str=args.device,
        plot_types=args.plot_types,
    )
    print(f"\nCompleted successfully: {len(res['figures'])} figures created.")


if __name__ == "__main__":
    main()
