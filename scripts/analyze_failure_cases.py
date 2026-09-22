"""Analyze failure and boundary cases across test trajectories under canonical pipeline.

Extracts best, median, and worst failure cases across 30-step autoregressive rollouts,
computing trajectory metrics in denormalized physical space, and exports publication figures
and diagnostic JSON metadata.
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.pipeline import create_flow_dataloaders
from src.metrics.field import evaluate_field_metrics
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.utils.fft_derivatives import compute_divergence, compute_vorticity
from src.utils.reproducibility import seed_everything
from src.utils.physics_contract import PHYSICS_PROTOCOL, SPATIAL_AXIS_CONTRACT, SHEAR_FLOW_DOMAIN_SIZE_XY


def analyze_failure_cases(
    model_path: str = "outputs/checkpoints/dynamics/closure_r4/ablation_E4_full_physics/best_vrmse_mean.pt",
    data_dir: str = "/root/autodl-tmp/datasets/shear_flow",
    split_type: str = "grouped",
    split_file: str = None,
    output_dir: str = "outputs",
    max_horizon: int = 30,
    stride: int = 20,
    downsample_factor: int = 2,
    normalize: bool = True,
    allow_legacy_checkpoint: bool = False,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    seed_everything(42)
    device = torch.device(device_str)
    fig_dir = os.path.join(output_dir, "figures")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Resolve candidate model path with strict Closure-R4 fail-closed protocol.
    # Pre-R4 E2/E3/E4 checkpoints are permanently invalid because their
    # divergence/vorticity losses used the swapped-axis physics operator.
    is_legacy = False
    checkpoint_training_protocol = PHYSICS_PROTOCOL

    canonical_default = "outputs/checkpoints/dynamics/closure_r4/ablation_E4_full_physics/best_vrmse_mean.pt"
    canonical_alt = "outputs/checkpoints/dynamics/closure_r4/ablation_E4_full_physics/latent_transformer/best_vrmse_mean.pt"
    if not os.path.exists(model_path):
        if model_path == canonical_default and os.path.exists(canonical_alt):
            model_path = canonical_alt

    invalid_axis_candidates = [
        "outputs/checkpoints/dynamics/ablation_E2_plus_L_div/best_vrmse_mean.pt",
        "outputs/checkpoints/dynamics/ablation_E2_plus_L_div/latent_transformer/best_vrmse_mean.pt",
        "outputs/checkpoints/dynamics/ablation_E3_plus_L_vort/best_vrmse_mean.pt",
        "outputs/checkpoints/dynamics/ablation_E3_plus_L_vort/latent_transformer/best_vrmse_mean.pt",
        "outputs/checkpoints/dynamics/ablation_E4_full_physics/best_vrmse_mean.pt",
        "outputs/checkpoints/dynamics/ablation_E4_full_physics/latent_transformer/best_vrmse_mean.pt",
        "outputs/checkpoints/dynamics/ablation_plus_L_div/best_vrmse_mean.pt",
        "outputs/checkpoints/dynamics/ablation_plus_L_div/latent_transformer/best_vrmse_mean.pt",
        "outputs/checkpoints/dynamics/ablation_plus_L_vort/best_vrmse_mean.pt",
        "outputs/checkpoints/dynamics/ablation_plus_L_vort/latent_transformer/best_vrmse_mean.pt",
        "outputs/checkpoints/dynamics/ablation_plus_L_div_vort/best_vrmse_mean.pt",
        "outputs/checkpoints/dynamics/ablation_plus_L_div_vort/latent_transformer/best_vrmse_mean.pt",
    ]

    if not os.path.exists(model_path):
        safe_legacy_candidates = [
            "outputs/checkpoints/dynamics/ablation_E1_rollout_field/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_E1_rollout_field/latent_transformer/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_E0_single_step/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_E0_single_step/latent_transformer/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/latent_transformer/best_vrmse_mean.pt",
        ]
        found_safe_legacy = next(
            (p for p in safe_legacy_candidates if os.path.exists(p)),
            None,
        )
        invalid_existing = [p for p in invalid_axis_candidates if os.path.exists(p)]

        if allow_legacy_checkpoint and found_safe_legacy:
            legacy_ckpt = torch.load(found_safe_legacy, map_location="cpu")
            legacy_cfg = legacy_ckpt.get("config", {})
            if legacy_cfg.get("lambda_div", 0.0) != 0.0 or legacy_cfg.get("lambda_vort", 0.0) != 0.0:
                raise RuntimeError(
                    f"Refusing legacy checkpoint '{found_safe_legacy}': non-zero physics-loss weights "
                    "make it invalid under Closure-R4."
                )
            model_path = found_safe_legacy
            is_legacy = True
            checkpoint_training_protocol = "pre-R4-field-only"
            print(
                f"Notice: Using pre-R4 field-only checkpoint {model_path}; "
                f"all diagnostics are recomputed under {PHYSICS_PROTOCOL}."
            )
        else:
            detail = ""
            if invalid_existing:
                detail += (
                    " Invalid pre-R4 physics-loss checkpoint(s) exist but are permanently blocked: "
                    f"{invalid_existing}."
                )
            if found_safe_legacy and not allow_legacy_checkpoint:
                detail += (
                    f" Safe field-only legacy checkpoint '{found_safe_legacy}' exists, but explicit "
                    "--allow_legacy_checkpoint is required."
                )
            raise FileNotFoundError(
                f"Checkpoint not found for Closure-R4 at '{model_path}'.{detail}"
            )

    print(
        f"Loading checkpoint config from {model_path} "
        f"[evaluation={PHYSICS_PROTOCOL}; checkpoint={checkpoint_training_protocol}]..."
    )
    ckpt = torch.load(model_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    # Self-describing config recovery
    embed_dim = cfg.get("embed_dim", 256)
    depth = cfg.get("depth", 6)
    num_heads = cfg.get("num_heads", 8)
    prediction_mode = cfg.get("prediction_mode", ckpt.get("prediction_mode", "direct"))
    use_condition = cfg.get("use_condition", ckpt.get("use_condition", True))
    ds_factor = cfg.get("downsample_factor", downsample_factor)
    norm_flag = cfg.get("normalize", normalize)
    s_type = cfg.get("split_type", split_type)

    print(f"Recovered model config: embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}, "
          f"prediction_mode='{prediction_mode}', use_condition={use_condition}, downsample={ds_factor}")

    if split_file is None:
        if s_type.endswith(".json"):
            split_file = s_type
        elif s_type.startswith("outputs/splits/"):
            split_file = s_type
        else:
            cand = f"outputs/splits/{s_type}.json"
            split_file = cand if os.path.exists(cand) else f"outputs/splits/{s_type}_split.json"

    # Canonical data pipeline
    _, _, test_loader, normalizer = create_flow_dataloaders(
        split_type=s_type,
        split_file=split_file,
        data_root=data_dir,
        history_length=4,
        horizon=max_horizon,
        stride=stride,
        downsample_factor=ds_factor,
        batch_size=1,
        num_workers=0,
        normalize=norm_flag,
    )

    print(f"Loaded {len(test_loader.dataset)} test trajectories for failure case analysis...")

    # Load Model with unprojected decoder
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=embed_dim,
        cond_dim=128,
        depth=depth,
        num_heads=num_heads,
        history_length=4,
        prediction_mode=prediction_mode,
    )
    model = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder).to(device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "encoder_state_dict" in ckpt and "transformer_state_dict" in ckpt:
        model.encoder.load_state_dict(ckpt["encoder_state_dict"])
        model.transformer.load_state_dict(ckpt["transformer_state_dict"])
        model.decoder.load_state_dict(ckpt["decoder_state_dict"])
    model.eval()

    trajectory_records = []

    with torch.no_grad():
        for traj_idx, batch in enumerate(test_loader):
            q_hist = batch["history"].to(device)
            q_future = batch["future"].to(device)
            re = batch["re"].to(device) if use_condition and "re" in batch else None
            sc = batch["sc"].to(device) if use_condition and "sc" in batch else None

            pred_traj = model.forward_rollout(q_hist, re, sc, horizon=max_horizon)

            # Denormalize to physical units for diagnostic analysis
            if normalizer is not None:
                pred_eval = normalizer.denormalize(pred_traj)
                gt_eval = normalizer.denormalize(q_future)
            else:
                pred_eval = pred_traj
                gt_eval = q_future

            # Enforce zero-mean pressure gauge in physical space
            pred_eval[:, :, 2:3, :, :] = pred_eval[:, :, 2:3, :, :] - pred_eval[:, :, 2:3, :, :].mean(dim=(-2, -1), keepdim=True)
            gt_eval[:, :, 2:3, :, :] = gt_eval[:, :, 2:3, :, :] - gt_eval[:, :, 2:3, :, :].mean(dim=(-2, -1), keepdim=True)

            # Evaluate trajectory level metrics
            step30_pred = pred_eval[:, -1]
            step30_gt = gt_eval[:, -1]
            metrics_step30 = evaluate_field_metrics(step30_pred, step30_gt)

            # Cumulative VRMSE across all 30 steps
            all_vrmse = []
            for t in range(max_horizon):
                step_m = evaluate_field_metrics(pred_eval[:, t], gt_eval[:, t])
                all_vrmse.append(step_m["vrmse_mean"])

            div_err = compute_divergence(step30_pred[0, 0], step30_pred[0, 1]).pow(2).mean().sqrt().item()
            vort_pred = compute_vorticity(step30_pred[0, 0], step30_pred[0, 1])
            vort_gt = compute_vorticity(step30_gt[0, 0], step30_gt[0, 1])
            vort_err = (vort_pred - vort_gt).pow(2).mean().sqrt().item()

            rec = {
                "traj_idx": traj_idx,
                "re": float(re[0].item()) if re is not None else 1e4,
                "sc": float(sc[0].item()) if sc is not None else 1.0,
                "mean_vrmse": float(np.mean(all_vrmse)),
                "step30_vrmse": float(metrics_step30["vrmse_mean"]),
                "step30_rmse": float(metrics_step30["rmse_mean"]),
                "step30_div_err": float(div_err),
                "step30_vort_rmse": float(vort_err),
                "all_vrmse": all_vrmse,
                "pred_step30": step30_pred[0].cpu().numpy(),
                "gt_step30": step30_gt[0].cpu().numpy(),
            }
            trajectory_records.append(rec)

    # Sort trajectories by cumulative 30-step VRMSE
    sorted_trajs = sorted(trajectory_records, key=lambda x: x["mean_vrmse"])
    best_case = sorted_trajs[0]
    median_case = sorted_trajs[len(sorted_trajs) // 2]
    worst_case = sorted_trajs[-1]

    print("\n" + "=" * 80)
    print("FAILURE CASE AUDIT RESULTS (30-STEP AUTOREGRESSIVE ROLLOUT)")
    print("=" * 80)
    print(f"BEST CASE   (# {best_case['traj_idx']:02d}): Re={best_case['re']:.0f}, Sc={best_case['sc']:.1f} | Mean VRMSE: {best_case['mean_vrmse']:.4f} | Step30 VRMSE: {best_case['step30_vrmse']:.4f} | DivErr: {best_case['step30_div_err']:.6f}")
    print(f"MEDIAN CASE (# {median_case['traj_idx']:02d}): Re={median_case['re']:.0f}, Sc={median_case['sc']:.1f} | Mean VRMSE: {median_case['mean_vrmse']:.4f} | Step30 VRMSE: {median_case['step30_vrmse']:.4f} | DivErr: {median_case['step30_div_err']:.6f}")
    print(f"WORST CASE  (# {worst_case['traj_idx']:02d}): Re={worst_case['re']:.0f}, Sc={worst_case['sc']:.1f} | Mean VRMSE: {worst_case['mean_vrmse']:.4f} | Step30 VRMSE: {worst_case['step30_vrmse']:.4f} | DivErr: {worst_case['step30_div_err']:.6f}")

    # Visualization
    cases_to_plot = [
        ("Best Case (Lowest VRMSE)", best_case),
        ("Median Typical Case", median_case),
        ("Worst Failure Case (Highest VRMSE)", worst_case),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(20, 11), dpi=150)

    for row_idx, (case_title, case_data) in enumerate(cases_to_plot):
        gt = case_data["gt_step30"]
        pred = case_data["pred_step30"]

        gt_t = torch.from_numpy(gt).unsqueeze(0)
        pred_t = torch.from_numpy(pred).unsqueeze(0)
        vort_gt = compute_vorticity(gt_t[:, 0], gt_t[:, 1])[0].numpy()
        vort_pred = compute_vorticity(pred_t[:, 0], pred_t[:, 1])[0].numpy()
        vort_diff = np.abs(vort_pred - vort_gt)

        # 1. Vorticity GT
        im0 = axes[row_idx, 0].imshow(vort_gt, cmap="RdBu_r")
        axes[row_idx, 0].set_title(f"{case_title}\nVorticity $\\omega$ Ground Truth (Step 30)", fontsize=10)
        axes[row_idx, 0].axis("off")
        plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        # 2. Vorticity Pred
        im1 = axes[row_idx, 1].imshow(vort_pred, cmap="RdBu_r")
        axes[row_idx, 1].set_title(f"{case_title}\nVorticity $\\omega$ Predicted (Step 30)", fontsize=10)
        axes[row_idx, 1].axis("off")
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # 3. Vorticity Absolute Error
        im2 = axes[row_idx, 2].imshow(vort_diff, cmap="inferno")
        axes[row_idx, 2].set_title(f"Vorticity Absolute Error\nRMSE: {case_data['step30_vort_rmse']:.4f}", fontsize=10)
        axes[row_idx, 2].axis("off")
        plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        # 4. Rollout Curve
        axes[row_idx, 3].plot(range(1, max_horizon + 1), case_data["all_vrmse"], color="#1f77b4", linewidth=2.0, marker=".")
        axes[row_idx, 3].set_title(f"Rollout Mean VRMSE Evolution\nRe={case_data['re']:.0f}, Sc={case_data['sc']:.1f}", fontsize=10)
        axes[row_idx, 3].set_xlabel("Horizon (Step)", fontsize=9)
        axes[row_idx, 3].set_ylabel("VRMSE", fontsize=9)
        axes[row_idx, 3].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("The Well Shear Flow V1: Failure and Boundary Case Analysis (30-Step Rollout)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])

    save_fig_path = os.path.join(fig_dir, "closure_r4_failure_cases_analysis.png")
    plt.savefig(save_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Exported failure case visualization to: {save_fig_path}")

    # Serialize JSON
    summary_data = {
        "metadata": {
            "model_path": model_path,
            "evaluation_protocol": PHYSICS_PROTOCOL,
            "checkpoint_training_protocol": checkpoint_training_protocol,
            "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
            "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
            "is_legacy": is_legacy,
            "split_type": s_type,
            "downsample_factor": ds_factor,
        },
        "best_case": {k: v for k, v in best_case.items() if k not in ["pred_step30", "gt_step30"]},
        "median_case": {k: v for k, v in median_case.items() if k not in ["pred_step30", "gt_step30"]},
        "worst_case": {k: v for k, v in worst_case.items() if k not in ["pred_step30", "gt_step30"]},
        "worst_3_trajectories": [
            {k: v for k, v in t.items() if k not in ["pred_step30", "gt_step30"]} for t in sorted_trajs[-3:]
        ],
        "best_3_trajectories": [
            {k: v for k, v in t.items() if k not in ["pred_step30", "gt_step30"]} for t in sorted_trajs[:3]
        ],
    }

    save_json_path = os.path.join(metrics_dir, "closure_r4_failure_cases_analysis.json")
    with open(save_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Exported failure case metadata to: {save_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Failure and boundary case analysis.")
    parser.add_argument("--model_path", type=str, default="outputs/checkpoints/dynamics/closure_r4/ablation_E4_full_physics/best_vrmse_mean.pt")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--split_type", type=str, default="grouped")
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--allow_legacy_checkpoint", action="store_true", default=False, help="Allow explicit re-evaluation of a pre-R4 field-only checkpoint; pre-R4 physics-loss checkpoints remain blocked")
    args = parser.parse_args()

    analyze_failure_cases(
        model_path=args.model_path,
        data_dir=args.data_dir,
        split_type=args.split_type,
        split_file=args.split_file,
        output_dir=args.output_dir,
        max_horizon=args.horizon,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
    )
