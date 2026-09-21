"""Analyze failure cases and boundary scenarios on 30-step rollout trajectories.

Identifies worst (highest error) and best trajectories across the test set,
visualizes spatial error fields (vorticity, tracer, velocity), and analyzes
physical root causes of degradation.
"""

import argparse
import glob
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.shear_flow_dataset import ShearFlowDataset
from src.metrics.field import evaluate_field_metrics
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.utils.fft_derivatives import compute_divergence, compute_vorticity
from src.utils.reproducibility import seed_everything


def analyze_failure_cases(
    model_path: str = "outputs/checkpoints/dynamics/ablation_plus_L_div_vort/latent_transformer/best_vrmse_mean.pt",
    data_dir: str = "/root/autodl-tmp/datasets/shear_flow",
    output_dir: str = "outputs",
    max_horizon: int = 30,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    seed_everything(42)
    device = torch.device(device_str)
    fig_dir = os.path.join(output_dir, "figures")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    test_files = sorted(glob.glob(os.path.join(data_dir, "**/test/*.hdf5"), recursive=True))
    if not test_files:
        test_files = sorted(glob.glob(os.path.join(data_dir, "**/valid/*.hdf5"), recursive=True))

    # Stride=20 ensures we get individual distinct trajectories across the test partition
    test_dataset = ShearFlowDataset(test_files, history_length=4, horizon=max_horizon, stride=20)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Analyzing {len(test_dataset)} test trajectories with model {model_path}...")

    # Load Model
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    transformer = LatentSTTransformer(
        latent_channels=64, embed_dim=256, cond_dim=128, depth=6, num_heads=8, history_length=4
    )
    model = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder).to(device)

    ckpt = torch.load(model_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    trajectory_records = []

    with torch.no_grad():
        for traj_idx, batch in enumerate(test_loader):
            q_hist = batch["history"].to(device)
            q_future = batch["future"].to(device)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)

            pred_traj = model.forward_rollout(q_hist, re, sc, horizon=max_horizon)
            # Enforce zero-mean pressure gauge in physical space
            pred_traj[:, :, 2:3, :, :] = pred_traj[:, :, 2:3, :, :] - pred_traj[:, :, 2:3, :, :].mean(dim=(-2, -1), keepdim=True)
            q_future_gauge = q_future.clone()
            q_future_gauge[:, :, 2:3, :, :] = q_future_gauge[:, :, 2:3, :, :] - q_future_gauge[:, :, 2:3, :, :].mean(dim=(-2, -1), keepdim=True)

            # Evaluate trajectory level metrics
            step30_pred = pred_traj[:, -1]
            step30_gt = q_future_gauge[:, -1]
            metrics_step30 = evaluate_field_metrics(step30_pred, step30_gt)

            # Also evaluate cumulative VRMSE across all 30 steps
            all_vrmse = []
            for t in range(max_horizon):
                step_m = evaluate_field_metrics(pred_traj[:, t], q_future_gauge[:, t])
                all_vrmse.append(step_m["vrmse_mean"])

            div_err = compute_divergence(step30_pred[0, 0], step30_pred[0, 1]).pow(2).mean().sqrt().item()
            vort_pred = compute_vorticity(step30_pred[0, 0], step30_pred[0, 1])
            vort_gt = compute_vorticity(step30_gt[0, 0], step30_gt[0, 1])
            vort_err = (vort_pred - vort_gt).pow(2).mean().sqrt().item()

            trajectory_records.append(
                {
                    "traj_idx": traj_idx,
                    "re": re.item(),
                    "sc": sc.item(),
                    "step30_vrmse": metrics_step30["vrmse_mean"],
                    "step30_rmse": metrics_step30["rmse_mean"],
                    "mean_rollout_vrmse": float(np.mean(all_vrmse)),
                    "step30_div_rmse": div_err,
                    "step30_vort_rmse": vort_err,
                    "all_vrmse": all_vrmse,
                    "pred_step30": step30_pred.cpu().numpy()[0],
                    "gt_step30": step30_gt.cpu().numpy()[0],
                }
            )

    # Sort trajectories by mean_rollout_vrmse
    sorted_trajs = sorted(trajectory_records, key=lambda x: x["mean_rollout_vrmse"])
    best_case = sorted_trajs[0]
    worst_case = sorted_trajs[-1]
    median_case = sorted_trajs[len(sorted_trajs) // 2]

    print(f"\n--- Trajectory Error Ranking ---")
    print(f"Best  Traj #{best_case['traj_idx']}  | Re={best_case['re']:.0f}, Sc={best_case['sc']:.1f} | Mean VRMSE: {best_case['mean_rollout_vrmse']:.4f} | Step30 VRMSE: {best_case['step30_vrmse']:.4f}")
    print(f"Median Traj #{median_case['traj_idx']} | Re={median_case['re']:.0f}, Sc={median_case['sc']:.1f} | Mean VRMSE: {median_case['mean_rollout_vrmse']:.4f} | Step30 VRMSE: {median_case['step30_vrmse']:.4f}")
    print(f"Worst Traj #{worst_case['traj_idx']} | Re={worst_case['re']:.0f}, Sc={worst_case['sc']:.1f} | Mean VRMSE: {worst_case['mean_rollout_vrmse']:.4f} | Step30 VRMSE: {worst_case['step30_vrmse']:.4f}")

    # Plot failure case spatial analysis figure
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), dpi=200)

    cases = [
        ("Best Case (Traj #" + str(best_case["traj_idx"]) + ")", best_case),
        ("Median Case (Traj #" + str(median_case["traj_idx"]) + ")", median_case),
        ("Worst Failure Case (Traj #" + str(worst_case["traj_idx"]) + ")", worst_case),
    ]

    for row_idx, (case_title, case_data) in enumerate(cases):
        gt = case_data["gt_step30"]
        pred = case_data["pred_step30"]

        # Calculate vorticity for ground truth and prediction
        # u is channel 0, v is channel 1
        gt_t = torch.from_numpy(gt).unsqueeze(0)
        pred_t = torch.from_numpy(pred).unsqueeze(0)
        vort_gt = compute_vorticity(gt_t[:, 0], gt_t[:, 1])[0].numpy()
        vort_pred = compute_vorticity(pred_t[:, 0], pred_t[:, 1])[0].numpy()
        vort_diff = np.abs(vort_pred - vort_gt)

        tracer_gt = gt[3]
        tracer_pred = pred[3]
        tracer_diff = np.abs(tracer_pred - tracer_gt)

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

    save_fig_path = os.path.join(fig_dir, "failure_cases_analysis.png")
    plt.savefig(save_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Exported failure case visualization to: {save_fig_path}")

    # Serialize JSON
    summary_data = {
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

    save_json_path = os.path.join(metrics_dir, "failure_cases_analysis.json")
    with open(save_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Exported failure case metadata to: {save_json_path}")


if __name__ == "__main__":
    analyze_failure_cases()
