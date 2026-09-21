"""Evaluate 4-group physical loss ablation models on 30-step rollout benchmark."""

import argparse
import glob
import json
import os
import sys
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.shear_flow_dataset import ShearFlowDataset
from src.metrics.rollout import evaluate_rollout_trajectory
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_transformer import LatentSTTransformer
from src.models.latent_forecaster import LatentForecaster
from src.utils.reproducibility import seed_everything


def evaluate_single_ablation(model, test_loader, device, max_horizon=30, eval_steps=[1, 5, 10, 20, 30]):
    model.eval()
    accumulated = {f"step_{s}": {} for s in eval_steps}
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            q_hist = batch["history"].to(device)
            q_future = batch["future"].to(device)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)
            b = len(q_hist)
            total_samples += b

            pred_traj = model.forward_rollout(q_hist, re, sc, horizon=max_horizon)
            pred_traj[:, :, 2:3, :, :] = pred_traj[:, :, 2:3, :, :] - pred_traj[:, :, 2:3, :, :].mean(dim=(-2, -1), keepdim=True)
            q_future_gauge = q_future.clone()
            q_future_gauge[:, :, 2:3, :, :] = q_future_gauge[:, :, 2:3, :, :] - q_future_gauge[:, :, 2:3, :, :].mean(dim=(-2, -1), keepdim=True)
            batch_res = evaluate_rollout_trajectory(pred_traj, q_future_gauge, evaluation_steps=eval_steps)

            for step_key, step_data in batch_res.items():
                for m_key, m_val in step_data.items():
                    accumulated[step_key][m_key] = accumulated[step_key].get(m_key, 0.0) + m_val * b

    averaged = {}
    for step_key, step_data in accumulated.items():
        averaged[step_key] = {k: v / total_samples for k, v in step_data.items()}
    return averaged


def run_physics_ablation_eval(
    data_dir: str = "/root/autodl-tmp/datasets/shear_flow",
    output_file: str = "outputs/metrics/physics_ablation_benchmark.json",
    max_horizon: int = 30,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    seed_everything(42)
    device = torch.device(device_str)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    test_files = sorted(glob.glob(os.path.join(data_dir, "**/test/*.hdf5"), recursive=True))
    if not test_files:
        test_files = sorted(glob.glob(os.path.join(data_dir, "**/valid/*.hdf5"), recursive=True))

    test_dataset = ShearFlowDataset(test_files, history_length=4, horizon=max_horizon, stride=20)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    print(f"Loaded {len(test_dataset)} test trajectories for physics ablation {max_horizon}-step rollout evaluation.")

    ablation_ckpts = {
        "L_field": [
            "outputs/checkpoints/dynamics/ablation_L_field/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_L_field/latent_transformer/best_vrmse_mean.pt",
        ],
        "plus_L_div": [
            "outputs/checkpoints/dynamics/ablation_plus_L_div/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_plus_L_div/latent_transformer/best_vrmse_mean.pt",
        ],
        "plus_L_vort": [
            "outputs/checkpoints/dynamics/ablation_plus_L_vort/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_plus_L_vort/latent_transformer/best_vrmse_mean.pt",
        ],
        "plus_L_div_vort": [
            "outputs/checkpoints/dynamics/ablation_plus_L_div_vort/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_plus_L_div_vort/latent_transformer/best_vrmse_mean.pt",
        ],
    }

    results = {}

    for name, candidate_paths in ablation_ckpts.items():
        ckpt_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                ckpt_path = p
                break

        if ckpt_path is None:
            print(f"Warning: No valid checkpoint found for {name} in {candidate_paths}, skipping")
            continue

        print(f"\n--- Evaluating Ablation Model [{name}] ({ckpt_path}) ---")
        encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
        decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
        transformer = LatentSTTransformer(
            latent_channels=64, embed_dim=256, cond_dim=128, depth=6, num_heads=8, history_length=4
        )
        forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder).to(device)
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in ckpt_data:
            forecaster.load_state_dict(ckpt_data["model_state_dict"])

        results[name] = evaluate_single_ablation(forecaster, test_loader, device, max_horizon=max_horizon)

    # Print summary table
    all_physics_metrics = [
        ("vrmse_mean", "Field Mean VRMSE"),
        ("rmse_mean", "Field Mean RMSE"),
        ("div_rmse", "Divergence RMSE"),
        ("vort_rmse", "Vorticity RMSE"),
        ("ke_rel_err", "Kinetic Energy Rel Err"),
        ("enstrophy_rel_err", "Enstrophy Rel Err"),
        ("energy_spectrum_mae", "Energy Spectrum MAE"),
        ("tracer_var_retention", "Tracer Var Retention"),
        ("tracer_mass_error", "Tracer Mass Error"),
    ]

    print("\n" + "=" * 98)
    print(f"{'Ablation Group':<20} | {'Physical Metric':<24} | {'Step 1':<9} | {'Step 5':<9} | {'Step 10':<9} | {'Step 20':<9} | {'Step 30':<9}")
    print("-" * 98)
    for m_name, m_res in results.items():
        for metric_key, metric_title in all_physics_metrics:
            row = [f"{m_res.get(f'step_{s}', {}).get(metric_key, 0.0):.4f}" for s in [1, 5, 10, 20, 30]]
            print(f"{m_name:<20} | {metric_title:<24} | {row[0]:<9} | {row[1]:<9} | {row[2]:<9} | {row[3]:<9} | {row[4]:<9}")
        print("-" * 98)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved physics ablation benchmark metrics to: {output_file}")

    # Auto-plot
    try:
        from scripts.plot_physics_ablation import plot_physics_ablation_curves
        plot_physics_ablation_curves(json_path=output_file)
    except Exception as e:
        print(f"Plotting notice: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--output_file", type=str, default="outputs/metrics/physics_ablation_benchmark.json")
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()

    run_physics_ablation_eval(data_dir=args.data_dir, output_file=args.output_file, max_horizon=args.horizon)
