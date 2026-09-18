"""Stage 8: Multi-Step Autoregressive Rollout Benchmark (h in {1, 5, 10, 20, 30})."""

import argparse
import glob
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from src.baselines.fno import FNO2D
from src.baselines.persistence import PersistenceBaseline
from src.data.shear_flow_dataset import ShearFlowDataset
from src.metrics.compute import benchmark_inference, count_parameters
from src.metrics.rollout import evaluate_rollout_trajectory
from src.models.decoder import Decoder2D
from src.models.direct_transformer import DirectSTTransformer
from src.models.encoder import Encoder2D
from src.models.history_buffer import HistoryBuffer
from src.models.latent_transformer import LatentSTTransformer
from src.utils.checkpoint import load_checkpoint
from src.utils.reproducibility import seed_everything


def evaluate_model_rollout(
    model_name: str,
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    max_horizon: int = 30,
    eval_steps: list = [1, 5, 10, 20, 30],
) -> dict:
    """Roll out model for max_horizon steps and compute evaluation metrics."""
    model.eval()
    accumulated_metrics = {f"step_{s}": {} for s in eval_steps}
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            q_hist = batch["history"].to(device)  # (B, L, 4, Ny, Nx)
            q_future = batch["future"].to(device)  # (B, H, 4, Ny, Nx)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)
            b = len(q_hist)
            total_samples += b

            # Generate rollout trajectory
            if model_name == "persistence":
                pred_traj = model(q_hist, horizon=max_horizon)

            elif model_name == "latent_transformer":
                pred_traj = model.forward_rollout(q_hist, re, sc, horizon=max_horizon)

            elif model_name == "direct_transformer":
                buf = HistoryBuffer(history_length=q_hist.shape[1])
                buf.reset(q_hist)
                pred_traj = buf.rollout(lambda hist, _c: model(hist, re=re, sc=sc), steps=max_horizon)

            elif model_name == "fno":
                buf = HistoryBuffer(history_length=q_hist.shape[1])
                buf.reset(q_hist)
                pred_traj = buf.rollout(lambda hist, _c: model(hist), steps=max_horizon)

            # Evaluate metrics at designated steps
            batch_res = evaluate_rollout_trajectory(pred_traj, q_future, evaluation_steps=eval_steps)
            for step_key, step_data in batch_res.items():
                for m_key, m_val in step_data.items():
                    accumulated_metrics[step_key][m_key] = (
                        accumulated_metrics[step_key].get(m_key, 0.0) + m_val * b
                    )

    # Average
    averaged = {}
    for step_key, step_data in accumulated_metrics.items():
        averaged[step_key] = {k: v / total_samples for k, v in step_data.items()}

    return averaged


def run_benchmark(
    data_dir: str = "/root/autodl-tmp/datasets/shear_flow",
    output_file: str = "outputs/metrics/rollout_benchmark.json",
    max_horizon: int = 30,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    seed_everything(42)
    device = torch.device(device_str)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    test_files = sorted(glob.glob(os.path.join(data_dir, "test", "*.hdf5")))
    if not test_files:
        print(f"No test files found in {data_dir}/test.")
        return

    test_dataset = ShearFlowDataset(test_files, history_length=4, horizon=max_horizon, stride=15)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    print(f"Loaded {len(test_dataset)} test trajectories for {max_horizon}-step rollout evaluation.")

    results = {}

    # 1. Baseline: Persistence
    print("\n--- Evaluating Persistence Baseline ---")
    persistence = PersistenceBaseline().to(device)
    results["persistence"] = evaluate_model_rollout("persistence", persistence, test_loader, device, max_horizon)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Model':<20} | {'Metric':<15} | {'Step 1':<10} | {'Step 5':<10} | {'Step 10':<10} | {'Step 20':<10} | {'Step 30':<10}")
    print("-" * 90)
    for m_name, m_res in results.items():
        for metric in ["vrmse_mean", "div_rmse", "tracer_var_retention"]:
            row = [f"{m_res.get(f'step_{s}', {}).get(metric, 0.0):.4f}" for s in [1, 5, 10, 20, 30]]
            print(f"{m_name:<20} | {metric:<15} | {row[0]:<10} | {row[1]:<10} | {row[2]:<10} | {row[3]:<10} | {row[4]:<10}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark metrics to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--output_file", type=str, default="outputs/metrics/rollout_benchmark.json")
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()

    run_benchmark(data_dir=args.data_dir, output_file=args.output_file, max_horizon=args.horizon)
