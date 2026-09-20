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
from src.models.latent_forecaster import LatentForecaster
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

            elif "latent" in model_name:
                pred_traj = model.forward_rollout(q_hist, re, sc, horizon=max_horizon)

            elif model_name == "direct_transformer":
                buf = HistoryBuffer(history_length=q_hist.shape[1])
                buf.reset(q_hist)
                pred_traj = buf.rollout(lambda hist, _c: model(hist, re=re, sc=sc), steps=max_horizon)

            elif model_name == "fno":
                buf = HistoryBuffer(history_length=q_hist.shape[1])
                buf.reset(q_hist)
                pred_traj = buf.rollout(lambda hist, _c: model(hist), steps=max_horizon)

            else:
                raise ValueError(f"Unknown model_name: {model_name}")


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
    latent_ckpt_arg: str = None,
):

    seed_everything(42)
    device = torch.device(device_str)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    test_files = sorted(glob.glob(os.path.join(data_dir, "**/test/*.hdf5"), recursive=True))
    if not test_files:
        test_files = sorted(glob.glob(os.path.join(data_dir, "**/valid/*.hdf5"), recursive=True))
        if test_files:
            print(f"Notice: No test files found. Using available valid files for benchmark evaluation.")
        else:
            print(f"No test/valid files found in {data_dir}.")
            return

    test_dataset = ShearFlowDataset(test_files, history_length=4, horizon=max_horizon, stride=20)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    print(f"Loaded {len(test_dataset)} test trajectories for {max_horizon}-step rollout evaluation.")

    results = {}

    # 1. Baseline: Persistence
    print("\n--- Evaluating Persistence Baseline ---")
    persistence = PersistenceBaseline().to(device)
    results["persistence"] = evaluate_model_rollout("persistence", persistence, test_loader, device, max_horizon)

    # 2. Main Model: Latent ST Transformer(s)
    candidate_ckpts = []
    if latent_ckpt_arg:
        candidate_ckpts.append(("latent_transformer", latent_ckpt_arg))
    else:
        step1_ckpt = "outputs/checkpoints/dynamics/latent_transformer/best_vrmse_mean.pt"
        rollout_ckpt = "outputs/checkpoints/dynamics/stage4_latent_rollout_ddp/latent_transformer/best_vrmse_mean.pt"
        if os.path.exists(step1_ckpt) and os.path.exists(rollout_ckpt):
            candidate_ckpts.append(("latent_step1", step1_ckpt))
            candidate_ckpts.append(("latent_rollout", rollout_ckpt))
        elif os.path.exists(rollout_ckpt):
            candidate_ckpts.append(("latent_transformer", rollout_ckpt))
        elif os.path.exists(step1_ckpt):
            candidate_ckpts.append(("latent_transformer", step1_ckpt))

    for m_label, ckpt_path in candidate_ckpts:
        if os.path.exists(ckpt_path):
            print(f"\n--- Evaluating Latent ST Transformer [{m_label}] ({ckpt_path}) ---")
            encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
            decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=True)
            transformer = LatentSTTransformer(
                latent_channels=64, embed_dim=256, cond_dim=128, depth=6, num_heads=8, history_length=4
            )
            forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder).to(device)
            ckpt_data = torch.load(ckpt_path, map_location="cpu")
            if "model_state_dict" in ckpt_data:
                forecaster.load_state_dict(ckpt_data["model_state_dict"])
            results[m_label] = evaluate_model_rollout(m_label, forecaster, test_loader, device, max_horizon)


    # 3. Ablation Baseline: Direct ST Transformer
    direct_ckpt = "outputs/checkpoints/dynamics/direct_transformer/best_vrmse_mean.pt"
    if os.path.exists(direct_ckpt):
        print("\n--- Evaluating Direct ST Transformer (Ablation Q1 Baseline) ---")
        direct_model = DirectSTTransformer(
            in_channels=4,
            patch_size=(8, 8),
            embed_dim=256,
            cond_dim=128,
            depth=6,
            num_heads=8,
            history_length=4,
            prediction_mode="direct",
        ).to(device)
        ckpt_data = torch.load(direct_ckpt, map_location="cpu")
        if "model_state_dict" in ckpt_data:
            direct_model.load_state_dict(ckpt_data["model_state_dict"])
        results["direct_transformer"] = evaluate_model_rollout("direct_transformer", direct_model, test_loader, device, max_horizon)

    # 4. Neural Operator Baseline: FNO-2D
    fno_ckpt = "outputs/checkpoints/dynamics/fno_baseline/fno/best_vrmse_mean.pt"
    if not os.path.exists(fno_ckpt):
        fno_ckpt = "outputs/checkpoints/dynamics/fno/best_vrmse_mean.pt"
    if os.path.exists(fno_ckpt):
        print("\n--- Evaluating FNO-2D Baseline ---")
        fno_model = FNO2D(
            in_channels=16,
            out_channels=4,
            modes1=16,
            modes2=16,
            width=64,
            num_layers=4,
        ).to(device)
        ckpt_data = torch.load(fno_ckpt, map_location="cpu")
        if "model_state_dict" in ckpt_data:
            fno_model.load_state_dict(ckpt_data["model_state_dict"])
        results["fno"] = evaluate_model_rollout("fno", fno_model, test_loader, device, max_horizon)

    # Print comprehensive physical summary table
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
    print(f"{'Model':<20} | {'Physical Metric':<24} | {'Step 1':<9} | {'Step 5':<9} | {'Step 10':<9} | {'Step 20':<9} | {'Step 30':<9}")
    print("-" * 98)
    for m_name, m_res in results.items():
        for metric_key, metric_title in all_physics_metrics:
            row = [f"{m_res.get(f'step_{s}', {}).get(metric_key, 0.0):.4f}" for s in [1, 5, 10, 20, 30]]
            print(f"{m_name:<20} | {metric_title:<24} | {row[0]:<9} | {row[1]:<9} | {row[2]:<9} | {row[3]:<9} | {row[4]:<9}")
        print("-" * 98)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark metrics to: {output_file}")

    # Auto-generate publication rollout benchmark curves
    try:
        from scripts.plot_rollout_comparison import plot_benchmark_curves
        plot_benchmark_curves(json_path=output_file)
    except Exception as e:
        print(f"Notice: Plotting failed with error: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--output_file", type=str, default="outputs/metrics/rollout_benchmark.json")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--latent_ckpt", type=str, default=None, help="Custom checkpoint path for Latent ST Transformer")
    args = parser.parse_args()

    run_benchmark(
        data_dir=args.data_dir,
        output_file=args.output_file,
        max_horizon=args.horizon,
        latent_ckpt_arg=args.latent_ckpt,
    )

