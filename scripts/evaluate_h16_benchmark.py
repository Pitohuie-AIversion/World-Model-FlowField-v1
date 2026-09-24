#!/usr/bin/env python3
"""Unified Benchmark Evaluation: Parent H8 Ep11 vs H16 Short-Best vs H16 Long-Best.

Evaluates:
  1. Parent H8 Saved Long-Best (checkpoint_step_11_vrmse_mean_0.2186.pt)
  2. H16 Short-Best (best_vrmse_mean.pt / checkpoint_step_8_vrmse_mean_0.2466.pt)
  3. H16 Long-Best (best_long_vrmse.pt / checkpoint_step_12_long_vrmse_0.3961.pt)

On canonical Test partition across rollout horizons h in [1, 5, 10, 16, 20, 30].

Metrics extracted per horizon:
  - Field errors: VRMSE (mean, u, v, p, s), RMSE (mean)
  - Physical conservation / invariants:
      * Divergence: div_rmse, div_max
      * Vorticity: vort_rmse
      * Enstrophy: enstrophy_rel_err, ens_pred, ens_targ
      * Kinetic Energy: ke_rel_err, ke_pred, ke_targ
      * Energy Spectrum: energy_spectrum_mae, spec_err_low, spec_err_mid, spec_err_high
      * Tracer: tracer_oob_rate, tracer_integral_drift, tracer_mass_rel_err
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.pipeline import create_flow_dataloaders, FieldNormalizer
from src.metrics.rollout import evaluate_rollout_trajectory
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.utils.physics_contract import (
    PHYSICS_PROTOCOL,
    SPATIAL_AXIS_CONTRACT,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
)
from src.utils.provenance import (
    get_git_commit,
    is_git_dirty,
    compute_file_sha256,
    compute_split_hash_from_file,
    compute_normalizer_hash,
    resolve_checkpoint_provenance,
    validate_evaluation_provenance,
)
from src.utils.reproducibility import seed_everything

# Target candidate checkpoints
BENCHMARK_TARGETS = {
    "parent_h8_ep11": {
        "title": "Parent H8 Saved Long-Best (Ep 11)",
        "path": "outputs/checkpoints/dynamics/horizon_r1/seed_42/E4_H8/latent_transformer/checkpoint_step_11_vrmse_mean_0.2186.pt",
        "expected_horizon": 8,
    },
    "h16_short_best_ep8": {
        "title": "H16 Short-Best (Ep 8)",
        "path": "outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H16/latent_transformer/best_vrmse_mean.pt",
        "expected_horizon": 16,
    },
    "h16_long_best_ep12": {
        "title": "H16 Long-Best (Ep 12)",
        "path": "outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H16/latent_transformer/best_long_vrmse.pt",
        "expected_horizon": 16,
    },
}

EVAL_HORIZONS = [1, 5, 10, 16, 20, 30]
DEFAULT_OUTPUT_METRICS = "outputs/metrics/h16_benchmark_evaluation.json"


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[LatentForecaster, dict, dict]:
    """Load LatentForecaster and return (model, config, full_checkpoint_dict)."""
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt_data.get("config", {})

    pred_mode = cfg.get("prediction_mode", ckpt_data.get("prediction_mode", "direct"))
    emb_dim = cfg.get("embed_dim", 256)
    depth = cfg.get("depth", 6)
    num_heads = cfg.get("num_heads", 8)

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
    )
    forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder).to(device)

    state_dict = ckpt_data.get("model_state_dict", ckpt_data)
    # Strip any DDP prefix if present
    cleaned_state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }
    load_msg = forecaster.load_state_dict(cleaned_state_dict, strict=True)
    forecaster.eval()
    return forecaster, cfg, ckpt_data


def evaluate_model_full_physical(
    model: LatentForecaster,
    data_loader: DataLoader,
    device: torch.device,
    eval_horizons: List[int],
    normalizer: Optional[FieldNormalizer] = None,
    use_condition: bool = True,
    domain_size: tuple = SHEAR_FLOW_DOMAIN_SIZE_XY,
) -> Dict[str, Dict[str, float]]:
    """Run full physical multi-step rollout evaluation on the test partition."""
    model.eval()
    max_h = max(eval_horizons)

    # Accumulate metrics keyed by step_{h}
    accumulated = {f"step_{h}": {} for h in eval_horizons}
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            q_hist = batch["history"].to(device)
            q_future = batch["future"].to(device)
            re = batch["re"].to(device) if use_condition and "re" in batch else None
            sc = batch["sc"].to(device) if use_condition and "sc" in batch else None
            b = len(q_hist)
            total_samples += b

            # Autoregressive forward rollout
            pred_traj = model.forward_rollout(q_hist, re, sc, horizon=max_h)

            # Denormalize to physical units
            if normalizer is not None:
                pred_eval = normalizer.denormalize(pred_traj)
                future_eval = normalizer.denormalize(q_future[:, :max_h])
                hist_eval = normalizer.denormalize(q_hist)
            else:
                pred_eval = pred_traj
                future_eval = q_future[:, :max_h]
                hist_eval = q_hist

            # Initial condition (last frame of history window)
            initial_state = hist_eval[:, -1].clone()

            # Zero-mean gauge pressure normalization in physical coordinates
            pred_eval[:, :, 2:3, :, :] = pred_eval[:, :, 2:3, :, :] - pred_eval[:, :, 2:3, :, :].mean(dim=(-2, -1), keepdim=True)
            future_eval[:, :, 2:3, :, :] = future_eval[:, :, 2:3, :, :] - future_eval[:, :, 2:3, :, :].mean(dim=(-2, -1), keepdim=True)
            initial_state[:, 2:3, :, :] = initial_state[:, 2:3, :, :] - initial_state[:, 2:3, :, :].mean(dim=(-2, -1), keepdim=True)

            # Compute field, spectral, conservation and tracer metrics
            batch_res = evaluate_rollout_trajectory(
                pred_trajectory=pred_eval,
                target_trajectory=future_eval,
                evaluation_steps=eval_horizons,
                domain_size=domain_size,
                initial_state=initial_state,
            )

            for step_key, step_data in batch_res.items():
                for m_key, m_val in step_data.items():
                    accumulated[step_key][m_key] = accumulated[step_key].get(m_key, 0.0) + float(m_val) * b

    averaged = {}
    for step_key, step_data in accumulated.items():
        averaged[step_key] = {k: v / total_samples for k, v in step_data.items()}

    return averaged


def print_markdown_tables(results: dict):
    """Print structured, publication-grade markdown comparison tables."""
    models = list(results.keys())

    print("\n" + "=" * 90)
    print("### TABLE 1: ROLLOUT FIELD VRMSE COMPARISON (Test Set)")
    print("=" * 90)
    header = "| Model | h=1 | h=5 | h=10 | h=16 | h=20 | h=30 | J_long (mean 10,20,30) |"
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    print(header)
    print(sep)
    for m in models:
        m_res = results[m]["metrics"]
        h1 = m_res.get("step_1", {}).get("vrmse_mean", float("nan"))
        h5 = m_res.get("step_5", {}).get("vrmse_mean", float("nan"))
        h10 = m_res.get("step_10", {}).get("vrmse_mean", float("nan"))
        h16 = m_res.get("step_16", {}).get("vrmse_mean", float("nan"))
        h20 = m_res.get("step_20", {}).get("vrmse_mean", float("nan"))
        h30 = m_res.get("step_30", {}).get("vrmse_mean", float("nan"))
        j_long = (h10 + h20 + h30) / 3.0 if not any(map(lambda x: x != x, [h10, h20, h30])) else float("nan")
        print(f"| {BENCHMARK_TARGETS[m]['title']} | {h1:.4f} | {h5:.4f} | {h10:.4f} | {h16:.4f} | {h20:.4f} | {h30:.4f} | **{j_long:.4f}** |")

    print("\n" + "=" * 90)
    print("### TABLE 2: PHYSICAL CONSERVATION & SPECTRAL INVARIANTS (Test Set)")
    print("=" * 90)
    header2 = "| Model | Metric | h=1 | h=10 | h=20 | h=30 |"
    sep2 = "| :--- | :--- | :---: | :---: | :---: | :---: |"
    print(header2)
    print(sep2)
    for m in models:
        m_res = results[m]["metrics"]
        title = BENCHMARK_TARGETS[m]["title"]
        for metric_name, display in [
            ("div_rmse", "Divergence RMSE"),
            ("vort_rmse", "Vorticity RMSE"),
            ("enstrophy_rel_err", "Enstrophy Rel Err"),
            ("energy_spectrum_mae", "Energy Spec MAE"),
            ("tracer_out_of_bounds_rate", "Tracer OOB Rate"),
            ("tracer_mass_error", "Tracer Mass Err"),
            ("tracer_mean_err", "Tracer Mean Drift"),
        ]:
            v1 = m_res.get("step_1", {}).get(metric_name, float("nan"))
            v10 = m_res.get("step_10", {}).get(metric_name, float("nan"))
            v20 = m_res.get("step_20", {}).get(metric_name, float("nan"))
            v30 = m_res.get("step_30", {}).get(metric_name, float("nan"))
            print(f"| {title} | {display} | {v1:.4e} | {v10:.4e} | {v20:.4e} | {v30:.4e} |")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate H8 Long-Best vs H16 Short-Best vs H16 Long-Best on physical metrics")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_METRICS)
    parser.add_argument("--formal", action="store_true", help="Enforce fail-closed provenance validation (clean git tree)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device(args.device)

    # 1. Formal preflight provenance
    if args.formal:
        dirty = is_git_dirty()
        if dirty:
            raise RuntimeError("Formal evaluation failed-closed: working tree is dirty.")

    current_commit = get_git_commit()
    print(f"Executing H16 benchmark evaluation on {device} (Git Commit: {current_commit})...")

    # 2. Setup dataset and read-only normalizer
    split_file = "outputs/splits/grouped_split.json"
    if not os.path.exists(split_file):
        split_file = "outputs/splits/grouped.json"

    stats_path = "outputs/normalization/stats_grouped.pt"
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Required normalizer statistics not found: {stats_path}")
    normalizer = FieldNormalizer()
    normalizer.load_state_dict(torch.load(stats_path, weights_only=True, map_location="cpu"))
    normalizer_hash = compute_normalizer_hash(normalizer)
    split_hash = compute_split_hash_from_file(split_file)

    _, _, test_loader, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file=split_file,
        data_root=args.data_dir,
        history_length=4,
        horizon=max(EVAL_HORIZONS),
        stride=20,
        downsample_factor=2,
        batch_size=args.batch_size,
        num_workers=0,
        normalize=True,
        normalizer=normalizer,
        preload_to_memory=False,
        seed=42,
    )
    print(f"Loaded test dataset: {len(test_loader.dataset)} samples ({len(test_loader)} batches).")

    # 3. Run evaluation across the 3 target models
    benchmark_results = {}
    for target_key, target_info in BENCHMARK_TARGETS.items():
        ckpt_path = target_info["path"]
        print(f"\n--- Evaluating: {target_info['title']} ({ckpt_path}) ---")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Target checkpoint not found: {ckpt_path}")

        model, cfg, ckpt_data = load_model_from_checkpoint(ckpt_path, device=device)
        sha256 = compute_file_sha256(ckpt_path)
        prov = resolve_checkpoint_provenance(ckpt_path, ckpt_data)
        t0 = time.time()
        metrics = evaluate_model_full_physical(
            model=model,
            data_loader=test_loader,
            device=device,
            eval_horizons=EVAL_HORIZONS,
            normalizer=normalizer,
            use_condition=True,
            domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY,
        )
        elapsed = time.time() - t0
        print(f"Completed in {elapsed:.1f}s.")

        benchmark_results[target_key] = {
            "title": target_info["title"],
            "checkpoint_path": ckpt_path,
            "sha256": sha256,
            "provenance": prov,
            "config": cfg,
            "metrics": metrics,
        }

    # 4. Save results to output_file
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_git_commit": current_commit,
                "evaluation_git_dirty": is_git_dirty(),
                "split_hash": split_hash,
                "normalizer_hash": normalizer_hash,
                "eval_horizons": EVAL_HORIZONS,
                "results": benchmark_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {out_path}")

    # 5. Print summary tables
    print_markdown_tables(benchmark_results)


if __name__ == "__main__":
    main()
