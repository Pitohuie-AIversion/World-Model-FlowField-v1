"""Stage 6: Multi-Step Evaluation for 5-Group Physical Loss Ablation (E0 - E4).

Evaluates the 5 physics ablation groups (per README spec Section 9.5 & Section 18):
- E0_single_step: Single-step baseline (H=1, L_field)
- E1_rollout_field: Multi-step rollout field loss (H=2, L_field)
- E2_plus_L_div: + Incompressibility divergence penalty (H=2, +L_div)
- E3_plus_L_vort: + Vorticity consistency penalty (H=2, +L_vort)
- E4_full_physics: Full physics coupling (H=2, +L_div + L_vort)

Evaluates on test partition up to max_horizon steps (default: 30) using the
canonical data pipeline, self-describing checkpoint recovery, and physical-space gauge.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from src.data.pipeline import create_flow_dataloaders
from src.metrics.rollout import evaluate_rollout_trajectory
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.utils.reproducibility import seed_everything
from src.utils.physics_contract import PHYSICS_PROTOCOL, SPATIAL_AXIS_CONTRACT, SHEAR_FLOW_DOMAIN_SIZE_XY


ABLATION_GROUPS = {
    "E0_single_step": {
        "title": "E0: Single-Step Pure Field Loss",
        "candidates": [
            "outputs/checkpoints/dynamics/closure_r4/ablation_E0_single_step/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/closure_r4/ablation_E0_single_step/latent_transformer/best_vrmse_mean.pt",
        ],
        "legacy_candidates": [
            "outputs/checkpoints/dynamics/ablation_E0_single_step/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_E0_single_step/latent_transformer/best_vrmse_mean.pt",
        ],
        "invalid_axis_candidates": [],
    },
    "E1_rollout_field": {
        "title": "E1: Rollout-Aware Field Loss",
        "candidates": [
            "outputs/checkpoints/dynamics/closure_r4/ablation_E1_rollout_field/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/closure_r4/ablation_E1_rollout_field/latent_transformer/best_vrmse_mean.pt",
        ],
        "legacy_candidates": [
            "outputs/checkpoints/dynamics/ablation_E1_rollout_field/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_E1_rollout_field/latent_transformer/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_L_field/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_L_field/latent_transformer/best_vrmse_mean.pt",
        ],
        "invalid_axis_candidates": [],
    },
    "E2_plus_L_div": {
        "title": "E2: + Divergence Loss",
        "candidates": [
            "outputs/checkpoints/dynamics/closure_r4/ablation_E2_plus_L_div/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/closure_r4/ablation_E2_plus_L_div/latent_transformer/best_vrmse_mean.pt",
        ],
        "legacy_candidates": [],
        "invalid_axis_candidates": [
            "outputs/checkpoints/dynamics/ablation_E2_plus_L_div/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_E2_plus_L_div/latent_transformer/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_plus_L_div/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_plus_L_div/latent_transformer/best_vrmse_mean.pt",
        ],
    },
    "E3_plus_L_vort": {
        "title": "E3: + Vorticity Loss",
        "candidates": [
            "outputs/checkpoints/dynamics/closure_r4/ablation_E3_plus_L_vort/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/closure_r4/ablation_E3_plus_L_vort/latent_transformer/best_vrmse_mean.pt",
        ],
        "legacy_candidates": [],
        "invalid_axis_candidates": [
            "outputs/checkpoints/dynamics/ablation_E3_plus_L_vort/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_E3_plus_L_vort/latent_transformer/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_plus_L_vort/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_plus_L_vort/latent_transformer/best_vrmse_mean.pt",
        ],
    },
    "E4_full_physics": {
        "title": "E4: Full Physics Coupling",
        "candidates": [
            "outputs/checkpoints/dynamics/closure_r4/ablation_E4_full_physics/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/closure_r4/ablation_E4_full_physics/latent_transformer/best_vrmse_mean.pt",
        ],
        "legacy_candidates": [],
        "invalid_axis_candidates": [
            "outputs/checkpoints/dynamics/ablation_E4_full_physics/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_E4_full_physics/latent_transformer/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_plus_L_div_vort/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/ablation_plus_L_div_vort/latent_transformer/best_vrmse_mean.pt",
        ],
    },
}


def evaluate_single_ablation(
    model: LatentForecaster,
    test_loader: DataLoader,
    device: torch.device,
    max_horizon: int = 30,
    eval_steps: list = [1, 5, 10, 20, 30],
    normalizer=None,
    use_condition: bool = True,
) -> dict:
    """Evaluate single ablation model for max_horizon steps in physical space."""
    model.eval()
    accumulated = {f"step_{s}": {} for s in eval_steps}
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            q_hist = batch["history"].to(device)
            q_future = batch["future"].to(device)
            re = batch["re"].to(device) if use_condition and "re" in batch else None
            sc = batch["sc"].to(device) if use_condition and "sc" in batch else None
            b = len(q_hist)
            total_samples += b

            pred_traj = model.forward_rollout(q_hist, re, sc, horizon=max_horizon)

            # Denormalize to physical units before computing physical metrics
            if normalizer is not None:
                pred_eval = normalizer.denormalize(pred_traj)
                future_eval = normalizer.denormalize(q_future)
            else:
                pred_eval = pred_traj
                future_eval = q_future

            # Enforce physical zero-mean pressure gauge
            pred_eval[:, :, 2:3, :, :] = pred_eval[:, :, 2:3, :, :] - pred_eval[:, :, 2:3, :, :].mean(dim=(-2, -1), keepdim=True)
            future_eval[:, :, 2:3, :, :] = future_eval[:, :, 2:3, :, :] - future_eval[:, :, 2:3, :, :].mean(dim=(-2, -1), keepdim=True)

            batch_res = evaluate_rollout_trajectory(pred_eval, future_eval, evaluation_steps=eval_steps)

            for step_key, step_data in batch_res.items():
                for m_key, m_val in step_data.items():
                    accumulated[step_key][m_key] = accumulated[step_key].get(m_key, 0.0) + m_val * b

    averaged = {}
    for step_key, step_data in accumulated.items():
        averaged[step_key] = {k: v / total_samples for k, v in step_data.items()}
    return averaged


def run_physics_ablation_eval(
    data_dir: str = "/root/autodl-tmp/datasets/shear_flow",
    output_file: str = "outputs/metrics/closure_r4_physics_ablation.json",
    split_type: str = "grouped",
    split_file: Optional[str] = None,
    downsample_factor: int = 2,
    normalize: bool = True,
    max_horizon: int = 30,
    stride: int = 20,
    allow_legacy_checkpoints: bool = False,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    seed_everything(42)
    device = torch.device(device_str)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if split_file is None:
        if split_type.endswith(".json"):
            split_file = split_type
        elif split_type.startswith("outputs/splits/"):
            split_file = split_type
        else:
            candidate = f"outputs/splits/{split_type}.json"
            if os.path.exists(candidate):
                split_file = candidate
            else:
                split_file = f"outputs/splits/{split_type}_split.json"

    print(f"Loading test dataset via unified pipeline ({split_type} split: {split_file})...")
    _, _, test_loader, normalizer = create_flow_dataloaders(
        split_type=split_type,
        split_file=split_file,
        data_root=data_dir,
        history_length=4,
        horizon=max_horizon,
        stride=stride,
        downsample_factor=downsample_factor,
        batch_size=2,
        num_workers=0,
        normalize=normalize,
    )

    print(f"Loaded {len(test_loader.dataset)} test trajectories for physics ablation {max_horizon}-step evaluation.")

    results = {}

    for group_key, group_info in ABLATION_GROUPS.items():
        candidate_paths = list(group_info["candidates"])
        if allow_legacy_checkpoints:
            candidate_paths.extend(group_info.get("legacy_candidates", []))

        ckpt_path = None
        is_legacy = False
        for p in candidate_paths:
            if os.path.exists(p):
                ckpt_path = p
                is_legacy = p in group_info.get("legacy_candidates", [])
                break

        if ckpt_path is None:
            invalid_existing = [
                p for p in group_info.get("invalid_axis_candidates", []) if os.path.exists(p)
            ]
            if invalid_existing:
                print(
                    f"Notice: pre-Closure-R4 checkpoint(s) for {group_key} were trained "
                    f"with the invalid swapped-axis physics operator and are blocked: {invalid_existing}"
                )
            print(f"Warning: No Closure-R4-valid checkpoint found for {group_key}; skipping")
            continue

        checkpoint_training_protocol = "pre-R4-field-only" if is_legacy else PHYSICS_PROTOCOL
        print(
            f"\n--- Evaluating Physics Ablation [{group_key}]: {group_info['title']} "
            f"({ckpt_path}) [evaluation={PHYSICS_PROTOCOL}; checkpoint={checkpoint_training_protocol}] ---"
        )
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt_data.get("config", {})

        pred_mode = cfg.get("prediction_mode", ckpt_data.get("prediction_mode", "direct"))
        use_cond = cfg.get("use_condition", ckpt_data.get("use_condition", True))
        emb_dim = cfg.get("embed_dim", 256)
        d_depth = cfg.get("depth", 6)
        n_heads = cfg.get("num_heads", 8)
        print(f"  [Checkpoint Config] prediction_mode='{pred_mode}', use_condition={use_cond}, embed_dim={emb_dim}, depth={d_depth}, num_heads={n_heads}")

        encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
        decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
        transformer = LatentSTTransformer(
            latent_channels=64,
            embed_dim=emb_dim,
            cond_dim=128,
            depth=d_depth,
            num_heads=n_heads,
            history_length=4,
            prediction_mode=pred_mode,
        )
        forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder).to(device)

        if "model_state_dict" in ckpt_data:
            forecaster.load_state_dict(ckpt_data["model_state_dict"])
        elif "encoder_state_dict" in ckpt_data and "transformer_state_dict" in ckpt_data:
            forecaster.encoder.load_state_dict(ckpt_data["encoder_state_dict"])
            forecaster.transformer.load_state_dict(ckpt_data["transformer_state_dict"])
            forecaster.decoder.load_state_dict(ckpt_data["decoder_state_dict"])

        eval_res = evaluate_single_ablation(
            forecaster, test_loader, device, max_horizon=max_horizon, normalizer=normalizer, use_condition=use_cond
        )
        eval_res["__metadata__"] = {
            "checkpoint": ckpt_path,
            "evaluation_protocol": PHYSICS_PROTOCOL,
            "checkpoint_training_protocol": checkpoint_training_protocol,
            "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
            "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
            "is_legacy": is_legacy,
            "prediction_mode": pred_mode,
            "use_condition": use_cond,
        }
        results[group_key] = eval_res

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
    print("PHYSICS ABLATION BENCHMARK SUMMARY (E0 - E4)")
    print("=" * 98)
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
    parser = argparse.ArgumentParser(description="Evaluate 5-group physics loss ablation benchmark.")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--split_type", type=str, default="grouped")
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="outputs/metrics/closure_r4_physics_ablation.json")
    parser.add_argument("--downsample_factor", type=int, default=2)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--allow_legacy_checkpoints", action="store_true", default=False, help="Allow opt-in re-evaluation of pre-R4 field-only checkpoints; poisoned physics-loss checkpoints remain blocked")
    args = parser.parse_args()

    run_physics_ablation_eval(
        data_dir=args.data_dir,
        output_file=args.output_file,
        split_type=args.split_type,
        split_file=args.split_file,
        downsample_factor=args.downsample_factor,
        normalize=args.normalize,
        max_horizon=args.horizon,
        allow_legacy_checkpoints=args.allow_legacy_checkpoints,
    )
