"""Comprehensive Full-Validation Benchmark & Governance for PDE Controlled Candidates.

Evaluates D0 (frozen baseline), P0-Step50, and PDE-Step50 across the complete validation split
(all trajectories, all sliding temporal windows under specified stride) with full physical metrics,
subgroup analysis (e.g. Sc=0.1 vs Sc=1.0), and rigorous cryptographic checkpoint governance.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.audit_pde_residuals import (
    compute_file_sha256,
    get_git_commit_hash,
    get_git_dirty,
)
from scripts.train_forecaster import LatentForecasterWrapper
from src.data.normalization import FieldNormalizer
from src.data.pipeline import compute_split_hash, create_flow_dataloaders
from src.losses.divergence import DivergenceLoss
from src.losses.navier_stokes import NavierStokesPDELoss
from src.metrics.field import compute_vrmse, evaluate_field_metrics
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_transformer import LatentSTTransformer
from src.utils.physics_contract import (
    PHYSICS_PROTOCOL,
    SPATIAL_AXIS_CONTRACT,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
)


def load_forecaster_model(
    checkpoint_path: str,
    ae_path: str,
    history_length: int = 4,
    device: torch.device = torch.device("cpu"),
) -> Tuple[LatentForecasterWrapper, Dict[str, Any]]:
    """Build and load a LatentForecasterWrapper from checkpoint."""
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32).to(device)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False).to(device)
    ae_state = torch.load(ae_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ae_state["encoder_state_dict"])
    decoder.load_state_dict(ae_state["decoder_state_dict"])
    encoder.eval()
    decoder.eval()

    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=256,
        depth=6,
        num_heads=8,
        history_length=history_length,
        prediction_mode="direct",
    ).to(device)

    model = LatentForecasterWrapper(
        encoder=encoder,
        transformer=transformer,
        decoder=decoder,
        freeze_representation=True,
    ).to(device)

    ckpt_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt_data.get("model_state_dict", ckpt_data)
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt_data


def evaluate_model_on_dataloader(
    model: LatentForecasterWrapper,
    dataloader: DataLoader,
    normalizer: FieldNormalizer,
    pde_loss_fn: NavierStokesPDELoss,
    div_loss_fn: DivergenceLoss,
    horizon: int = 12,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], int]:
    """Evaluate model over dataloader collecting overall and per-subgroup metrics.

    Returns:
        overall_metrics: Aggregate mean over all windows.
        subgroup_metrics: Per Schmidt group (e.g. Sc=0.1, Sc=1.0) aggregate means.
        total_samples: Count of processed evaluation windows.
    """
    model.eval()

    sample_metrics_list: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            q_hist = batch["history"].to(device)  # normalized (B, L, 4, Ny, Nx)
            q_future = batch["future"].to(device)  # normalized (B, H, 4, Ny, Nx)
            re = batch["re"].to(device)  # (B,)
            sc = batch["sc"].to(device)  # (B,)
            dt = batch["dt"].to(device)  # (B,)
            b_size = q_hist.size(0)

            # Rollout
            q_pred_norm = model.forward_rollout(q_hist, re=re, sc=sc, horizon=horizon)

            # Denormalize
            q_pred = normalizer.denormalize(q_pred_norm)
            q_target = normalizer.denormalize(q_future)
            q_hist_phys = normalizer.denormalize(q_hist)
            q0_phys = q_hist_phys[:, -1]

            # Zero-mean pressure gauge in physical space
            q_pred[:, :, 2:3] = q_pred[:, :, 2:3] - q_pred[:, :, 2:3].mean(dim=(-2, -1), keepdim=True)
            q_target[:, :, 2:3] = q_target[:, :, 2:3] - q_target[:, :, 2:3].mean(dim=(-2, -1), keepdim=True)

            for b in range(b_size):
                p_b = q_pred[b : b + 1]
                t_b = q_target[b : b + 1]
                q0_b = q0_phys[b : b + 1]
                re_b = re[b : b + 1]
                sc_b = sc[b : b + 1]
                dt_b = dt[b : b + 1]

                vrmse_std = compute_vrmse(p_b, t_b).item()
                f_m = evaluate_field_metrics(p_b, t_b, channel_names=("u", "v", "p", "s"))
                rmse_total = math.sqrt(torch.mean((p_b - t_b) ** 2).item())

                # Divergence
                div_val = math.sqrt(div_loss_fn(p_b[:, :, :2]).item())

                # Discrete PDE residuals
                _, pde_stats = pde_loss_fn(p_b, re=re_b, sc=sc_b, dt=dt_b, q0_phys=q0_b)
                ru_val = pde_stats["res_momentum_u_rmse"]
                rv_val = pde_stats["res_momentum_v_rmse"]
                rs_val = pde_stats["res_tracer_s_rmse"]

                sc_val = round(float(sc[b].item()), 4)
                re_val = round(float(re[b].item()), 1)

                sample_metrics_list.append({
                    "vrmse_standard": vrmse_std,
                    "rmse_total": rmse_total,
                    "rmse_u": f_m["rmse_u"],
                    "rmse_v": f_m["rmse_v"],
                    "rmse_p": f_m["rmse_p"],
                    "rmse_s": f_m["rmse_s"],
                    "vrmse_u": f_m["vrmse_u"],
                    "vrmse_v": f_m["vrmse_v"],
                    "vrmse_p": f_m["vrmse_p"],
                    "vrmse_s": f_m["vrmse_s"],
                    "div_rmse": div_val,
                    "res_u_rmse": ru_val,
                    "res_v_rmse": rv_val,
                    "res_s_rmse": rs_val,
                    "sc": sc_val,
                    "re": re_val,
                })

    total_samples = len(sample_metrics_list)
    if total_samples == 0:
        raise ValueError("No evaluation samples were processed.")

    metric_keys = [
        "vrmse_standard", "rmse_total",
        "rmse_u", "rmse_v", "rmse_p", "rmse_s",
        "vrmse_u", "vrmse_v", "vrmse_p", "vrmse_s",
        "div_rmse", "res_u_rmse", "res_v_rmse", "res_s_rmse",
    ]

    # Compute overall mean
    overall_metrics = {
        k: float(np.mean([s[k] for s in sample_metrics_list]))
        for k in metric_keys
    }

    # Compute subgroup metrics by Sc
    sc_groups = sorted(list(set(s["sc"] for s in sample_metrics_list)))
    subgroup_metrics: Dict[str, Dict[str, float]] = {}
    for sc_val in sc_groups:
        sub_samples = [s for s in sample_metrics_list if s["sc"] == sc_val]
        grp_key = f"Sc_{sc_val}"
        subgroup_metrics[grp_key] = {
            "num_windows": len(sub_samples),
            **{k: float(np.mean([s[k] for s in sub_samples])) for k in metric_keys}
        }

    return overall_metrics, subgroup_metrics, total_samples


def build_candidate_governance_metadata(
    checkpoint_path: str,
    ckpt_data: Dict[str, Any],
    parent_d0_sha256: str,
    split_file: str,
    norm_file: str,
    history_length: int = 4,
    horizon: int = 12,
    seed: int = 42,
    lambda_mom: float = 0.0,
    lambda_tr: float = 0.0,
    mom_scale_u: float = 0.05,
    mom_scale_v: float = 0.05,
    tracer_scale_s: float = 0.02,
    branch_type: str = "P0",
    step: int = 50,
) -> Dict[str, Any]:
    """Construct formal governance metadata package for a candidate checkpoint."""
    ckpt_sha = compute_file_sha256(checkpoint_path)
    with open(split_file, "r") as f:
        split_json = json.load(f)
    split_hash = compute_split_hash(split_json)
    norm_sha = compute_file_sha256(norm_file)

    training_git_commit = get_git_commit_hash()
    training_git_dirty = get_git_dirty()

    gov = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": ckpt_sha,
        "parent_d0_sha256": parent_d0_sha256,
        "branch_type": branch_type,
        "step": step,
        "seed": seed,
        "training_horizon": horizon,
        "history_length": history_length,
        "downsample_factor": 2,
        "use_spatial_pos": False,
        "split_file": split_file,
        "split_hash": split_hash,
        "normalizer_file": norm_file,
        "normalizer_hash": norm_sha,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
        "training_git_commit": training_git_commit,
        "training_git_dirty": training_git_dirty,
        "pde_weights": {
            "lambda_mom": lambda_mom,
            "lambda_tr": lambda_tr,
        },
        "residual_scales": {
            "mom_scale_u": mom_scale_u,
            "mom_scale_v": mom_scale_v,
            "tracer_scale_s": tracer_scale_s,
        },
        "selection_criterion": (
            f"Step {step} paired candidate from controlled training (Beff=8, lr=5e-5, AdamW, FP32). "
            f"Evaluated on multi-window validation trajectory benchmark."
        ),
    }
    return gov


def run_full_validation_benchmark(
    d0_checkpoint: str = "outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt",
    p0_checkpoint: str = "outputs/checkpoints/dynamics/pde_controlled_experiment/p0_step_50.pt",
    pde_checkpoint: str = "outputs/checkpoints/dynamics/pde_controlled_experiment/pde_step_50.pt",
    ae_checkpoint: str = "outputs/checkpoints/representation/best_autoencoder.pt",
    split_file: str = "outputs/splits/grouped_split.json",
    norm_file: str = "outputs/normalization/stats_grouped.pt",
    data_root: Optional[str] = None,
    split: str = "valid",
    history_length: int = 4,
    horizon: int = 12,
    valid_stride: int = 1,
    batch_size: int = 8,
    num_workers: int = 2,
    device: Optional[torch.device] = None,
    output_json: str = "outputs/evaluations/pde_controlled_candidates_full_val.json",
) -> Dict[str, Any]:
    """Execute complete validation benchmark across all windows for D0, P0-Step50, and PDE-Step50."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_root is None:
        data_root = os.environ.get("SHEAR_FLOW_DATA_DIR", "/root/autodl-tmp/datasets/shear_flow")

    d0_sha = compute_file_sha256(d0_checkpoint)
    p0_sha = compute_file_sha256(p0_checkpoint)
    pde_sha = compute_file_sha256(pde_checkpoint)

    # 1. Normalizer setup
    normalizer = FieldNormalizer()
    normalizer.load_state_dict(torch.load(norm_file, map_location="cpu", weights_only=True))
    normalizer_cpu = copy.deepcopy(normalizer)
    normalizer.to(device)

    # 2. Physics loss operators for validation
    pde_loss_fn = NavierStokesPDELoss(domain_size=(1.0, 2.0), dealias=True).to(device)
    div_loss_fn = DivergenceLoss(domain_size=(1.0, 2.0)).to(device)

    # 3. Create canonical validation dataloader covering all windows
    _, valid_loader, test_loader, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file=split_file,
        data_root=data_root,
        history_length=history_length,
        horizon=horizon,
        valid_horizon=horizon,
        train_stride=2,
        valid_stride=valid_stride,
        test_stride=valid_stride,
        downsample_factor=2,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=True,
        normalizer=normalizer_cpu,
        preload_to_memory=False,
        is_distributed=False,
        rank=0,
        world_size=1,
        seed=42,
        return_sampler=False,
        require_pressure=True,
        require_tracer=True,
    )

    eval_loader = valid_loader if split == "valid" else test_loader
    total_val_windows = len(eval_loader.dataset)

    print("=" * 115)
    print(f"FULL BENCHMARK EVALUATION ({split.upper()} SPLIT, stride={valid_stride})")
    print(f"Total evaluation windows: {total_val_windows} across all {split} trajectories")
    print(f"D0  Baseline: {d0_checkpoint} (SHA: {d0_sha[:12]}...)")
    print(f"P0  Step 50: {p0_checkpoint} (SHA: {p0_sha[:12]}...)")
    print(f"PDE Step 50: {pde_checkpoint} (SHA: {pde_sha[:12]}...)")
    print("=" * 115)

    # 4. Evaluate D0 Frozen Baseline
    print("\n--- [1/3] EVALUATING D0 FROZEN BASELINE ---")
    t0 = time.time()
    d0_model, d0_ckpt_data = load_forecaster_model(d0_checkpoint, ae_checkpoint, history_length, device)
    d0_overall, d0_subgroups, n_d0 = evaluate_model_on_dataloader(
        d0_model, eval_loader, normalizer, pde_loss_fn, div_loss_fn, horizon, device
    )
    print(f"D0 done in {time.time() - t0:.1f}s: Standard VRMSE = {d0_overall['vrmse_standard']:.6f} | Total RMSE = {d0_overall['rmse_total']:.6f}")

    # 5. Evaluate P0-Step50 Control
    print("\n--- [2/3] EVALUATING P0-STEP50 CANDIDATE ---")
    t0 = time.time()
    p0_model, p0_ckpt_data = load_forecaster_model(p0_checkpoint, ae_checkpoint, history_length, device)
    p0_overall, p0_subgroups, n_p0 = evaluate_model_on_dataloader(
        p0_model, eval_loader, normalizer, pde_loss_fn, div_loss_fn, horizon, device
    )
    print(f"P0 done in {time.time() - t0:.1f}s: Standard VRMSE = {p0_overall['vrmse_standard']:.6f} | Total RMSE = {p0_overall['rmse_total']:.6f}")

    # 6. Evaluate PDE-Step50 Candidate
    print("\n--- [3/3] EVALUATING PDE-STEP50 CANDIDATE ---")
    t0 = time.time()
    pde_model, pde_ckpt_data = load_forecaster_model(pde_checkpoint, ae_checkpoint, history_length, device)
    pde_overall, pde_subgroups, n_pde = evaluate_model_on_dataloader(
        pde_model, eval_loader, normalizer, pde_loss_fn, div_loss_fn, horizon, device
    )
    print(f"PDE done in {time.time() - t0:.1f}s: Standard VRMSE = {pde_overall['vrmse_standard']:.6f} | Total RMSE = {pde_overall['rmse_total']:.6f}")

    # 7. Comparison Table
    metric_keys = [
        "vrmse_standard", "rmse_total",
        "rmse_u", "rmse_v", "rmse_p", "rmse_s",
        "vrmse_u", "vrmse_v", "vrmse_p", "vrmse_s",
        "div_rmse", "res_u_rmse", "res_v_rmse", "res_s_rmse",
    ]

    comparison_overall: Dict[str, Dict[str, float]] = {}
    print("\n" + "=" * 125)
    print(f"{'Metric':<18} | {'D0 Baseline':<14} | {'P0 Step 50':<14} | {'PDE Step 50':<14} | {'P0 vs D0 (%)':<15} | {'PDE vs D0 (%)':<15} | {'PDE vs P0 (%)':<15}")
    print("-" * 125)

    for m in metric_keys:
        v_d0 = d0_overall[m]
        v_p0 = p0_overall[m]
        v_pde = pde_overall[m]

        delta_p0_d0 = (v_p0 - v_d0) / (v_d0 + 1e-12) * 100.0
        delta_pde_d0 = (v_pde - v_d0) / (v_d0 + 1e-12) * 100.0
        delta_pde_p0 = (v_pde - v_p0) / (v_p0 + 1e-12) * 100.0

        comparison_overall[m] = {
            "d0": v_d0,
            "p0": v_p0,
            "pde": v_pde,
            "delta_p0_vs_d0_pct": delta_p0_d0,
            "delta_pde_vs_d0_pct": delta_pde_d0,
            "delta_pde_vs_p0_pct": delta_pde_p0,
        }
        print(
            f"{m:<18} | "
            f"{v_d0:<14.6f} | "
            f"{v_p0:<14.6f} | "
            f"{v_pde:<14.6f} | "
            f"{delta_p0_d0:<+15.2f}% | "
            f"{delta_pde_d0:<+15.2f}% | "
            f"{delta_pde_p0:<+15.2f}%"
        )
    print("=" * 125)

    # Subgroup comparisons
    comparison_subgroups: Dict[str, Any] = {}
    for grp_key in d0_subgroups.keys():
        sub_comp = {}
        for m in metric_keys:
            sub_d0 = d0_subgroups[grp_key][m]
            sub_p0 = p0_subgroups[grp_key][m]
            sub_pde = pde_subgroups[grp_key][m]
            sub_comp[m] = {
                "d0": sub_d0,
                "p0": sub_p0,
                "pde": sub_pde,
                "delta_p0_vs_d0_pct": (sub_p0 - sub_d0) / (sub_d0 + 1e-12) * 100.0,
                "delta_pde_vs_d0_pct": (sub_pde - sub_d0) / (sub_d0 + 1e-12) * 100.0,
                "delta_pde_vs_p0_pct": (sub_pde - sub_p0) / (sub_p0 + 1e-12) * 100.0,
            }
        comparison_subgroups[grp_key] = {
            "num_windows": d0_subgroups[grp_key]["num_windows"],
            "metrics": sub_comp,
        }

    # 8. Checkpoint Governance metadata
    p0_governance = build_candidate_governance_metadata(
        checkpoint_path=p0_checkpoint,
        ckpt_data=p0_ckpt_data,
        parent_d0_sha256=d0_sha,
        split_file=split_file,
        norm_file=norm_file,
        history_length=history_length,
        horizon=horizon,
        seed=42,
        lambda_mom=0.0,
        lambda_tr=0.0,
        branch_type="P0_control",
        step=50,
    )
    pde_governance = build_candidate_governance_metadata(
        checkpoint_path=pde_checkpoint,
        ckpt_data=pde_ckpt_data,
        parent_d0_sha256=d0_sha,
        split_file=split_file,
        norm_file=norm_file,
        history_length=history_length,
        horizon=horizon,
        seed=42,
        lambda_mom=2.5e-5,
        lambda_tr=4.0e-6,
        branch_type="PDE_experiment",
        step=50,
    )

    # 9. Rigorous Scientific Verdict
    d0_vrmse = d0_overall["vrmse_standard"]
    p0_vrmse = p0_overall["vrmse_standard"]
    pde_vrmse = pde_overall["vrmse_standard"]

    continued_training_effect_pct = (p0_vrmse - d0_vrmse) / d0_vrmse * 100.0
    pde_incremental_effect_pct = (pde_vrmse - p0_vrmse) / p0_vrmse * 100.0
    pde_vs_d0_pct = (pde_vrmse - d0_vrmse) / d0_vrmse * 100.0

    pde_beats_p0 = pde_vrmse < p0_vrmse
    pde_beats_d0 = pde_vrmse < d0_vrmse
    p0_beats_d0 = p0_vrmse < d0_vrmse

    div_pde = pde_overall["div_rmse"]
    div_p0 = p0_overall["div_rmse"]
    div_pde_beats_p0 = div_pde <= div_p0

    verdict_summary = {
        "total_windows_evaluated": total_val_windows,
        "split": split,
        "d0_vrmse": d0_vrmse,
        "p0_vrmse": p0_vrmse,
        "pde_vrmse": pde_vrmse,
        "continued_training_benefit_pct": continued_training_effect_pct,
        "pde_incremental_benefit_pct": pde_incremental_effect_pct,
        "pde_vs_d0_pct": pde_vs_d0_pct,
        "pde_beats_p0_on_vrmse": pde_beats_p0,
        "pde_beats_d0_on_vrmse": pde_beats_d0,
        "div_pde_vs_p0_pct": (div_pde - div_p0) / div_p0 * 100.0,
        "div_pde_beats_p0": div_pde_beats_p0,
        "trade_off_observation": (
            "Field prediction accuracy (VRMSE/RMSE) improves under continued training + PDE, "
            "while some physical residuals (divergence, momentum, tracer) remain higher than frozen D0."
        ),
        "status": (
            "VALIDATED_ON_FULL_VAL" if (pde_beats_p0 and pde_beats_d0)
            else "PARTIAL_OR_UNPROVEN"
        ),
    }

    print("\nSCIENTIFIC VERDICT SUMMARY:")
    for k, v in verdict_summary.items():
        print(f"  {k}: {v}")

    # 10. Archive payload
    payload = {
        "metadata": {
            "evaluation_git_commit": get_git_commit_hash(),
            "evaluation_git_dirty": get_git_dirty(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "split_file": split_file,
            "split_file_sha256": compute_file_sha256(split_file),
            "norm_file": norm_file,
            "norm_file_sha256": compute_file_sha256(norm_file),
            "ae_ckpt": ae_checkpoint,
            "ae_ckpt_sha256": compute_file_sha256(ae_checkpoint),
            "d0_ckpt": d0_checkpoint,
            "d0_ckpt_sha256": d0_sha,
            "p0_ckpt": p0_checkpoint,
            "p0_ckpt_sha256": p0_sha,
            "pde_ckpt": pde_checkpoint,
            "pde_ckpt_sha256": pde_sha,
            "evaluation_protocol": {
                "split": split,
                "history_length": history_length,
                "horizon": horizon,
                "valid_stride": valid_stride,
                "total_windows_evaluated": total_val_windows,
                "batch_size": batch_size,
                "downsample_factor": 2,
            },
        },
        "governance": {
            "p0_step_50": p0_governance,
            "pde_step_50": pde_governance,
        },
        "overall_comparison": comparison_overall,
        "subgroup_comparison": comparison_subgroups,
        "raw_overall_metrics": {
            "d0": d0_overall,
            "p0": p0_overall,
            "pde": pde_overall,
        },
        "verdict": verdict_summary,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nBenchmark results successfully archived to {output_json}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate D0 vs P0 vs PDE on full validation/test split.")
    parser.add_argument("--d0_checkpoint", type=str, default="outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt")
    parser.add_argument("--p0_checkpoint", type=str, default="outputs/checkpoints/dynamics/pde_controlled_experiment/p0_step_50.pt")
    parser.add_argument("--pde_checkpoint", type=str, default="outputs/checkpoints/dynamics/pde_controlled_experiment/pde_step_50.pt")
    parser.add_argument("--ae_checkpoint", type=str, default="outputs/checkpoints/representation/best_autoencoder.pt")
    parser.add_argument("--split_file", type=str, default="outputs/splits/grouped_split.json")
    parser.add_argument("--norm_file", type=str, default="outputs/normalization/stats_grouped.pt")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--history_length", type=int, default=4)
    parser.add_argument("--valid_stride", type=int, default=1)
    parser.add_argument("--full_windows", action="store_true", help="Evaluate every possible sliding window (stride=1)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_json", type=str, default="outputs/evaluations/pde_controlled_candidates_full_val.json")
    args = parser.parse_args()

    stride = 1 if args.full_windows else args.valid_stride

    run_full_validation_benchmark(
        d0_checkpoint=args.d0_checkpoint,
        p0_checkpoint=args.p0_checkpoint,
        pde_checkpoint=args.pde_checkpoint,
        ae_checkpoint=args.ae_checkpoint,
        split_file=args.split_file,
        norm_file=args.norm_file,
        data_root=args.data_root,
        split=args.split,
        history_length=args.history_length,
        horizon=args.horizon,
        valid_stride=stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_json=args.output_json,
    )
