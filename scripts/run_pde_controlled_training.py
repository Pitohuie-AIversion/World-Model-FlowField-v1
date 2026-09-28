"""Controlled continued training experiment for Navier-Stokes & Tracer PDE residual losses.

Strict Scientific & Engineering Governance:
1. Strict Configuration Parity with Approved H12 Baseline:
   - Training horizon H=12, history length L=4 (total window 16).
   - Effective batch size Beff=8 (batch_size=2, grad_accum_steps=4 per update step on 1 GPU).
   - Downsample factor=2 (128x256 spatial grid).
   - Constant learning rate lr=5e-5, AdamW, weight_decay=1e-4, grad clip max_norm=1.0.
   - Baseline loss terms: L_P0 = L_MSE + 0.01 * L_div + 0.05 * L_vort (field_loss_space="normalized").
   - PDE residual scales: mom_scale_u=0.05, mom_scale_v=0.05, tracer_scale_s=0.02, pde_dealias=True.
   - PDE candidate weights: lambda_mom=2.5e-5, lambda_tr=4.0e-6.
2. Fair Paired Control Setup:
   - Identical initialization: both P0 and PDE start from deepcopied D0 baseline weights.
   - Identical data trajectory: pre-sampled stream of batches shared identically between P0 and PDE.
   - P0 Group: updates with L_P0 only.
   - PDE Group: updates with L_P0 + lambda_mom * L_mom + lambda_tr * L_tr.
3. Multi-Node Checkpointing & Fixed Validation Window:
   - Evaluated at Step 0 (frozen D0 benchmark), Step 20, Step 50, Step 100.
   - Fixed validation window evaluated with canonical compute_vrmse() and evaluate_field_metrics().
4. Non-Destructive Storage & Fail-Safe Selection:
   - Original D0/G0/G1 checkpoints are never modified or overwritten.
   - Candidate checkpoints saved to outputs/checkpoints/dynamics/pde_controlled_experiment/.
   - Independent report archived to outputs/evaluations/pde_controlled_training_h12.json.
   - Fail-safe selection rule: if no candidate outperforms Step 0 D0 baseline, explicitly records
     that no candidate surpassed D0 and retains D0 as the recommended deployment checkpoint.
"""

import copy
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn

from scripts.audit_pde_residuals import (
    compute_file_sha256,
    get_git_commit_hash,
    get_git_dirty,
    precheck_audit_manifest_and_samples,
)
from scripts.probe_pde_gradient_scales import (
    evaluate_on_fixed_validation_window,
)
from scripts.train_forecaster import (
    LatentForecasterWrapper,
    _compute_batch_loss,
)
from src.data.normalization import FieldNormalizer
from src.data.pipeline import create_flow_dataloaders
from src.losses.divergence import DivergenceLoss
from src.losses.navier_stokes import (
    NavierStokesMomentumResidualLoss,
    NavierStokesPDELoss,
    TracerAdvectionDiffusionResidualLoss,
)
from src.losses.vorticity import VorticityLoss
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_transformer import LatentSTTransformer


def run_pde_controlled_training(
    split_file: str = "outputs/splits/grouped_split.json",
    norm_file: str = "outputs/normalization/stats_grouped.pt",
    ae_ckpt: str = "outputs/checkpoints/representation/best_autoencoder.pt",
    d0_ckpt: str = "outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt",
    data_root: Optional[str] = None,
    total_steps: int = 100,
    eval_steps: Optional[List[int]] = None,
    history_length: int = 4,
    horizon: int = 12,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    mom_scale_u: float = 0.05,
    mom_scale_v: float = 0.05,
    tracer_scale_s: float = 0.02,
    pde_dealias: bool = True,
    lambda_div: float = 0.01,
    lambda_vort: float = 0.05,
    lambda_mom: float = 2.5e-5,
    lambda_tr: float = 4.0e-6,
    seed: int = 42,
    num_workers: int = 2,
    output_ckpt_dir: str = "outputs/checkpoints/dynamics/pde_controlled_experiment",
    output_json: str = "outputs/evaluations/pde_controlled_training_h12.json",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute fair paired controlled training comparing Step 0 D0, P0, and PDE across evaluation nodes."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_root is None:
        data_root = os.environ.get("SHEAR_FLOW_DATA_DIR", "/root/autodl-tmp/datasets/shear_flow")

    if eval_steps is None:
        eval_steps = [0, 20, 50, 100]
    eval_steps = sorted(list(set(eval_steps)))

    effective_batch_size = batch_size * grad_accum_steps

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Precheck validation window contracts (H=12, L=4 -> total_window=16)
    total_window = history_length + horizon
    validated_samples = precheck_audit_manifest_and_samples(
        split_file=split_file,
        data_root=data_root,
        total_window=total_window,
        num_samples=4,
    )

    # 2. Load Normalizer (CPU copy for workers, CUDA for models)
    normalizer = FieldNormalizer()
    normalizer_state = torch.load(norm_file, map_location="cpu", weights_only=True)
    normalizer.load_state_dict(normalizer_state)
    normalizer_cpu = copy.deepcopy(normalizer)
    normalizer.to(device)

    # 3. Load Representation and Dynamics Architecture
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32).to(device)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False).to(device)
    ae_state = torch.load(ae_ckpt, map_location=device, weights_only=False)
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

    base_model = LatentForecasterWrapper(
        encoder=encoder,
        transformer=transformer,
        decoder=decoder,
        freeze_representation=True,
    ).to(device)

    d0_state = torch.load(d0_ckpt, map_location=device, weights_only=False)
    base_model.load_state_dict(d0_state["model_state_dict"])

    # 4. Construct Loss Functions matching exact H12 baseline and PDE residual scales
    mom_loss_fn = NavierStokesMomentumResidualLoss(
        domain_size=(1.0, 2.0),
        scale_u=mom_scale_u,
        scale_v=mom_scale_v,
        dealias=pde_dealias,
    ).to(device)
    tracer_loss_fn = TracerAdvectionDiffusionResidualLoss(
        domain_size=(1.0, 2.0),
        scale_s=tracer_scale_s,
        dealias=pde_dealias,
    ).to(device)
    pde_loss_fn = NavierStokesPDELoss(domain_size=(1.0, 2.0), dealias=pde_dealias).to(device)
    rollout_loss_fn = nn.MSELoss()
    div_loss_fn = DivergenceLoss(domain_size=(1.0, 2.0)).to(device) if lambda_div > 0 else None
    vort_loss_fn = VorticityLoss(domain_size=(1.0, 2.0)).to(device) if lambda_vort > 0 else None

    # 5. Dataloader for H=12 training
    train_loader, _, _, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file=split_file,
        data_root=data_root,
        history_length=history_length,
        horizon=horizon,
        valid_horizon=horizon,
        train_stride=2,
        valid_stride=8,
        downsample_factor=2,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=True,
        normalizer=normalizer_cpu,
        preload_to_memory=False,
        is_distributed=False,
        rank=0,
        world_size=1,
        seed=seed,
        return_sampler=False,
        require_pressure=True,
        require_tracer=True,
    )

    print("=" * 110)
    print("CONTROLLED TRAINING EXPERIMENT: H=12, Beff=8, Step 0 vs. P0 vs. PDE")
    print(f"Alignment: horizon={horizon}, batch_size={batch_size}, grad_accum={grad_accum_steps} (Beff={effective_batch_size})")
    print(f"Optimizer: lr={lr}, weight_decay={weight_decay}, grad_clip={grad_clip_norm}")
    print(f"Loss parameters: lambda_div={lambda_div}, lambda_vort={lambda_vort}")
    print(f"PDE residual parameters: scale_u={mom_scale_u}, scale_v={mom_scale_v}, scale_s={tracer_scale_s}")
    print(f"PDE weights: lambda_mom={lambda_mom}, lambda_tr={lambda_tr}")
    print(f"Evaluation checkpoint nodes: {eval_steps}")
    print("=" * 110)

    # 6. Step 0 Baseline Evaluation (Frozen D0 Benchmark)
    print("\n--- NODE 0: EVALUATING FROZEN D0 BASELINE (STEP 0) ---")
    metrics_step_0 = evaluate_on_fixed_validation_window(
        model=base_model,
        validated_samples=validated_samples,
        normalizer=normalizer,
        pde_loss_fn=pde_loss_fn,
        history_length=history_length,
        horizon=horizon,
        device=device,
    )
    print(f"Step 0 D0 Baseline Standard VRMSE: {metrics_step_0['vrmse_standard']:.6e} | Total RMSE: {metrics_step_0['rmse_total']:.6e}")

    # 7. Collect microbatches for paired deterministic training
    total_microbatches_needed = total_steps * grad_accum_steps
    print(f"\nCollecting {total_microbatches_needed} microbatches for strictly identical paired execution...")
    shared_microbatches: List[Dict[str, Any]] = []
    data_iter = iter(train_loader)
    for _ in range(total_microbatches_needed):
        try:
            b = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            b = next(data_iter)
        shared_microbatches.append(b)

    os.makedirs(output_ckpt_dir, exist_ok=True)

    # 8. P0 Control Group: L_P0 only (lambda_mom=0, lambda_tr=0)
    print("\n" + "=" * 110)
    print(f"EXECUTING P0 CONTROL GROUP (L_existing only, lambda_mom=0, lambda_tr=0, {total_steps} steps)...")
    print("=" * 110)

    p0_model = copy.deepcopy(base_model)
    p0_model.train()
    p0_optimizer = torch.optim.AdamW(
        [p for p in p0_model.transformer.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    p0_eval_records: Dict[int, Dict[str, float]] = {0: metrics_step_0}
    p0_step_losses: List[float] = []

    t_p0_start = time.time()
    for step in range(1, total_steps + 1):
        p0_optimizer.zero_grad()
        step_accum_loss = 0.0

        for m_idx in range(grad_accum_steps):
            mb = shared_microbatches[(step - 1) * grad_accum_steps + m_idx]
            q_hist = mb["history"].to(device)
            q_future = mb["future"].to(device)
            re = mb["re"].to(device)
            sc = mb["sc"].to(device)
            dt = mb["dt"].to(device)

            pred_norm = p0_model.forward_rollout(q_hist, re=re, sc=sc, horizon=horizon)
            loss_mb = _compute_batch_loss(
                pred=pred_norm,
                q_future=q_future,
                normalizer=normalizer,
                field_loss_space="normalized",
                lambda_div=lambda_div,
                lambda_vort=lambda_vort,
                rollout_loss_fn=rollout_loss_fn,
                div_loss_fn=div_loss_fn,
                vort_loss_fn=vort_loss_fn,
                lambda_mom=0.0,
                lambda_tr=0.0,
            )
            if not torch.isfinite(loss_mb):
                raise FloatingPointError(f"Non-finite loss in P0 control at step {step}, microbatch {m_idx}: {loss_mb.item()}")

            loss_scaled = loss_mb / grad_accum_steps
            loss_scaled.backward()
            step_accum_loss += float(loss_mb.item()) / grad_accum_steps

        for p in p0_model.transformer.parameters():
            if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                raise FloatingPointError(f"Non-finite gradient in P0 control at step {step}!")

        torch.nn.utils.clip_grad_norm_(
            [p for p in p0_model.transformer.parameters() if p.requires_grad],
            max_norm=grad_clip_norm,
        )
        p0_optimizer.step()
        p0_step_losses.append(step_accum_loss)

        if step in eval_steps:
            metrics_eval = evaluate_on_fixed_validation_window(
                model=p0_model,
                validated_samples=validated_samples,
                normalizer=normalizer,
                pde_loss_fn=pde_loss_fn,
                history_length=history_length,
                horizon=horizon,
                device=device,
            )
            p0_eval_records[step] = metrics_eval
            ckpt_path = os.path.join(output_ckpt_dir, f"p0_step_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": p0_model.state_dict(),
                    "metrics": metrics_eval,
                    "loss_type": "P0_control",
                },
                ckpt_path,
            )
            print(f"P0 Step {step:3d}/{total_steps} | Loss={step_accum_loss:.5e} | VRMSE={metrics_eval['vrmse_standard']:.5e} | Checkpoint saved: {ckpt_path}")

    t_p0_elapsed = time.time() - t_p0_start
    print(f"P0 Control Group finished in {t_p0_elapsed:.1f} s.")

    # 9. PDE Experiment Group: L_P0 + PDE (lambda_mom, lambda_tr)
    print("\n" + "=" * 110)
    print(f"EXECUTING PDE EXPERIMENT GROUP (L_existing + PDE, lambda_mom={lambda_mom}, lambda_tr={lambda_tr}, {total_steps} steps)...")
    print("=" * 110)

    pde_model = copy.deepcopy(base_model)
    pde_model.train()
    pde_optimizer = torch.optim.AdamW(
        [p for p in pde_model.transformer.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    pde_eval_records: Dict[int, Dict[str, float]] = {0: metrics_step_0}
    pde_step_losses: List[float] = []

    t_pde_start = time.time()
    for step in range(1, total_steps + 1):
        pde_optimizer.zero_grad()
        step_accum_loss = 0.0

        for m_idx in range(grad_accum_steps):
            mb = shared_microbatches[(step - 1) * grad_accum_steps + m_idx]
            q_hist = mb["history"].to(device)
            q_future = mb["future"].to(device)
            re = mb["re"].to(device)
            sc = mb["sc"].to(device)
            dt = mb["dt"].to(device)

            pred_norm = pde_model.forward_rollout(q_hist, re=re, sc=sc, horizon=horizon)
            q_hist_phys = normalizer.denormalize(q_hist)
            q0_phys = q_hist_phys[:, -1]

            loss_mb = _compute_batch_loss(
                pred=pred_norm,
                q_future=q_future,
                normalizer=normalizer,
                field_loss_space="normalized",
                lambda_div=lambda_div,
                lambda_vort=lambda_vort,
                rollout_loss_fn=rollout_loss_fn,
                div_loss_fn=div_loss_fn,
                vort_loss_fn=vort_loss_fn,
                lambda_mom=lambda_mom,
                lambda_tr=lambda_tr,
                mom_loss_fn=mom_loss_fn,
                tracer_loss_fn=tracer_loss_fn,
                re=re,
                sc=sc,
                dt=dt,
                q0_phys=q0_phys,
            )
            if not torch.isfinite(loss_mb):
                raise FloatingPointError(f"Non-finite loss in PDE experiment at step {step}, microbatch {m_idx}: {loss_mb.item()}")

            loss_scaled = loss_mb / grad_accum_steps
            loss_scaled.backward()
            step_accum_loss += float(loss_mb.item()) / grad_accum_steps

        for p in pde_model.transformer.parameters():
            if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                raise FloatingPointError(f"Non-finite gradient in PDE experiment at step {step}!")

        torch.nn.utils.clip_grad_norm_(
            [p for p in pde_model.transformer.parameters() if p.requires_grad],
            max_norm=grad_clip_norm,
        )
        pde_optimizer.step()
        pde_step_losses.append(step_accum_loss)

        if step in eval_steps:
            metrics_eval = evaluate_on_fixed_validation_window(
                model=pde_model,
                validated_samples=validated_samples,
                normalizer=normalizer,
                pde_loss_fn=pde_loss_fn,
                history_length=history_length,
                horizon=horizon,
                device=device,
            )
            pde_eval_records[step] = metrics_eval
            ckpt_path = os.path.join(output_ckpt_dir, f"pde_step_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": pde_model.state_dict(),
                    "metrics": metrics_eval,
                    "loss_type": "PDE_experiment",
                },
                ckpt_path,
            )
            print(f"PDE Step {step:3d}/{total_steps} | Loss={step_accum_loss:.5e} | VRMSE={metrics_eval['vrmse_standard']:.5e} | Checkpoint saved: {ckpt_path}")

    t_pde_elapsed = time.time() - t_pde_start
    print(f"PDE Experiment Group finished in {t_pde_elapsed:.1f} s.")

    # 10. Multi-Node Trajectory Analysis & Decision Contract
    print("\n" + "=" * 125)
    print(f"{'Node':<8} | {'P0 VRMSE':<14} | {'PDE VRMSE':<14} | {'D0 Baseline':<14} | {'P0 vs D0 (%)':<16} | {'PDE vs D0 (%)':<16} | {'PDE vs P0 (%)':<16}")
    print("-" * 125)

    trajectory_summary: Dict[str, Any] = {}
    base_vrmse = metrics_step_0["vrmse_standard"]

    for s in eval_steps:
        m_p0 = p0_eval_records[s]
        m_pde = pde_eval_records[s]
        vrmse_p0 = m_p0["vrmse_standard"]
        vrmse_pde = m_pde["vrmse_standard"]

        delta_p0_d0 = (vrmse_p0 - base_vrmse) / base_vrmse * 100.0
        delta_pde_d0 = (vrmse_pde - base_vrmse) / base_vrmse * 100.0
        delta_pde_p0 = (vrmse_pde - vrmse_p0) / (vrmse_p0 + 1e-12) * 100.0

        trajectory_summary[str(s)] = {
            "step": s,
            "p0_vrmse": vrmse_p0,
            "pde_vrmse": vrmse_pde,
            "d0_baseline_vrmse": base_vrmse,
            "delta_p0_vs_d0_pct": delta_p0_d0,
            "delta_pde_vs_d0_pct": delta_pde_d0,
            "delta_pde_vs_p0_pct": delta_pde_p0,
            "p0_metrics": m_p0,
            "pde_metrics": m_pde,
        }

        print(
            f"Step {s:<3d} | "
            f"{vrmse_p0:<14.5e} | "
            f"{vrmse_pde:<14.5e} | "
            f"{base_vrmse:<14.5e} | "
            f"{delta_p0_d0:<+16.2f}% | "
            f"{delta_pde_d0:<+16.2f}% | "
            f"{delta_pde_p0:<+16.2f}%"
        )
    print("=" * 125)

    # 11. Fail-Safe Decision Rule
    # Find best candidate across all updated steps (excluding Step 0 itself)
    best_candidate_vrmse = float("inf")
    best_candidate_branch = None
    best_candidate_step = None
    best_candidate_path = None

    for s in eval_steps:
        if s == 0:
            continue
        v_p0 = p0_eval_records[s]["vrmse_standard"]
        v_pde = pde_eval_records[s]["vrmse_standard"]

        if v_p0 < best_candidate_vrmse:
            best_candidate_vrmse = v_p0
            best_candidate_branch = "P0"
            best_candidate_step = s
            best_candidate_path = os.path.join(output_ckpt_dir, f"p0_step_{s}.pt")

        if v_pde < best_candidate_vrmse:
            best_candidate_vrmse = v_pde
            best_candidate_branch = "PDE"
            best_candidate_step = s
            best_candidate_path = os.path.join(output_ckpt_dir, f"pde_step_{s}.pt")

    superior_to_d0 = best_candidate_vrmse < base_vrmse
    if superior_to_d0:
        decision_verdict = f"SUPERIOR_FOUND: {best_candidate_branch} step {best_candidate_step} surpassed D0 (VRMSE {best_candidate_vrmse:.5e} < {base_vrmse:.5e})"
        recommended_checkpoint = best_candidate_path
    else:
        decision_verdict = f"NO_SUPERIOR_CANDIDATE: best candidate ({best_candidate_branch} step {best_candidate_step} VRMSE {best_candidate_vrmse:.5e}) did not outperform frozen D0 (VRMSE {base_vrmse:.5e}). Retaining frozen D0."
        recommended_checkpoint = d0_ckpt

    print(f"\nDECISION CONTRACT VERDICT: {decision_verdict}")
    print(f"RECOMMENDED CHECKPOINT: {recommended_checkpoint}")

    # 12. Build Rigorous Provenance & Evaluation Report
    script_path = os.path.abspath(__file__)
    payload = {
        "metadata": {
            "git_commit": get_git_commit_hash(),
            "git_dirty": get_git_dirty(),
            "experiment_script": os.path.relpath(script_path, os.getcwd()) if script_path.startswith(os.getcwd()) else script_path,
            "experiment_script_sha256": compute_file_sha256(script_path),
            "split_file": split_file,
            "split_file_sha256": compute_file_sha256(split_file),
            "norm_file": norm_file,
            "norm_file_sha256": compute_file_sha256(norm_file),
            "ae_ckpt": ae_ckpt,
            "ae_ckpt_sha256": compute_file_sha256(ae_ckpt),
            "d0_ckpt": d0_ckpt,
            "d0_ckpt_sha256": compute_file_sha256(d0_ckpt),
            "alignment_configuration": {
                "training_horizon": horizon,
                "history_length": history_length,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": effective_batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "grad_clip_norm": grad_clip_norm,
                "seed": seed,
                "lambda_div": lambda_div,
                "lambda_vort": lambda_vort,
                "residual_scales": {
                    "mom_scale_u": mom_scale_u,
                    "mom_scale_v": mom_scale_v,
                    "tracer_scale_s": tracer_scale_s,
                },
                "pde_weights": {
                    "lambda_mom": lambda_mom,
                    "lambda_tr": lambda_tr,
                },
            },
            "eval_nodes": eval_steps,
            "total_steps": total_steps,
            "validation_window_manifest": validated_samples,
        },
        "step_0_d0_baseline": metrics_step_0,
        "evaluation_trajectory": trajectory_summary,
        "loss_histories": {
            "p0_step_losses": p0_step_losses,
            "pde_step_losses": pde_step_losses,
        },
        "decision_contract": {
            "superior_to_d0": superior_to_d0,
            "verdict": decision_verdict,
            "d0_baseline_vrmse": base_vrmse,
            "best_candidate_vrmse": best_candidate_vrmse if math.isfinite(best_candidate_vrmse) else None,
            "best_candidate_branch": best_candidate_branch,
            "best_candidate_step": best_candidate_step,
            "best_candidate_checkpoint": best_candidate_path,
            "recommended_checkpoint": recommended_checkpoint,
        },
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nControlled training results successfully archived to {output_json}")

    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run H=12 controlled continued training experiment.")
    parser.add_argument("--output_json", type=str, default="outputs/evaluations/pde_controlled_training_h12.json")
    parser.add_argument("--output_ckpt_dir", type=str, default="outputs/checkpoints/dynamics/pde_controlled_experiment")
    parser.add_argument("--total_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=str, default="0,20,50,100")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lambda_div", type=float, default=0.01)
    parser.add_argument("--lambda_vort", type=float, default=0.05)
    parser.add_argument("--mom_scale_u", type=float, default=0.05)
    parser.add_argument("--mom_scale_v", type=float, default=0.05)
    parser.add_argument("--tracer_scale_s", type=float, default=0.02)
    parser.add_argument("--lambda_mom", type=float, default=2.5e-5)
    parser.add_argument("--lambda_tr", type=float, default=4.0e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    steps_list = [int(s.strip()) for s in args.eval_steps.split(",") if s.strip()]

    run_pde_controlled_training(
        output_json=args.output_json,
        output_ckpt_dir=args.output_ckpt_dir,
        total_steps=args.total_steps,
        eval_steps=steps_list,
        horizon=args.horizon,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lambda_div=args.lambda_div,
        lambda_vort=args.lambda_vort,
        mom_scale_u=args.mom_scale_u,
        mom_scale_v=args.mom_scale_v,
        tracer_scale_s=args.tracer_scale_s,
        lambda_mom=args.lambda_mom,
        lambda_tr=args.lambda_tr,
        seed=args.seed,
        num_workers=args.num_workers,
    )
