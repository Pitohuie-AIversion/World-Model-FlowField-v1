"""Small-budget gradient norm and loss scale probe for PDE residual losses.

Strict Protocol Guarantees:
1. Q1: Verify whether PDE momentum and tracer losses genuinely participate in optimization:
   - Check gradients arriving at trainable latent dynamics parameters theta are strictly finite (no NaN/Inf).
   - Verify frozen encoder and decoder receive zero/None gradient updates.
2. Q2: Verify whether PDE loss scales overpower the existing task:
   - Compute decoupled gradients on the exact same batch and identical parameter set theta:
     ||grad_theta(L_existing)||_2 vs ||grad_theta(lambda_mom * L_mom)||_2 vs ||grad_theta(lambda_tr * L_tr)||_2.
   - Record gradient norm ratios and cosine directional alignment across multiple representative training batches.
3. Q3: Verify whether short updates preserve or break basic forecasting performance:
   - Evaluate field VRMSE/RMSE, divergence, and discrete PDE residuals on a fixed validation window
     before and after a small-budget (e.g. 20-step) fine-tuning update.
4. Non-destructive: Existing checkpoints (D0/G0/G1) are untouched; probe runs on isolated in-memory copies.
"""

import copy
import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.audit_pde_residuals import (
    compute_file_sha256,
    get_git_commit_hash,
    get_git_dirty,
    precheck_audit_manifest_and_samples,
)
from scripts.train_forecaster import (
    LatentForecasterWrapper,
    _compute_batch_loss,
    validate_forecaster_pde_config,
)
from src.data.normalization import FieldNormalizer
from src.data.pipeline import create_flow_dataloaders
from src.losses.navier_stokes import (
    NavierStokesMomentumResidualLoss,
    NavierStokesPDELoss,
    TracerAdvectionDiffusionResidualLoss,
)
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_transformer import LatentSTTransformer
from src.utils.fft_derivatives import compute_divergence


def compute_gradient_norm(parameters: List[torch.Tensor], grads: List[Optional[torch.Tensor]]) -> float:
    """Compute global L2 norm of gradient vectors across parameters."""
    total_norm_sq = 0.0
    for g in grads:
        if g is not None:
            if not torch.all(torch.isfinite(g)):
                return float("nan")
            total_norm_sq += float((g**2).sum().item())
    return math.sqrt(total_norm_sq)


def compute_gradient_cosine_similarity(
    grads1: List[Optional[torch.Tensor]],
    grads2: List[Optional[torch.Tensor]],
) -> float:
    """Compute cosine similarity between two gradient vectors in parameter space."""
    dot_product = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    for g1, g2 in zip(grads1, grads2):
        if g1 is not None and g2 is not None:
            dot_product += float((g1 * g2).sum().item())
            norm1_sq += float((g1**2).sum().item())
            norm2_sq += float((g2**2).sum().item())
    denom = math.sqrt(norm1_sq) * math.sqrt(norm2_sq)
    if denom <= 1e-12:
        return 0.0
    return float(dot_product / denom)


def compute_decoupled_batch_gradients(
    model: LatentForecasterWrapper,
    batch: Dict[str, Any],
    normalizer: FieldNormalizer,
    mom_loss_fn: NavierStokesMomentumResidualLoss,
    tracer_loss_fn: TracerAdvectionDiffusionResidualLoss,
    rollout_loss_fn: nn.Module,
    lambda_div: float = 0.0,
    lambda_vort: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Compute unweighted loss values and decoupled gradient vectors for existing, momentum, and tracer losses.

    This function executes on the identical forward pass and computes separate gradients with respect
    to the exact same trainable parameters theta = model.transformer.parameters().
    """
    q_hist = batch["history"].to(device)
    q_future = batch["future"].to(device)
    re = batch["re"].to(device)
    sc = batch["sc"].to(device)
    dt = batch["dt"].to(device)
    horizon = q_future.shape[1]

    # Model prediction
    pred_norm = model.forward_rollout(q_hist, re=re, sc=sc, horizon=horizon)
    pred_phys = normalizer.denormalize(pred_norm)
    target_phys = normalizer.denormalize(q_future)

    # Historical connection state q0 in physical space (frame L-1)
    q_hist_phys = normalizer.denormalize(q_hist)
    q0_phys = q_hist_phys[:, -1]

    # 1. Existing base loss (normalized MSE + divergence + vorticity)
    loss_existing = _compute_batch_loss(
        pred=pred_norm,
        q_future=q_future,
        normalizer=normalizer,
        field_loss_space="normalized",
        lambda_div=lambda_div,
        lambda_vort=lambda_vort,
        rollout_loss_fn=rollout_loss_fn,
        lambda_spec=0.0,
        lambda_mom=0.0,
        lambda_tr=0.0,
    )

    # 2. Raw unweighted momentum residual loss
    loss_mom, stats_mom = mom_loss_fn(pred_phys, re=re, dt=dt, q0_phys=q0_phys)

    # 3. Raw unweighted tracer residual loss
    loss_tr, stats_tr = tracer_loss_fn(pred_phys, re=re, sc=sc, dt=dt, q0_phys=q0_phys)

    # Trainable dynamics parameters theta
    trainable_params = [p for p in model.transformer.parameters() if p.requires_grad]

    # Compute decoupled gradients with retain_graph=True
    grads_existing = torch.autograd.grad(
        loss_existing, trainable_params, retain_graph=True, allow_unused=True
    )
    grads_mom = torch.autograd.grad(
        loss_mom, trainable_params, retain_graph=True, allow_unused=True
    )
    grads_tr = torch.autograd.grad(
        loss_tr, trainable_params, retain_graph=False, allow_unused=True
    )

    # Verify frozen encoder and decoder parameters receive no gradients
    frozen_encoder_active = any(p.requires_grad for p in model.encoder.parameters())
    frozen_decoder_active = any(p.requires_grad for p in model.decoder.parameters())

    norm_existing = compute_gradient_norm(trainable_params, grads_existing)
    norm_mom_raw = compute_gradient_norm(trainable_params, grads_mom)
    norm_tr_raw = compute_gradient_norm(trainable_params, grads_tr)

    cos_sim_mom = compute_gradient_cosine_similarity(grads_existing, grads_mom)
    cos_sim_tr = compute_gradient_cosine_similarity(grads_existing, grads_tr)
    cos_sim_mom_tr = compute_gradient_cosine_similarity(grads_mom, grads_tr)

    return {
        "loss_existing": float(loss_existing.item()),
        "loss_mom_raw": float(loss_mom.item()),
        "loss_tr_raw": float(loss_tr.item()),
        "norm_existing": norm_existing,
        "norm_mom_raw": norm_mom_raw,
        "norm_tr_raw": norm_tr_raw,
        "cos_sim_existing_mom": cos_sim_mom,
        "cos_sim_existing_tr": cos_sim_tr,
        "cos_sim_mom_tr": cos_sim_mom_tr,
        "frozen_encoder_active": frozen_encoder_active,
        "frozen_decoder_active": frozen_decoder_active,
        "stats_mom": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in stats_mom.items()},
        "stats_tr": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in stats_tr.items()},
    }


def evaluate_on_fixed_validation_window(
    model: LatentForecasterWrapper,
    validated_samples: List[Dict[str, Any]],
    normalizer: FieldNormalizer,
    pde_loss_fn: NavierStokesPDELoss,
    history_length: int = 4,
    horizon: int = 4,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Evaluate physical VRMSE, RMSE, divergence, and discrete PDE residuals on fixed validation samples."""
    model.eval()
    total_window = history_length + horizon
    q0_idx = history_length - 1
    pred_slice = slice(history_length, total_window)

    vrmse_list: List[float] = []
    rmse_u_list: List[float] = []
    rmse_v_list: List[float] = []
    rmse_p_list: List[float] = []
    rmse_s_list: List[float] = []
    div_rmse_list: List[float] = []
    res_u_list: List[float] = []
    res_v_list: List[float] = []
    res_s_list: List[float] = []

    import h5py

    with torch.no_grad():
        for s_info in validated_samples:
            h5_path = s_info["file_path"]
            sim_idx = s_info["traj_idx"]
            re_val = s_info["re"]
            sc_val = s_info["sc"]
            dt_val = s_info["dt"]

            with h5py.File(h5_path, "r") as f:
                vel = f["t1_fields/velocity"][sim_idx, :total_window]
                p = f["t0_fields/pressure"][sim_idx, :total_window]
                s = f["t0_fields/tracer"][sim_idx, :total_window]
                u = vel[..., 0]
                v = vel[..., 1]
                q_full_np = np.stack([u, v, p, s], axis=1)

            q_full = torch.from_numpy(q_full_np).float().to(device).unsqueeze(0)
            q_down = q_full[..., ::2, ::2]  # (1, T, 4, 128, 256)

            q0_down = q_down[:, q0_idx]
            q_target_pred = q_down[:, pred_slice]

            q_down_norm = normalizer.normalize(q_down)
            q_hist_norm = q_down_norm[:, :history_length]
            re_tensor = torch.tensor([re_val], device=device, dtype=torch.float32)
            sc_tensor = torch.tensor([sc_val], device=device, dtype=torch.float32)

            q_pred_norm = model.forward_rollout(
                q_hist_norm, re=re_tensor, sc=sc_tensor, horizon=horizon
            )
            q_pred = normalizer.denormalize(q_pred_norm)

            # Enforce zero-mean pressure gauge
            q_pred[:, :, 2:3] = q_pred[:, :, 2:3] - q_pred[:, :, 2:3].mean(dim=(-2, -1), keepdim=True)
            q_target_pred[:, :, 2:3] = q_target_pred[:, :, 2:3] - q_target_pred[:, :, 2:3].mean(dim=(-2, -1), keepdim=True)

            # Field errors
            diff = q_pred - q_target_pred
            rmse_u = torch.sqrt(torch.mean(diff[:, :, 0] ** 2)).item()
            rmse_v = torch.sqrt(torch.mean(diff[:, :, 1] ** 2)).item()
            rmse_p = torch.sqrt(torch.mean(diff[:, :, 2] ** 2)).item()
            rmse_s = torch.sqrt(torch.mean(diff[:, :, 3] ** 2)).item()
            var_target = torch.var(q_target_pred, dim=(-2, -1), unbiased=False).mean().item()
            vrmse = math.sqrt(torch.mean(diff**2).item()) / (math.sqrt(var_target) + 1e-8)

            # Divergence
            div = compute_divergence(q_pred[0, :, 0], q_pred[0, :, 1], domain_size=(1.0, 2.0))
            div_rmse = torch.sqrt(torch.mean(div**2)).item()

            # Discrete PDE residuals
            _, stats = pde_loss_fn(
                q_pred,
                re=re_val,
                sc=sc_val,
                dt=dt_val,
                q0_phys=q0_down,
            )

            vrmse_list.append(vrmse)
            rmse_u_list.append(rmse_u)
            rmse_v_list.append(rmse_v)
            rmse_p_list.append(rmse_p)
            rmse_s_list.append(rmse_s)
            div_rmse_list.append(div_rmse)
            res_u_list.append(stats["res_momentum_u_rmse"])
            res_v_list.append(stats["res_momentum_v_rmse"])
            res_s_list.append(stats["res_tracer_s_rmse"])

    return {
        "vrmse": float(np.mean(vrmse_list)),
        "rmse_u": float(np.mean(rmse_u_list)),
        "rmse_v": float(np.mean(rmse_v_list)),
        "rmse_p": float(np.mean(rmse_p_list)),
        "rmse_s": float(np.mean(rmse_s_list)),
        "div_rmse": float(np.mean(div_rmse_list)),
        "res_u_rmse": float(np.mean(res_u_list)),
        "res_v_rmse": float(np.mean(res_v_list)),
        "res_s_rmse": float(np.mean(res_s_list)),
    }


def run_pde_gradient_probe(
    split_file: str = "outputs/splits/grouped_split.json",
    norm_file: str = "outputs/normalization/stats_grouped.pt",
    ae_ckpt: str = "outputs/checkpoints/representation/best_autoencoder.pt",
    d0_ckpt: str = "outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt",
    data_root: Optional[str] = None,
    num_probe_batches: int = 5,
    update_steps: int = 20,
    lr: float = 1e-4,
    history_length: int = 4,
    horizon: int = 4,
    candidate_weights: Optional[List[Tuple[float, float]]] = None,
    output_json: str = "outputs/evaluations/pde_gradient_probe.json",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute rigorous gradient norm and small-budget training probe."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_root is None:
        data_root = os.environ.get("SHEAR_FLOW_DATA_DIR", "/root/autodl-tmp/datasets/shear_flow")

    if candidate_weights is None:
        # Default exploration grid:
        # 1. Zero PDE baseline
        # 2. Conservative scaling: 0.005, 0.002
        # 3. Intermediate scaling: 0.01, 0.01
        # 4. Standard default scaling: 0.05, 0.02
        # 5. High scaling: 0.1, 0.05
        candidate_weights = [
            (0.005, 0.002),
            (0.01, 0.01),
            (0.05, 0.02),
            (0.1, 0.05),
        ]

    # Precheck contracts
    total_window = history_length + horizon
    validated_samples = precheck_audit_manifest_and_samples(
        split_file=split_file,
        data_root=data_root,
        total_window=total_window,
        num_samples=4,
    )

    # Load Normalizer
    normalizer = FieldNormalizer()
    normalizer_state = torch.load(norm_file, map_location=device, weights_only=True)
    normalizer.load_state_dict(normalizer_state)
    normalizer.to(device)

    # Load Base Architecture
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

    model = LatentForecasterWrapper(
        encoder=encoder,
        transformer=transformer,
        decoder=decoder,
        freeze_representation=True,
    ).to(device)

    d0_state = torch.load(d0_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(d0_state["model_state_dict"])

    # Loss Functions
    mom_loss_fn = NavierStokesMomentumResidualLoss(domain_size=(1.0, 2.0), dealias=True).to(device)
    tracer_loss_fn = TracerAdvectionDiffusionResidualLoss(domain_size=(1.0, 2.0), dealias=True).to(device)
    pde_loss_fn = NavierStokesPDELoss(domain_size=(1.0, 2.0), dealias=True).to(device)
    rollout_loss_fn = nn.MSELoss()

    # Dataloader for probe
    train_loader, _, _, _, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file=split_file,
        data_root=data_root,
        history_length=history_length,
        horizon=horizon,
        valid_horizon=horizon,
        train_stride=2,
        valid_stride=8,
        downsample_factor=2,
        batch_size=4,
        num_workers=2,
        normalize=True,
        normalizer=normalizer,
        preload_to_memory=False,
        is_distributed=False,
        rank=0,
        world_size=1,
        seed=42,
        return_sampler=True,
        require_pressure=True,
        require_tracer=True,
    )

    print("=" * 90)
    print("PHASE 1: PROBING DECOUPLED LOSS SCALES AND GRADIENT NORMS")
    print(f"Sampling {num_probe_batches} training batches to compute parameter-level gradient dynamics...")
    print("=" * 90)

    raw_batch_records: List[Dict[str, Any]] = []
    probe_iter = iter(train_loader)

    model.eval()  # ensure dropout/eval consistency during measurement
    for b_idx in range(num_probe_batches):
        batch = next(probe_iter)
        rec = compute_decoupled_batch_gradients(
            model=model,
            batch=batch,
            normalizer=normalizer,
            mom_loss_fn=mom_loss_fn,
            tracer_loss_fn=tracer_loss_fn,
            rollout_loss_fn=rollout_loss_fn,
            device=device,
        )
        raw_batch_records.append(rec)

    # Aggregate unweighted statistics across probe batches
    mean_loss_existing = float(np.mean([r["loss_existing"] for r in raw_batch_records]))
    mean_loss_mom_raw = float(np.mean([r["loss_mom_raw"] for r in raw_batch_records]))
    mean_loss_tr_raw = float(np.mean([r["loss_tr_raw"] for r in raw_batch_records]))

    mean_norm_existing = float(np.mean([r["norm_existing"] for r in raw_batch_records]))
    mean_norm_mom_raw = float(np.mean([r["norm_mom_raw"] for r in raw_batch_records]))
    mean_norm_tr_raw = float(np.mean([r["norm_tr_raw"] for r in raw_batch_records]))

    mean_cos_existing_mom = float(np.mean([r["cos_sim_existing_mom"] for r in raw_batch_records]))
    mean_cos_existing_tr = float(np.mean([r["cos_sim_existing_tr"] for r in raw_batch_records]))
    mean_cos_mom_tr = float(np.mean([r["cos_sim_mom_tr"] for r in raw_batch_records]))

    # Verify Q1: PDE terms truly participate in optimization
    is_q1_finite = (
        math.isfinite(mean_norm_existing)
        and math.isfinite(mean_norm_mom_raw)
        and math.isfinite(mean_norm_tr_raw)
        and mean_norm_mom_raw > 0.0
        and mean_norm_tr_raw > 0.0
    )
    is_representation_frozen = not any(r["frozen_encoder_active"] or r["frozen_decoder_active"] for r in raw_batch_records)

    print("\n--- UNWEIGHTED BASELINE GRADIENT SUMMARY ---")
    print(f"L_existing loss: {mean_loss_existing:.6e} | Grad norm: {mean_norm_existing:.6e}")
    print(f"L_momentum loss: {mean_loss_mom_raw:.6e} | Grad norm: {mean_norm_mom_raw:.6e}")
    print(f"L_tracer   loss: {mean_loss_tr_raw:.6e} | Grad norm: {mean_norm_tr_raw:.6e}")
    print(f"Gradient Cosine Alignments: cos(exist, mom)={mean_cos_existing_mom:+.4f}, cos(exist, tr)={mean_cos_existing_tr:+.4f}, cos(mom, tr)={mean_cos_mom_tr:+.4f}")
    print(f"Q1 Verification: Finite & non-zero gradients={is_q1_finite}, Representation frozen={is_representation_frozen}")

    # Evaluate Candidate Weights (Q2)
    weight_probe_results: List[Dict[str, Any]] = []
    print("\n" + "=" * 95)
    print("PHASE 2: EVALUATING CANDIDATE WEIGHT PAIRS ON GRADIENT RATIOS (Q2)")
    print(f"{'lambda_mom':<12} | {'lambda_tr':<12} | {'Loss Ratio (Mom)':<18} | {'Loss Ratio (Tr)':<18} | {'Grad Ratio (Mom)':<18} | {'Grad Ratio (Tr)':<18}")
    print("-" * 95)

    for (lmom, ltr) in candidate_weights:
        # Effective losses and effective gradient norms
        eff_loss_mom = lmom * mean_loss_mom_raw
        eff_loss_tr = ltr * mean_loss_tr_raw

        eff_norm_mom = lmom * mean_norm_mom_raw
        eff_norm_tr = ltr * mean_norm_tr_raw

        grad_ratio_mom = eff_norm_mom / (mean_norm_existing + 1e-12)
        grad_ratio_tr = eff_norm_tr / (mean_norm_existing + 1e-12)

        loss_ratio_mom = eff_loss_mom / (mean_loss_existing + 1e-12)
        loss_ratio_tr = eff_loss_tr / (mean_loss_existing + 1e-12)

        print(
            f"{lmom:<12.4e} | {ltr:<12.4e} | "
            f"{loss_ratio_mom:<18.4e} | {loss_ratio_tr:<18.4e} | "
            f"{grad_ratio_mom:<18.4e} | {grad_ratio_tr:<18.4e}"
        )

        weight_probe_results.append({
            "lambda_mom": lmom,
            "lambda_tr": ltr,
            "effective_loss_mom": eff_loss_mom,
            "effective_loss_tr": eff_loss_tr,
            "loss_ratio_mom": loss_ratio_mom,
            "loss_ratio_tr": loss_ratio_tr,
            "effective_grad_norm_mom": eff_norm_mom,
            "effective_grad_norm_tr": eff_norm_tr,
            "grad_ratio_mom": grad_ratio_mom,
            "grad_ratio_tr": grad_ratio_tr,
        })
    print("=" * 95)

    # Phase 3: Small-budget Training Probe on Fixed Validation Window (Q3)
    print("\n" + "=" * 90)
    print("PHASE 3: SMALL-BUDGET TRAINING UPDATE PROBE (Q3)")
    print(f"Evaluating fixed validation window before and after {update_steps} optimization steps...")
    print("=" * 90)

    # 1. Baseline Evaluation (Step 0)
    metrics_before = evaluate_on_fixed_validation_window(
        model=model,
        validated_samples=validated_samples,
        normalizer=normalizer,
        pde_loss_fn=pde_loss_fn,
        history_length=history_length,
        horizon=horizon,
        device=device,
    )

    # 2. Isolated Update with representative candidate (e.g. lambda_mom=0.01, lambda_tr=0.01)
    probe_model = copy.deepcopy(model)
    probe_model.train()
    probe_optimizer = torch.optim.AdamW(
        [p for p in probe_model.transformer.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )

    probe_lmom = 0.01
    probe_ltr = 0.01

    step_losses: List[float] = []
    update_iter = iter(train_loader)

    for step in range(update_steps):
        try:
            batch = next(update_iter)
        except StopIteration:
            update_iter = iter(train_loader)
            batch = next(update_iter)

        q_hist = batch["history"].to(device)
        q_future = batch["future"].to(device)
        re = batch["re"].to(device)
        sc = batch["sc"].to(device)
        dt = batch["dt"].to(device)

        probe_optimizer.zero_grad()
        pred_norm = probe_model.forward_rollout(q_hist, re=re, sc=sc, horizon=horizon)
        q_hist_phys = normalizer.denormalize(q_hist)
        q0_phys = q_hist_phys[:, -1]

        step_loss = _compute_batch_loss(
            pred=pred_norm,
            q_future=q_future,
            normalizer=normalizer,
            field_loss_space="normalized",
            lambda_div=0.0,
            lambda_vort=0.0,
            rollout_loss_fn=rollout_loss_fn,
            lambda_mom=probe_lmom,
            lambda_tr=probe_ltr,
            mom_loss_fn=mom_loss_fn,
            tracer_loss_fn=tracer_loss_fn,
            re=re,
            sc=sc,
            dt=dt,
            q0_phys=q0_phys,
        )

        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in probe_model.transformer.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        probe_optimizer.step()
        step_losses.append(float(step_loss.item()))

    # 3. Post-update Evaluation
    metrics_after = evaluate_on_fixed_validation_window(
        model=probe_model,
        validated_samples=validated_samples,
        normalizer=normalizer,
        pde_loss_fn=pde_loss_fn,
        history_length=history_length,
        horizon=horizon,
        device=device,
    )

    print(f"\n{update_steps}-Step Fine-Tuning Loss: Initial={step_losses[0]:.4e} -> Final={step_losses[-1]:.4e}")
    print(f"{'Metric':<20} | {'Before Update (Baseline)':<25} | {'After 20 Steps (PDE Probe)':<25} | {'Delta (%)':<15}")
    print("-" * 90)

    comparison_summary: Dict[str, Dict[str, float]] = {}
    for metric_key in ["vrmse", "rmse_u", "rmse_v", "rmse_p", "rmse_s", "div_rmse", "res_u_rmse", "res_v_rmse", "res_s_rmse"]:
        val_before = metrics_before[metric_key]
        val_after = metrics_after[metric_key]
        delta_pct = (val_after - val_before) / (val_before + 1e-12) * 100.0
        comparison_summary[metric_key] = {
            "before": val_before,
            "after": val_after,
            "delta_pct": delta_pct,
        }
        print(f"{metric_key:<20} | {val_before:<25.4e} | {val_after:<25.4e} | {delta_pct:<+15.2f}%")
    print("=" * 90)

    # Build provenance metadata
    script_path = os.path.abspath(__file__)
    payload = {
        "metadata": {
            "git_commit": get_git_commit_hash(),
            "git_dirty": get_git_dirty(),
            "probe_script": os.path.relpath(script_path, os.getcwd()) if script_path.startswith(os.getcwd()) else script_path,
            "probe_script_sha256": compute_file_sha256(script_path),
            "split_file": split_file,
            "split_file_sha256": compute_file_sha256(split_file),
            "norm_file": norm_file,
            "norm_file_sha256": compute_file_sha256(norm_file),
            "ae_ckpt": ae_ckpt,
            "ae_ckpt_sha256": compute_file_sha256(ae_ckpt),
            "d0_ckpt": d0_ckpt,
            "d0_ckpt_sha256": compute_file_sha256(d0_ckpt),
            "num_probe_batches": num_probe_batches,
            "update_steps": update_steps,
            "update_lr": lr,
            "probed_weights": {"lambda_mom": probe_lmom, "lambda_tr": probe_ltr},
        },
        "q1_verification": {
            "gradients_finite": is_q1_finite,
            "representation_frozen": is_representation_frozen,
            "mean_norm_existing": mean_norm_existing,
            "mean_norm_mom_raw": mean_norm_mom_raw,
            "mean_norm_tr_raw": mean_norm_tr_raw,
        },
        "q2_loss_and_gradient_scales": {
            "mean_loss_existing": mean_loss_existing,
            "mean_loss_mom_raw": mean_loss_mom_raw,
            "mean_loss_tr_raw": mean_loss_tr_raw,
            "cosine_alignments": {
                "cos_existing_mom": mean_cos_existing_mom,
                "cos_existing_tr": mean_cos_existing_tr,
                "cos_mom_tr": mean_cos_mom_tr,
            },
            "candidate_weight_evaluations": weight_probe_results,
        },
        "q3_validation_performance_comparison": {
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "comparison": comparison_summary,
            "step_losses": step_losses,
        },
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nComprehensive gradient probe results archived to {output_json}")

    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run gradient scale and small-budget training probe.")
    parser.add_argument("--output_json", type=str, default="outputs/evaluations/pde_gradient_probe.json")
    parser.add_argument("--num_probe_batches", type=int, default=5)
    parser.add_argument("--update_steps", type=int, default=20)
    args = parser.parse_args()

    run_pde_gradient_probe(
        output_json=args.output_json,
        num_probe_batches=args.num_probe_batches,
        update_steps=args.update_steps,
    )
