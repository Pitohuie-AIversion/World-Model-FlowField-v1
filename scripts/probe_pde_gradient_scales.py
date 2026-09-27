"""Small-budget gradient norm, loss scale, and paired P0/PDE control probe.

Strict Protocol Guarantees:
1. Strict Configuration Parity with train_forecaster():
   - Residual scales explicitly matched: scale_u=0.05, scale_v=0.05, scale_s=0.02.
   - Decoupled gradient computation supports full existing loss (MSE + divergence + vorticity).
2. Standard Repository Metrics:
   - Uses canonical src.metrics.field.compute_vrmse() and evaluate_field_metrics().
   - Eliminates all ad-hoc mixed-variance formulas.
3. Rigorous Vector Space Operations:
   - Cosine similarity computed across complete parameter space R^D (treating None as exact zeros).
   - Fail-closed: halts immediately upon non-finite gradients or representation freeze contract violations.
4. Paired P0 vs. PDE Control Probe:
   - Identical initialization from baseline D0.
   - Identical training batch sequence and random seeds.
   - Identical optimizer settings (AdamW, lr, weight_decay, grad clip).
   - P0 Group: updates with L_existing only.
   - PDE Group: updates with L_existing + lambda_mom * L_mom + lambda_tr * L_tr.
   - Evaluated on identical fixed validation window.
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
from src.losses.divergence import DivergenceLoss
from src.losses.navier_stokes import (
    NavierStokesMomentumResidualLoss,
    NavierStokesPDELoss,
    TracerAdvectionDiffusionResidualLoss,
)
from src.losses.vorticity import VorticityLoss
from src.metrics.field import compute_vrmse, evaluate_field_metrics
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_transformer import LatentSTTransformer
from src.utils.fft_derivatives import compute_divergence


def compute_gradient_norm(grads: List[Optional[torch.Tensor]]) -> float:
    """Compute global L2 norm of gradient vectors across parameters in full space R^D."""
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
    """Compute exact cosine similarity between two gradient vectors in full parameter space R^D.

    Treats None entries (e.g. parameters not receiving gradients from one loss) as exact zeros.
    Both vector norms are accumulated across all parameters independently, preventing subset projection errors.
    """
    if len(grads1) != len(grads2):
        raise ValueError(f"Gradient list length mismatch: {len(grads1)} vs {len(grads2)}")

    dot_product = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0

    for g1, g2 in zip(grads1, grads2):
        if g1 is not None:
            if not torch.all(torch.isfinite(g1)):
                return float("nan")
            norm1_sq += float((g1**2).sum().item())
        if g2 is not None:
            if not torch.all(torch.isfinite(g2)):
                return float("nan")
            norm2_sq += float((g2**2).sum().item())
        if g1 is not None and g2 is not None:
            dot_product += float((g1 * g2).sum().item())

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
    div_loss_fn: Optional[nn.Module] = None,
    vort_loss_fn: Optional[nn.Module] = None,
    lambda_div: float = 0.0,
    lambda_vort: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Compute decoupled gradients for existing, momentum, and tracer losses on identical parameter set theta."""
    q_hist = batch["history"].to(device)
    q_future = batch["future"].to(device)
    re = batch["re"].to(device)
    sc = batch["sc"].to(device)
    dt = batch["dt"].to(device)
    horizon = q_future.shape[1]

    # Model prediction
    pred_norm = model.forward_rollout(q_hist, re=re, sc=sc, horizon=horizon)
    pred_phys = normalizer.denormalize(pred_norm)

    # Historical connection state q0 in physical space (frame L-1)
    q_hist_phys = normalizer.denormalize(q_hist)
    q0_phys = q_hist_phys[:, -1]

    # 1. Existing base loss with full physical regularization terms if requested
    loss_existing = _compute_batch_loss(
        pred=pred_norm,
        q_future=q_future,
        normalizer=normalizer,
        field_loss_space="normalized",
        lambda_div=lambda_div,
        lambda_vort=lambda_vort,
        rollout_loss_fn=rollout_loss_fn,
        div_loss_fn=div_loss_fn,
        vort_loss_fn=vort_loss_fn,
        lambda_spec=0.0,
        lambda_mom=0.0,
        lambda_tr=0.0,
    )

    # 2. Raw unweighted momentum residual loss (using configured scales in mom_loss_fn)
    loss_mom, stats_mom = mom_loss_fn(pred_phys, re=re, dt=dt, q0_phys=q0_phys)

    # 3. Raw unweighted tracer residual loss (using configured scales in tracer_loss_fn)
    loss_tr, stats_tr = tracer_loss_fn(pred_phys, re=re, sc=sc, dt=dt, q0_phys=q0_phys)

    # Trainable dynamics parameters theta
    trainable_params = [p for p in model.transformer.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("Model has 0 trainable transformer parameters! Failing closed.")

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

    # Fail-closed checks on freeze status
    frozen_encoder_active = any(p.requires_grad for p in model.encoder.parameters())
    frozen_decoder_active = any(p.requires_grad for p in model.decoder.parameters())
    if frozen_encoder_active or frozen_decoder_active:
        raise RuntimeError("Representation freeze contract violated: encoder or decoder has requires_grad=True.")

    norm_existing = compute_gradient_norm(grads_existing)
    norm_mom_raw = compute_gradient_norm(grads_mom)
    norm_tr_raw = compute_gradient_norm(grads_tr)

    # Fail-closed checks on gradient finiteness
    if not (math.isfinite(norm_existing) and math.isfinite(norm_mom_raw) and math.isfinite(norm_tr_raw)):
        raise FloatingPointError(
            f"Non-finite gradient norm detected: norm_exist={norm_existing}, "
            f"norm_mom={norm_mom_raw}, norm_tr={norm_tr_raw}."
        )

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
    """Evaluate standard VRMSE, per-channel RMSE, divergence, and discrete PDE residuals on fixed validation samples."""
    model.eval()
    total_window = history_length + horizon
    q0_idx = history_length - 1
    pred_slice = slice(history_length, total_window)

    vrmse_standard_list: List[float] = []
    rmse_total_list: List[float] = []
    rmse_u_list: List[float] = []
    rmse_v_list: List[float] = []
    rmse_p_list: List[float] = []
    rmse_s_list: List[float] = []
    vrmse_u_list: List[float] = []
    vrmse_v_list: List[float] = []
    vrmse_p_list: List[float] = []
    vrmse_s_list: List[float] = []
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

            # Enforce zero-mean pressure gauge in physical space
            q_pred[:, :, 2:3] = q_pred[:, :, 2:3] - q_pred[:, :, 2:3].mean(dim=(-2, -1), keepdim=True)
            q_target_pred[:, :, 2:3] = q_target_pred[:, :, 2:3] - q_target_pred[:, :, 2:3].mean(dim=(-2, -1), keepdim=True)

            # 1. Standard VRMSE via canonical src.metrics.field.compute_vrmse
            vrmse_std = compute_vrmse(q_pred, q_target_pred).item()
            vrmse_standard_list.append(vrmse_std)

            # 2. Per-channel errors via standard evaluate_field_metrics
            metrics_dict = evaluate_field_metrics(q_pred, q_target_pred, channel_names=("u", "v", "p", "s"))
            rmse_total = math.sqrt(torch.mean((q_pred - q_target_pred) ** 2).item())

            rmse_total_list.append(rmse_total)
            rmse_u_list.append(metrics_dict["rmse_u"])
            rmse_v_list.append(metrics_dict["rmse_v"])
            rmse_p_list.append(metrics_dict["rmse_p"])
            rmse_s_list.append(metrics_dict["rmse_s"])
            vrmse_u_list.append(metrics_dict["vrmse_u"])
            vrmse_v_list.append(metrics_dict["vrmse_v"])
            vrmse_p_list.append(metrics_dict["vrmse_p"])
            vrmse_s_list.append(metrics_dict["vrmse_s"])

            # 3. Divergence
            div = compute_divergence(q_pred[0, :, 0], q_pred[0, :, 1], domain_size=(1.0, 2.0))
            div_rmse = torch.sqrt(torch.mean(div**2)).item()
            div_rmse_list.append(div_rmse)

            # 4. Discrete PDE residuals
            _, stats = pde_loss_fn(
                q_pred,
                re=re_val,
                sc=sc_val,
                dt=dt_val,
                q0_phys=q0_down,
            )
            res_u_list.append(stats["res_momentum_u_rmse"])
            res_v_list.append(stats["res_momentum_v_rmse"])
            res_s_list.append(stats["res_tracer_s_rmse"])

    return {
        "vrmse_standard": float(np.mean(vrmse_standard_list)),
        "rmse_total": float(np.mean(rmse_total_list)),
        "rmse_u": float(np.mean(rmse_u_list)),
        "rmse_v": float(np.mean(rmse_v_list)),
        "rmse_p": float(np.mean(rmse_p_list)),
        "rmse_s": float(np.mean(rmse_s_list)),
        "vrmse_u": float(np.mean(vrmse_u_list)),
        "vrmse_v": float(np.mean(vrmse_v_list)),
        "vrmse_p": float(np.mean(vrmse_p_list)),
        "vrmse_s": float(np.mean(vrmse_s_list)),
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
    lr: float = 5e-5,
    history_length: int = 4,
    horizon: int = 4,
    mom_scale_u: float = 0.05,
    mom_scale_v: float = 0.05,
    tracer_scale_s: float = 0.02,
    pde_dealias: bool = True,
    lambda_div: float = 0.01,
    lambda_vort: float = 0.05,
    candidate_weights: Optional[List[Tuple[float, float]]] = None,
    pde_probe_weights: Tuple[float, float] = (2.5e-5, 4.0e-6),
    output_json: str = "outputs/evaluations/pde_gradient_probe.json",
    num_workers: int = 2,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute rigorous gradient norm and paired P0/PDE small-budget training probe."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_root is None:
        data_root = os.environ.get("SHEAR_FLOW_DATA_DIR", "/root/autodl-tmp/datasets/shear_flow")

    if candidate_weights is None:
        # Candidate weights tailored for train_forecaster scale (scale_u=0.05, scale_v=0.05, scale_s=0.02)
        # Recall that with scale a=0.05, loss and gradient are amplified by 1/(0.05^2) = 400x for momentum,
        # and 1/(0.02^2) = 2500x for tracer.
        # (2.5e-5, 4.0e-6) provides exact algebraic scale-equivalence to (0.01, 0.01) at scale=1.0.
        candidate_weights = [
            (2.5e-5, 4.0e-6),
            (5.0e-5, 1.0e-5),
            (1.0e-4, 2.0e-5),
            (2.5e-4, 5.0e-5),
            (1.0e-3, 1.0e-4),
        ]

    torch.manual_seed(42)
    np.random.seed(42)

    # Precheck contracts
    total_window = history_length + horizon
    validated_samples = precheck_audit_manifest_and_samples(
        split_file=split_file,
        data_root=data_root,
        total_window=total_window,
        num_samples=4,
    )

    # Load Normalizer (without overwriting disk files)
    normalizer = FieldNormalizer()
    normalizer_state = torch.load(norm_file, map_location="cpu", weights_only=True)
    normalizer.load_state_dict(normalizer_state)
    normalizer_cpu = copy.deepcopy(normalizer)
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

    base_model = LatentForecasterWrapper(
        encoder=encoder,
        transformer=transformer,
        decoder=decoder,
        freeze_representation=True,
    ).to(device)

    d0_state = torch.load(d0_ckpt, map_location=device, weights_only=False)
    base_model.load_state_dict(d0_state["model_state_dict"])

    # Loss Functions matching train_forecaster exact scales and baseline terms
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

    # Dataloader for probe passing preloaded CPU normalizer
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
        num_workers=num_workers,
        normalize=True,
        normalizer=normalizer_cpu,
        preload_to_memory=False,
        is_distributed=False,
        rank=0,
        world_size=1,
        seed=42,
        return_sampler=True,
        require_pressure=True,
        require_tracer=True,
    )

    print("=" * 95)
    print("PHASE 1: PROBING DECOUPLED LOSS SCALES AND GRADIENT NORMS")
    print(f"Sampling {num_probe_batches} training batches to compute parameter-level gradient dynamics...")
    print(f"Residual scale parity: scale_u={mom_scale_u}, scale_v={mom_scale_v}, scale_s={tracer_scale_s}")
    print(f"Existing loss parity: lambda_div={lambda_div}, lambda_vort={lambda_vort}")
    print("=" * 95)

    raw_batch_records: List[Dict[str, Any]] = []
    probe_iter = iter(train_loader)

    base_model.eval()
    for b_idx in range(num_probe_batches):
        batch = next(probe_iter)
        rec = compute_decoupled_batch_gradients(
            model=base_model,
            batch=batch,
            normalizer=normalizer,
            mom_loss_fn=mom_loss_fn,
            tracer_loss_fn=tracer_loss_fn,
            rollout_loss_fn=rollout_loss_fn,
            div_loss_fn=div_loss_fn,
            vort_loss_fn=vort_loss_fn,
            lambda_div=lambda_div,
            lambda_vort=lambda_vort,
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

    is_q1_finite = (
        math.isfinite(mean_norm_existing)
        and math.isfinite(mean_norm_mom_raw)
        and math.isfinite(mean_norm_tr_raw)
        and mean_norm_mom_raw > 0.0
        and mean_norm_tr_raw > 0.0
    )
    is_representation_frozen = not any(
        r["frozen_encoder_active"] or r["frozen_decoder_active"] for r in raw_batch_records
    )

    if not (is_q1_finite and is_representation_frozen):
        raise RuntimeError("Q1 Verification FAILED: non-finite gradients or unfreezed representation.")

    print("\n--- UNWEIGHTED BASELINE GRADIENT SUMMARY ---")
    print(f"L_existing loss: {mean_loss_existing:.6e} | Grad norm: {mean_norm_existing:.6e}")
    print(f"L_momentum loss: {mean_loss_mom_raw:.6e} | Grad norm: {mean_norm_mom_raw:.6e} (scale=0.05)")
    print(f"L_tracer   loss: {mean_loss_tr_raw:.6e} | Grad norm: {mean_norm_tr_raw:.6e} (scale=0.02)")
    print(
        f"Gradient Cosine Alignments: cos(exist, mom)={mean_cos_existing_mom:+.4f}, "
        f"cos(exist, tr)={mean_cos_existing_tr:+.4f}, cos(mom, tr)={mean_cos_mom_tr:+.4f}"
    )

    # Phase 2: Estimate Gradient Ratios for Candidate Weights (Q2)
    weight_probe_results: List[Dict[str, Any]] = []
    print("\n" + "=" * 95)
    print("PHASE 2: INITIAL GRADIENT RATIO ESTIMATES FOR CANDIDATE WEIGHTS (Q2)")
    print(f"{'lambda_mom':<12} | {'lambda_tr':<12} | {'Loss Ratio (Mom)':<18} | {'Loss Ratio (Tr)':<18} | {'Grad Ratio (Mom)':<18} | {'Grad Ratio (Tr)':<18}")
    print("-" * 95)

    for (lmom, ltr) in candidate_weights:
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

    # Phase 3: P0 Control vs. PDE Experiment Paired Small-budget Training (Q3)
    print("\n" + "=" * 95)
    print("PHASE 3: FAIR P0 CONTROL VS. PDE EXPERIMENT PAIRED TRAINING (Q3)")
    print(f"Comparing D0 Baseline vs. P0 Control ({update_steps} steps) vs. PDE Experiment ({update_steps} steps)")
    print(f"Update hyperparams: lr={lr}, lambda_div={lambda_div}, lambda_vort={lambda_vort}")
    print(f"Target PDE probe weights: lambda_mom={pde_probe_weights[0]}, lambda_tr={pde_probe_weights[1]}")
    print("=" * 95)

    # Pre-update evaluation on fixed validation window (Initial D0 Baseline)
    metrics_baseline = evaluate_on_fixed_validation_window(
        model=base_model,
        validated_samples=validated_samples,
        normalizer=normalizer,
        pde_loss_fn=pde_loss_fn,
        history_length=history_length,
        horizon=horizon,
        device=device,
    )

    # 1. Collect identical batches for both runs to ensure strict batch-for-batch paired alignment
    update_batches: List[Dict[str, Any]] = []
    batch_iter = iter(train_loader)
    for _ in range(update_steps):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(train_loader)
            batch = next(batch_iter)
        update_batches.append(batch)

    # 2. P0 Control Group: same initialization, same data, L_existing only
    print(f"\nExecuting P0 Control Group (L_existing only, lambda_mom=0, lambda_tr=0, lr={lr})...")
    p0_model = copy.deepcopy(base_model)
    p0_model.train()
    p0_optimizer = torch.optim.AdamW(
        [p for p in p0_model.transformer.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )

    p0_losses: List[float] = []
    for step, b in enumerate(update_batches):
        q_hist = b["history"].to(device)
        q_future = b["future"].to(device)
        re = b["re"].to(device)
        sc = b["sc"].to(device)
        dt = b["dt"].to(device)

        p0_optimizer.zero_grad()
        pred_norm = p0_model.forward_rollout(q_hist, re=re, sc=sc, horizon=horizon)

        loss = _compute_batch_loss(
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
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss detected in P0 control at step {step}: {loss.item()}")
        loss.backward()

        for p in p0_model.transformer.parameters():
            if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                raise FloatingPointError(f"Non-finite gradient in P0 control training at step {step}!")

        torch.nn.utils.clip_grad_norm_(
            [p for p in p0_model.transformer.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        p0_optimizer.step()
        p0_losses.append(float(loss.item()))

    metrics_p0 = evaluate_on_fixed_validation_window(
        model=p0_model,
        validated_samples=validated_samples,
        normalizer=normalizer,
        pde_loss_fn=pde_loss_fn,
        history_length=history_length,
        horizon=horizon,
        device=device,
    )

    # 3. PDE Experiment Group: same initialization, exact same batch sequence, L_existing + PDE
    print(f"Executing PDE Experiment Group (lambda_mom={pde_probe_weights[0]}, lambda_tr={pde_probe_weights[1]}, lr={lr})...")
    pde_model = copy.deepcopy(base_model)
    pde_model.train()
    pde_optimizer = torch.optim.AdamW(
        [p for p in pde_model.transformer.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )

    pde_losses: List[float] = []
    for step, b in enumerate(update_batches):
        q_hist = b["history"].to(device)
        q_future = b["future"].to(device)
        re = b["re"].to(device)
        sc = b["sc"].to(device)
        dt = b["dt"].to(device)

        pde_optimizer.zero_grad()
        pred_norm = pde_model.forward_rollout(q_hist, re=re, sc=sc, horizon=horizon)
        q_hist_phys = normalizer.denormalize(q_hist)
        q0_phys = q_hist_phys[:, -1]

        loss = _compute_batch_loss(
            pred=pred_norm,
            q_future=q_future,
            normalizer=normalizer,
            field_loss_space="normalized",
            lambda_div=lambda_div,
            lambda_vort=lambda_vort,
            rollout_loss_fn=rollout_loss_fn,
            div_loss_fn=div_loss_fn,
            vort_loss_fn=vort_loss_fn,
            lambda_mom=pde_probe_weights[0],
            lambda_tr=pde_probe_weights[1],
            mom_loss_fn=mom_loss_fn,
            tracer_loss_fn=tracer_loss_fn,
            re=re,
            sc=sc,
            dt=dt,
            q0_phys=q0_phys,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss detected in PDE experiment at step {step}: {loss.item()}")
        loss.backward()

        for p in pde_model.transformer.parameters():
            if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                raise FloatingPointError(f"Non-finite gradient in PDE experiment training at step {step}!")

        torch.nn.utils.clip_grad_norm_(
            [p for p in pde_model.transformer.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        pde_optimizer.step()
        pde_losses.append(float(loss.item()))

    metrics_pde = evaluate_on_fixed_validation_window(
        model=pde_model,
        validated_samples=validated_samples,
        normalizer=normalizer,
        pde_loss_fn=pde_loss_fn,
        history_length=history_length,
        horizon=horizon,
        device=device,
    )

    # 4. Print paired 3-way comparison table
    print("\n" + "=" * 120)
    print(f"{'Metric':<18} | {'D0 Baseline':<14} | {'P0 Control':<14} | {'PDE Group':<14} | {'P0 vs Base (%)':<16} | {'PDE vs P0 (%)':<16}")
    print("-" * 120)

    comparison_summary: Dict[str, Dict[str, float]] = {}
    metric_keys = [
        "vrmse_standard",
        "rmse_total",
        "rmse_u",
        "rmse_v",
        "rmse_p",
        "rmse_s",
        "vrmse_u",
        "vrmse_v",
        "vrmse_p",
        "vrmse_s",
        "div_rmse",
        "res_u_rmse",
        "res_v_rmse",
        "res_s_rmse",
    ]

    for k in metric_keys:
        val_base = metrics_baseline[k]
        val_p0 = metrics_p0[k]
        val_pde = metrics_pde[k]
        delta_p0_vs_base_pct = (val_p0 - val_base) / (val_base + 1e-12) * 100.0
        delta_pde_vs_p0_pct = (val_pde - val_p0) / (val_p0 + 1e-12) * 100.0

        comparison_summary[k] = {
            "d0_baseline": val_base,
            "p0_control": val_p0,
            "pde_experiment": val_pde,
            "delta_p0_vs_base_pct": delta_p0_vs_base_pct,
            "delta_pde_vs_p0_pct": delta_pde_vs_p0_pct,
        }

        print(
            f"{k:<18} | "
            f"{val_base:<14.4e} | "
            f"{val_p0:<14.4e} | "
            f"{val_pde:<14.4e} | "
            f"{delta_p0_vs_base_pct:<+16.2f}% | "
            f"{delta_pde_vs_p0_pct:<+16.2f}%"
        )
    print("=" * 120)

    # Provenance Payload
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
            "residual_scales": {
                "mom_scale_u": mom_scale_u,
                "mom_scale_v": mom_scale_v,
                "tracer_scale_s": tracer_scale_s,
            },
            "baseline_training_config": {
                "lambda_div": lambda_div,
                "lambda_vort": lambda_vort,
            },
            "pde_probe_weights": {
                "lambda_mom": pde_probe_weights[0],
                "lambda_tr": pde_probe_weights[1],
            },
            "num_probe_batches": num_probe_batches,
            "update_steps": update_steps,
            "update_lr": lr,
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
            "batch_records": [
                {
                    "batch_idx": idx,
                    "loss_existing": r["loss_existing"],
                    "loss_mom_raw": r["loss_mom_raw"],
                    "loss_tr_raw": r["loss_tr_raw"],
                    "norm_existing": r["norm_existing"],
                    "norm_mom_raw": r["norm_mom_raw"],
                    "norm_tr_raw": r["norm_tr_raw"],
                }
                for idx, r in enumerate(raw_batch_records)
            ],
        },
        "q3_paired_control_comparison": {
            "metrics_baseline": metrics_baseline,
            "metrics_p0_control": metrics_p0,
            "metrics_pde_experiment": metrics_pde,
            "comparison": comparison_summary,
            "p0_losses": p0_losses,
            "pde_losses": pde_losses,
        },
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nPaired control probe results successfully archived to {output_json}")

    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run gradient scale and paired P0/PDE control probe.")
    parser.add_argument("--output_json", type=str, default="outputs/evaluations/pde_gradient_probe.json")
    parser.add_argument("--num_probe_batches", type=int, default=5)
    parser.add_argument("--update_steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--mom_scale_u", type=float, default=0.05)
    parser.add_argument("--mom_scale_v", type=float, default=0.05)
    parser.add_argument("--tracer_scale_s", type=float, default=0.02)
    parser.add_argument("--lambda_div", type=float, default=0.01)
    parser.add_argument("--lambda_vort", type=float, default=0.05)
    parser.add_argument("--probe_lambda_mom", type=float, default=2.5e-5)
    parser.add_argument("--probe_lambda_tr", type=float, default=4.0e-6)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    run_pde_gradient_probe(
        output_json=args.output_json,
        num_probe_batches=args.num_probe_batches,
        update_steps=args.update_steps,
        lr=args.lr,
        mom_scale_u=args.mom_scale_u,
        mom_scale_v=args.mom_scale_v,
        tracer_scale_s=args.tracer_scale_s,
        lambda_div=args.lambda_div,
        lambda_vort=args.lambda_vort,
        pde_probe_weights=(args.probe_lambda_mom, args.probe_lambda_tr),
        num_workers=args.num_workers,
    )
