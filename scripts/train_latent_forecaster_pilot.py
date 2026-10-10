#!/usr/bin/env python3
"""Stage 2 A: Deterministic Latent Space Forecaster Pilot Training & Evaluation.

Trains a LatentSTTransformer on latent states extracted by a frozen VorticityAutoencoder (Cz=64)
over periodic scalar advection-diffusion trajectories.

Scientific & Verification Protocol:
1. Frozen Representation: Encoder and Decoder parameters are frozen (requires_grad=False).
   Checkpoint SHA-256 is verified before and after training to prove zero mutation.
2. Trajectory-Isolated Evaluation: Evaluates single-step prediction on the strictly isolated validation set.
3. State Transition Proof: Compares model single-step error against the Persistence Baseline (omega_t).
   Model must demonstrate state transition capability (Model error < Persistence error).
4. Triple-Trajectory Error Decomposition:
   - Analytical Ground Truth: omega_{t+1}
   - Frozen Autoencoder Reference: D(E(omega_{t+1}))
   - World Model Dynamical Prediction: D(hat{z}_{t+1})
   - Decomposes Total Field Error into Representation Error and Pure Dynamics Error.
5. Physical Enstrophy & Budget Tracking:
   - Measures enstrophy retention and enstrophy budget residuals for predicted vs GT trajectories.
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_temporal_vorticity_representation import (
    compute_file_sha256,
    compute_tensor_digest,
)
from src.data.synthetic_advection_diffusion import (
    AdvectionDiffusionConfig,
    PeriodicScalarAdvectionDiffusion,
    build_trajectory_dataset_manifest,
    build_trajectory_windows,
    compute_enstrophy_budget_residual,
    generate_trajectory_dataset,
    verify_trajectory_split_isolation,
)
from src.models.latent_forecaster_pilot import VorticityLatentForecaster
from src.utils.reproducibility import seed_everything


def run_latent_forecaster_pilot(
    config_path: Path,
    device_str: Optional[str] = None,
    output_dir_override: Optional[Path] = None,
    epochs_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute complete Phase 2 A deterministic latent forecaster training and evaluation."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(42)

    dev_str = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_str)

    out_dir = output_dir_override or Path(cfg["checkpoints"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Physics & Data Configuration
    dom_cfg = cfg["domain"]
    phys_cfg = cfg["physical_params"]
    time_cfg = cfg["temporal_params"]
    traj_cfg = cfg["trajectories"]
    win_cfg = cfg.get("windowing", {"history_len": 4, "future_len": 1, "stride": 1})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    adv_cfg = AdvectionDiffusionConfig(
        nx=dom_cfg["nx"],
        ny=dom_cfg["ny"],
        lx=dom_cfg["lx"],
        ly=dom_cfg["ly"],
        u0=phys_cfg["u0"],
        v0=phys_cfg["v0"],
        nu=phys_cfg["nu"],
        base_wavenumber=phys_cfg["base_wavenumber"],
        perturbation_modes=tuple(tuple(m) for m in phys_cfg["perturbation_modes"]),
        perturbation_amplitude=phys_cfg["perturbation_amplitude"],
    )
    solver = PeriodicScalarAdvectionDiffusion(cfg=adv_cfg)

    dt = float(time_cfg["dt"])
    num_steps = int(time_cfg["num_steps"])

    # 2. Synthesize Partitions with Strict Trajectory Isolation
    print("\n[1/5] Synthesizing partitioned trajectories and dataset manifest...")
    train_trajs, train_seeds = generate_trajectory_dataset(
        num_trajectories=int(traj_cfg["num_train"]), num_steps=num_steps, dt=dt,
        seed_base=int(traj_cfg["train_seed_base"]), cfg=adv_cfg, device=device,
    )
    val_trajs, val_seeds = generate_trajectory_dataset(
        num_trajectories=int(traj_cfg["num_val"]), num_steps=num_steps, dt=dt,
        seed_base=int(traj_cfg["val_seed_base"]), cfg=adv_cfg, device=device,
    )
    test_trajs, test_seeds = generate_trajectory_dataset(
        num_trajectories=int(traj_cfg["num_test"]), num_steps=num_steps, dt=dt,
        seed_base=int(traj_cfg["test_seed_base"]), cfg=adv_cfg, device=device,
    )

    isolation_metrics = verify_trajectory_split_isolation(train_trajs, val_trajs, test_trajs)
    manifest = build_trajectory_dataset_manifest(
        train_trajectories=train_trajs,
        val_trajectories=val_trajs,
        test_trajectories=test_trajs,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
        adv_cfg=adv_cfg,
        time_cfg=time_cfg,
        window_cfg=win_cfg,
    )
    manifest_path = out_dir / "pilot_dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 3. Slice into History and Target Forecasting Windows
    h_len = int(win_cfg["history_len"])
    f_len = int(win_cfg["future_len"])
    stride = int(win_cfg.get("stride", 1))

    train_hist, train_target = build_trajectory_windows(train_trajs, h_len, f_len, stride)
    val_hist, val_target = build_trajectory_windows(val_trajs, h_len, f_len, stride)

    print(f"      Train windows: {train_hist.shape[0]} samples (hist: {train_hist.shape}, target: {train_target.shape})")
    print(f"      Val windows:   {val_hist.shape[0]} samples (hist: {val_hist.shape}, target: {val_target.shape})")

    # 4. Load Frozen Vorticity Autoencoder Checkpoint (Verification Contract 1)
    print("\n[2/5] Loading frozen Autoencoder and verifying checkpoint integrity...")
    ae_ckpt_path = Path(cfg["checkpoints"]["ae_checkpoint"])
    if not ae_ckpt_path.is_file():
        raise FileNotFoundError(f"Required Autoencoder checkpoint not found: {ae_ckpt_path}")

    sha_before = compute_file_sha256(ae_ckpt_path)
    latent_channels = int(model_cfg.get("latent_channels", 64))

    forecaster = VorticityLatentForecaster.from_checkpoint(
        checkpoint_path=ae_ckpt_path,
        latent_channels=latent_channels,
        transformer_kwargs={
            "embed_dim": int(model_cfg.get("embed_dim", 128)),
            "cond_dim": int(model_cfg.get("cond_dim", 64)),
            "depth": int(model_cfg.get("depth", 4)),
            "num_heads": int(model_cfg.get("num_heads", 4)),
            "history_length": h_len,
            "prediction_mode": str(model_cfg.get("prediction_mode", "direct")),
        },
        device=device,
    )

    # Assert freeze: encoder/decoder parameters requires_grad must be False
    for name, p in forecaster.autoencoder.named_parameters():
        assert not p.requires_grad, f"Autoencoder parameter {name} is not frozen!"

    # 5. Pre-encode Latent States with Frozen Encoder
    print("\n[3/5] Pre-encoding physical trajectories into latent space...")
    with torch.no_grad():
        z_train_hist = forecaster.autoencoder.encoder(train_hist)       # (N_tr, L, C_z, H_z, W_z)
        z_train_target = forecaster.autoencoder.encoder(train_target)   # (N_tr, 1, C_z, H_z, W_z)
        z_val_hist = forecaster.autoencoder.encoder(val_hist)           # (N_val, L, C_z, H_z, W_z)
        z_val_target = forecaster.autoencoder.encoder(val_target)       # (N_val, 1, C_z, H_z, W_z)

    print(f"      Latent spatial resolution: {z_train_hist.shape[-2]}x{z_train_hist.shape[-1]}")
    print(f"      Latent channel dimension: {z_train_hist.shape[2]}")

    # 6. Train LatentSTTransformer
    print("\n[4/5] Training LatentSTTransformer for deterministic state transition...")
    batch_size = int(train_cfg.get("batch_size", 16))
    train_dataset = TensorDataset(z_train_hist, z_train_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    lr = float(train_cfg.get("lr", 5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(forecaster.transformer.parameters(), lr=lr, weight_decay=weight_decay)

    num_epochs = epochs_override if epochs_override is not None else int(train_cfg.get("epochs", 20))
    loss_history: List[float] = []

    forecaster.transformer.train()
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        for batch_z_hist, batch_z_target in train_loader:
            optimizer.zero_grad()
            z_pred = forecaster.transformer(batch_z_hist)
            loss = F.mse_loss(z_pred, batch_z_target)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        loss_history.append(avg_loss)
        if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
            print(f"      Epoch {epoch:02d}/{num_epochs:02d} | Latent MSE Loss: {avg_loss:.6e}")

    train_duration = time.time() - start_time
    print(f"      Training completed in {train_duration:.2f}s | Initial Loss: {loss_history[0]:.6e} -> Final: {loss_history[-1]:.6e}")

    # Verify zero gradient leakage to autoencoder
    for name, p in forecaster.autoencoder.named_parameters():
        assert p.grad is None, f"Autoencoder parameter {name} received unwanted gradients during training!"

    # 7. Rigorous Evaluation on Trajectory-Isolated Validation Set
    print("\n[5/5] Evaluating on isolated Validation set (Triple-Trajectory & Baseline Comparison)...")
    forecaster.transformer.eval()
    forecaster.autoencoder.eval()

    with torch.no_grad():
        # A. Predictions
        z_val_pred = forecaster.transformer(z_val_hist)       # (N_val, 1, C_z, H_z, W_z)
        q_val_pred = forecaster.decode(z_val_pred)            # (N_val, 1, 1, H, W)
        q_val_ae_ref = forecaster.decode(z_val_target)        # (N_val, 1, 1, H, W)
        q_val_persist = val_hist[:, -1:]                      # (N_val, 1, 1, H, W)
        q_val_gt = val_target                                 # (N_val, 1, 1, H, W)

        # B. Spatial Relative L2 Errors (Per Sample)
        def batch_rel_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
            diff_norm = torch.norm(pred - ref, p=2, dim=(-2, -1))
            ref_norm = torch.norm(ref, p=2, dim=(-2, -1))
            return float(torch.mean(diff_norm / (ref_norm + 1e-8)).item())

        model_rel_l2 = batch_rel_l2(q_val_pred, q_val_gt)
        persist_rel_l2 = batch_rel_l2(q_val_persist, q_val_gt)
        pure_dyn_rel_l2 = batch_rel_l2(q_val_pred, q_val_ae_ref)
        rep_rel_l2 = batch_rel_l2(q_val_ae_ref, q_val_gt)

        # C. Latent Space Relative Error
        latent_rel_l2 = float(torch.mean(
            torch.norm(z_val_pred - z_val_target, p=2, dim=(-3, -2, -1)) /
            (torch.norm(z_val_target, p=2, dim=(-3, -2, -1)) + 1e-8)
        ).item())

        # D. Physical Enstrophy & Dissipation Tracking
        z_gt = solver.compute_enstrophy(q_val_gt)
        z_pred = solver.compute_enstrophy(q_val_pred)
        z_ae = solver.compute_enstrophy(q_val_ae_ref)
        z_hist_last = solver.compute_enstrophy(q_val_persist)

        mean_z_gt = float(torch.mean(z_gt).item())
        mean_z_pred = float(torch.mean(z_pred).item())
        mean_z_ae = float(torch.mean(z_ae).item())
        mean_z_hist = float(torch.mean(z_hist_last).item())

        diss_gt = solver.compute_enstrophy_dissipation_rate(q_val_gt)
        diss_pred = solver.compute_enstrophy_dissipation_rate(q_val_pred)
        diss_hist = solver.compute_enstrophy_dissipation_rate(q_val_persist)

        mean_diss_gt = float(torch.mean(diss_gt).item())
        mean_diss_pred = float(torch.mean(diss_pred).item())
        mean_diss_hist = float(torch.mean(diss_hist).item())

        # Enstrophy budget residual across the single transition step
        r_gt = compute_enstrophy_budget_residual(mean_z_hist, mean_z_gt, mean_diss_hist, mean_diss_gt, dt=dt)
        r_pred = compute_enstrophy_budget_residual(mean_z_hist, mean_z_pred, mean_diss_hist, mean_diss_pred, dt=dt)

    # Checkpoint Immutability Verification (Verification Contract 1)
    sha_after = compute_file_sha256(ae_ckpt_path)
    assert sha_before == sha_after, "CRITICAL: Autoencoder checkpoint on disk was modified during pilot training!"

    model_beats_persistence = bool(model_rel_l2 < persist_rel_l2)
    error_reduction_pct = float((persist_rel_l2 - model_rel_l2) / persist_rel_l2 * 100.0)

    print("\n=================== Phase 2 A Pilot Evaluation Results ===================")
    print(f"    Persistence Baseline Rel L2 : {persist_rel_l2 * 100:.2f}% (Identity naive forecast)")
    print(f"    Transformer Model Rel L2    : {model_rel_l2 * 100:.2f}% (Learned state transition)")
    print(f"    Relative Error Reduction    : {error_reduction_pct:.2f}% vs Persistence")
    print(f"    State Transition Verified   : {'PASS (Outperforms Persistence)' if model_beats_persistence else 'FAIL'}")
    print(f"    ----------------------------------------------------------------------")
    print(f"    Error Decomposition:")
    print(f"      - Frozen AE Rep Error     : {rep_rel_l2 * 100:.2f}% (||D(E(q)) - q|| / ||q||)")
    print(f"      - Pure Dynamics Error     : {pure_dyn_rel_l2 * 100:.2f}% (||D(hat_z) - D(E(q))|| / ||D(E(q))||)")
    print(f"      - Latent State Rel L2     : {latent_rel_l2 * 100:.2f}% (||hat_z - E(q)|| / ||E(q)||)")
    print(f"    ----------------------------------------------------------------------")
    print(f"    Enstrophy Physics:")
    print(f"      - GT Mean Enstrophy       : {mean_z_gt:.6f}")
    print(f"      - Pred Mean Enstrophy     : {mean_z_pred:.6f}")
    print(f"      - AE Ref Mean Enstrophy   : {mean_z_ae:.6f}")
    print(f"      - GT Budget Residual      : {r_gt:.6e}")
    print(f"      - Pred Budget Residual    : {r_pred:.6e}")
    print("==========================================================================")

    # 8. Save Transformer Checkpoint & Summary Artifacts
    model_ckpt_path = out_dir / "latest_checkpoint.pt"
    torch.save(
        {
            "transformer_state_dict": forecaster.transformer.state_dict(),
            "latent_channels": latent_channels,
            "embed_dim": forecaster.transformer.embed_dim,
            "history_length": h_len,
            "completed_epoch": num_epochs,
            "final_train_loss": loss_history[-1],
            "val_model_rel_l2": model_rel_l2,
            "ae_checkpoint_sha256": sha_after,
        },
        model_ckpt_path,
    )

    summary: Dict[str, Any] = {
        "metadata": {
            "experiment_name": cfg["experiment"]["name"],
            "protocol": cfg["experiment"]["protocol"],
            "phase": "2A_deterministic_pilot",
            "execution_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "device": dev_str,
            "domain": dom_cfg,
            "physical_parameters": phys_cfg,
            "temporal_parameters": time_cfg,
            "window_parameters": win_cfg,
            "training_parameters": train_cfg,
        },
        "checkpoint_integrity": {
            "ae_checkpoint_path": str(ae_ckpt_path),
            "ae_checkpoint_sha256": sha_after,
            "ae_checkpoint_unmutated": bool(sha_before == sha_after),
            "transformer_checkpoint_path": str(model_ckpt_path),
        },
        "training_metrics": {
            "num_epochs": num_epochs,
            "initial_loss": loss_history[0],
            "final_loss": loss_history[-1],
            "training_duration_seconds": train_duration,
            "loss_history": loss_history,
        },
        "evaluation_metrics": {
            "model_rel_l2": model_rel_l2,
            "persistence_rel_l2": persist_rel_l2,
            "error_reduction_percent_vs_persistence": error_reduction_pct,
            "model_beats_persistence": model_beats_persistence,
            "state_transition_verdict": "PASS" if model_beats_persistence else "FAIL",
            "pure_dynamics_rel_l2": pure_dyn_rel_l2,
            "representation_rel_l2": rep_rel_l2,
            "latent_rel_l2": latent_rel_l2,
            "physical_enstrophy": {
                "mean_gt_enstrophy": mean_z_gt,
                "mean_pred_enstrophy": mean_z_pred,
                "mean_ae_ref_enstrophy": mean_z_ae,
                "gt_budget_residual": r_gt,
                "pred_budget_residual": r_pred,
            },
        },
    }

    summary_file = out_dir / "latent_forecaster_pilot_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved pilot model checkpoint to: {model_ckpt_path}")
    print(f"Saved complete pilot summary to: {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 2 A Deterministic Latent Forecaster Pilot")
    parser.add_argument("--config", type=str, default="configs/experiment/latent_forecaster_pilot.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = PROJECT_ROOT / args.config

    out_override = Path(args.output_dir) if args.output_dir else None
    run_latent_forecaster_pilot(
        config_path=cfg_path,
        device_str=args.device,
        output_dir_override=out_override,
        epochs_override=args.epochs,
    )


if __name__ == "__main__":
    main()
