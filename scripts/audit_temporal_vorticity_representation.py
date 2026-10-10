#!/usr/bin/env python3
"""Temporal Representation Audit for Frozen Vorticity Autoencoders.

Evaluates pre-trained, frozen autoencoders (Cz=64 and Cz=16) on dynamically evolving
periodic scalar advection-diffusion trajectories.

Scientific Objective:
Examine whether the frozen representation manifold (trained strictly on static background fields)
maintains reconstruction fidelity and physical conservation when subjected to spatial translation
and viscous dissipation over continuous time t in [0, T].

Engineering Contracts:
1. Zero retraining or weight updates: autoencoders are loaded in eval() mode with all
   gradients disabled (requires_grad = False).
2. Checkpoint immutability assertion: SHA-256 of checkpoint files is verified before and after execution.
3. Pre-declared diagnostic threshold: if frozen AE reconstruction error exceeds 20% (threshold_diagnosis_rel_l2),
   the audit flags an out-of-distribution diagnostic alert in the summary without silently masking it.
4. Physical enstrophy budget tracking: monitors both instantaneous enstrophy Z(t), instantaneous dissipation
   rate diss(t) = -nu * <|grad omega|^2>, and the enstrophy budget residual:
       R_Z(t_k) = (Z(t_{k+1}) - Z(t_k)) / dt - 0.5 * (diss(t_k) + diss(t_{k+1}))
5. Comprehensive dataset manifest: saves complete partition splits, seeds, digests, and window-to-trajectory mappings.
"""

import argparse
import datetime
import hashlib
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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_vorticity_autoencoder import VorticityAutoencoder
from src.data.synthetic_advection_diffusion import (
    AdvectionDiffusionConfig,
    PeriodicScalarAdvectionDiffusion,
    build_trajectory_dataset_manifest,
    compute_enstrophy_budget_residual,
    compute_tensor_digest,
    generate_trajectory_dataset,
    verify_trajectory_split_isolation,
)
from src.metrics.vorticity_representation import (
    compute_enstrophy_spectrum_ratio,
    compute_pairwise_difference_error,
    compute_pointwise_variance_ratio,
    compute_relative_l2_error,
)
from src.utils.fft_derivatives import spectral_grad_2d


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_frozen_autoencoder(
    checkpoint_path: Path,
    latent_channels: int,
    device: torch.device,
) -> Tuple[VorticityAutoencoder, Dict[str, Any], str]:
    """Load pre-trained VorticityAutoencoder in strictly frozen evaluation mode.

    Returns:
        model: Evaluated VorticityAutoencoder with requires_grad=False on all parameters.
        checkpoint_data: Loaded checkpoint dictionary.
        sha256_hash: SHA-256 digest of the checkpoint file.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    sha256_before = compute_file_sha256(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = VorticityAutoencoder(
        in_channels=1,
        out_channels=1,
        latent_channels=latent_channels,
        base_channels=32,
        project_pressure=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Strict freeze assertion
    for p in model.parameters():
        p.requires_grad = False

    return model, ckpt, sha256_before


def classify_enstrophy_decay_behavior(
    gt_z_total_decay_pct: float,
    recon_z_total_decay_pct: float,
    zero_decay_tolerance: float = 1e-4,
) -> Tuple[str, bool]:
    """Classify enstrophy decay behavior comparing reconstructed flow against ground truth.

    Priority Order:
        1. Unphysical Energy Growth: recon decay is negative (enstrophy increases over time)
        2. Attenuated Dissipation Bias: recon decay is less than 20% of true viscous decay
        3. Excessive Dissipation Bias: recon decay exceeds 200% of true viscous decay
        4. Decay Magnitude Comparable: recon decay falls within [20%, 200%] of true decay

    For near-zero ground truth decay (|gt_decay| < zero_decay_tolerance):
        - recon < -zero_decay_tolerance -> UNPHYSICAL_ENERGY_GROWTH
        - recon > zero_decay_tolerance -> EXCESSIVE_DISSIPATION_BIAS
        - otherwise -> DECAY_MAGNITUDE_COMPARABLE

    Returns:
        decay_diagnosis (str): One of UNPHYSICAL_ENERGY_GROWTH, ATTENUATED_DISSIPATION_BIAS,
                               EXCESSIVE_DISSIPATION_BIAS, DECAY_MAGNITUDE_COMPARABLE.
        dissipation_attenuated (bool): True if ATTENUATED_DISSIPATION_BIAS.
    """
    if abs(gt_z_total_decay_pct) < zero_decay_tolerance:
        if recon_z_total_decay_pct < -zero_decay_tolerance:
            return "UNPHYSICAL_ENERGY_GROWTH", False
        if recon_z_total_decay_pct > zero_decay_tolerance:
            return "EXCESSIVE_DISSIPATION_BIAS", False
        return "DECAY_MAGNITUDE_COMPARABLE", False

    # Check unphysical energy growth first (priority 1)
    if recon_z_total_decay_pct < 0.0:
        return "UNPHYSICAL_ENERGY_GROWTH", False
    # Check attenuated dissipation bias (priority 2)
    if recon_z_total_decay_pct < 0.20 * gt_z_total_decay_pct:
        return "ATTENUATED_DISSIPATION_BIAS", True
    # Check excessive dissipation bias (priority 3)
    if recon_z_total_decay_pct > 2.0 * gt_z_total_decay_pct:
        return "EXCESSIVE_DISSIPATION_BIAS", False
    # Decay magnitude is comparable (priority 4)
    return "DECAY_MAGNITUDE_COMPARABLE", False


def classify_physical_budget_verdict(
    decay_diagnosis: str,
    dissipation_attenuated: bool,
    max_recon_residual: float,
    threshold_budget_residual: Optional[float] = None,
) -> str:
    """Determine physical budget verdict based on residual and decay behavior.

    - If threshold_budget_residual is specified:
        - If max residual <= threshold and decay is comparable -> PHYSICALLY_BALANCED
        - Else if decay has a specific bias -> returns that diagnosis
        - Else -> PHYSICAL_BUDGET_RESIDUAL_EXCEEDED
    - If threshold_budget_residual is None:
        - If dissipation attenuated -> ATTENUATED_DISSIPATION_BIAS
        - Else if decay has a specific bias (e.g. growth/excessive) -> returns that diagnosis
        - Else -> BUDGET_CONSISTENCY_NOT_ASSESSED (never blindly returns PHYSICALLY_BALANCED)
    """
    if threshold_budget_residual is not None:
        thresh = float(threshold_budget_residual)
        if not math.isfinite(thresh) or thresh < 0.0:
            raise ValueError(f"threshold_budget_residual must be a non-negative finite float, got {thresh}")
        if max_recon_residual <= thresh and decay_diagnosis == "DECAY_MAGNITUDE_COMPARABLE":
            return "PHYSICALLY_BALANCED"
        if dissipation_attenuated:
            return "ATTENUATED_DISSIPATION_BIAS"
        if decay_diagnosis != "DECAY_MAGNITUDE_COMPARABLE":
            return decay_diagnosis
        return "PHYSICAL_BUDGET_RESIDUAL_EXCEEDED"

    if dissipation_attenuated:
        return "ATTENUATED_DISSIPATION_BIAS"
    if decay_diagnosis != "DECAY_MAGNITUDE_COMPARABLE":
        return decay_diagnosis
    return "BUDGET_CONSISTENCY_NOT_ASSESSED"


def run_temporal_representation_audit(
    config_path: Path,
    device_str: Optional[str] = None,
    override_threshold: Optional[float] = None,
    output_dir_override: Optional[Path] = None,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """Execute complete temporal representation audit pipeline."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dev_str = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_str)

    out_dir = output_dir_override or Path(cfg["evaluation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize physical dynamics configuration
    dom_cfg = cfg["domain"]
    phys_cfg = cfg["physical_params"]
    time_cfg = cfg["temporal_params"]
    traj_cfg = cfg["trajectories"]
    eval_cfg = cfg["evaluation"]
    win_cfg = cfg.get("windowing", {"history_len": 4, "future_len": 16, "stride": 1})

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
    total_time = float(time_cfg.get("total_time", dt * num_steps))
    time_points = [float(k * dt) for k in range(num_steps + 1)]

    # 2. Synthesize partitioned datasets (Train, Val, Test) strictly by trajectory seeds
    num_train = int(traj_cfg["num_train"])
    num_val = int(traj_cfg["num_val"])
    num_test = int(traj_cfg["num_test"])

    train_seed_base = int(traj_cfg["train_seed_base"])
    val_seed_base = int(traj_cfg["val_seed_base"])
    test_seed_base = int(traj_cfg["test_seed_base"])

    train_trajs, train_seeds = generate_trajectory_dataset(
        num_trajectories=num_train, num_steps=num_steps, dt=dt,
        seed_base=train_seed_base, cfg=adv_cfg, device=dev,
    )
    val_trajs, val_seeds = generate_trajectory_dataset(
        num_trajectories=num_val, num_steps=num_steps, dt=dt,
        seed_base=val_seed_base, cfg=adv_cfg, device=dev,
    )
    test_trajs, test_seeds = generate_trajectory_dataset(
        num_trajectories=num_test, num_steps=num_steps, dt=dt,
        seed_base=test_seed_base, cfg=adv_cfg, device=dev,
    )

    # Build and persist complete dataset manifest
    dataset_manifest = build_trajectory_dataset_manifest(
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
    manifest_path = out_dir / "temporal_dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    val_digest = compute_tensor_digest(val_trajs)

    threshold = float(override_threshold if override_threshold is not None else eval_cfg.get("threshold_diagnosis_rel_l2", 0.20))
    capacities = list(eval_cfg.get("capacities", [64, 16]))
    ckpt_root = Path(eval_cfg.get("checkpoint_dir", "outputs/experiments/vorticity_capacity"))

    results: Dict[str, Any] = {
        "metadata": {
            "experiment_name": cfg["experiment"]["name"],
            "protocol": cfg["experiment"]["protocol"],
            "date": str(cfg["experiment"].get("date", datetime.date.today().isoformat())),
            "execution_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "device": dev_str,
            "domain": dom_cfg,
            "physical_parameters": phys_cfg,
            "temporal_parameters": time_cfg,
            "time_points": time_points,
            "threshold_diagnosis_rel_l2": threshold,
            "dataset_manifest_file": str(manifest_path),
            "dataset_isolation": dataset_manifest["isolation_metrics"],
            "validation_dataset_digest": val_digest,
            "scientific_note": (
                "Threshold diagnosis specifically checks field-level relative L2 reconstruction error. "
                "Physical enstrophy dissipation fidelity is tracked separately via enstrophy budget residuals."
            ),
        },
        "capacities_evaluated": capacities,
        "models": {},
    }

    # 3. Evaluate each frozen capacity across time points
    plot_data: Dict[int, Dict[str, Any]] = {}

    for cz in capacities:
        ckpt_path = ckpt_root / f"cz_{cz}" / "latest_checkpoint.pt"
        model, ckpt_data, sha_before = load_frozen_autoencoder(ckpt_path, cz, dev)

        # Confirm zero gradient update capability
        for name, param in model.named_parameters():
            if param.requires_grad:
                raise RuntimeError(f"Parameter {name} has requires_grad=True in frozen model Cz={cz}!")

        # Step-by-step metrics tracking
        rel_l2_series = []
        abs_l2_series = []
        pde_series = []
        vr_series = []
        gt_enstrophy_series = []
        recon_enstrophy_series = []
        enstrophy_ratio_series = []
        gt_dissipation_series = []
        recon_dissipation_series = []
        dissipation_ratio_series = []

        with torch.no_grad():
            for k in range(num_steps + 1):
                t_k = time_points[k]
                gt_frame = val_trajs[:, k]  # (N_val, 1, nx, ny)
                recon_frame = model(gt_frame)  # (N_val, 1, nx, ny)

                # Accuracy metrics
                rel_res = compute_relative_l2_error(recon_frame, gt_frame)
                rel_l2_series.append(float(rel_res["relative_l2"]))
                abs_l2_series.append(float(rel_res["absolute_l2"]))

                pde_res = compute_pairwise_difference_error(recon_frame, gt_frame)
                pde_series.append(float(pde_res["pairwise_difference_error"]))

                vr_res = compute_pointwise_variance_ratio(recon_frame, gt_frame)
                vr_series.append(float(vr_res["variance_ratio"]))

                # Physical Enstrophy: Z = 0.5 * mean(omega^2)
                gt_z = solver.compute_enstrophy(gt_frame).mean().item()
                recon_z = solver.compute_enstrophy(recon_frame).mean().item()
                gt_enstrophy_series.append(float(gt_z))
                recon_enstrophy_series.append(float(recon_z))
                enstrophy_ratio_series.append(float(recon_z / (gt_z + 1e-12)))

                # Enstrophy Dissipation Rate: diss = -nu * mean(|grad omega|^2) (<= 0)
                gt_diss = solver.compute_enstrophy_dissipation_rate(gt_frame).mean().item()
                recon_diss = solver.compute_enstrophy_dissipation_rate(recon_frame).mean().item()
                gt_dissipation_series.append(float(gt_diss))
                recon_dissipation_series.append(float(recon_diss))
                dissipation_ratio_series.append(float(recon_diss / (gt_diss - 1e-12)))

        # Immutability check
        sha_after = compute_file_sha256(ckpt_path)
        if sha_before != sha_after:
            raise RuntimeError(
                f"Checkpoint SHA-256 mutation detected for Cz={cz}! "
                f"Before: {sha_before}, After: {sha_after}."
            )

        # 4. Enstrophy budget residual calculation
        # R_Z(t_k) = (Z(t_{k+1}) - Z(t_k)) / dt - 0.5 * (diss_k + diss_{k+1})
        gt_delta_rate_series = []
        recon_delta_rate_series = []
        gt_budget_residuals = []
        recon_budget_residuals = []

        for k in range(num_steps):
            gt_delta_rate = (gt_enstrophy_series[k + 1] - gt_enstrophy_series[k]) / dt
            recon_delta_rate = (recon_enstrophy_series[k + 1] - recon_enstrophy_series[k]) / dt
            gt_delta_rate_series.append(float(gt_delta_rate))
            recon_delta_rate_series.append(float(recon_delta_rate))

            r_gt = compute_enstrophy_budget_residual(
                gt_enstrophy_series[k], gt_enstrophy_series[k + 1],
                gt_dissipation_series[k], gt_dissipation_series[k + 1],
                dt=dt,
            )
            r_recon = compute_enstrophy_budget_residual(
                recon_enstrophy_series[k], recon_enstrophy_series[k + 1],
                recon_dissipation_series[k], recon_dissipation_series[k + 1],
                dt=dt,
            )
            gt_budget_residuals.append(float(r_gt))
            recon_budget_residuals.append(float(r_recon))

        max_rel_l2 = max(rel_l2_series)
        mean_rel_l2 = float(np.mean(rel_l2_series))
        min_rel_l2 = min(rel_l2_series)
        error_drift = float(rel_l2_series[-1] - rel_l2_series[0])

        is_alert = bool(max_rel_l2 > threshold)

        # Physical enstrophy decay comparison
        gt_z_total_decay_pct = float((gt_enstrophy_series[0] - gt_enstrophy_series[-1]) / gt_enstrophy_series[0] * 100.0)
        recon_z_total_decay_pct = float((recon_enstrophy_series[0] - recon_enstrophy_series[-1]) / recon_enstrophy_series[0] * 100.0)
        max_recon_residual = float(max(abs(r) for r in recon_budget_residuals))
        mean_recon_residual = float(np.mean(recon_budget_residuals))

        # 1. Decay magnitude diagnosis (prioritizes unphysical energy growth before attenuation)
        decay_diagnosis, dissipation_attenuated = classify_enstrophy_decay_behavior(
            gt_z_total_decay_pct=gt_z_total_decay_pct,
            recon_z_total_decay_pct=recon_z_total_decay_pct,
        )

        # 2. Physical budget consistency verdict (requires explicit threshold to grant PHYSICALLY_BALANCED)
        threshold_budget_res = eval_cfg.get("threshold_budget_residual", None)
        budget_verdict = classify_physical_budget_verdict(
            decay_diagnosis=decay_diagnosis,
            dissipation_attenuated=dissipation_attenuated,
            max_recon_residual=max_recon_residual,
            threshold_budget_residual=threshold_budget_res,
        )

        results["models"][str(cz)] = {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": sha_after,
            "frozen_verification": {
                "eval_mode": True,
                "requires_grad_all_false": True,
                "weights_immutable": True,
            },
            "summary_metrics": {
                "t0_relative_l2": rel_l2_series[0],
                "t_final_relative_l2": rel_l2_series[-1],
                "min_relative_l2": min_rel_l2,
                "max_relative_l2": max_rel_l2,
                "mean_relative_l2": mean_rel_l2,
                "error_drift": error_drift,
                "threshold_declared": threshold,
                "diagnostic_alert_triggered": is_alert,
                "diagnostic_verdict": (
                    "REPRESENTATION_ERROR_EXCEEDED_THRESHOLD"
                    if is_alert else "WITHIN_ACCEPTABLE_BOUNDS"
                ),
                "enstrophy_physics": {
                    "gt_total_decay_percent": gt_z_total_decay_pct,
                    "recon_total_decay_percent": recon_z_total_decay_pct,
                    "max_enstrophy_budget_residual": max_recon_residual,
                    "mean_enstrophy_budget_residual": mean_recon_residual,
                    "decay_magnitude_diagnosis": decay_diagnosis,
                    "dissipation_attenuation_bias": dissipation_attenuated,
                    "threshold_budget_residual_declared": threshold_budget_res,
                    "physical_budget_verdict": budget_verdict,
                },
            },
            "time_series": {
                "relative_l2": rel_l2_series,
                "absolute_l2": abs_l2_series,
                "pairwise_difference_error": pde_series,
                "variance_ratio": vr_series,
                "ground_truth_enstrophy": gt_enstrophy_series,
                "reconstructed_enstrophy": recon_enstrophy_series,
                "enstrophy_retention_ratio": enstrophy_ratio_series,
                "ground_truth_dissipation_rate": gt_dissipation_series,
                "reconstructed_dissipation_rate": recon_dissipation_series,
                "dissipation_rate_ratio": dissipation_ratio_series,
                "gt_enstrophy_delta_rate": gt_delta_rate_series,
                "recon_enstrophy_delta_rate": recon_delta_rate_series,
                "gt_enstrophy_budget_residuals": gt_budget_residuals,
                "recon_enstrophy_budget_residuals": recon_budget_residuals,
            },
        }

        plot_data[cz] = {
            "rel_l2": rel_l2_series,
            "pde": pde_series,
            "gt_enstrophy": gt_enstrophy_series,
            "recon_enstrophy": recon_enstrophy_series,
            "gt_diss": gt_dissipation_series,
            "recon_diss": recon_dissipation_series,
            "gt_residuals": gt_budget_residuals,
            "recon_residuals": recon_budget_residuals,
            "gt_delta_rate": gt_delta_rate_series,
            "recon_delta_rate": recon_delta_rate_series,
        }

        print(f"=== Evaluated Frozen Capacity Cz={cz} on Dynamic Trajectories ===")
        print(f"    t=0.00s Rel L2: {rel_l2_series[0]*100:.2f}% | PDE: {pde_series[0]*100:.2f}%")
        print(f"    t={total_time:.2f}s Rel L2: {rel_l2_series[-1]*100:.2f}% | PDE: {pde_series[-1]*100:.2f}%")
        print(f"    GT Enstrophy Decay: {gt_z_total_decay_pct:.2f}% | Recon Enstrophy Decay: {recon_z_total_decay_pct:.2f}% -> {decay_diagnosis}")
        print(f"    Physical Budget Verdict: {budget_verdict}")

    # 5. Generate Visualization Plots
    generated_plots = []
    if generate_plots:
        # Plot 1: Reconstruction Error & PDE over Time
        plt.figure(figsize=(10, 5), dpi=150)
        t_arr = np.array(time_points)
        for cz, pdata in plot_data.items():
            plt.plot(t_arr, np.array(pdata["rel_l2"]) * 100, label=f"Cz={cz} Rel L2 (%)", lw=2)
            plt.plot(t_arr, np.array(pdata["pde"]) * 100, label=f"Cz={cz} PDE (%)", lw=1.5, linestyle="--")

        plt.axhline(threshold * 100, color="red", linestyle=":", lw=1.5, label=f"Diagnosis Threshold ({threshold*100:.0f}%)")
        plt.xlabel("Physical Evolution Time t (seconds)")
        plt.ylabel("Error (%)")
        plt.title("Frozen Autoencoder Reconstruction Error vs Time\n(Periodic Scalar Advection-Diffusion)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot1_path = out_dir / "reconstruction_error_vs_time.png"
        plt.savefig(plot1_path)
        plt.close()
        generated_plots.append(str(plot1_path))

        # Plot 2: Enstrophy Evolution & Time Rate vs Instantaneous Dissipation
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        ref_gt_z = plot_data[capacities[0]]["gt_enstrophy"]
        axes[0].plot(t_arr, ref_gt_z, "k-", lw=2.5, label="Analytical Ground Truth")
        for cz, pdata in plot_data.items():
            axes[0].plot(t_arr, pdata["recon_enstrophy"], label=f"Frozen Recon Cz={cz}", lw=1.8, linestyle="--")
        axes[0].set_xlabel("Time t (s)")
        axes[0].set_ylabel("Mean Enstrophy Z(t)")
        axes[0].set_title("Instantaneous Enstrophy Evolution Z(t)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        t_mid = t_arr[:-1] + 0.5 * dt
        ref_gt_delta = plot_data[capacities[0]]["gt_delta_rate"]
        ref_gt_diss = plot_data[capacities[0]]["gt_diss"]
        axes[1].plot(t_mid, ref_gt_delta, "k-", lw=2.5, label="GT dZ/dt (Delta Z / dt)")
        axes[1].plot(t_arr, ref_gt_diss, "k:", lw=1.5, label="GT Dissipation Rate diss(t)")
        for cz, pdata in plot_data.items():
            axes[1].plot(t_mid, pdata["recon_delta_rate"], label=f"Recon Cz={cz} dZ/dt", lw=1.8)
            axes[1].plot(t_arr, pdata["recon_diss"], label=f"Recon Cz={cz} diss(t)", lw=1.2, linestyle=":")
        axes[1].set_xlabel("Time t (s)")
        axes[1].set_ylabel("Rate")
        axes[1].set_title("Actual Time Derivative dZ/dt vs Gradient Dissipation diss(t)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        plot2_path = out_dir / "enstrophy_and_dissipation_vs_time.png"
        plt.savefig(plot2_path)
        plt.close()
        generated_plots.append(str(plot2_path))

        # Plot 3: Enstrophy Budget Residual R_Z(t)
        plt.figure(figsize=(10, 5), dpi=150)
        ref_gt_res = plot_data[capacities[0]]["gt_residuals"]
        plt.plot(t_mid, ref_gt_res, "k-o", lw=2.0, label="Ground Truth Residual R_Z(t) (Analytical O(dt^2))")
        for cz, pdata in plot_data.items():
            plt.plot(t_mid, pdata["recon_residuals"], label=f"Recon Cz={cz} Budget Residual R_Z(t)", lw=1.8, marker="x")
        plt.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
        plt.xlabel("Physical Evolution Time t (seconds)")
        plt.ylabel("Budget Residual R_Z(t)")
        plt.title("Enstrophy Budget Residual R_Z(t_k) = Delta Z / dt - 0.5 * (diss_k + diss_{k+1})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot3_path = out_dir / "enstrophy_budget_residuals_vs_time.png"
        plt.savefig(plot3_path)
        plt.close()
        generated_plots.append(str(plot3_path))

    results["generated_plots"] = generated_plots

    summary_file = out_dir / "temporal_audit_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved complete temporal audit summary to: {summary_file}")
    print(f"Saved complete dataset manifest to: {manifest_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen vorticity autoencoders on temporal dynamics.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/temporal_representation_audit.yaml",
        help="Path to experiment configuration YAML.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda').")
    parser.add_argument("--threshold", type=float, default=None, help="Override diagnostic threshold.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else None
    run_temporal_representation_audit(
        config_path=cfg_path,
        device_str=args.device,
        override_threshold=args.threshold,
        output_dir_override=out_dir,
        generate_plots=not args.no_plot,
    )


if __name__ == "__main__":
    main()
