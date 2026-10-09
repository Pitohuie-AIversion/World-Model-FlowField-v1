#!/usr/bin/env python3
"""Latent representation capacity comparison experiment for single-channel physical vorticity autoencoder.

Systematically trains and evaluates VorticityAutoencoder across three latent capacity levels:
- C_z = 64: Latent shape [64, 8, 8], 4096 elements, element ratio 1.0x (Baseline, spatial-to-channel rearrangement)
- C_z = 32: Latent shape [32, 8, 8], 2048 elements, element ratio 2.0x (2x compression, elements halved)
- C_z = 16: Latent shape [16, 8, 8], 1024 elements, element ratio 4.0x (4x aggressive compression)

Evaluates 5 standardized scientific metric dimensions under strictly controlled data and training parameters:
1. Relative and absolute L2 reconstruction error
2. Pairwise sample difference retention error (PDE)
3. Pointwise sample variance retention ratio (VR)
4. Radial enstrophy spectrum preservation ratio & spurious high-frequency energy
5. Computational overhead (parameter count, training duration, inference latency)

Zero dependency on real StocBench data; strictly operates on synthetic dev pipeline.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
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

from scripts.train_vorticity_autoencoder import (
    generate_synthetic_vorticity_dataset,
    train_vorticity_autoencoder,
)
from src.metrics.vorticity_representation import (
    compute_batched_radial_enstrophy_spectrum,
    compute_enstrophy_spectrum_ratio,
    compute_pairwise_difference_error,
    compute_pointwise_variance_ratio,
    compute_relative_l2_error,
)


def run_single_capacity_evaluation(
    model: nn.Module,
    val_data: torch.Tensor,
    device: torch.device,
    domain_size: tuple = (1.0, 1.0),
) -> Dict[str, Any]:
    """Perform rigorous 5-dimensional representation evaluation on frozen validation data."""
    model.eval()
    val_data_dev = val_data.to(device)

    # Measure inference latency over validation set
    t0 = time.perf_counter()
    with torch.no_grad():
        recon_dev = model(val_data_dev)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_infer_time = time.perf_counter() - t0
    latency_ms_per_sample = (total_infer_time / val_data.shape[0]) * 1000.0

    recon_cpu = recon_dev.detach().cpu()
    target_cpu = val_data.cpu()

    # 1. Spatial reconstruction error
    l2_metrics = compute_relative_l2_error(recon_cpu, target_cpu)

    # 2. Pointwise variance retention
    var_metrics = compute_pointwise_variance_ratio(recon_cpu, target_cpu)

    # 3. Pairwise difference preservation error
    pair_metrics = compute_pairwise_difference_error(recon_cpu, target_cpu)

    # 4. Radial enstrophy spectrum preservation
    spec_ratio_res = compute_enstrophy_spectrum_ratio(recon_cpu, target_cpu, domain_size=domain_size)

    valid_ratios = [r for r in spec_ratio_res["spectrum_ratio"] if r is not None]
    mean_valid_spectrum_ratio = float(np.mean(valid_ratios)) if valid_ratios else None

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    return {
        "relative_l2": l2_metrics["relative_l2"],
        "absolute_l2": l2_metrics["absolute_l2"],
        "variance_ratio": var_metrics["variance_ratio"],
        "mean_target_variance": var_metrics["mean_target_variance"],
        "mean_pred_variance": var_metrics["mean_pred_variance"],
        "pairwise_difference_error": pair_metrics["pairwise_difference_error"],
        "mean_absolute_difference_error": pair_metrics["mean_absolute_difference_error"],
        "mean_valid_spectrum_ratio": mean_valid_spectrum_ratio,
        "spurious_energy_in_zero_bins": spec_ratio_res["spurious_energy_in_zero_bins"],
        "k_bins": spec_ratio_res["k_bins"],
        "spectrum_target": spec_ratio_res["spectrum_target"],
        "spectrum_pred": spec_ratio_res["spectrum_pred"],
        "spectrum_ratio": spec_ratio_res["spectrum_ratio"],
        "total_parameters": total_params,
        "latency_ms_per_sample": latency_ms_per_sample,
        "reconstruction_tensor": recon_cpu,
    }


def generate_comparison_plots(
    results: Dict[int, Dict[str, Any]],
    val_data: torch.Tensor,
    output_dir: Path,
) -> List[str]:
    """Generate publication-grade figures comparing metric degradation, field reconstruction, and spectra."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cz_list = sorted(results.keys(), reverse=True)  # [64, 32, 16]
    generated_files = []

    # --------------------------------------------------------------------------
    # Figure 1: Quantitative Bar Chart Comparison
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=200)

    labels = [f"Cz={cz}\n({results[cz]['capacity_metadata']['element_compression_ratio']}x)" for cz in cz_list]
    rel_l2s = [results[cz]["evaluation"]["relative_l2"] * 100.0 for cz in cz_list]
    pdes = [results[cz]["evaluation"]["pairwise_difference_error"] * 100.0 for cz in cz_list]
    vrs = [results[cz]["evaluation"]["variance_ratio"] for cz in cz_list]
    params_k = [results[cz]["evaluation"]["total_parameters"] / 1000.0 for cz in cz_list]

    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    # (a) Relative L2 Error
    bars1 = axes[0, 0].bar(labels, rel_l2s, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    axes[0, 0].set_title("(a) Relative Reconstruction Error (% L2)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Relative L2 Error (%)")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9)

    # (b) Pairwise Difference Error (PDE)
    bars2 = axes[0, 1].bar(labels, pdes, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    axes[0, 1].set_title("(b) Pairwise Difference Error (% PDE)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Pairwise Difference Error (%)")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9)

    # (c) Variance Retention Ratio
    bars3 = axes[1, 0].bar(labels, vrs, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    axes[1, 0].axhline(1.0, color="gray", linestyle=":", label="Ideal (1.0)")
    axes[1, 0].set_title("(c) Ensemble Variance Retention (VR)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Variance Ratio (Var_pred / Var_gt)")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1, 0].legend(loc="lower right")
    for bar in bars3:
        yval = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom", fontsize=9)

    # (d) Parameter Count
    bars4 = axes[1, 1].bar(labels, params_k, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    axes[1, 1].set_title("(d) Model Parameter Count (k)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Parameters (x 1,000)")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars4:
        yval = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2.0, yval + 2.0, f"{yval:.1f}k", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Representation Capacity Trade-Off on Synthetic Vorticity Fields", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    metrics_path = output_dir / "capacity_comparison_metrics.png"
    plt.savefig(metrics_path, bbox_inches="tight")
    plt.close()
    generated_files.append(str(metrics_path))

    # --------------------------------------------------------------------------
    # Figure 2: Radial Enstrophy Spectra Comparison
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    ref_k = results[cz_list[0]]["evaluation"]["k_bins"]
    ref_spec_tgt = results[cz_list[0]]["evaluation"]["spectrum_target"]

    # Target ground truth spectrum
    ax.plot(ref_k, ref_spec_tgt, color="black", linewidth=2.2, label="Ground Truth (Target)", linestyle="-")

    styles = [("--", "#2b5c8f"), ("-.", "#d95f02"), (":", "#7570b3")]
    for cz, (ls, col) in zip(cz_list, styles):
        spec_pred = results[cz]["evaluation"]["spectrum_pred"]
        ratio_str = f"Ratio: {results[cz]['evaluation']['mean_valid_spectrum_ratio']:.3f}" if results[cz]["evaluation"]["mean_valid_spectrum_ratio"] else ""
        ax.plot(ref_k, spec_pred, color=col, linestyle=ls, linewidth=1.8, label=f"Recon Cz={cz} ({ratio_str})")

    ax.set_yscale("log")
    ax.set_title("Radial Enstrophy Spectrum E(k) Across Latent Capacities", fontsize=12, fontweight="bold")
    ax.set_xlabel("Wavenumber k", fontsize=10)
    ax.set_ylabel("Radial Enstrophy Spectrum E(k) [log scale]", fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    spectra_path = output_dir / "capacity_comparison_spectra.png"
    plt.savefig(spectra_path, bbox_inches="tight")
    plt.close()
    generated_files.append(str(spectra_path))

    # --------------------------------------------------------------------------
    # Figure 3: Spatial Vorticity Reconstruction & Absolute Error Fields
    # --------------------------------------------------------------------------
    sample_idx = 0
    gt_sample = val_data[sample_idx, 0].cpu().numpy()
    vmin, vmax = gt_sample.min(), gt_sample.max()

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), dpi=200)

    # Row 0: Ground Truth & Reconstructed Fields
    im0 = axes[0, 0].imshow(gt_sample, cmap="PRGn", vmin=vmin, vmax=vmax, origin="lower")
    axes[0, 0].set_title("Ground Truth ω", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    for col_idx, cz in enumerate(cz_list, start=1):
        recon_sample = results[cz]["evaluation"]["reconstruction_tensor"][sample_idx, 0].numpy()
        im = axes[0, col_idx].imshow(recon_sample, cmap="PRGn", vmin=vmin, vmax=vmax, origin="lower")
        axes[0, col_idx].set_title(f"Recon Cz={cz} ({results[cz]['capacity_metadata']['element_compression_ratio']}x)", fontsize=11)
        axes[0, col_idx].axis("off")
        plt.colorbar(im, ax=axes[0, col_idx], fraction=0.046, pad=0.04)

    # Row 1: Absolute Error Maps (Blank for GT, Heatmaps for models)
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "Absolute Error\n|ω - ω_recon|", ha="center", va="center", fontsize=12, fontweight="bold")

    max_err = max(
        np.abs(gt_sample - results[cz]["evaluation"]["reconstruction_tensor"][sample_idx, 0].numpy()).max()
        for cz in cz_list
    )

    for col_idx, cz in enumerate(cz_list, start=1):
        err_sample = np.abs(gt_sample - results[cz]["evaluation"]["reconstruction_tensor"][sample_idx, 0].numpy())
        im_err = axes[1, col_idx].imshow(err_sample, cmap="inferno", vmin=0, vmax=max_err, origin="lower")
        rel_l2 = results[cz]["evaluation"]["relative_l2"] * 100.0
        axes[1, col_idx].set_title(f"Err Cz={cz} (Rel L2: {rel_l2:.2f}%)", fontsize=10)
        axes[1, col_idx].axis("off")
        plt.colorbar(im_err, ax=axes[1, col_idx], fraction=0.046, pad=0.04)

    plt.suptitle("Qualitative Vorticity Reconstruction vs Absolute Error Across Latent Capacities", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fields_path = output_dir / "capacity_comparison_fields.png"
    plt.savefig(fields_path, bbox_inches="tight")
    plt.close()
    generated_files.append(str(fields_path))

    return generated_files


def run_capacity_comparison_experiment(
    config_path: str = "configs/train/vorticity_autoencoder.yaml",
    capacities: List[int] = (64, 32, 16),
    output_root: str = "outputs/experiments/vorticity_capacity",
    epochs: int = 20,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete capacity comparison experiment across C_z in capacities."""
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # Device selection
    if device is None:
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev_str = device
    dev = torch.device(dev_str)

    # 1. Synthesize canonical fixed validation dataset
    synth_cfg = base_cfg.get("synthetic_data", {})
    val_data = generate_synthetic_vorticity_dataset(
        num_samples=synth_cfg.get("num_val_samples", 32),
        nx=base_cfg.get("domain", {}).get("nx", 64),
        ny=base_cfg.get("domain", {}).get("ny", 64),
        lx=base_cfg.get("domain", {}).get("lx", 1.0),
        ly=base_cfg.get("domain", {}).get("ly", 1.0),
        base_wavenumber=synth_cfg.get("base_wavenumber", 1),
        perturbation_modes=synth_cfg.get("perturbation_modes", [[1, 0], [0, 1], [1, 1], [2, 1]]),
        perturbation_amplitude=synth_cfg.get("perturbation_amplitude", 0.1),
        seed=synth_cfg.get("seed", 42),
    )

    domain_size = (
        float(base_cfg.get("domain", {}).get("lx", 1.0)),
        float(base_cfg.get("domain", {}).get("ly", 1.0)),
    )

    experiment_results = {}

    print(f"\n{'='*80}")
    print(f"STARTING VORTICITY AUTOENCODER CAPACITY COMPARISON EXPERIMENT")
    print(f"Target Latent Capacities C_z: {capacities}")
    print(f"Training Epochs per Model: {epochs}")
    print(f"Execution Device: {dev_str}")
    print(f"{'='*80}\n")

    for cz in capacities:
        print(f"\n>>> Running Training & Evaluation for C_z = {cz} ...")
        cz_dir = out_root / f"cz_{cz}"
        cz_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.perf_counter()
        train_res = train_vorticity_autoencoder(
            config=base_cfg,
            output_dir=str(cz_dir),
            override_epochs=epochs,
            override_latent_channels=cz,
            device=dev_str,
        )
        train_duration = time.perf_counter() - t_start

        # Perform comprehensive representation evaluation
        eval_res = run_single_capacity_evaluation(
            model=train_res["model"],
            val_data=val_data,
            device=dev,
            domain_size=domain_size,
        )

        experiment_results[cz] = {
            "latent_channels": cz,
            "capacity_metadata": train_res["capacity_metadata"],
            "training_summary": train_res["summary"],
            "training_duration_seconds": train_duration,
            "checkpoint_path": train_res["checkpoint_path"],
            "evaluation": eval_res,
        }

        print(f"--- Finished C_z = {cz} in {train_duration:.1f}s ---")
        print(f"    Relative L2 Error: {eval_res['relative_l2']*100:.2f}%")
        print(f"    Pairwise Difference Error (PDE): {eval_res['pairwise_difference_error']*100:.2f}%")
        print(f"    Variance Ratio (VR): {eval_res['variance_ratio']:.4f}")
        print(f"    Mean Spectrum Ratio: {eval_res['mean_valid_spectrum_ratio']:.4f}")
        print(f"    Model Parameters: {eval_res['total_parameters']:,}")

    # Generate comparison plots
    plot_files = generate_comparison_plots(experiment_results, val_data, out_root)

    # Build clean JSON serializable dictionary (omitting raw tensors)
    summary_dict = {
        "metadata": {
            "experiment_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "device": dev_str,
            "capacities_evaluated": list(capacities),
            "generated_plots": plot_files,
        },
        "models": {},
    }

    for cz, data in experiment_results.items():
        eval_copy = dict(data["evaluation"])
        eval_copy.pop("reconstruction_tensor", None)
        summary_dict["models"][str(cz)] = {
            "latent_channels": cz,
            "capacity_metadata": data["capacity_metadata"],
            "training_duration_seconds": data["training_duration_seconds"],
            "final_step_loss": data["training_summary"]["final_step_loss"],
            "checkpoint_path": data["checkpoint_path"],
            "metrics": eval_copy,
        }

    summary_file = out_root / "capacity_comparison_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"Summary JSON saved to: {summary_file}")
    for p in plot_files:
        print(f"Generated Plot: {p}")
    print(f"{'='*80}\n")

    return summary_dict


def main():
    parser = argparse.ArgumentParser(description="Run Vorticity Autoencoder Capacity Comparison Experiment")
    parser.add_argument("--config", type=str, default="configs/train/vorticity_autoencoder.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments/vorticity_capacity")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--capacities", type=int, nargs="+", default=[64, 32, 16])
    args = parser.parse_args()

    run_capacity_comparison_experiment(
        config_path=args.config,
        capacities=args.capacities,
        output_root=args.output_dir,
        epochs=args.epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
