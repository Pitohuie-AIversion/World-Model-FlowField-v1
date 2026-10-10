#!/usr/bin/env python3
"""Latent representation capacity comparison experiment for single-channel physical vorticity autoencoder.

Systematically trains and evaluates VorticityAutoencoder across three latent capacity levels:
- C_z = 64: Latent shape [64, 8, 8], 4096 elements, element ratio 1.0x (Reference configuration, spatial-to-channel rearrangement)
- C_z = 32: Latent shape [32, 8, 8], 2048 elements, element ratio 2.0x (2x compression, elements halved)
- C_z = 16: Latent shape [16, 8, 8], 1024 elements, element ratio 4.0x (4x compact representation)

Rigorous Experimental Controls:
- Training seed: 42 (128 samples); Independent Validation seed: 1042 (32 samples, seed + 1000)
- Fail-closed dataset separation verification: zero sample overlap mathematically enforced
- Separate benchmarking for single-request latency (batch_size=1) and batch efficiency (batch_size=32)
- Distinct reporting of arithmetic mean of valid spectral bins vs total enstrophy retention ratio
- Supports --eval-only mode to evaluate pre-existing checkpoints without retraining.

Zero dependency on real StocBench data; strictly operates on synthetic dev pipeline.
"""

import argparse
import hashlib
import json
import os
import subprocess
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
    VorticityAutoencoder,
    compute_capacity_metadata,
    generate_synthetic_vorticity_dataset,
    train_vorticity_autoencoder,
)
from src.metrics.vorticity_representation import (
    compute_enstrophy_spectrum_ratio,
    compute_pairwise_difference_error,
    compute_pointwise_variance_ratio,
    compute_relative_l2_error,
)


def get_git_commit_hash() -> str:
    """Retrieve current Git commit SHA or fallback if not available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_tensor_digest(tensor: torch.Tensor) -> Dict[str, Any]:
    """Compute deterministic SHA-256 fingerprint and summary statistics of a PyTorch tensor."""
    raw_bytes = tensor.contiguous().cpu().numpy().tobytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    return {
        "sha256": sha256,
        "shape": list(tensor.shape),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def verify_dataset_separation(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    min_dist_threshold: float = 1e-3,
) -> float:
    """Fail-closed assertion that no validation sample is identical or near-identical to any training sample.

    Args:
        train_data: Tensor of shape (N_train, C, H, W)
        val_data: Tensor of shape (N_val, C, H, W)
        min_dist_threshold: Minimum allowed pairwise L2 distance.

    Returns:
        float: Minimum pairwise L2 distance found across all (val, train) pairs.

    Raises:
        ValueError: If any validation sample has L2 distance < min_dist_threshold to a training sample.
    """
    n_train = train_data.shape[0]
    n_val = val_data.shape[0]

    train_flat = train_data.view(n_train, -1)
    val_flat = val_data.view(n_val, -1)

    min_l2_dist = float("inf")
    closest_pair = (-1, -1)

    for i in range(n_val):
        diff = train_flat - val_flat[i : i + 1]  # (N_train, D)
        dists = torch.linalg.norm(diff, dim=1)  # (N_train,)
        min_d = float(torch.min(dists).item())
        min_idx = int(torch.argmin(dists).item())
        if min_d < min_l2_dist:
            min_l2_dist = min_d
            closest_pair = (i, min_idx)

    if min_l2_dist < min_dist_threshold:
        raise ValueError(
            f"Data leakage detected! Validation sample {closest_pair[0]} is identical or near-identical "
            f"to training sample {closest_pair[1]} (L2 distance = {min_l2_dist:.6e} < {min_dist_threshold}). "
            f"Independent validation integrity compromised."
        )

    return min_l2_dist


def benchmark_model_latency(
    model: nn.Module,
    device: torch.device,
    sample_shape: Tuple[int, ...] = (1, 1, 64, 64),
    batch_shape: Tuple[int, ...] = (32, 1, 64, 64),
    warmup_runs: int = 10,
    repeat_runs: int = 50,
) -> Dict[str, Any]:
    """Measure single-request latency (batch_size=1) and batched efficiency (batch_size=32) with warmup and synchronization."""
    model.eval()

    # 1. Single-sample latency (batch_size=1)
    x_single = torch.randn(*sample_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x_single)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times_single = []
        for _ in range(repeat_runs):
            t0 = time.perf_counter()
            _ = model(x_single)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_single.append((time.perf_counter() - t0) * 1000.0)

    # 2. Batched throughput and latency (batch_size=32)
    x_batch = torch.randn(*batch_shape, device=device)
    with torch.no_grad():
        for _ in range(min(5, warmup_runs)):
            _ = model(x_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times_batch = []
        for _ in range(min(20, repeat_runs)):
            t0 = time.perf_counter()
            _ = model(x_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_batch.append((time.perf_counter() - t0) * 1000.0)

    median_single = float(np.median(times_single))
    mean_single = float(np.mean(times_single))
    std_single = float(np.std(times_single))
    p95_single = float(np.percentile(times_single, 95))

    median_batch = float(np.median(times_batch))
    batch_size = batch_shape[0]
    amortized_ms = median_batch / batch_size
    throughput_fps = (batch_size * 1000.0) / median_batch if median_batch > 0 else 0.0

    return {
        "single_request_latency": {
            "batch_size": 1,
            "median_ms": median_single,
            "mean_ms": mean_single,
            "std_ms": std_single,
            "p95_ms": p95_single,
            "num_runs": repeat_runs,
        },
        "batched_efficiency": {
            "batch_size": batch_size,
            "batch_median_ms": median_batch,
            "amortized_ms_per_sample": amortized_ms,
            "throughput_samples_per_sec": throughput_fps,
            "num_runs": len(times_batch),
        },
    }


def run_single_capacity_evaluation(
    model: nn.Module,
    val_data: torch.Tensor,
    device: torch.device,
    domain_size: Tuple[float, float] = (1.0, 1.0),
) -> Dict[str, Any]:
    """Perform rigorous 5-dimensional representation evaluation on frozen independent validation data."""
    model.eval()
    val_data_dev = val_data.to(device)

    with torch.no_grad():
        recon_dev = model(val_data_dev)
    if device.type == "cuda":
        torch.cuda.synchronize()

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

    # Calculate Band-Integrated Enstrophy Spectrum Ratio: sum(E_pred) / sum(E_target) across valid wavenumber shells
    total_e_target = sum(spec_ratio_res["spectrum_target"])
    total_e_pred = sum(spec_ratio_res["spectrum_pred"])
    band_integrated_enstrophy_ratio = float(total_e_pred / total_e_target) if total_e_target > 0 else None

    # Cross-verify with physical-space domain-integrated enstrophy ratio: 0.5 * mean(recon^2) / (0.5 * mean(target^2))
    target_phys_ens = float((0.5 * torch.mean(target_cpu ** 2)).item())
    recon_phys_ens = float((0.5 * torch.mean(recon_cpu ** 2)).item())
    physical_enstrophy_ratio = float(recon_phys_ens / target_phys_ens) if target_phys_ens > 0 else None

    # Benchmark latency
    h, w = val_data.shape[-2], val_data.shape[-1]
    latency_info = benchmark_model_latency(
        model, device, sample_shape=(1, 1, h, w), batch_shape=(val_data.shape[0], 1, h, w)
    )

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
        "band_integrated_enstrophy_ratio": band_integrated_enstrophy_ratio,
        "physical_enstrophy_ratio": physical_enstrophy_ratio,
        "total_enstrophy_retention_ratio": band_integrated_enstrophy_ratio,  # Maintained for contract compatibility
        "target_physical_enstrophy": target_phys_ens,
        "reconstructed_physical_enstrophy": recon_phys_ens,
        "spurious_energy_in_zero_bins": spec_ratio_res["spurious_energy_in_zero_bins"],
        "k_bins": spec_ratio_res["k_bins"],
        "spectrum_target": spec_ratio_res["spectrum_target"],
        "spectrum_pred": spec_ratio_res["spectrum_pred"],
        "spectrum_ratio": spec_ratio_res["spectrum_ratio"],
        "total_parameters": total_params,
        "latency_benchmarks": latency_info,
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

    # Figure 1: Quantitative Bar Chart Comparison
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=200)

    labels = [f"Cz={cz}\n({results[cz]['capacity_metadata']['element_compression_ratio']}x)" for cz in cz_list]
    rel_l2s = [results[cz]["evaluation"]["relative_l2"] * 100.0 for cz in cz_list]
    pdes = [results[cz]["evaluation"]["pairwise_difference_error"] * 100.0 for cz in cz_list]
    vrs = [results[cz]["evaluation"]["variance_ratio"] for cz in cz_list]
    single_lats = [results[cz]["evaluation"]["latency_benchmarks"]["single_request_latency"]["median_ms"] for cz in cz_list]

    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    # (a) Relative L2 Error
    bars1 = axes[0, 0].bar(labels, rel_l2s, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    axes[0, 0].set_title("(a) Relative Reconstruction Error (% L2)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Relative L2 Error (%)")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9)

    # (b) Pairwise Difference Error (PDE)
    bars2 = axes[0, 1].bar(labels, pdes, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    axes[0, 1].set_title("(b) Pairwise Difference Error (% PDE)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Pairwise Difference Error (%)")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9)

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

    # (d) Single-Sample Latency (Median ms, B=1)
    bars4 = axes[1, 1].bar(labels, single_lats, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    axes[1, 1].set_title("(d) Single-Sample Latency (B=1 Median ms)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Latency (ms)")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars4:
        yval = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.2f}ms", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Representation Capacity Trade-Off on Independent Validation Set", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    metrics_path = output_dir / "capacity_comparison_metrics.png"
    plt.savefig(metrics_path, bbox_inches="tight")
    plt.close()
    generated_files.append(str(metrics_path))

    # Figure 2: Radial Enstrophy Spectra Comparison
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    ref_k = results[cz_list[0]]["evaluation"]["k_bins"]
    ref_spec_tgt = results[cz_list[0]]["evaluation"]["spectrum_target"]

    ax.plot(ref_k, ref_spec_tgt, color="black", linewidth=2.2, label="Ground Truth (Validation Target)", linestyle="-")

    styles = [("--", "#2b5c8f"), ("-.", "#d95f02"), (":", "#7570b3")]
    for cz, (ls, col) in zip(cz_list, styles):
        spec_pred = results[cz]["evaluation"]["spectrum_pred"]
        ratio_str = f"Avg Ratio: {results[cz]['evaluation']['mean_valid_spectrum_ratio']:.3f}" if results[cz]["evaluation"]["mean_valid_spectrum_ratio"] else ""
        ax.plot(ref_k, spec_pred, color=col, linestyle=ls, linewidth=1.8, label=f"Recon Cz={cz} ({ratio_str})")

    ax.set_yscale("log")
    ax.set_title("Radial Enstrophy Spectrum E(k) Across Latent Capacities (Independent Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Wavenumber k", fontsize=10)
    ax.set_ylabel("Radial Enstrophy Spectrum E(k) [log scale]", fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    spectra_path = output_dir / "capacity_comparison_spectra.png"
    plt.savefig(spectra_path, bbox_inches="tight")
    plt.close()
    generated_files.append(str(spectra_path))

    # Figure 3: Spatial Vorticity Reconstruction & Absolute Error Fields
    sample_idx = 0
    gt_sample = val_data[sample_idx, 0].cpu().numpy()
    vmin, vmax = gt_sample.min(), gt_sample.max()

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), dpi=200)

    im0 = axes[0, 0].imshow(gt_sample, cmap="PRGn", vmin=vmin, vmax=vmax, origin="lower")
    axes[0, 0].set_title("Independent GT ω", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    for col_idx, cz in enumerate(cz_list, start=1):
        recon_sample = results[cz]["evaluation"]["reconstruction_tensor"][sample_idx, 0].numpy()
        im = axes[0, col_idx].imshow(recon_sample, cmap="PRGn", vmin=vmin, vmax=vmax, origin="lower")
        axes[0, col_idx].set_title(f"Recon Cz={cz} ({results[cz]['capacity_metadata']['element_compression_ratio']}x)", fontsize=11)
        axes[0, col_idx].axis("off")
        plt.colorbar(im, ax=axes[0, col_idx], fraction=0.046, pad=0.04)

    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "Absolute Error\n|ω_val - ω_recon|", ha="center", va="center", fontsize=12, fontweight="bold")

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

    plt.suptitle("Qualitative Vorticity Reconstruction on Independent Validation Sample", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fields_path = output_dir / "capacity_comparison_fields.png"
    plt.savefig(fields_path, bbox_inches="tight")
    plt.close()
    generated_files.append(str(fields_path))

    return generated_files


def evaluate_existing_checkpoints(
    checkpoint_dir: str = "outputs/experiments/vorticity_capacity",
    capacities: List[int] = (64, 32, 16),
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Load existing trained checkpoints and re-evaluate strictly on independent validation set."""
    ckpt_root = Path(checkpoint_dir)
    out_path = Path(output_dir) if output_dir else ckpt_root / "independent_eval"
    out_path.mkdir(parents=True, exist_ok=True)

    dev_str = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_str)

    results = {}

    for cz in capacities:
        ckpt_file = ckpt_root / f"cz_{cz}" / "latest_checkpoint.pt"
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"Checkpoint for Cz={cz} not found at {ckpt_file}")

        ckpt = torch.load(ckpt_file, map_location=dev, weights_only=False)
        cfg = ckpt.get("config", {})
        synth_cfg = cfg.get("synthetic_data", {})
        domain_cfg = cfg.get("domain", {})

        nx = domain_cfg.get("nx", 64)
        ny = domain_cfg.get("ny", 64)
        lx = domain_cfg.get("lx", 1.0)
        ly = domain_cfg.get("ly", 1.0)
        domain_size = (lx, ly)

        train_seed = synth_cfg.get("seed", 42)
        val_seed = train_seed + 1000  # Strictly separate seed

        # Generate canonical train set and independent val set
        train_data = generate_synthetic_vorticity_dataset(
            num_samples=synth_cfg.get("num_train_samples", 128),
            nx=nx, ny=ny, lx=lx, ly=ly,
            base_wavenumber=synth_cfg.get("base_wavenumber", 1),
            perturbation_modes=synth_cfg.get("perturbation_modes", [[1, 0], [0, 1], [1, 1], [2, 1]]),
            perturbation_amplitude=synth_cfg.get("perturbation_amplitude", 0.1),
            seed=train_seed,
        )
        val_data = generate_synthetic_vorticity_dataset(
            num_samples=synth_cfg.get("num_val_samples", 32),
            nx=nx, ny=ny, lx=lx, ly=ly,
            base_wavenumber=synth_cfg.get("base_wavenumber", 1),
            perturbation_modes=synth_cfg.get("perturbation_modes", [[1, 0], [0, 1], [1, 1], [2, 1]]),
            perturbation_amplitude=synth_cfg.get("perturbation_amplitude", 0.1),
            seed=val_seed,
        )

        min_dist = verify_dataset_separation(train_data, val_data)

        # Build model and load weights
        model_cfg = cfg.get("model", {})
        model = VorticityAutoencoder(
            in_channels=1,
            out_channels=1,
            latent_channels=cz,
            base_channels=model_cfg.get("base_channels", 32),
            project_pressure=False,
        ).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])

        eval_res = run_single_capacity_evaluation(model, val_data, device=dev, domain_size=domain_size)
        capacity_meta = compute_capacity_metadata(nx, ny, 1, cz)

        train_digest = compute_tensor_digest(train_data)
        val_digest = compute_tensor_digest(val_data)
        ckpt_sha256 = compute_file_sha256(ckpt_file)
        ckpt_size_bytes = ckpt_file.stat().st_size

        results[cz] = {
            "latent_channels": cz,
            "capacity_metadata": capacity_meta,
            "checkpoint_path": str(ckpt_file),
            "checkpoint_sha256": ckpt_sha256,
            "checkpoint_size_bytes": ckpt_size_bytes,
            "checkpoint_completed_epoch": ckpt.get("epoch"),
            "dataset_separation_min_l2": min_dist,
            "evaluation": eval_res,
        }

        print(f"--- Evaluated Checkpoint Cz={cz} on Independent Set (seed={val_seed}) ---")
        print(f"    Relative L2 Error: {eval_res['relative_l2']*100:.2f}%")
        print(f"    Pairwise Difference Error (PDE): {eval_res['pairwise_difference_error']*100:.2f}%")
        print(f"    Variance Ratio (VR): {eval_res['variance_ratio']:.4f}")
        print(f"    Mean Valid Spectrum Ratio: {eval_res['mean_valid_spectrum_ratio']:.4f}")
        print(f"    Band-Integrated Enstrophy Ratio: {eval_res['band_integrated_enstrophy_ratio']:.4f}")
        print(f"    Physical-Space Enstrophy Ratio: {eval_res['physical_enstrophy_ratio']:.4f}")
        print(f"    Single-Request Latency (B=1): {eval_res['latency_benchmarks']['single_request_latency']['median_ms']:.2f}ms")

    # Generate plots
    plot_files = generate_comparison_plots(results, val_data, out_path)

    # Build summary JSON
    summary_dict = {
        "metadata": {
            "evaluation_type": "independent_validation_set",
            "git_commit": get_git_commit_hash(),
            "experiment_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_identity": {
                "protocol": "synthetic_vorticity_streamfunction_v1",
                "train_seed": train_seed,
                "val_seed": val_seed,
                "train_samples": int(train_data.shape[0]),
                "val_samples": int(val_data.shape[0]),
                "train_data_digest": train_digest,
                "val_data_digest": val_digest,
                "dataset_separation": {
                    "min_pairwise_l2": min_dist,
                    "threshold": 1e-3,
                    "identical_samples_count": 0,
                    "status": "PASSED_FAIL_CLOSED",
                },
            },
            "device": dev_str,
            "capacities_evaluated": list(capacities),
            "generated_plots": plot_files,
            "note": "Re-evaluation conducted on strictly separated validation set (seed 1042) without retraining.",
        },
        "models": {},
    }

    for cz, data in results.items():
        eval_copy = dict(data["evaluation"])
        eval_copy.pop("reconstruction_tensor", None)
        summary_dict["models"][str(cz)] = {
            "latent_channels": cz,
            "capacity_metadata": data["capacity_metadata"],
            "checkpoint_path": data["checkpoint_path"],
            "checkpoint_sha256": data["checkpoint_sha256"],
            "checkpoint_size_bytes": data["checkpoint_size_bytes"],
            "checkpoint_completed_epoch": data["checkpoint_completed_epoch"],
            "dataset_separation_min_l2": data["dataset_separation_min_l2"],
            "metrics": eval_copy,
        }

    summary_file = out_path / "capacity_comparison_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    return summary_dict


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

    dev_str = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_str)

    # 1. Synthesize canonical fixed datasets with STRICT independence
    synth_cfg = base_cfg.get("synthetic_data", {})
    train_seed = synth_cfg.get("seed", 42)
    val_seed = train_seed + 1000

    nx = base_cfg.get("domain", {}).get("nx", 64)
    ny = base_cfg.get("domain", {}).get("ny", 64)
    lx = base_cfg.get("domain", {}).get("lx", 1.0)
    ly = base_cfg.get("domain", {}).get("ly", 1.0)
    base_k = synth_cfg.get("base_wavenumber", 1)
    modes = synth_cfg.get("perturbation_modes", [[1, 0], [0, 1], [1, 1], [2, 1]])
    amp = synth_cfg.get("perturbation_amplitude", 0.1)

    train_data = generate_synthetic_vorticity_dataset(
        num_samples=synth_cfg.get("num_train_samples", 128),
        nx=nx, ny=ny, lx=lx, ly=ly,
        base_wavenumber=base_k, perturbation_modes=modes, perturbation_amplitude=amp,
        seed=train_seed,
    )
    val_data = generate_synthetic_vorticity_dataset(
        num_samples=synth_cfg.get("num_val_samples", 32),
        nx=nx, ny=ny, lx=lx, ly=ly,
        base_wavenumber=base_k, perturbation_modes=modes, perturbation_amplitude=amp,
        seed=val_seed,
    )

    min_dist = verify_dataset_separation(train_data, val_data)
    domain_size = (float(lx), float(ly))
    experiment_results = {}

    for cz in capacities:
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

        eval_res = run_single_capacity_evaluation(
            model=train_res["model"],
            val_data=val_data,
            device=dev,
            domain_size=domain_size,
        )

        ckpt_path_p = Path(train_res["checkpoint_path"])
        ckpt_sha = compute_file_sha256(ckpt_path_p) if ckpt_path_p.is_file() else "unknown"
        ckpt_sz = ckpt_path_p.stat().st_size if ckpt_path_p.is_file() else 0

        experiment_results[cz] = {
            "latent_channels": cz,
            "capacity_metadata": train_res["capacity_metadata"],
            "training_summary": train_res["summary"],
            "training_duration_seconds": train_duration,
            "checkpoint_path": train_res["checkpoint_path"],
            "checkpoint_sha256": ckpt_sha,
            "checkpoint_size_bytes": ckpt_sz,
            "dataset_separation_min_l2": min_dist,
            "evaluation": eval_res,
        }

    plot_files = generate_comparison_plots(experiment_results, val_data, out_root)

    train_digest = compute_tensor_digest(train_data)
    val_digest = compute_tensor_digest(val_data)

    summary_dict = {
        "metadata": {
            "evaluation_type": "independent_validation_set",
            "git_commit": get_git_commit_hash(),
            "experiment_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "dataset_identity": {
                "protocol": "synthetic_vorticity_streamfunction_v1",
                "train_seed": train_seed,
                "val_seed": val_seed,
                "train_samples": int(train_data.shape[0]),
                "val_samples": int(val_data.shape[0]),
                "train_data_digest": train_digest,
                "val_data_digest": val_digest,
                "dataset_separation": {
                    "min_pairwise_l2": min_dist,
                    "threshold": 1e-3,
                    "identical_samples_count": 0,
                    "status": "PASSED_FAIL_CLOSED",
                },
            },
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
            "checkpoint_sha256": data["checkpoint_sha256"],
            "checkpoint_size_bytes": data["checkpoint_size_bytes"],
            "dataset_separation_min_l2": data["dataset_separation_min_l2"],
            "metrics": eval_copy,
        }

    summary_file = out_root / "capacity_comparison_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    return summary_dict


def main():
    parser = argparse.ArgumentParser(description="Run Vorticity Autoencoder Capacity Comparison Experiment")
    parser.add_argument("--config", type=str, default="configs/train/vorticity_autoencoder.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--capacities", type=int, nargs="+", default=[64, 32, 16])
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing checkpoints without retraining")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/experiments/vorticity_capacity")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_existing_checkpoints(
            checkpoint_dir=args.checkpoint_dir,
            capacities=args.capacities,
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        out_root = args.output_dir if args.output_dir else "outputs/experiments/vorticity_capacity"
        run_capacity_comparison_experiment(
            config_path=args.config,
            capacities=args.capacities,
            output_root=out_root,
            epochs=args.epochs,
            device=args.device,
        )


if __name__ == "__main__":
    main()
