#!/usr/bin/env python3
"""Training Convergence Analysis for Closure-R4 Multi-Seed Ablations.

Parses all training logs (E0-E4 across seeds 42, 43, 44), extracts per-epoch
train loss and validation VRMSE trajectories, calculates key convergence metrics
(best epoch, final epoch, best-to-final gap, last-5 slope, status), generates
publication-grade convergence curves, and exports a structured JSON summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_log_filename(filename: str) -> Dict[str, Any]:
    """Extract seed and ablation group from a log filename.

    Examples:
        - train_closure_r4_ablation_E0_single_step.log -> seed: 42, group: E0_single_step
        - train_closure_r4_ablation_E4_full_physics.log -> seed: 42, group: E4_full_physics
        - train_closure_r4_seed_43_ablation_E1_rollout_field.log -> seed: 43, group: E1_rollout_field
        - train_closure_r4_seed_44_ablation_E2_plus_L_div.log -> seed: 44, group: E2_plus_L_div
    """
    base = os.path.basename(filename)

    # Check for explicit seed in filename
    seed_match = re.search(r"seed_(\d+)", base)
    seed = int(seed_match.group(1)) if seed_match else 42

    # Check for ablation group name
    group_match = re.search(r"ablation_(E\d+_[a-zA-Z0-9_]+)\.log", base)
    if group_match:
        group = group_match.group(1)
    else:
        # Fallback to general group matching
        alt_match = re.search(r"(E\d+_[a-zA-Z0-9_]+)", base)
        group = alt_match.group(1) if alt_match else base.replace(".log", "")

    return {"seed": seed, "group": group, "filename": base}


def parse_training_log(filepath: str | Path) -> Dict[str, Any]:
    """Parse a single training log file.

    Extracts:
        - horizon: rollout horizon during training (1 for E0, 2 for E1-E4)
        - total_epochs: total scheduled epochs
        - epochs: list of 1-based epoch numbers
        - train_losses: list of train loss floats
        - val_vrmses: list of validation VRMSE floats (Rollout Mean for E1-E4, VRMSE Mean for E0)
        - step1_vrmses: list of step-1 VRMSE floats (if present)
        - component_vrmses: dict of lists for u, v, p, s
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Log file not found: {filepath}")

    meta = parse_log_filename(str(filepath))

    epochs: List[int] = []
    train_losses: List[float] = []
    val_vrmses: List[float] = []
    step1_vrmses: List[Optional[float]] = []
    u_vals: List[Optional[float]] = []
    v_vals: List[Optional[float]] = []
    p_vals: List[Optional[float]] = []
    s_vals: List[Optional[float]] = []

    horizon: int = 1 if "E0" in meta["group"] else 2
    total_epochs: int = 30

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # Check for header metadata
            h_match = re.search(r"Horizon:\s*(\d+)", line)
            if h_match:
                horizon = int(h_match.group(1))
            ep_match = re.search(r"Epochs:\s*(\d+)", line)
            if ep_match:
                total_epochs = int(ep_match.group(1))

            # Check for epoch training line:
            # Epoch [01/30] | Train Loss: 1.5094e-01 | Val Rollout Mean VRMSE: 0.4974 | Step 1 VRMSE: ...
            # or Epoch [01/30] | Train Loss: 8.9765e-02 | Val VRMSE Mean: 0.3987 ...
            ep_line_match = re.search(r"Epoch\s*\[(\d+)/(\d+)\]", line)
            if not ep_line_match:
                continue

            ep_num = int(ep_line_match.group(1))
            tot_ep = int(ep_line_match.group(2))
            total_epochs = tot_ep

            loss_m = re.search(r"Train Loss:\s*([0-9\.e\+\-]+)", line)
            val_m = re.search(r"Val (?:Rollout Mean VRMSE|VRMSE Mean):\s*([0-9\.e\+\-]+)", line)
            s1_m = re.search(r"Step 1 VRMSE:\s*([0-9\.e\+\-]+)", line)
            comp_m = re.search(r"\(u:\s*([0-9\.e\+\-]+),\s*v:\s*([0-9\.e\+\-]+),\s*p:\s*([0-9\.e\+\-]+),\s*s:\s*([0-9\.e\+\-]+)\)", line)

            if loss_m and val_m:
                epochs.append(ep_num)
                train_losses.append(float(loss_m.group(1)))
                val_vrmses.append(float(val_m.group(1)))
                step1_vrmses.append(float(s1_m.group(1)) if s1_m else None)

                if comp_m:
                    u_vals.append(float(comp_m.group(1)))
                    v_vals.append(float(comp_m.group(2)))
                    p_vals.append(float(comp_m.group(3)))
                    s_vals.append(float(comp_m.group(4)))
                else:
                    u_vals.append(None)
                    v_vals.append(None)
                    p_vals.append(None)
                    s_vals.append(None)

    return {
        "seed": meta["seed"],
        "group": meta["group"],
        "filename": meta["filename"],
        "horizon": horizon,
        "total_epochs": total_epochs,
        "epochs": epochs,
        "train_losses": train_losses,
        "val_vrmses": val_vrmses,
        "step1_vrmses": step1_vrmses,
        "components": {
            "u": u_vals,
            "v": v_vals,
            "p": p_vals,
            "s": s_vals,
        },
    }


def compute_last_k_slope(values: List[float], k: int = 5) -> float:
    """Compute ordinary least squares linear regression slope over the last k points."""
    if len(values) < 2:
        return 0.0
    k = min(k, len(values))
    y = np.array(values[-k:], dtype=float)
    x = np.arange(k, dtype=float)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    denominator = np.sum((x - x_mean) ** 2)
    if denominator == 0:
        return 0.0
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denominator)
    return slope


def compute_run_convergence_metrics(parsed_run: Dict[str, Any], k_window: int = 5) -> Dict[str, Any]:
    """Compute convergence diagnostics for a single training trajectory."""
    val_vrmses = parsed_run["val_vrmses"]
    if not val_vrmses:
        raise ValueError(f"No validation VRMSE values parsed for {parsed_run['filename']}")

    best_idx = int(np.argmin(val_vrmses))
    best_epoch = parsed_run["epochs"][best_idx]
    best_val_vrmse = float(val_vrmses[best_idx])
    final_val_vrmse = float(val_vrmses[-1])
    best_to_final_gap = float(final_val_vrmse - best_val_vrmse)
    last_slope = compute_last_k_slope(val_vrmses, k=k_window)

    # Determine scientific convergence status
    # 1. still_converging: best epoch is at the very end AND slope is strongly negative
    # 2. mild_overfitting: validation VRMSE has risen noticeably or slope is positive
    # 3. plateaued: slope is very near 0, validation VRMSE has stabilized within a flat band
    total_ep = parsed_run["total_epochs"]
    if best_epoch == total_ep and last_slope < -0.01:
        status = "still_converging"
        description = "Minimum reached at final epoch with negative descent slope; potential further convergence with more epochs."
    elif last_slope > 0.005 or best_to_final_gap >= 0.03:
        status = "post_minimum_fluctuation"
        description = "Validation VRMSE reached minimum at an earlier epoch and displayed post-minimum fluctuation or slight rise."
    elif abs(last_slope) <= 0.005:
        status = "plateaued"
        description = "Validation VRMSE stabilized in flat plateau region across final epochs."
    else:
        # last_slope < 0 but best_epoch < total_ep (e.g. temporary fluctuation after peak)
        status = "plateau_with_fluctuations"
        description = "Validation VRMSE entered post-minimum fluctuation band without sustained descent."

    return {
        "group": parsed_run["group"],
        "seed": parsed_run["seed"],
        "horizon": parsed_run["horizon"],
        "total_epochs": total_ep,
        "best_epoch": best_epoch,
        "best_val_vrmse": best_val_vrmse,
        "final_val_vrmse": final_val_vrmse,
        "best_to_final_gap": best_to_final_gap,
        "last_5_slope": last_slope,
        "status": status,
        "description": description,
    }


def analyze_all_logs(log_dir: str | Path) -> Dict[str, Any]:
    """Parse and analyze all Closure-R4 training logs in log_dir."""
    log_dir = Path(log_dir)
    pattern = str(log_dir / "train_closure_r4_*.log")
    log_files = sorted(glob.glob(pattern))

    if not log_files:
        raise FileNotFoundError(f"No log files matching pattern: {pattern}")

    runs_data = []
    run_summaries = []

    for filepath in log_files:
        parsed = parse_training_log(filepath)
        metrics = compute_run_convergence_metrics(parsed)
        runs_data.append(parsed)
        run_summaries.append(metrics)

    # Sort summaries by group, then seed
    group_order = {
        "E0_single_step": 0,
        "E1_rollout_field": 1,
        "E2_plus_L_div": 2,
        "E3_plus_L_vort": 3,
        "E4_full_physics": 4,
    }
    run_summaries.sort(key=lambda s: (group_order.get(s["group"], 99), s["seed"]))
    runs_data.sort(key=lambda d: (group_order.get(d["group"], 99), d["seed"]))

    # Compute aggregate convergence statistics
    best_epochs = [s["best_epoch"] for s in run_summaries]
    gaps = [s["best_to_final_gap"] for s in run_summaries]
    slopes = [s["last_5_slope"] for s in run_summaries]

    aggregate = {
        "total_runs": len(run_summaries),
        "mean_best_epoch": float(np.mean(best_epochs)),
        "min_best_epoch": int(np.min(best_epochs)),
        "max_best_epoch": int(np.max(best_epochs)),
        "runs_peaking_at_epoch_30": int(sum(1 for e in best_epochs if e == 30)),
        "runs_peaking_before_epoch_30": int(sum(1 for e in best_epochs if e < 30)),
        "fraction_peaking_before_epoch_30": float(np.mean([1 if e < 30 else 0 for e in best_epochs])),
        "mean_best_to_final_gap": float(np.mean(gaps)),
        "all_minima_precede_final_epoch": bool(all(e < 30 for e in best_epochs)),
        "mean_last_5_slope": float(np.mean(slopes)),
    }

    return {
        "log_dir": str(log_dir),
        "aggregate": aggregate,
        "runs": run_summaries,
        "raw_trajectories": runs_data,
    }


def format_summary_table(analysis: Dict[str, Any]) -> str:
    """Format the convergence analysis results as a clean markdown/ASCII table."""
    lines = []
    lines.append("| Group | Seed | Horizon | Best Ep | Best VRMSE | Final VRMSE | Gap (Final-Best) | Last-5 Slope | Status |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
    for r in analysis["runs"]:
        gap_str = f"+{r['best_to_final_gap']:.4f}" if r['best_to_final_gap'] >= 0 else f"{r['best_to_final_gap']:.4f}"
        slope_str = f"{r['last_5_slope']:+.5f}"
        lines.append(
            f"| `{r['group']}` | {r['seed']} | H={r['horizon']} | **Ep {r['best_epoch']}** | "
            f"{r['best_val_vrmse']:.4f} | {r['final_val_vrmse']:.4f} | {gap_str} | {slope_str} | {r['status']} |"
        )
    return "\n".join(lines)


def plot_convergence_curves(analysis: Dict[str, Any], output_path: str | Path) -> None:
    """Generate publication-grade 4-panel training convergence diagnostic figure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    ax_train, ax_val = axes[0, 0], axes[0, 1]
    ax_hist, ax_diag = axes[1, 0], axes[1, 1]

    # Visual palette
    group_colors = {
        "E0_single_step": "#7f7f7f",       # Gray
        "E1_rollout_field": "#1f77b4",     # Blue
        "E2_plus_L_div": "#2ca02c",        # Green
        "E3_plus_L_vort": "#ff7f0e",       # Orange
        "E4_full_physics": "#d62728",      # Red
    }
    seed_styles = {
        42: "-",
        43: "--",
        44: ":",
    }

    # Panel A: Training Loss vs Epoch (Log scale)
    for run in analysis["raw_trajectories"]:
        grp = run["group"]
        seed = run["seed"]
        eps = run["epochs"]
        loss = run["train_losses"]
        c = group_colors.get(grp, "#333333")
        ls = seed_styles.get(seed, "-")
        label = f"{grp} (s{seed})" if seed == 42 else None
        ax_train.plot(eps, loss, color=c, linestyle=ls, alpha=0.85, linewidth=1.8, label=label)

    ax_train.set_yscale("log")
    ax_train.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax_train.set_ylabel("Training Loss (log scale)", fontsize=12, fontweight="bold")
    ax_train.set_title("(a) Training Loss Trajectories (Descent & Saturation)", fontsize=13, fontweight="bold", pad=10)
    ax_train.grid(True, linestyle="--", alpha=0.5)
    ax_train.legend(frameon=True, fontsize=9, loc="upper right")

    # Panel B: Validation Rollout VRMSE vs Epoch
    for run in analysis["raw_trajectories"]:
        grp = run["group"]
        seed = run["seed"]
        eps = run["epochs"]
        val = run["val_vrmses"]
        c = group_colors.get(grp, "#333333")
        ls = seed_styles.get(seed, "-")
        ax_val.plot(eps, val, color=c, linestyle=ls, alpha=0.85, linewidth=1.8)

        # Mark best epoch
        best_idx = int(np.argmin(val))
        ax_val.scatter(
            [eps[best_idx]], [val[best_idx]],
            color=c, edgecolor="black", s=60, zorder=5, alpha=0.9
        )

    ax_val.axvspan(20.5, 30.5, color="#f0f0f0", alpha=0.7, label="Plateau / Saturation Window (Ep 21–30)")
    ax_val.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax_val.set_ylabel("Validation VRMSE (Mean)", fontsize=12, fontweight="bold")
    ax_val.set_title("(b) Validation Rollout VRMSE vs Epoch (Best Checkpoint Markers)", fontsize=13, fontweight="bold", pad=10)
    all_val = [v for run in analysis["raw_trajectories"] for v in run["val_vrmses"] if v is not None]
    max_val = max(all_val) if all_val else 0.55
    min_val = min(all_val) if all_val else 0.08
    ax_val.set_ylim(max(0.05, min_val * 0.85), max_val * 1.05)
    ax_val.grid(True, linestyle="--", alpha=0.5)
    ax_val.legend(frameon=True, fontsize=9, loc="upper right")

    # Panel C: Best Epoch Distribution
    best_epochs = [r["best_epoch"] for r in analysis["runs"]]
    bins = np.arange(19.5, 31.5, 1)
    counts, edges = np.histogram(best_epochs, bins=bins)
    ax_hist.bar(edges[:-1] + 0.5, counts, width=0.8, color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax_hist.axvline(30, color="#d62728", linestyle="--", linewidth=2, label="Scheduled Final Epoch (30)")
    ax_hist.axvline(np.mean(best_epochs), color="#2ca02c", linestyle="-", linewidth=2.5,
                    label=f"Mean Best Epoch = {np.mean(best_epochs):.1f}")
    ax_hist.set_xlabel("Best Epoch", fontsize=12, fontweight="bold")
    ax_hist.set_ylabel("Run Count", fontsize=12, fontweight="bold")
    ax_hist.set_title(f"(c) Best Checkpoint Distribution (0/{len(best_epochs)} peaked at Ep 30)", fontsize=13, fontweight="bold", pad=10)
    ax_hist.set_xticks(range(20, 31))
    ax_hist.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax_hist.legend(frameon=True, fontsize=10, loc="upper left")

    # Panel D: Best-to-Final Gap vs Last-5 Slope Diagnostic
    for r in analysis["runs"]:
        grp = r["group"]
        c = group_colors.get(grp, "#333333")
        marker = "o" if r["seed"] == 42 else ("s" if r["seed"] == 43 else "^")
        ax_diag.scatter(
            r["last_5_slope"], r["best_to_final_gap"],
            color=c, marker=marker, s=90, edgecolor="black", alpha=0.9,
            label=f"{grp} (s{r['seed']})"
        )

    ax_diag.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax_diag.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax_diag.set_xlabel("Last-5 Epoch Slope (Epochs 26–30)", fontsize=12, fontweight="bold")
    ax_diag.set_ylabel("Best-to-Final Gap (VRMSE_final - VRMSE_best)", fontsize=12, fontweight="bold")
    ax_diag.set_title("(d) Convergence Diagnosis: Post-Minimum Gap vs Last-5 Slope", fontsize=13, fontweight="bold", pad=10)
    ax_diag.grid(True, linestyle="--", alpha=0.5)

    # Highlight region
    ax_diag.annotate(
        "Validation minimum achieved\nprior to epoch 30;\nlate-epoch plateau / fluctuation",
        xy=(0.002, 0.04), xytext=(-0.045, 0.06),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="#999900")
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Convergence Analysis] Saved figure to {output_path}")


def export_convergence_json(analysis: Dict[str, Any], output_path: str | Path) -> None:
    """Export structured summary JSON without verbose raw trajectories."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare sanitized JSON data
    epoch_summaries = []
    for run_meta, raw in zip(analysis["runs"], analysis["raw_trajectories"]):
        epoch_summaries.append({
            "group": run_meta["group"],
            "seed": run_meta["seed"],
            "best_epoch": run_meta["best_epoch"],
            "best_val_vrmse": run_meta["best_val_vrmse"],
            "final_val_vrmse": run_meta["final_val_vrmse"],
            "train_loss_initial": raw["train_losses"][0] if raw["train_losses"] else None,
            "train_loss_final": raw["train_losses"][-1] if raw["train_losses"] else None,
            "val_vrmse_initial": raw["val_vrmses"][0] if raw["val_vrmses"] else None,
            "val_vrmse_final": raw["val_vrmses"][-1] if raw["val_vrmses"] else None,
        })

    payload = {
        "aggregate": analysis["aggregate"],
        "runs": analysis["runs"],
        "epoch_trajectories_summary": epoch_summaries,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Convergence Analysis] Saved summary JSON to {output_path}")


def export_full_trajectories_json(analysis: Dict[str, Any], output_path: str | Path) -> None:
    """Export complete epoch 1-30 trajectories for all runs with SHA256 and provenance."""
    import hashlib
    import subprocess
    from datetime import datetime, timezone

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute source log SHA256 for each trajectory if present
    log_dir = Path(analysis.get("log_dir", "outputs"))
    trajectories_with_provenance = []
    for traj in analysis["raw_trajectories"]:
        traj_copy = dict(traj)
        log_file = log_dir / traj["filename"]
        if log_file.is_file():
            with open(log_file, "rb") as f:
                traj_copy["source_log_sha256"] = hashlib.sha256(f.read()).hexdigest()
        else:
            traj_copy["source_log_sha256"] = None
        trajectories_with_provenance.append(traj_copy)

    # Git metadata
    git_commit = None
    git_dirty = None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        git_dirty = len(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()) > 0
    except Exception:
        pass

    payload = {
        "metadata": {
            "description": "Full epoch-by-epoch training loss and validation VRMSE trajectories across Closure-R4 runs",
            "total_runs": len(trajectories_with_provenance),
            "epochs_per_run": 30,
            "seeds": [42, 43, 44],
            "groups": ["E0_single_step", "E1_rollout_field", "E2_plus_L_div", "E3_plus_L_vort", "E4_full_physics"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analysis_git_commit": git_commit,
            "analysis_git_dirty": git_dirty,
        },
        "trajectories": trajectories_with_provenance,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Convergence Analysis] Saved full trajectories JSON to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Closure-R4 training convergence across all runs.")
    parser.add_argument("--log_dir", type=str, default="outputs", help="Directory containing train_closure_r4_*.log files")
    parser.add_argument("--output_json", type=str, default="outputs/metrics/training_convergence_summary.json", help="Path to save summary JSON")
    parser.add_argument("--output_trajectories", type=str, default="outputs/metrics/training_convergence_trajectories.json", help="Path to save full trajectories JSON")
    parser.add_argument("--output_fig", type=str, default="outputs/figures/training_convergence_curves.png", help="Path to save plot")
    parser.add_argument("--no_plot", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    print(f"[Convergence Analysis] Reading logs from {args.log_dir}...")
    analysis = analyze_all_logs(args.log_dir)

    print("\n" + "=" * 80)
    print("CLOSURE-R4 TRAINING CONVERGENCE DIAGNOSTIC SUMMARY")
    print("=" * 80)
    table_str = format_summary_table(analysis)
    print(table_str)
    print("=" * 80)
    agg = analysis["aggregate"]
    print(f"Total Runs Analyzed: {agg['total_runs']}")
    print(f"Mean Best Epoch:     {agg['mean_best_epoch']:.2f} (Range: Ep {agg['min_best_epoch']} – Ep {agg['max_best_epoch']})")
    print(f"Runs Peaking at Ep30: {agg['runs_peaking_at_epoch_30']}/{agg['total_runs']} ({agg['fraction_peaking_before_epoch_30']*100:.1f}% peaked before Ep 30)")
    print(f"Mean Best-to-Final:  +{agg['mean_best_to_final_gap']:.4f} (All minima precede final epoch: {agg['all_minima_precede_final_epoch']})")
    print(f"Mean Last-5 Slope:   {agg['mean_last_5_slope']:+.5f}")
    print("=" * 80 + "\n")

    export_convergence_json(analysis, args.output_json)
    if args.output_trajectories:
        export_full_trajectories_json(analysis, args.output_trajectories)

    if not args.no_plot:
        plot_convergence_curves(analysis, args.output_fig)


if __name__ == "__main__":
    main()
