#!/usr/bin/env python3
"""Horizon-R1 Formal Analysis: Parse training logs, compute J_long, generate figures.

This script:
1. Parses H2-control / H4 / H8 training logs
2. Computes J_long = mean(h10, h20, h30) per epoch
3. Identifies dual checkpoint selection: short-best vs long-best
4. Generates comparison figures
5. Outputs formal JSON summaries
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGS = {
    "H2-control": "outputs/train_horizon_r1_seed_42_E4_H2_control.log",
    "H4": "outputs/train_horizon_r1_seed_42_E4_H4.log",
    "H8": "outputs/train_horizon_r1_seed_42_E4_H8.log",
}

CKPT_DIRS = {
    "H2-control": "outputs/checkpoints/dynamics/horizon_r1/seed_42/E4_H2_control/latent_transformer",
    "H4": "outputs/checkpoints/dynamics/horizon_r1/seed_42/E4_H4/latent_transformer",
    "H8": "outputs/checkpoints/dynamics/horizon_r1/seed_42/E4_H8/latent_transformer",
}


def parse_log(path: str) -> list:
    """Parse training log, return list of dicts per epoch."""
    pattern = re.compile(
        r"Epoch \[(\d+)/(\d+)\] \| Train Loss: ([\d.e+-]+) \| "
        r"Val Rollout Mean VRMSE: ([\d.]+) \| "
        r"Step 1 VRMSE: ([\d.]+) \(u: ([\d.]+), v: ([\d.]+), p: ([\d.]+), s: ([\d.]+)\) \| "
        r"Diag \[h10: ([\d.]+), h20: ([\d.]+), h30: ([\d.]+)\]"
    )
    results = []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                h10, h20, h30 = float(m.group(10)), float(m.group(11)), float(m.group(12))
                results.append({
                    "epoch": int(m.group(1)),
                    "train_loss": float(m.group(3)),
                    "val_vrmse": float(m.group(4)),
                    "step1_vrmse": float(m.group(5)),
                    "step1_u": float(m.group(6)),
                    "step1_v": float(m.group(7)),
                    "step1_p": float(m.group(8)),
                    "step1_s": float(m.group(9)),
                    "diag_h10": h10,
                    "diag_h20": h20,
                    "diag_h30": h30,
                    "j_long": (h10 + h20 + h30) / 3.0,
                })
    return results


def find_saved_checkpoints(ckpt_dir: str) -> Dict[int, str]:
    """Find saved checkpoint files and map epoch -> path."""
    saved = {}
    if not os.path.isdir(ckpt_dir):
        return saved
    for fname in os.listdir(ckpt_dir):
        m = re.match(r"checkpoint_step_(\d+)_vrmse_mean_([\d.]+)\.pt", fname)
        if m:
            saved[int(m.group(1))] = os.path.join(ckpt_dir, fname)
    return saved


def analyze_group(name: str, epochs: list, ckpt_dir: str) -> dict:
    """Analyze a single training group: find short-best and long-best."""
    saved_ckpts = find_saved_checkpoints(ckpt_dir)

    # Short-best: the saved best_vrmse_mean.pt (lowest val_vrmse)
    best_val = min(epochs, key=lambda e: e["val_vrmse"])

    # Long-best among ALL epochs
    best_j_all = min(epochs, key=lambda e: e["j_long"])

    # Long-best among SAVED epochs only
    saved_epochs_data = [e for e in epochs if e["epoch"] in saved_ckpts]
    best_j_saved = min(saved_epochs_data, key=lambda e: e["j_long"]) if saved_epochs_data else None

    return {
        "group": name,
        "total_epochs": len(epochs),
        "saved_checkpoint_epochs": sorted(saved_ckpts.keys()),
        "short_best": {
            "epoch": best_val["epoch"],
            "val_vrmse": best_val["val_vrmse"],
            "step1_vrmse": best_val["step1_vrmse"],
            "diag_h10": best_val["diag_h10"],
            "diag_h20": best_val["diag_h20"],
            "diag_h30": best_val["diag_h30"],
            "j_long": best_val["j_long"],
            "checkpoint": os.path.join(ckpt_dir, "best_vrmse_mean.pt"),
        },
        "long_best_all": {
            "epoch": best_j_all["epoch"],
            "j_long": best_j_all["j_long"],
            "val_vrmse": best_j_all["val_vrmse"],
            "diag_h10": best_j_all["diag_h10"],
            "diag_h20": best_j_all["diag_h20"],
            "diag_h30": best_j_all["diag_h30"],
            "checkpoint_saved": best_j_all["epoch"] in saved_ckpts,
        },
        "long_best_saved": {
            "epoch": best_j_saved["epoch"],
            "j_long": best_j_saved["j_long"],
            "val_vrmse": best_j_saved["val_vrmse"],
            "diag_h10": best_j_saved["diag_h10"],
            "diag_h20": best_j_saved["diag_h20"],
            "diag_h30": best_j_saved["diag_h30"],
            "checkpoint": saved_ckpts[best_j_saved["epoch"]],
        } if best_j_saved else None,
    }


def generate_figures(all_data: Dict[str, list], output_dir: str = "outputs/figures"):
    """Generate comparison figures for Horizon-R1 ablation."""
    os.makedirs(output_dir, exist_ok=True)
    colors = {"H2-control": "#2196F3", "H4": "#FF9800", "H8": "#E91E63"}

    # Figure 1: 2x3 training curves + diagnostics
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for name, epochs_data in all_data.items():
        ep = [e["epoch"] for e in epochs_data]
        c = colors[name]

        axes[0, 0].plot(ep, [e["train_loss"] for e in epochs_data], '-o', color=c, label=name, markersize=4)
        axes[0, 1].plot(ep, [e["step1_vrmse"] for e in epochs_data], '-o', color=c, label=name, markersize=4)
        axes[0, 2].plot(ep, [e["j_long"] for e in epochs_data], '-o', color=c, label=name, markersize=4, linewidth=2)

        axes[1, 0].plot(ep, [e["diag_h10"] for e in epochs_data], '-o', color=c, label=name, markersize=4)
        axes[1, 1].plot(ep, [e["diag_h20"] for e in epochs_data], '-o', color=c, label=name, markersize=4)
        axes[1, 2].plot(ep, [e["diag_h30"] for e in epochs_data], '-o', color=c, label=name, markersize=4)

    titles = [
        "Train Loss", "Step-1 VRMSE (fair short-range)", "J_long = mean(h10,h20,h30)",
        "Diagnostic h=10 VRMSE", "Diagnostic h=20 VRMSE", "Diagnostic h=30 VRMSE",
    ]
    for i, ax in enumerate(axes.flat):
        ax.set_title(titles[i], fontsize=12, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Horizon-R1 Ablation: H2-control vs H4 vs H8\n"
        "(Seed 42, Parent=E4 canonical, Beff=8, 12 epochs each)",
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()
    path1 = os.path.join(output_dir, "horizon_r1_comparison.png")
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path1}")
    plt.close(fig)

    # Figure 2: J_long bar chart with dual selection
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    groups = list(all_data.keys())
    x = np.arange(len(groups))
    w = 0.25

    # Short-best J_long
    short_j = []
    long_saved_j = []
    long_all_j = []
    for name, epochs in all_data.items():
        best_val = min(epochs, key=lambda e: e["val_vrmse"])
        short_j.append(best_val["j_long"])

        saved_ckpts = find_saved_checkpoints(CKPT_DIRS[name])
        saved_data = [e for e in epochs if e["epoch"] in saved_ckpts]
        if saved_data:
            long_saved_j.append(min(saved_data, key=lambda e: e["j_long"])["j_long"])
        else:
            long_saved_j.append(0)

        long_all_j.append(min(epochs, key=lambda e: e["j_long"])["j_long"])

    ax2.bar(x - w, short_j, w, label="Short-best (val VRMSE)", color=[colors[g] for g in groups], alpha=0.7)
    ax2.bar(x, long_saved_j, w, label="Long-best saved (J_long)", color=[colors[g] for g in groups], alpha=0.5, hatch='//')
    ax2.bar(x + w, long_all_j, w, label="Long-best all (J_long)", color=[colors[g] for g in groups], alpha=0.3, hatch='\\\\')

    for i, (s, ls, la) in enumerate(zip(short_j, long_saved_j, long_all_j)):
        ax2.text(i - w, s, f'{s:.2f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i, ls, f'{ls:.2f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i + w, la, f'{la:.2f}', ha='center', va='bottom', fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(groups)
    ax2.set_ylabel("J_long = mean(h10, h20, h30) VRMSE")
    ax2.set_title("Checkpoint Selection: Short-best vs Long-best\n(Lower is better)", fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path2 = os.path.join(output_dir, "horizon_r1_checkpoint_selection.png")
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path2}")
    plt.close(fig2)


def main():
    print("=" * 90)
    print("HORIZON-R1 FORMAL ANALYSIS")
    print("=" * 90)

    # Parse logs
    all_data = {}
    for name, path in LOGS.items():
        if os.path.exists(path):
            all_data[name] = parse_log(path)
            print(f"  Parsed {name}: {len(all_data[name])} epochs")
        else:
            print(f"  WARNING: Log not found: {path}")

    if not all_data:
        print("ERROR: No training logs found!")
        sys.exit(1)

    # Analyze each group
    print("\n--- Dual Checkpoint Selection ---")
    summary = {}
    for name, epochs in all_data.items():
        ckpt_dir = CKPT_DIRS[name]
        analysis = analyze_group(name, epochs, ckpt_dir)
        summary[name] = analysis

        print(f"\n  {name}:")
        print(f"    Short-best: Ep{analysis['short_best']['epoch']}, "
              f"ValVRMSE={analysis['short_best']['val_vrmse']:.4f}, "
              f"J_long={analysis['short_best']['j_long']:.4f}")
        lb = analysis.get('long_best_saved')
        if lb:
            print(f"    Long-best (saved): Ep{lb['epoch']}, "
                  f"ValVRMSE={lb['val_vrmse']:.4f}, "
                  f"J_long={lb['j_long']:.4f}")
        lba = analysis['long_best_all']
        print(f"    Long-best (all):   Ep{lba['epoch']}, "
              f"J_long={lba['j_long']:.4f}, "
              f"saved={'YES' if lba['checkpoint_saved'] else 'NO'}")

    # Generate figures
    print("\n--- Generating Figures ---")
    generate_figures(all_data)

    # Save outputs
    os.makedirs("outputs/metrics", exist_ok=True)

    traj_path = "outputs/metrics/horizon_r1_epoch_trajectories.json"
    with open(traj_path, "w") as f:
        json.dump({name: epochs for name, epochs in all_data.items()}, f, indent=2)
    print(f"\n  Saved trajectories: {traj_path}")

    summary_path = "outputs/metrics/horizon_r1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")

    # Print comparison table
    print("\n" + "=" * 90)
    print("COMPARISON (Step-1 VRMSE is the fair short-range metric)")
    print("=" * 90)
    print(f"{'Group':<14} {'Selection':<14} {'Epoch':>5} {'Step1':>8} {'h10':>8} {'h20':>8} {'h30':>8} {'J_long':>8}")
    print("-" * 82)
    for name in all_data:
        a = summary[name]
        sb = a["short_best"]
        print(f"{name:<14} {'short-best':<14} {sb['epoch']:>5} {sb['step1_vrmse']:>8.4f} "
              f"{sb['diag_h10']:>8.4f} {sb['diag_h20']:>8.4f} {sb['diag_h30']:>8.4f} {sb['j_long']:>8.4f}")
        lb = a.get("long_best_saved")
        if lb:
            # Need to get step1_vrmse from epoch data
            ep_data = [e for e in all_data[name] if e["epoch"] == lb["epoch"]][0]
            print(f"{'':<14} {'long-best':<14} {lb['epoch']:>5} {ep_data['step1_vrmse']:>8.4f} "
                  f"{lb['diag_h10']:>8.4f} {lb['diag_h20']:>8.4f} {lb['diag_h30']:>8.4f} {lb['j_long']:>8.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
