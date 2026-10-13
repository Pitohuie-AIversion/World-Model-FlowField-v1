#!/usr/bin/env python3
"""Plot publication-quality figures for the Horizon controlled variable experiment.

Analyzes the effect of training rollout horizon H (controlled variable) on
future rollout step h (prediction target, h in [1, 30]).

Generates:
1. outputs/figures/horizon_r1/figure_horizon_controlled_experiment.png:
   4-panel publication-grade figure focusing on the strict Horizon-R1 controlled study
   (H2-control vs H4 vs H8, both short-best and long-best selection, component breakdown).
2. outputs/figures/horizon_r1/figure_grand_horizon_ladder_test_curves.png:
   Full ladder scaling analysis across H in {1, 2, 4, 8, 12, 16} predicting h in [1, 30].
3. outputs/figures/horizon_r1/figure_horizon_controlled_experiment_metadata.json:
   Machine-readable quantitative metadata and provenance tracking.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.provenance import compute_file_sha256, get_git_commit, is_git_dirty


def set_publication_style():
    """Apply clean, elegant scientific publication styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.1,
    })


def load_all_metrics() -> Dict[str, Any]:
    """Load verified evaluation JSON files across the repository."""
    h_r1_path = PROJECT_ROOT / "outputs/metrics/horizon_r1_test_evaluation.json"
    h_16_path = PROJECT_ROOT / "outputs/metrics/h16_benchmark_evaluation_v3.json"
    h_12_path = PROJECT_ROOT / "outputs/metrics/h12_benchmark_evaluation.json"
    e0_path = PROJECT_ROOT / "outputs/metrics/closure_r4_physics_ablation_seed42_v2.json"

    with open(h_r1_path, "r", encoding="utf-8") as f:
        h_r1 = json.load(f)

    with open(h_16_path, "r", encoding="utf-8") as f:
        h_16 = json.load(f)

    with open(h_12_path, "r", encoding="utf-8") as f:
        h_12 = json.load(f)

    with open(e0_path, "r", encoding="utf-8") as f:
        e0_full = json.load(f)

    return {
        "h_r1": h_r1,
        "h_16": h_16,
        "h_12": h_12,
        "e0": e0_full.get("E0_single_step", {}),
    }


def plot_horizon_controlled_4panel(metrics: Dict[str, Any], out_dir: Path) -> Path:
    """Generate 4-panel publication figure for Horizon-R1 controlled study."""
    set_publication_style()
    h_r1 = metrics["h_r1"]
    horizons = [1, 5, 10, 20, 30]

    # Models data
    groups = {
        "Parent (H=2 pre-ablation)": {
            "vrmse": [h_r1["parent_test"][f"h{h}"]["vrmse_mean"] for h in horizons],
            "color": "#7f7f7f", "ls": ":", "marker": "x", "lw": 2.0, "alpha": 0.85
        },
        "H=2 Control (Short-Best)": {
            "vrmse": [h_r1["H2_short"][f"h{h}"]["vrmse_mean"] for h in horizons],
            "color": "#1f77b4", "ls": "--", "marker": "o", "lw": 2.2, "alpha": 0.95
        },
        "H=2 Control (Long-Best)": {
            "vrmse": [h_r1["H2_long"][f"h{h}"]["vrmse_mean"] for h in horizons],
            "color": "#1f77b4", "ls": "-", "marker": "s", "lw": 2.4, "alpha": 0.95
        },
        "H=4 (Short-Best)": {
            "vrmse": [h_r1["H4_short"][f"h{h}"]["vrmse_mean"] for h in horizons],
            "color": "#2ca02c", "ls": "--", "marker": "^", "lw": 2.2, "alpha": 0.95
        },
        "H=4 (Long-Best)": {
            "vrmse": [h_r1["H4_long"][f"h{h}"]["vrmse_mean"] for h in horizons],
            "color": "#2ca02c", "ls": "-", "marker": "v", "lw": 2.4, "alpha": 0.95
        },
        "H=8 (Short-Best)": {
            "vrmse": [h_r1["H8_short"][f"h{h}"]["vrmse_mean"] for h in horizons],
            "color": "#d62728", "ls": "--", "marker": "D", "lw": 2.4, "alpha": 0.95
        },
        "H=8 (Long-Best)": {
            "vrmse": [h_r1["H8_long"][f"h{h}"]["vrmse_mean"] for h in horizons],
            "color": "#d62728", "ls": "-", "marker": "*", "lw": 2.8, "alpha": 1.0
        },
    }

    fig = plt.figure(figsize=(18, 13), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # ----------------------------------------------------
    # Panel (a): Rollout Field Mean VRMSE vs Future Horizon h
    # ----------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    for label, g in groups.items():
        ax_a.plot(
            horizons, g["vrmse"],
            label=label,
            color=g["color"],
            linestyle=g["ls"],
            marker=g["marker"],
            linewidth=g["lw"],
            markersize=8,
            alpha=g["alpha"]
        )

    # Annotate key gain at h=10
    v_h2 = h_r1["H2_short"]["h10"]["vrmse_mean"]
    v_h8 = h_r1["H8_short"]["h10"]["vrmse_mean"]
    gain_pct = (v_h2 - v_h8) / v_h2 * 100
    ax_a.annotate(
        f"$h=10$ Error Drop:\n$\\mathbf{{-{gain_pct:.1f}\\%}}$ ($0.690 \\to 0.178$)",
        xy=(10, v_h8),
        xytext=(11.5, 0.38),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef3c7", edgecolor="#f59e0b", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.5),
    )

    # Annotate key gain at h=30
    v_h2_30 = h_r1["H2_long"]["h30"]["vrmse_mean"]
    v_h8_30 = h_r1["H8_long"]["h30"]["vrmse_mean"]
    gain_30 = (v_h2_30 - v_h8_30) / v_h2_30 * 100
    ax_a.annotate(
        f"$h=30$ Long-Best Drop:\n$\\mathbf{{-{gain_30:.1f}\\%}}$ ($1.369 \\to 0.855$)",
        xy=(30, v_h8_30),
        xytext=(21, 0.98),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcfce7", edgecolor="#16a34a", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.5),
    )

    ax_a.set_xticks(horizons)
    ax_a.set_xlabel("Future Rollout Prediction Horizon $h$ (steps)")
    ax_a.set_ylabel("Field Mean VRMSE (Test Set)")
    ax_a.set_title("(a) Autoregressive Rollout Degradation: Future Horizon $h$ vs Controlled $H$")
    ax_a.legend(loc="upper left", framealpha=0.95)
    ax_a.set_ylim(-0.02, 1.55)

    # ----------------------------------------------------
    # Panel (b): Channel-wise Error at h=10 (Medium Horizon)
    # ----------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    channels = ["Streamwise $u$", "Transverse $v$", "Pressure $p$", "Tracer $s$"]
    ch_keys = ["vrmse_u", "vrmse_v", "vrmse_p", "vrmse_s"]

    x = np.arange(len(channels))
    w = 0.22

    h2_vals = [h_r1["H2_short"]["h10"][k] for k in ch_keys]
    h4_vals = [h_r1["H4_short"]["h10"][k] for k in ch_keys]
    h8_vals = [h_r1["H8_short"]["h10"][k] for k in ch_keys]

    ax_b.bar(x - w, h2_vals, width=w, label="H=2 Control (Short-Best)", color="#1f77b4", edgecolor="black", alpha=0.85)
    ax_b.bar(x, h4_vals, width=w, label="H=4 (Short-Best)", color="#2ca02c", edgecolor="black", alpha=0.85)
    ax_b.bar(x + w, h8_vals, width=w, label="H=8 (Short-Best)", color="#d62728", edgecolor="black", alpha=0.85)

    # Value tags above H8 bars
    for i, (v2, v8) in enumerate(zip(h2_vals, h8_vals)):
        drop = (v2 - v8) / v2 * 100
        ax_b.text(i + w, v8 + 0.02, f"-{drop:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#b91c1c")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(channels)
    ax_b.set_ylabel("Channel VRMSE at $h=10$")
    ax_b.set_title("(b) Multi-Channel Physical Error Suppression at Intermediate Horizon ($h=10$)")
    ax_b.legend(loc="upper right", framealpha=0.95)
    ax_b.set_ylim(0, 1.05)

    # ----------------------------------------------------
    # Panel (c): Terminal Horizon h=30 Channel-wise Comparison
    # ----------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    h2_long_vals = [h_r1["H2_long"]["h30"][k] for k in ch_keys]
    h4_long_vals = [h_r1["H4_long"]["h30"][k] for k in ch_keys]
    h8_long_vals = [h_r1["H8_long"]["h30"][k] for k in ch_keys]
    parent_vals = [h_r1["parent_test"]["h30"][k] for k in ch_keys]

    w_c = 0.18
    ax_c.bar(x - 1.5*w_c, parent_vals, width=w_c, label="Parent (H=2 Baseline)", color="#7f7f7f", edgecolor="black", alpha=0.8)
    ax_c.bar(x - 0.5*w_c, h2_long_vals, width=w_c, label="H=2 Control (Long-Best)", color="#1f77b4", edgecolor="black", alpha=0.85)
    ax_c.bar(x + 0.5*w_c, h4_long_vals, width=w_c, label="H=4 (Long-Best)", color="#2ca02c", edgecolor="black", alpha=0.85)
    ax_c.bar(x + 1.5*w_c, h8_long_vals, width=w_c, label="H=8 (Long-Best)", color="#d62728", edgecolor="black", alpha=0.85)

    # Highlight tracer suppression
    s_idx = 3
    s_parent = parent_vals[s_idx]
    s_h8 = h8_long_vals[s_idx]
    s_drop = (s_parent - s_h8) / s_parent * 100
    ax_c.annotate(
        f"Tracer $s$ Divergence Suppressed:\n$\\mathbf{{-{s_drop:.1f}\\%}}$ ($2.214 \\to 0.902$)",
        xy=(x[s_idx] + 1.5*w_c, s_h8),
        xytext=(x[s_idx] - 0.5, 1.85),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3e8ff", edgecolor="#9333ea", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#7e22ce", lw=1.5),
    )

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(channels)
    ax_c.set_ylabel("Channel VRMSE at Terminal $h=30$")
    ax_c.set_title("(c) Terminal Horizon ($h=30$) Robustness: Preventing Passive Scalar Explosion")
    ax_c.legend(loc="upper left", framealpha=0.95)
    ax_c.set_ylim(0, 2.5)

    # ----------------------------------------------------
    # Panel (d): 2D Error Heatmap Matrix (Controlled H vs Future h)
    # ----------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    h_labels = [f"h={h}" for h in horizons]
    ctrl_rows = ["H=2 Control", "H=4 Expansion", "H=8 Expansion"]

    matrix_data = np.array([
        [h_r1["H2_short"][f"h{h}"]["vrmse_mean"] for h in horizons],
        [h_r1["H4_short"][f"h{h}"]["vrmse_mean"] for h in horizons],
        [h_r1["H8_short"][f"h{h}"]["vrmse_mean"] for h in horizons],
    ])

    im = ax_d.imshow(matrix_data, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.35)
    ax_d.set_xticks(range(len(horizons)))
    ax_d.set_xticklabels(h_labels, fontsize=11, fontweight="bold")
    ax_d.set_yticks(range(len(ctrl_rows)))
    ax_d.set_yticklabels(ctrl_rows, fontsize=11, fontweight="bold")

    # Add numeric labels in each cell
    for r in range(len(ctrl_rows)):
        for c in range(len(horizons)):
            val = matrix_data[r, c]
            text_color = "white" if val > 0.75 else "black"
            ax_d.text(c, r, f"{val:.4f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=11)

    # Mark training horizons cutoff on heatmap
    # H=2 is at step index 0.5 (between 1 and 5)
    # H=4 is around step index 0.8
    # H=8 is around step index 1.8
    ax_d.scatter([0.25], [0], color="#2563eb", marker="|", s=400, linewidths=3.5, label="Training Horizon $H$ Boundary")
    ax_d.scatter([0.75], [1], color="#2563eb", marker="|", s=400, linewidths=3.5)
    ax_d.scatter([1.75], [2], color="#2563eb", marker="|", s=400, linewidths=3.5)

    cbar = plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    cbar.set_label("Field Mean VRMSE", fontsize=11, fontweight="bold")

    ax_d.set_title("(d) Quantitative Error Matrix: Controlled Variable $H$ vs Future Horizon $h$")
    ax_d.set_xlabel("Future Prediction Horizon $h$ (steps)")
    ax_d.set_ylabel("Controlled Variable $H$")
    ax_d.legend(loc="lower right", framealpha=0.92, fontsize=9)

    fig.suptitle(
        "Controlled Variable Experiment: Training Horizon $H$ Predicting Future Horizon $h$ Trajectories\n"
        "Strict Horizon-R1 Protocol: Parent E4-H2 Seed 42, $B_{\\mathrm{eff}}=8$, 12 Epochs, Constant $\\lambda_{\\mathrm{div}}, \\lambda_\\omega$",
        fontsize=15, fontweight="bold"
    )

    out_file = out_dir / "figure_horizon_controlled_experiment.png"
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved publication figure to: {out_file}")
    return out_file


def plot_grand_horizon_ladder_test_curves(metrics: Dict[str, Any], out_dir: Path) -> Path:
    """Generate comprehensive scaling analysis for full ladder H in {1, 2, 4, 8, 12, 16}."""
    set_publication_style()
    h_r1 = metrics["h_r1"]
    h_16 = metrics["h_16"]
    h_12 = metrics["h_12"]
    e0 = metrics["e0"]

    # Unified ladder definition across test set
    ladder_models = {
        "H=1 (E0 Single-Step)": {
            "H": 1,
            "steps": [1, 5, 10, 20, 30],
            "vrmse": [e0[f"step_{s}"]["vrmse_mean"] for s in [1, 5, 10, 20, 30]],
            "color": "#7f7f7f", "ls": ":", "marker": "o", "lw": 2.0
        },
        "H=2 (E4-H2 Control)": {
            "H": 2,
            "steps": [1, 5, 10, 20, 30],
            "vrmse": [h_r1["H2_short"][f"h{s}"]["vrmse_mean"] for s in [1, 5, 10, 20, 30]],
            "color": "#1f77b4", "ls": "--", "marker": "s", "lw": 2.2
        },
        "H=4 (E4-H4 Curriculum)": {
            "H": 4,
            "steps": [1, 5, 10, 20, 30],
            "vrmse": [h_r1["H4_short"][f"h{s}"]["vrmse_mean"] for s in [1, 5, 10, 20, 30]],
            "color": "#2ca02c", "ls": "-.", "marker": "^", "lw": 2.2
        },
        "H=8 (E4-H8 Long-Best)": {
            "H": 8,
            "steps": [1, 5, 10, 16, 20, 30],
            "vrmse": [
                h_16["results"]["parent_h8_ep11"]["metrics"][f"step_{s}"]["vrmse_mean"]
                for s in [1, 5, 10, 16, 20, 30]
            ],
            "color": "#ff7f0e", "ls": "-", "marker": "v", "lw": 2.4
        },
        "H=12 (E4-H12 Long-Best)": {
            "H": 12,
            "steps": [1, 5, 10, 16, 20, 30],
            "vrmse": [
                h_12["results"]["h12_long_best"]["metrics"][f"step_{s}"]["vrmse_mean"]
                for s in [1, 5, 10, 16, 20, 30]
            ],
            "color": "#9467bd", "ls": "-", "marker": "D", "lw": 2.6
        },
        "H=16 (E4-H16 Long-Best)": {
            "H": 16,
            "steps": [1, 5, 10, 16, 20, 30],
            "vrmse": [
                h_16["results"]["h16_long_best_ep12"]["metrics"][f"step_{s}"]["vrmse_mean"]
                for s in [1, 5, 10, 16, 20, 30]
            ],
            "color": "#d62728", "ls": "-", "marker": "*", "lw": 3.0
        },
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.8), constrained_layout=True)

    # ----------------------------------------------------
    # Subplot 1: Full Ladder Test Degradation Curves
    # ----------------------------------------------------
    for name, m in ladder_models.items():
        ax1.plot(
            m["steps"], m["vrmse"],
            label=name,
            color=m["color"],
            linestyle=m["ls"],
            marker=m["marker"],
            linewidth=m["lw"],
            markersize=8,
        )

    ax1.set_xticks([1, 5, 10, 16, 20, 30])
    ax1.set_xlabel("Autoregressive Rollout Horizon $h$ (steps)")
    ax1.set_ylabel("Field Mean VRMSE (Test Set)")
    ax1.set_title("(a) Full Horizon Ladder Test Trajectories ($H \\in \\{1, 2, 4, 8, 12, 16\\}$)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.95)
    ax1.set_ylim(-0.02, 1.55)

    # Annotate total improvement at h=30
    h1_30 = ladder_models["H=1 (E0 Single-Step)"]["vrmse"][-1]
    h16_30 = ladder_models["H=16 (E4-H16 Long-Best)"]["vrmse"][-1]
    total_drop = (h1_30 - h16_30) / h1_30 * 100
    ax1.annotate(
        f"Terminal $h=30$ Error Compression:\n$\\mathbf{{-{total_drop:.1f}\\%}}$ ($1.257 \\to 0.289$)",
        xy=(30, h16_30),
        xytext=(16, 0.52),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcfce7", edgecolor="#16a34a", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.5),
    )

    # ----------------------------------------------------
    # Subplot 2: Terminal Error & Speedup Scaling vs Large-H
    # ----------------------------------------------------
    h_vals = [1, 2, 4, 8, 12, 16]
    terminal_vrmses = [
        ladder_models["H=1 (E0 Single-Step)"]["vrmse"][-1],
        ladder_models["H=2 (E4-H2 Control)"]["vrmse"][-1],
        ladder_models["H=4 (E4-H4 Curriculum)"]["vrmse"][-1],
        ladder_models["H=8 (E4-H8 Long-Best)"]["vrmse"][-1],
        ladder_models["H=12 (E4-H12 Long-Best)"]["vrmse"][-1],
        ladder_models["H=16 (E4-H16 Long-Best)"]["vrmse"][-1],
    ]
    h10_vrmses = [
        ladder_models["H=1 (E0 Single-Step)"]["vrmse"][2],
        ladder_models["H=2 (E4-H2 Control)"]["vrmse"][2],
        ladder_models["H=4 (E4-H4 Curriculum)"]["vrmse"][2],
        ladder_models["H=8 (E4-H8 Long-Best)"]["vrmse"][2],
        ladder_models["H=12 (E4-H12 Long-Best)"]["vrmse"][2],
        ladder_models["H=16 (E4-H16 Long-Best)"]["vrmse"][2],
    ]

    ax2.plot(h_vals, terminal_vrmses, "-o", color="#d62728", linewidth=2.5, markersize=8, label="Terminal Error at $h=30$")
    ax2.plot(h_vals, h10_vrmses, "-s", color="#1f77b4", linewidth=2.5, markersize=8, label="Medium Horizon Error at $h=10$")

    for x_h, y_term in zip(h_vals, terminal_vrmses):
        offset = 0.06 if x_h != 1 else -0.08
        va = "bottom" if x_h != 1 else "top"
        ax2.text(x_h, y_term + offset, f"{y_term:.3f}", ha="center", va=va, fontsize=9.5, fontweight="bold", color="#b91c1c")

    for x_h, y_10 in zip(h_vals, h10_vrmses):
        offset = -0.07 if x_h != 1 else 0.06
        va = "top" if x_h != 1 else "bottom"
        ax2.text(x_h, y_10 + offset, f"{y_10:.3f}", ha="center", va=va, fontsize=9.5, fontweight="bold", color="#1d4ed8")

    ax2.set_xticks(h_vals)
    ax2.set_xlabel("Training Horizon Controlled Variable $H$ (steps)")
    ax2.set_ylabel("Field Mean VRMSE")
    ax2.set_title("(b) Error Power Decay Scaling vs Training Horizon $H$")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.95)
    ax2.set_ylim(0.0, 1.55)

    fig.suptitle(
        "Grand Horizon Ladder: Controlled Scaling Analysis of Training Horizon $H$ for Autoregressive Rollouts $h \\in [1, 30]$\n"
        "The Well Incompressible Shear Flow Benchmark (Test Set $N=45$ Windows)",
        fontsize=15, fontweight="bold"
    )

    out_file = out_dir / "figure_grand_horizon_ladder_test_curves.png"
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved publication ladder figure to: {out_file}")
    return out_file


def save_manifest(out_dir: Path, fig1_path: Path, fig2_path: Path, metrics: Dict[str, Any]):
    """Save formal JSON metadata manifest for the generated figures."""
    h_r1 = metrics["h_r1"]
    h_16 = metrics["h_16"]
    h_12 = metrics["h_12"]
    e0 = metrics["e0"]

    manifest = {
        "metadata": {
            "title": "Horizon Controlled Variable Experiment: Training Horizon H vs Rollout Horizon h",
            "generation_timestamp": datetime.datetime.now().isoformat(),
            "git_commit": get_git_commit(),
            "git_dirty": is_git_dirty(),
            "figures_generated": [str(fig1_path), str(fig2_path)],
            "figure_1_sha256": compute_file_sha256(fig1_path),
            "figure_2_sha256": compute_file_sha256(fig2_path),
        },
        "quantitative_summary": {
            "horizon_r1_controlled_groups": {
                "H2_control_short": {h: h_r1["H2_short"][h]["vrmse_mean"] for h in ["h1", "h5", "h10", "h20", "h30"]},
                "H2_control_long": {h: h_r1["H2_long"][h]["vrmse_mean"] for h in ["h1", "h5", "h10", "h20", "h30"]},
                "H4_short": {h: h_r1["H4_short"][h]["vrmse_mean"] for h in ["h1", "h5", "h10", "h20", "h30"]},
                "H4_long": {h: h_r1["H4_long"][h]["vrmse_mean"] for h in ["h1", "h5", "h10", "h20", "h30"]},
                "H8_short": {h: h_r1["H8_short"][h]["vrmse_mean"] for h in ["h1", "h5", "h10", "h20", "h30"]},
                "H8_long": {h: h_r1["H8_long"][h]["vrmse_mean"] for h in ["h1", "h5", "h10", "h20", "h30"]},
                "parent_baseline": {h: h_r1["parent_test"][h]["vrmse_mean"] for h in ["h1", "h5", "h10", "h20", "h30"]},
            },
            "full_ladder_h30_terminal_vrmse": {
                "H1": e0["step_30"]["vrmse_mean"],
                "H2": h_r1["H2_short"]["h30"]["vrmse_mean"],
                "H4": h_r1["H4_short"]["h30"]["vrmse_mean"],
                "H8": h_16["results"]["parent_h8_ep11"]["metrics"]["step_30"]["vrmse_mean"],
                "H12": h_12["results"]["h12_long_best"]["metrics"]["step_30"]["vrmse_mean"],
                "H16": h_16["results"]["h16_long_best_ep12"]["metrics"]["step_30"]["vrmse_mean"],
            },
        },
    }

    manifest_path = out_dir / "figure_horizon_controlled_experiment_metadata.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved metadata manifest to: {manifest_path}")


def main():
    print("=" * 80)
    print("PLOTTING HORIZON CONTROLLED VARIABLE EXPERIMENT FIGURES")
    print("=" * 80)

    out_dir = PROJECT_ROOT / "outputs" / "figures" / "horizon_r1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metrics data...")
    metrics = load_all_metrics()

    print("Generating Figure 1: 4-Panel Controlled Experiment (Horizon-R1)...")
    fig1 = plot_horizon_controlled_4panel(metrics, out_dir)

    print("Generating Figure 2: Grand Horizon Ladder Scaling Analysis...")
    fig2 = plot_grand_horizon_ladder_test_curves(metrics, out_dir)

    print("Saving Metadata Manifest...")
    save_manifest(out_dir, fig1, fig2, metrics)

    print("\n[SUCCESS] Completed all controlled variable result figure generations!")


if __name__ == "__main__":
    main()
