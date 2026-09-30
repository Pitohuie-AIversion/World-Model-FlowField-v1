#!/usr/bin/env python3
"""Publication-grade visualization script for FM-R2 paper figures.

Generates:
1. Figure 2: Pre-registered primary multi-seed outcomes (h=10 Ensemble VRMSE,
   h=10 Energy Spectrum Error with honest Seed 45 reversal, h=5 Ensemble VRMSE),
   strictly separating Discovery Seed 42 from Confirmatory Replication Seeds 43-46.
2. Figure 3: Pre-specified secondary physical rollout metrics (h=10 Sample VRMSE,
   Sample Divergence RMS, Sample Vorticity RMSE) displaying both trajectory-level
   paired slope lines and seed-level clustered mean effects (4/4 seeds improved).

Outputs vector PDF, editable SVG, and 300 DPI publication PNG.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


# ---------------------------------------------------------------------------
# Aesthetic & Publication Style Defaults (Nature / Science style guidelines)
# ---------------------------------------------------------------------------

def setup_publication_style() -> None:
    """Configure matplotlib defaults for high-quality academic publishing."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Lucida Grande"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.8,
        "figure.titlesize": 10.0,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#ededed",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


# Strict Seed Palette
SEED_COLORS = {
    42: "#6c757d",  # Discovery: Neutral Gray
    43: "#1f77b4",  # Replication 1: Steel Blue
    44: "#2ca02c",  # Replication 2: Forest Green
    45: "#d62728",  # Replication 3: Crimson
    46: "#9467bd",  # Replication 4: Royal Purple
}

SEED_MARKERS = {
    42: "D",  # Diamond for discovery
    43: "o",  # Circle
    44: "s",  # Square
    45: "^",  # Triangle up
    46: "v",  # Triangle down
}


# ---------------------------------------------------------------------------
# Data Loading & Processing
# ---------------------------------------------------------------------------

def load_trajectory_paired_data(json_path: str | Path) -> Dict[str, Any]:
    """Load and validate the trajectory-level paired analysis JSON artifact."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Artifact not found at {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_seed_summary(
    data: Dict[str, Any], metric_key: str, seeds: List[int]
) -> List[Dict[str, Any]]:
    """Extract seed-level aggregate means and relative changes for a given metric."""
    results = []
    by_seed = data["by_seed_trajectory"]
    for s in seeds:
        seed_str = str(s)
        trajs = by_seed[seed_str]
        c2_vals = [trajs[t]["metrics"][metric_key]["C2"] for t in trajs]
        r2a_vals = [trajs[t]["metrics"][metric_key]["R2_A"] for t in trajs]
        delta_vals = [trajs[t]["metrics"][metric_key]["delta"] for t in trajs]

        c2_mean = float(np.mean(c2_vals))
        r2a_mean = float(np.mean(r2a_vals))
        delta_mean = r2a_mean - c2_mean
        rel_change_pct = (delta_mean / c2_mean) * 100.0 if c2_mean != 0 else 0.0

        results.append({
            "seed": s,
            "c2_mean": c2_mean,
            "r2a_mean": r2a_mean,
            "delta_mean": delta_mean,
            "rel_change_pct": rel_change_pct,
            "c2_std": float(np.std(c2_vals, ddof=1)),
            "r2a_std": float(np.std(r2a_vals, ddof=1)),
            "c2_vals": c2_vals,
            "r2a_vals": r2a_vals,
            "delta_vals": delta_vals,
        })
    return results


# ---------------------------------------------------------------------------
# Figure 2: Pre-Registered Primary Multi-Seed Outcomes
# ---------------------------------------------------------------------------

def plot_figure2_primary_multiseed(
    data: Dict[str, Any], output_dir: Path, prefix: str = "fig2_primary_multiseed"
) -> Dict[str, Path]:
    """Plot Figure 2: Pre-registered primary multi-seed outcomes (3 panels)."""
    setup_publication_style()

    all_seeds = [42, 43, 44, 45, 46]

    panels_cfg = [
        {
            "key": "primary1_h10_ens_vrmse",
            "panel_label": "(a) Ens VRMSE (h=10)",
            "subtitle": "Primary 1 • Modest / Seed-Variable",
            "summary_tag": "3/4 Seeds Improved\nMean: -3.14% (p=0.140)",
            "ylim": (-9.0, 5.5),
            "yticks": [-8, -6, -4, -2, 0, 2, 4],
            "badge_pos": (0.96, 0.94),
        },
        {
            "key": "primary2_h10_ens_spec_rel_err",
            "panel_label": "(b) Energy Spectrum Error (h=10)",
            "subtitle": "Primary 2 • Majority-Consistent, Sensitive",
            "summary_tag": "3/4 Improved (Mean -12.5%)\nSeed 45 Reversal (+11.99%)",
            "ylim": (-26.0, 24.0),
            "yticks": [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20],
            "badge_pos": (0.96, 0.94),
        },
        {
            "key": "primary4_h5_ens_vrmse",
            "panel_label": "(c) Ens VRMSE (h=5)",
            "subtitle": "Primary 4 • Short-Horizon Invariance",
            "summary_tag": "No Detectable Difference\nMean: -0.18% (p=0.885)",
            "ylim": (-5.0, 6.5),
            "yticks": [-4, -2, 0, 2, 4, 6],
            "badge_pos": (0.96, 0.94),
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.2), dpi=300)
    fig.subplots_adjust(wspace=0.36, top=0.82, bottom=0.23, left=0.08, right=0.98)

    for ax, cfg in zip(axes, panels_cfg):
        seed_stats = extract_seed_summary(data, cfg["key"], all_seeds)
        x_indices = np.arange(len(all_seeds))

        # 1. Shaded background for Discovery Seed 42
        ax.axvspan(-0.5, 0.5, color="#f1f3f5", alpha=1.0, zorder=0)
        ax.axvline(0.5, color="#adb5bd", linestyle="--", linewidth=0.9, zorder=1)

        # 2. Zero reference line (No Effect)
        ax.axhline(0.0, color="#495057", linestyle="-", linewidth=0.8, alpha=0.9, zorder=1)

        # 3. Beneficial direction indicator
        ax.text(
            0.03, 0.04, "▼ Lower is Better (R2-A improves)",
            transform=ax.transAxes, fontsize=6.5, color="#2b8a3e",
            fontweight="bold", va="bottom", ha="left"
        )

        # 4. Connect replication seeds with thin guideline
        rep_x = x_indices[1:]
        rep_y = [s["rel_change_pct"] for s in seed_stats[1:]]
        ax.plot(rep_x, rep_y, color="#ced4da", linestyle=":", linewidth=1.1, zorder=2)

        y_span = cfg["ylim"][1] - cfg["ylim"][0]

        # Plot individual seed deltas
        for i, s_stat in enumerate(seed_stats):
            s = s_stat["seed"]
            d = s_stat["rel_change_pct"]
            c = SEED_COLORS[s]
            m = SEED_MARKERS[s]

            # Stem line
            stem_c = c if s != 42 else "#6c757d"
            ax.vlines(i, 0, d, color=stem_c, linewidth=2.0, alpha=0.8, zorder=3)

            # Marker
            is_reversal = (s == 45 and "spec" in cfg["key"])
            edge_c = "#212529" if is_reversal else "white"
            ax.plot(
                i, d, marker=m, markersize=7.0, color=c,
                markeredgecolor=edge_c, markeredgewidth=1.1, zorder=4
            )

            # Value label
            y_offset = y_span * 0.045
            if d >= 0:
                va = "bottom"
                y_pos = d + y_offset * 0.35
            else:
                va = "top"
                y_pos = d - y_offset * 0.35

            txt_c = "#c92a2a" if is_reversal else ("#2b8a3e" if d < -1.0 else "#212529")
            ax.text(
                i, y_pos, f"{d:+.1f}%",
                ha="center", va=va, fontsize=6.8, fontweight="bold",
                color=txt_c, zorder=5
            )

        # 5. Titles & Header
        ax.set_title(cfg["panel_label"], pad=16, fontweight="bold", fontsize=8.8)
        ax.text(
            0.5, 1.02, cfg["subtitle"],
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.0, color="#495057", style="italic"
        )

        # Summary box
        bbox_props = dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#ced4da", lw=0.6, alpha=0.95)
        ax.text(
            cfg["badge_pos"][0], cfg["badge_pos"][1], cfg["summary_tag"],
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, bbox=bbox_props, zorder=6
        )

        # X-axis configuration
        ax.set_xticks(x_indices)
        ax.set_xticklabels(["Seed 42\n[Discovery]", "Seed 43", "Seed 44", "Seed 45", "Seed 46"], fontsize=7.2)
        ax.set_xlim(-0.5, len(all_seeds) - 0.5)

        # Y-axis configuration
        ax.set_ylabel("Relative Change Δ (%)", fontsize=8.0)
        ax.set_ylim(cfg["ylim"])
        ax.set_yticks(cfg["yticks"])

    # Output generation
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {}
    for ext in ["pdf", "png", "svg"]:
        p = output_dir / f"{prefix}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        out_paths[ext] = p
    plt.close(fig)
    return out_paths


# ---------------------------------------------------------------------------
# Figure 3: Pre-Specified Secondary Physical Rollout Metrics
# ---------------------------------------------------------------------------

def plot_figure3_secondary_physics(
    data: Dict[str, Any], output_dir: Path, prefix: str = "fig3_secondary_physics"
) -> Dict[str, Path]:
    """Plot Figure 3: Pre-specified secondary physical rollout metrics (4/4 seeds)."""
    setup_publication_style()

    rep_seeds = [43, 44, 45, 46]

    panels_cfg = [
        {
            "key": "secondary_h10_samp_vrmse",
            "panel_label": "(a) Sample VRMSE (h=10)",
            "subtitle": "Single-Trajectory Mean Squared Error",
            "summary_tag": "4/4 Seeds Improved\nReplication Δ: -3.79% (p=0.0469)",
            "ylabel": "Sample VRMSE vs GT",
            "ylim": (0.25, 2.75),
            "yticks": [0.5, 1.0, 1.5, 2.0, 2.5],
        },
        {
            "key": "secondary_h10_samp_div_rms",
            "panel_label": "(b) Sample Divergence RMS (h=10)",
            "subtitle": "Incompressibility Constraint (|∇·u|)",
            "summary_tag": "4/4 Seeds Improved\nReplication Δ: -5.34% (p=0.0107)",
            "ylabel": "Divergence RMS (s⁻¹)",
            "ylim": (0.2, 2.05),
            "yticks": [0.4, 0.8, 1.2, 1.6, 2.0],
        },
        {
            "key": "secondary_h10_samp_vort_rmse",
            "panel_label": "(c) Sample Vorticity RMSE (h=10)",
            "subtitle": "Vorticity Field Realism (ω)",
            "summary_tag": "4/4 Seeds Improved\nReplication Δ: -3.66% (p=0.0181)",
            "ylabel": "Vorticity RMSE vs GT",
            "ylim": (0.3, 2.95),
            "yticks": [0.5, 1.0, 1.5, 2.0, 2.5],
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.3), dpi=300)
    fig.subplots_adjust(wspace=0.36, top=0.82, bottom=0.22, left=0.08, right=0.98)

    by_seed = data["by_seed_trajectory"]

    for ax, cfg in zip(axes, panels_cfg):
        k = cfg["key"]

        # 1. Layer 1: Plot 24 trajectory-level paired slope lines (thin, transparent)
        for s in rep_seeds:
            c = SEED_COLORS[s]
            seed_trajs = by_seed[str(s)]
            for t_id, t_info in seed_trajs.items():
                m = t_info["metrics"][k]
                y_c2 = m["C2"]
                y_r2a = m["R2_A"]
                ax.plot(
                    [0, 1], [y_c2, y_r2a],
                    color=c, alpha=0.25, linewidth=0.9, zorder=2
                )

        # 2. Layer 2: Seed-level cluster mean slope lines (thick, solid)
        seed_means_c2 = []
        seed_means_r2a = []
        for s in rep_seeds:
            c = SEED_COLORS[s]
            m_symbol = SEED_MARKERS[s]
            seed_trajs = by_seed[str(s)]
            c2_vals = [t_info["metrics"][k]["C2"] for t_info in seed_trajs.values()]
            r2a_vals = [t_info["metrics"][k]["R2_A"] for t_info in seed_trajs.values()]

            mean_c2 = float(np.mean(c2_vals))
            mean_r2a = float(np.mean(r2a_vals))

            seed_means_c2.append(mean_c2)
            seed_means_r2a.append(mean_r2a)

            # Bold seed slope line showing downward trend
            ax.plot(
                [0, 1], [mean_c2, mean_r2a],
                color=c, linewidth=2.4, alpha=0.95, zorder=5
            )

            # Markers on C2 (left) and R2-A (right)
            ax.plot(0, mean_c2, marker="o", markersize=6.5, color=c, markeredgecolor="white", markeredgewidth=0.8, zorder=6)
            ax.plot(1, mean_r2a, marker=m_symbol, markersize=7.0, color=c, markeredgecolor="white", markeredgewidth=0.8, zorder=6)

        # Grand Mean across 4 seeds
        grand_c2 = float(np.mean(seed_means_c2))
        grand_r2a = float(np.mean(seed_means_r2a))
        ax.plot(
            [0, 1], [grand_c2, grand_r2a],
            color="#212529", linestyle="--", linewidth=1.8, zorder=7
        )

        # 3. Titles and subtitles
        ax.set_title(cfg["panel_label"], pad=16, fontweight="bold", fontsize=8.8)
        ax.text(
            0.5, 1.02, cfg["subtitle"],
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.0, color="#495057", style="italic"
        )

        # Annotations badge: 4/4 seeds improved placed in top headroom
        bbox_props = dict(boxstyle="round,pad=0.25", fc="#e6fcf5", ec="#20c997", lw=0.75, alpha=0.95)
        ax.text(
            0.96, 0.96, f"✓ {cfg['summary_tag']}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color="#087f5b", fontweight="bold",
            bbox=bbox_props, zorder=8
        )

        # Axes labels & limits
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["C2 (Control)\n[Teacher-Forced]", "R2-A (Treatment)\n[Self-Conditioned]"], fontsize=7.2)
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylabel(cfg["ylabel"], fontsize=8.0)
        ax.set_ylim(cfg["ylim"])
        ax.set_yticks(cfg["yticks"])

    # Shared Legend at bottom
    legend_elements = [
        Line2D([0], [0], color=SEED_COLORS[s], marker=SEED_MARKERS[s], linewidth=2.0, markersize=5.0, label=f"Seed {s}")
        for s in rep_seeds
    ]
    legend_elements.append(
        Line2D([0], [0], color="#212529", linestyle="--", linewidth=1.5, label="Grand Mean")
    )
    legend_elements.append(
        Line2D([0], [0], color="#adb5bd", alpha=0.6, linewidth=1.0, label="Trajectory Pair (N=24)")
    )

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#ced4da",
        fontsize=7.0,
    )

    # Save to multiple publication formats
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {}
    for ext in ["pdf", "png", "svg"]:
        p = output_dir / f"{prefix}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        out_paths[ext] = p
    plt.close(fig)
    return out_paths


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Render publication figures for FM-R2 paper.")
    parser.add_argument(
        "--input-json",
        type=str,
        default="outputs/metrics/fm_r2_multiseed_trajectory_paired_analysis.json",
        help="Path to multiseed trajectory paired analysis JSON artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures/paper",
        help="Directory to save rendered figure artifacts (PDF, PNG, SVG).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)

    print(f"[visualize_paper_figures] Loading paired trajectory data from: {input_path}")
    data = load_trajectory_paired_data(input_path)

    print(f"[visualize_paper_figures] Rendering Figure 2 (Primary Multi-Seed Endpoints)...")
    fig2_paths = plot_figure2_primary_multiseed(data, output_dir)
    for ext, p in fig2_paths.items():
        print(f"  -> Generated Fig 2 ({ext.upper()}): {p} ({p.stat().st_size / 1024:.1f} KB)")

    print(f"[visualize_paper_figures] Rendering Figure 3 (Secondary Physics Rollout Metrics)...")
    fig3_paths = plot_figure3_secondary_physics(data, output_dir)
    for ext, p in fig3_paths.items():
        print(f"  -> Generated Fig 3 ({ext.upper()}): {p} ({p.stat().st_size / 1024:.1f} KB)")

    print(f"[visualize_paper_figures] SUCCESS: All publication figures successfully generated in {output_dir}")


if __name__ == "__main__":
    main()
