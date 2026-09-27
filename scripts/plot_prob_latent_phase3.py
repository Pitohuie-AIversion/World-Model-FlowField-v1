"""Plot publication-quality benchmark curves and uncertainty calibration figures for Phase 3 probabilistic evaluation.

Reads frozen metrics from outputs/metrics/phase3_probabilistic_evaluation.json and generates:
1. figure_1_vrmse_evolution.png / .pdf: Standard benchmark VRMSE evolution across horizons h in {1, 5, 10, 20, 30}.
   (Field-averaged over 4 channels u, v, p, s; distinct from custom RMS diagnostic).
2. figure_2_interval_calibration.png / .pdf: Nominal coverage vs. empirical coverage (Reliability diagram)
   at nominal levels 50%, 80%, 90%, 95% with non-overlapping categorical bar layout and percentage point accounting.
3. figure_3_spread_skill_relationship.png / .pdf: Velocity vector Pooled RMS Spread vs. Pooled RMSE,
   and Pooled Spread-Skill Ratio (SSR) vs. Horizon h with Spread-RMSE parity reference line.
4. figure_summary_phase3.png / .pdf: 3-panel publication summary consolidating all evaluation dimensions.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Publication-grade aesthetic styling
STYLE = {
    "D0": {
        "label": "D0: Deterministic Baseline",
        "color": "#1f77b4",  # Blue
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.2,
        "markersize": 7,
    },
    "G0": {
        "label": "G0: Homoscedastic Baseline",
        "color": "#2ca02c",  # Green
        "marker": "s",
        "linestyle": "--",
        "linewidth": 2.2,
        "markersize": 7,
    },
    "G1": {
        "label": "G1: Heteroscedastic Model",
        "color": "#d62728",  # Red / Coral
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 2.4,
        "markersize": 7,
    },
    "ref": {
        "color": "#7f7f7f",
        "linestyle": "--",
        "linewidth": 1.6,
    },
}


def load_phase3_metrics(json_path: Path) -> Dict[str, Any]:
    """Load and validate the structure of Phase 3 evaluation report."""
    if not json_path.exists():
        raise FileNotFoundError(f"Evaluation report not found at {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "step1_single_step_test_evaluation" not in data:
        raise KeyError("Missing 'step1_single_step_test_evaluation' in report.")
    if "step2_autoregressive_rollout_evaluation" not in data:
        raise KeyError("Missing 'step2_autoregressive_rollout_evaluation' in report.")

    return data


def extract_metadata_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract rollout configuration parameters for dynamic annotations."""
    cfg = data.get("step2_autoregressive_rollout_evaluation", {}).get("configuration", {})
    windows_evaluated = cfg.get("windows_evaluated", len(cfg.get("window_manifest", [])))
    if windows_evaluated == 0:
        windows_evaluated = cfg.get("total_windows_in_dataset", 105)

    num_samples = cfg.get("num_samples", 32)
    inflation_factor = cfg.get("finite_k_inflation_factor")
    if inflation_factor is None:
        inflation_factor = math.sqrt((num_samples + 1.0) / num_samples)

    return {
        "windows_evaluated": int(windows_evaluated),
        "num_samples": int(num_samples),
        "finite_k_inflation_factor": float(inflation_factor),
    }


def extract_vrmse_data(data: Dict[str, Any]) -> Tuple[List[int], Dict[str, List[float]], Dict[str, List[float]]]:
    """Extract standard VRMSE and RMS diagnostic VRMSE across horizons for D0, G0, G1.

    Returns:
        horizons: List of horizon step integers, e.g. [1, 5, 10, 20, 30].
        std_vrmse: Dict mapping model name to list of standard VRMSE values.
        rms_diag_vrmse: Dict mapping model name to list of custom RMS diagnostic values.
    """
    rollout_data = data["step2_autoregressive_rollout_evaluation"]["per_horizon_metrics"]
    horizons = []
    std_vrmse = {"D0": [], "G0": [], "G1": []}
    rms_diag_vrmse = {"D0": [], "G0": [], "G1": []}

    for key, h_metrics in sorted(rollout_data.items(), key=lambda x: int(x[0].split("_")[1])):
        h = int(key.split("_")[1])
        horizons.append(h)
        for model in ("D0", "G0", "G1"):
            if model in h_metrics:
                m_data = h_metrics[model]
                std_vrmse[model].append(float(m_data["ensemble_mean_vrmse"]))
                rms_diag_vrmse[model].append(float(m_data.get("rms_of_window_mean_vrmse", float("nan"))))
            else:
                std_vrmse[model].append(float("nan"))
                rms_diag_vrmse[model].append(float("nan"))

    return horizons, std_vrmse, rms_diag_vrmse


def extract_calibration_data(data: Dict[str, Any]) -> Tuple[List[float], Dict[str, List[float]], Dict[str, List[float]]]:
    """Extract nominal and empirical interval coverages and signed calibration deviations in percentage points.

    Returns:
        nominal_pcts: Nominal confidence levels as percentages, e.g. [50.0, 80.0, 90.0, 95.0].
        empirical_pcts: Dict mapping model name ('G0', 'G1') to empirical coverages as percentages.
        deviation_pct_points: Dict mapping model name to signed deviation (empirical - nominal) in percentage points.
    """
    step1 = data["step1_single_step_test_evaluation"]
    levels = ["50", "80", "90", "95"]
    nominal_pcts = [float(lvl) for lvl in levels]

    empirical_pcts = {"G0": [], "G1": []}
    deviation_pct_points = {"G0": [], "G1": []}

    for model, m_key in [("G0", "G0_homoscedastic_baseline"), ("G1", "G1_heteroscedastic_model")]:
        intervals = step1[m_key]["intervals"]
        for lvl in levels:
            lvl_info = intervals[lvl]
            picp = float(lvl_info["picp"]) * 100.0  # Convert to %
            nom = float(lvl_info["nominal"]) * 100.0
            dev = picp - nom  # Signed deviation in percentage points
            empirical_pcts[model].append(picp)
            deviation_pct_points[model].append(dev)

    return nominal_pcts, empirical_pcts, deviation_pct_points


def extract_spread_skill_data(data: Dict[str, Any]) -> Tuple[List[int], Dict[str, Dict[str, List[float]]]]:
    """Extract Pooled Spread and Pooled RMSE for velocity vector across horizons.

    Returns:
        horizons: List of horizon step integers.
        ss_data: Dict mapping model name ('G0', 'G1') to dict with keys:
            'pooled_spread', 'pooled_spread_adj', 'pooled_rmse', 'pooled_ssr', 'pooled_ssr_adj'.
    """
    rollout_data = data["step2_autoregressive_rollout_evaluation"]["per_horizon_metrics"]
    horizons = []
    ss_data = {
        "G0": {
            "pooled_spread": [],
            "pooled_spread_adj": [],
            "pooled_rmse": [],
            "pooled_ssr": [],
            "pooled_ssr_adj": [],
        },
        "G1": {
            "pooled_spread": [],
            "pooled_spread_adj": [],
            "pooled_rmse": [],
            "pooled_ssr": [],
            "pooled_ssr_adj": [],
        },
    }

    for key, h_metrics in sorted(rollout_data.items(), key=lambda x: int(x[0].split("_")[1])):
        h = int(key.split("_")[1])
        horizons.append(h)
        for model in ("G0", "G1"):
            vel_ss = h_metrics[model]["spread_skill_velocity"]
            ss_data[model]["pooled_spread"].append(float(vel_ss["pooled_rms_spread_raw"]))
            ss_data[model]["pooled_spread_adj"].append(float(vel_ss["pooled_rms_spread_adjusted"]))
            ss_data[model]["pooled_rmse"].append(float(vel_ss["pooled_rmse"]))
            ss_data[model]["pooled_ssr"].append(float(vel_ss["pooled_ssr_raw"]))
            ss_data[model]["pooled_ssr_adj"].append(float(vel_ss["pooled_ssr_adjusted"]))

    return horizons, ss_data


def save_figure_dual_format(fig: plt.Figure, save_path: Path):
    """Save figure in both high-resolution raster (PNG) and publication vector (PDF) formats."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = save_path.with_suffix(".png")
    pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path} and {pdf_path}")


def plot_figure_1_vrmse(
    horizons: List[int],
    std_vrmse: Dict[str, List[float]],
    rms_diag_vrmse: Dict[str, List[float]],
    config_meta: Dict[str, Any],
    save_path: Path,
):
    """Plot Figure 1: Standard benchmark VRMSE evolution and RMS diagnostic."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Standard Benchmark VRMSE (Arithmetic Mean across windows)
    for model in ("D0", "G0", "G1"):
        ax1.plot(
            horizons,
            std_vrmse[model],
            label=STYLE[model]["label"],
            color=STYLE[model]["color"],
            marker=STYLE[model]["marker"],
            linestyle=STYLE[model]["linestyle"],
            linewidth=STYLE[model]["linewidth"],
            markersize=STYLE[model]["markersize"],
        )

    ax1.set_xlabel("Autoregressive Rollout Horizon $h$", fontsize=12)
    ax1.set_ylabel("Standard VRMSE (4-Channel Average: $u, v, p, s$)", fontsize=12)
    ax1.set_title("(a) Standard Benchmark VRMSE Evolution", fontsize=13, fontweight="bold")
    ax1.set_xticks(horizons)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # Dynamic annotation noting discrete evaluated steps and window count
    windows_count = config_meta.get("windows_evaluated", 105)
    ax1.text(
        0.05,
        0.65,
        "Standard Benchmark Metric:\n"
        r"$\mathrm{VRMSE} = \frac{1}{W} \sum_{w=1}^W \mathrm{vrmse}_w$" + "\n"
        f"Evaluated on {windows_count} full test windows\n"
        "(Markers at discrete evaluation steps)",
        transform=ax1.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9),
    )

    # Panel B: Standard VRMSE vs. RMS Diagnostic Comparison
    for model in ("D0", "G0", "G1"):
        ax2.plot(
            horizons,
            std_vrmse[model],
            label=f"{model} Standard Mean",
            color=STYLE[model]["color"],
            marker=STYLE[model]["marker"],
            linestyle="-",
            linewidth=2.0,
            markersize=6,
        )
        ax2.plot(
            horizons,
            rms_diag_vrmse[model],
            label=f"{model} RMS Diagnostic",
            color=STYLE[model]["color"],
            marker=STYLE[model]["marker"],
            linestyle=":",
            linewidth=1.8,
            markersize=5,
            alpha=0.75,
        )

    ax2.set_xlabel("Autoregressive Rollout Horizon $h$", fontsize=12)
    ax2.set_ylabel("Error Metric Value", fontsize=12)
    ax2.set_title("(b) Standard VRMSE vs. Custom RMS Diagnostic", fontsize=13, fontweight="bold")
    ax2.set_xticks(horizons)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8.5, ncol=2)

    # Diagnostic formula text box
    ax2.text(
        0.05,
        0.55,
        "Custom RMS Diagnostic:\n"
        r"$\mathrm{RMS\text{-}VRMSE} = \sqrt{\frac{1}{W} \sum_{w=1}^W \mathrm{vrmse}_w^2}$" + "\n"
        r"Notice: $\mathrm{RMS\text{-}VRMSE} \geq \mathrm{VRMSE}_{\mathrm{std}}$" + "\n"
        "(Preserved as auxiliary diagnostic)",
        transform=ax2.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9),
    )

    plt.tight_layout()
    save_figure_dual_format(fig, save_path)
    plt.close(fig)


def plot_figure_2_calibration(
    nominal_pcts: List[float],
    empirical_pcts: Dict[str, List[float]],
    deviation_pct_points: Dict[str, List[float]],
    save_path: Path,
):
    """Plot Figure 2: Reliability Diagram and Non-Overlapping Categorical Bar Layout."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Reliability Diagram (Real Percentage Coordinates)
    ax1.plot([0, 100], [0, 100], label="Ideal Calibration ($y = x$)", color=STYLE["ref"]["color"], linestyle="--", linewidth=1.8)

    # Plot G0 and G1
    for model in ("G0", "G1"):
        ax1.plot(
            nominal_pcts,
            empirical_pcts[model],
            label=STYLE[model]["label"],
            color=STYLE[model]["color"],
            marker=STYLE[model]["marker"],
            linestyle=STYLE[model]["linestyle"],
            linewidth=STYLE[model]["linewidth"],
            markersize=STYLE[model]["markersize"],
        )

    ax1.set_xlim(40, 100)
    ax1.set_ylim(20, 102)
    ax1.set_xlabel("Nominal Coverage Level (%)", fontsize=12)
    ax1.set_ylabel("Empirical Coverage PICP (%)", fontsize=12)
    ax1.set_title("(a) Single-Step Latent Reliability Diagram", fontsize=13, fontweight="bold")
    ax1.set_xticks(nominal_pcts)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # Dynamic annotation of under-coverage in central interval
    dev_first = deviation_pct_points["G1"][0]
    ax1.annotate(
        f"G1 {nominal_pcts[0]:g}% Coverage: {empirical_pcts['G1'][0]:.1f}%\n({dev_first:+.2f} percentage pts)",
        xy=(nominal_pcts[0], empirical_pcts["G1"][0]),
        xytext=(nominal_pcts[0] + 2, 33),
        arrowprops=dict(facecolor=STYLE["G1"]["color"], shrink=0.08, width=1.2, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color=STYLE["G1"]["color"],
    )

    # Panel B: Calibration Deviation in Percentage Points (Equidistant Categorical Coordinates)
    # Using categorical index prevents 90% and 95% bar collision!
    x_indices = np.arange(len(nominal_pcts))
    width = 0.35

    rects_g0 = ax2.bar(
        x_indices - width / 2,
        deviation_pct_points["G0"],
        width,
        label=STYLE["G0"]["label"],
        color=STYLE["G0"]["color"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )
    rects_g1 = ax2.bar(
        x_indices + width / 2,
        deviation_pct_points["G1"],
        width,
        label=STYLE["G1"]["label"],
        color=STYLE["G1"]["color"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )

    ax2.axhline(0, color="black", linestyle="-", linewidth=1.0)
    ax2.set_xlabel("Nominal Prediction-Interval Coverage (%)", fontsize=12)
    ax2.set_ylabel("Signed Calibration Deviation (Percentage Points)", fontsize=12)
    ax2.set_title("(b) Interval Calibration Error (Percentage Points)", fontsize=13, fontweight="bold")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([f"{p:g}%" for p in nominal_pcts], fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=10)

    # Add numeric labels on bars
    for rect in rects_g0:
        height = rect.get_height()
        va = "bottom" if height >= 0 else "top"
        ax2.annotate(
            f"{height:+.1f} pp",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -10),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
            color=STYLE["G0"]["color"],
            fontweight="bold",
        )
    for rect in rects_g1:
        height = rect.get_height()
        va = "bottom" if height >= 0 else "top"
        ax2.annotate(
            f"{height:+.1f} pp",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -10),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
            color=STYLE["G1"]["color"],
            fontweight="bold",
        )

    plt.tight_layout()
    save_figure_dual_format(fig, save_path)
    plt.close(fig)


def plot_figure_3_spread_skill(
    horizons: List[int],
    ss_data: Dict[str, Dict[str, List[float]]],
    config_meta: Dict[str, Any],
    save_path: Path,
):
    """Plot Figure 3: Velocity Vector Pooled RMS Spread vs. Pooled RMSE and Spread-Skill Ratio."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Pooled Spread vs. Pooled RMSE
    for model in ("G0", "G1"):
        ax1.plot(
            horizons,
            ss_data[model]["pooled_spread"],
            label=f"{model} Pooled Spread",
            color=STYLE[model]["color"],
            marker="o",
            linestyle="-",
            linewidth=2.2,
            markersize=7,
        )
        ax1.plot(
            horizons,
            ss_data[model]["pooled_rmse"],
            label=f"{model} Pooled RMSE",
            color=STYLE[model]["color"],
            marker="x",
            linestyle="--",
            linewidth=2.0,
            markersize=7,
        )

    ax1.set_xlabel("Autoregressive Rollout Horizon $h$", fontsize=12)
    ax1.set_ylabel("Velocity Error / Dispersion Scale (Physical Units)", fontsize=12)
    ax1.set_title("(a) Velocity Vector Pooled Spread vs. Pooled RMSE", fontsize=13, fontweight="bold")
    ax1.set_xticks(horizons)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    # Note on D0: Pooled RMSE was unarchived, Spread not applicable
    ax1.text(
        0.05,
        0.50,
        "Note on D0 Deterministic Baseline:\n"
        "• Pooled Spread: Not Applicable (Deterministic)\n"
        "• Velocity Pooled RMSE: Unarchived in Report JSON\n"
        "• Pooled SSR: Not Applicable",
        transform=ax1.transAxes,
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9),
    )

    # Panel B: Pooled Spread-Skill Ratio (SSR) Evolution
    # Wording strictly specifies Spread-RMSE parity rather than complete calibration
    ax2.axhline(
        1.0,
        color=STYLE["ref"]["color"],
        linestyle="--",
        linewidth=1.8,
        label=r"Spread–RMSE parity ($\mathrm{ratio} = 1$)",
    )

    num_samples = config_meta.get("num_samples", 32)
    inflation_factor = config_meta.get("finite_k_inflation_factor", math.sqrt((num_samples + 1.0) / num_samples))

    for model in ("G0", "G1"):
        ax2.plot(
            horizons,
            ss_data[model]["pooled_ssr"],
            label=f"{model} Raw Pooled SSR",
            color=STYLE[model]["color"],
            marker=STYLE[model]["marker"],
            linestyle="-",
            linewidth=2.2,
            markersize=7,
        )
        ax2.plot(
            horizons,
            ss_data[model]["pooled_ssr_adj"],
            label=f"{model} Finite-$K$ Adj. SSR ($K={num_samples}$)",
            color=STYLE[model]["color"],
            marker=STYLE[model]["marker"],
            linestyle=":",
            linewidth=1.8,
            markersize=5,
            alpha=0.8,
        )

    ax2.set_xlabel("Autoregressive Rollout Horizon $h$", fontsize=12)
    ax2.set_ylabel("Spread-Skill Ratio (SSR = Spread / RMSE)", fontsize=12)
    ax2.set_title("(b) Velocity Pooled Spread-Skill Ratio (SSR)", fontsize=13, fontweight="bold")
    ax2.set_xticks(horizons)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    # Formula annotations with dynamic K and factor
    ax2.text(
        0.05,
        0.18,
        r"$\mathrm{SSR}_{\mathrm{raw}} = \frac{S_{\mathrm{pooled}}}{R_{\mathrm{pooled}}}$" + "\n"
        r"$S_{\mathrm{pooled}} = \sqrt{\frac{1}{W}\sum_w v_w},\ R_{\mathrm{pooled}} = \sqrt{\frac{1}{W}\sum_w m_w}$" + "\n"
        rf"Raw SSR vs. Finite-$K$ Factor: $\sqrt{{\frac{{{num_samples}+1}}{{{num_samples}}}}} \approx {inflation_factor:.4f}$" + "\n"
        "(Note: Parity reflects dispersion-error scale match, not full distribution calibration)",
        transform=ax2.transAxes,
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9),
    )

    plt.tight_layout()
    save_figure_dual_format(fig, save_path)
    plt.close(fig)


def plot_figure_summary_3panel(
    horizons: List[int],
    std_vrmse: Dict[str, List[float]],
    nominal_pcts: List[float],
    empirical_pcts: Dict[str, List[float]],
    ss_data: Dict[str, Dict[str, List[float]]],
    save_path: Path,
):
    """Plot consolidated 3-panel publication summary figure."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    # Panel 1: Standard VRMSE
    for model in ("D0", "G0", "G1"):
        ax1.plot(
            horizons,
            std_vrmse[model],
            label=f"{model}",
            color=STYLE[model]["color"],
            marker=STYLE[model]["marker"],
            linestyle=STYLE[model]["linestyle"],
            linewidth=STYLE[model]["linewidth"],
            markersize=STYLE[model]["markersize"],
        )
    ax1.set_xlabel("Horizon $h$", fontsize=11)
    ax1.set_ylabel("Standard VRMSE (4-Channel Avg)", fontsize=11)
    ax1.set_title("(a) Field Mean Error (VRMSE)", fontsize=12, fontweight="bold")
    ax1.set_xticks(horizons)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    # Panel 2: Reliability Diagram
    ax2.plot([40, 100], [40, 100], label="Ideal ($y = x$)", color=STYLE["ref"]["color"], linestyle="--", linewidth=1.5)
    for model in ("G0", "G1"):
        ax2.plot(
            nominal_pcts,
            empirical_pcts[model],
            label=f"{model}",
            color=STYLE[model]["color"],
            marker=STYLE[model]["marker"],
            linestyle=STYLE[model]["linestyle"],
            linewidth=STYLE[model]["linewidth"],
            markersize=STYLE[model]["markersize"],
        )
    ax2.set_xlim(40, 100)
    ax2.set_ylim(20, 102)
    ax2.set_xlabel("Nominal Coverage (%)", fontsize=11)
    ax2.set_ylabel("Empirical Coverage PICP (%)", fontsize=11)
    ax2.set_title("(b) Interval Reliability Diagram", fontsize=12, fontweight="bold")
    ax2.set_xticks(nominal_pcts)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    # Panel 3: Spread-Skill Ratio with Parity label
    ax3.axhline(
        1.0,
        color=STYLE["ref"]["color"],
        linestyle="--",
        linewidth=1.5,
        label=r"Parity ($\mathrm{ratio} = 1$)",
    )
    for model in ("G0", "G1"):
        ax3.plot(
            horizons,
            ss_data[model]["pooled_ssr"],
            label=f"{model} Raw SSR",
            color=STYLE[model]["color"],
            marker=STYLE[model]["marker"],
            linestyle="-",
            linewidth=2.0,
            markersize=6,
        )
    ax3.set_xlabel("Horizon $h$", fontsize=11)
    ax3.set_ylabel("Pooled Spread-Skill Ratio (SSR)", fontsize=11)
    ax3.set_title("(c) Velocity Spread-Skill Ratio", fontsize=12, fontweight="bold")
    ax3.set_xticks(horizons)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    save_figure_dual_format(fig, save_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 3 probabilistic evaluation publication figures.")
    parser.add_argument(
        "--metrics_json",
        type=str,
        default="outputs/metrics/phase3_probabilistic_evaluation.json",
        help="Path to phase 3 evaluation metrics JSON report.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/figures/probabilistic",
        help="Directory to save generated figures.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Phase 3 metrics from: {metrics_path}")
    data = load_phase3_metrics(metrics_path)

    # 1. Extract Data and Configuration Metadata
    config_meta = extract_metadata_config(data)
    horizons, std_vrmse, rms_diag_vrmse = extract_vrmse_data(data)
    nominal_pcts, empirical_pcts, deviation_pct_points = extract_calibration_data(data)
    horizons_ss, ss_data = extract_spread_skill_data(data)

    # 2. Generate Figure 1: VRMSE evolution
    fig1_path = output_dir / "figure_1_vrmse_evolution.png"
    plot_figure_1_vrmse(horizons, std_vrmse, rms_diag_vrmse, config_meta, fig1_path)

    # 3. Generate Figure 2: Reliability & Interval Calibration Error
    fig2_path = output_dir / "figure_2_interval_calibration.png"
    plot_figure_2_calibration(nominal_pcts, empirical_pcts, deviation_pct_points, fig2_path)

    # 4. Generate Figure 3: Spread vs. Skill & SSR Evolution
    fig3_path = output_dir / "figure_3_spread_skill_relationship.png"
    plot_figure_3_spread_skill(horizons_ss, ss_data, config_meta, fig3_path)

    # 5. Generate Figure Summary: 3-Panel consolidated view
    fig_summary_path = output_dir / "figure_summary_phase3.png"
    plot_figure_summary_3panel(horizons, std_vrmse, nominal_pcts, empirical_pcts, ss_data, fig_summary_path)

    print("All Phase 3 figures successfully generated and saved in both PNG and PDF formats.")


if __name__ == "__main__":
    main()
