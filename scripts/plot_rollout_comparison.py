"""Plot multi-step autoregressive rollout benchmarks and physical field comparison.

Generates publication-quality comparison figures from outputs/metrics/rollout_benchmark.json.
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def plot_benchmark_curves(
    json_path: str = "outputs/metrics/rollout_benchmark.json",
    save_path: str = "outputs/figures/rollout_benchmark_curves.png",
):
    """Plot metric evolution curves across autoregressive prediction steps."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    steps = [1, 5, 10, 20, 30]
    metrics = [
        ("vrmse_mean", "Field Mean VRMSE", True),
        ("div_rmse", r"Divergence Error $\|\nabla \cdot \mathbf{u}\|$", True),
        ("vort_rmse", r"Vorticity Error $\|\omega - \omega^*\|$", True),
        ("energy_spectrum_mae", "Energy Spectrum MAE", True),
    ]

    styles = {
        "persistence": {"label": "Persistence Baseline (B0)", "color": "#7f7f7f", "linestyle": ":", "marker": "o"},
        "latent_transformer": {"label": "Latent ST Transformer (P0, World Model)", "color": "#1f77b4", "linestyle": "-", "marker": "s", "linewidth": 2.5},
        "direct_transformer": {"label": "Direct ST Transformer (B4, Grid Patch)", "color": "#d62728", "linestyle": "--", "marker": "^", "linewidth": 2.0},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes = axes.flatten()

    for idx, (m_key, m_title, is_log) in enumerate(metrics):
        ax = axes[idx]
        for model_name, m_dict in data.items():
            vals = [m_dict.get(f"step_{s}", {}).get(m_key, np.nan) for s in steps]
            cfg = styles.get(model_name, {"label": model_name, "color": "black", "linestyle": "-"})
            ax.plot(steps, vals, label=cfg["label"], color=cfg["color"], linestyle=cfg["linestyle"], marker=cfg.get("marker", "o"), linewidth=cfg.get("linewidth", 1.8), markersize=7)

        ax.set_title(m_title, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Autoregressive Prediction Horizon (Step)", fontsize=11)
        ax.set_ylabel("Metric Value", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        if is_log and m_key in ["div_rmse", "vort_rmse"]:
            ax.set_yscale("log")
        ax.legend(fontsize=9, loc="best", framealpha=0.9)

    plt.suptitle("The Well Shear Flow V1: 30-Step Autoregressive Rollout Benchmark\nLatent World Model vs Direct Patch Baseline vs Persistence", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Successfully exported benchmark curves to: {save_path}")


if __name__ == "__main__":
    plot_benchmark_curves()
