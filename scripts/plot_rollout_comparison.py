"""Plot multi-step autoregressive rollout benchmarks and physical field comparison.

Generates publication-quality 8-panel comparison figures from outputs/metrics/rollout_benchmark.json,
covering field errors, divergence, vorticity, kinetic energy, enstrophy, energy spectrum, and tracer metrics.
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
    """Plot comprehensive 8-panel physical metric evolution curves across prediction steps."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    steps = [1, 5, 10, 20, 30]

    # 8 Key Physical Evaluation Metrics
    metrics_config = [
        # (key, title, y_label, use_log_scale, ideal_ref_val)
        ("vrmse_mean", "Field Mean Error (VRMSE)", "Mean VRMSE", True, None),
        ("div_rmse", r"Divergence Error $\|\nabla \cdot \mathbf{u}\|$", "Divergence RMSE", True, 0.0),
        ("vort_rmse", r"Vorticity Error $\|\omega - \omega^*\|$", "Vorticity RMSE", True, 0.0),
        ("ke_rel_err", "Kinetic Energy Relative Error", r"$|K - K^*| / K^*$", True, 0.0),
        ("enstrophy_rel_err", "Enstrophy Relative Error", r"$|\Omega - \Omega^*| / \Omega^*$", True, 0.0),
        ("energy_spectrum_mae", "Energy Spectrum Error (MAE)", "Spectrum Log MAE", False, 0.0),
        ("tracer_var_retention", "Tracer Variance Retention Ratio", r"$\mathrm{Var}(s) / \mathrm{Var}(s^*)$", False, 1.0),
        ("tracer_mass_error", "Tracer Mass Conservation Error", r"$|M(s) - M(s^*)| / M(s^*)$", True, 0.0),
    ]

    styles = {
        "persistence": {
            "label": "Persistence Baseline (B0)",
            "color": "#7f7f7f",
            "linestyle": ":",
            "marker": "o",
            "linewidth": 1.6,
        },
        "fno": {
            "label": "FNO-2D (Neural Operator)",
            "color": "#2ca02c",
            "linestyle": "-.",
            "marker": "D",
            "linewidth": 1.8,
        },
        "direct_transformer": {
            "label": "Direct ST Transformer (Grid Patch)",
            "color": "#d62728",
            "linestyle": "--",
            "marker": "^",
            "linewidth": 1.8,
        },
        "latent_step1": {
            "label": "Latent Transformer (Step-1 Supervised)",
            "color": "#ff7f0e",
            "linestyle": "--",
            "marker": "v",
            "linewidth": 2.0,
        },
        "latent_rollout": {
            "label": "Latent World Model (Rollout-Aware H=2, Ours)",
            "color": "#1f77b4",
            "linestyle": "-",
            "marker": "s",
            "linewidth": 2.6,
        },
        "latent_transformer": {
            "label": "Latent ST Transformer",
            "color": "#1f77b4",
            "linestyle": "-",
            "marker": "s",
            "linewidth": 2.4,
        },
    }

    fig, axes = plt.subplots(4, 2, figsize=(16, 20), dpi=200)
    axes = axes.flatten()

    for idx, (m_key, m_title, y_label, is_log, ref_val) in enumerate(metrics_config):
        ax = axes[idx]

        # Plot ideal physical reference line if applicable
        if ref_val is not None and not is_log:
            ax.axhline(ref_val, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label="Ideal Reference" if idx == 6 else None)

        for model_name, m_dict in data.items():
            vals = [m_dict.get(f"step_{s}", {}).get(m_key, np.nan) for s in steps]
            cfg = styles.get(
                model_name,
                {"label": model_name, "color": "black", "linestyle": "-", "marker": "x", "linewidth": 1.5},
            )
            ax.plot(
                steps,
                vals,
                label=cfg["label"],
                color=cfg["color"],
                linestyle=cfg["linestyle"],
                marker=cfg.get("marker", "o"),
                linewidth=cfg.get("linewidth", 1.8),
                markersize=7,
                alpha=0.9,
            )

        ax.set_title(f"({chr(97 + idx)}) {m_title}", fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Autoregressive Horizon (Step)", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_xticks(steps)
        ax.grid(True, linestyle="--", alpha=0.5)

        if is_log:
            ax.set_yscale("log")

        # Place legend on first subplot and tracer variance subplot
        if idx == 0:
            ax.legend(fontsize=9, loc="upper left", framealpha=0.92)
        elif idx == 6:
            ax.legend(fontsize=9, loc="best", framealpha=0.92)

    plt.suptitle(
        "The Well Shear Flow V1: 30-Step Physical Rollout Benchmark\nComprehensive Evaluation of Conservation, Kinematics, and Multi-Scale Transport",
        fontsize=16,
        fontweight="bold",
        y=0.992,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.985])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Successfully exported 8-panel publication benchmark curves to: {save_path}")


if __name__ == "__main__":
    plot_benchmark_curves()
