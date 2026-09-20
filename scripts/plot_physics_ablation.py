"""Plot comparison figures for 4-group physical loss ablation study."""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def plot_physics_ablation_curves(
    json_path: str = "outputs/metrics/physics_ablation_benchmark.json",
    save_path: str = "outputs/figures/physics_ablation_curves.png",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    steps = [1, 5, 10, 20, 30]

    # 4 Key Comparative Panels
    panels = [
        ("vrmse_mean", "Field Mean VRMSE", "Mean VRMSE", True),
        ("div_rmse", r"Divergence Error $\|\nabla \cdot \mathbf{u}\|$", "Divergence RMSE", True),
        ("vort_rmse", r"Vorticity Error $\|\omega - \omega^*\|$", "Vorticity RMSE", True),
        ("energy_spectrum_mae", "Energy Spectrum MAE", "Spectrum MAE", False),
    ]

    styles = {
        "L_field": {
            "label": r"1. $L_{\mathrm{field}}$ (Baseline)",
            "color": "#7f7f7f",
            "linestyle": ":",
            "marker": "o",
            "linewidth": 2.0,
        },
        "plus_L_div": {
            "label": r"2. $+ L_{\mathrm{div}}$ (Divergence-Free)",
            "color": "#2ca02c",
            "linestyle": "--",
            "marker": "s",
            "linewidth": 2.2,
        },
        "plus_L_vort": {
            "label": r"3. $+ L_\omega$ (Vorticity Consistency)",
            "color": "#ff7f0e",
            "linestyle": "-.",
            "marker": "^",
            "linewidth": 2.2,
        },
        "plus_L_div_vort": {
            "label": r"4. $+ L_{\mathrm{div}} + L_\omega$ (Full Physics)",
            "color": "#1f77b4",
            "linestyle": "-",
            "marker": "D",
            "linewidth": 2.8,
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=200)
    axes = axes.flatten()

    for idx, (m_key, m_title, y_label, is_log) in enumerate(panels):
        ax = axes[idx]
        for grp_name, m_dict in data.items():
            vals = [m_dict.get(f"step_{s}", {}).get(m_key, np.nan) for s in steps]
            cfg = styles.get(
                grp_name,
                {"label": grp_name, "color": "black", "linestyle": "-", "marker": "x", "linewidth": 1.5},
            )
            ax.plot(
                steps,
                vals,
                label=cfg["label"],
                color=cfg["color"],
                linestyle=cfg["linestyle"],
                marker=cfg.get("marker", "o"),
                linewidth=cfg.get("linewidth", 2.0),
                markersize=8,
                alpha=0.9,
            )

        ax.set_title(f"({chr(97 + idx)}) {m_title}", fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Autoregressive Horizon (Step)", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_xticks(steps)
        ax.grid(True, linestyle="--", alpha=0.5)

        if is_log:
            ax.set_yscale("log")

        ax.legend(fontsize=10, loc="best", framealpha=0.9)

    plt.suptitle(
        "The Well Shear Flow V1: Physical Loss Ablation Study\nImpact of Divergence and Vorticity Regularization on Rollout Stability",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Successfully exported physics ablation curves to: {save_path}")


if __name__ == "__main__":
    plot_physics_ablation_curves()
