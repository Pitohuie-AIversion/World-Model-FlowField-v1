"""Generate publication-quality comparative evaluation figures for PDE-controlled experiment.

Reads outputs/evaluations/pde_controlled_candidates_full_val.json and renders
a 3-panel figure comparing D0 baseline, P0 control, and PDE candidate models across
validation metrics, physical residuals, and Schmidt number subgroups.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def plot_pde_evaluation_figure():
    json_path = PROJECT_ROOT / "outputs" / "evaluations" / "pde_controlled_candidates_full_val.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Evaluation JSON not found at {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data["overall_comparison"]
    subgroups = data["subgroup_comparison"]

    # Setup styles
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)

    # Color palette
    c_d0 = "#2563eb"   # Blue
    c_p0 = "#94a3b8"   # Slate gray
    c_pde = "#059669"  # Emerald green

    # ----------------------------------------------------
    # Panel A: VRMSE & Field RMSE Comparison (1110 Windows)
    # ----------------------------------------------------
    ax = axes[0]
    categories = ["Standard\nVRMSE", "Total\nRMSE", "u RMSE\n(x10)", "p RMSE\n(x10)", "s RMSE\n(x10)"]
    d0_vals = [
        metrics["vrmse_standard"]["d0"],
        metrics["rmse_total"]["d0"],
        metrics["rmse_u"]["d0"] * 10,
        metrics["rmse_p"]["d0"] * 10,
        metrics["rmse_s"]["d0"] * 10,
    ]
    p0_vals = [
        metrics["vrmse_standard"]["p0"],
        metrics["rmse_total"]["p0"],
        metrics["rmse_u"]["p0"] * 10,
        metrics["rmse_p"]["p0"] * 10,
        metrics["rmse_s"]["p0"] * 10,
    ]
    pde_vals = [
        metrics["vrmse_standard"]["pde"],
        metrics["rmse_total"]["pde"],
        metrics["rmse_u"]["pde"] * 10,
        metrics["rmse_p"]["pde"] * 10,
        metrics["rmse_s"]["pde"] * 10,
    ]

    x = np.arange(len(categories))
    w = 0.26
    ax.bar(x - w, d0_vals, width=w, label="D0 (Frozen Baseline)", color=c_d0, edgecolor="black", linewidth=0.5)
    ax.bar(x, p0_vals, width=w, label="P0 (Standard Finetune)", color=c_p0, edgecolor="black", linewidth=0.5)
    ax.bar(x + w, pde_vals, width=w, label="PDE (Physics Supervised)", color=c_pde, edgecolor="black", linewidth=0.5)

    # Annotate delta for standard VRMSE
    v_d0 = metrics["vrmse_standard"]["d0"]
    v_pde = metrics["vrmse_standard"]["pde"]
    v_p0 = metrics["vrmse_standard"]["p0"]
    ax.annotate(
        f"+2.42% vs D0\n-0.09% vs P0",
        xy=(x[0] + w, v_pde),
        xytext=(x[0] + 0.3, v_pde * 1.08),
        fontsize=8,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef3c7", edgecolor="#f59e0b", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#b45309", lw=0.8),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Error Metric Value")
    ax.set_title("(a) Field Prediction Accuracy (1110 Windows)", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, max(d0_vals[0], p0_vals[0], pde_vals[0]) * 1.3)

    # ----------------------------------------------------
    # Panel B: Physical Residual Invariants (Divergence, Momentum, Tracer)
    # ----------------------------------------------------
    ax = axes[1]
    res_cats = ["Divergence\nRMS", "Momentum r_u\nRMSE", "Momentum r_v\nRMSE", "Tracer r_s\nRMSE"]
    d0_res = [
        metrics["div_rmse"]["d0"],
        metrics["res_u_rmse"]["d0"],
        metrics["res_v_rmse"]["d0"],
        metrics["res_s_rmse"]["d0"],
    ]
    p0_res = [
        metrics["div_rmse"]["p0"],
        metrics["res_u_rmse"]["p0"],
        metrics["res_v_rmse"]["p0"],
        metrics["res_s_rmse"]["p0"],
    ]
    pde_res = [
        metrics["div_rmse"]["pde"],
        metrics["res_u_rmse"]["pde"],
        metrics["res_v_rmse"]["pde"],
        metrics["res_s_rmse"]["pde"],
    ]

    xr = np.arange(len(res_cats))
    ax.bar(xr - w, d0_res, width=w, label="D0 (Baseline)", color=c_d0, edgecolor="black", linewidth=0.5)
    ax.bar(xr, p0_res, width=w, label="P0 (Control)", color=c_p0, edgecolor="black", linewidth=0.5)
    ax.bar(xr + w, pde_res, width=w, label="PDE (Step 50)", color=c_pde, edgecolor="black", linewidth=0.5)

    # Annotate PDE damping on tracer residual
    r_p0 = metrics["res_s_rmse"]["p0"]
    r_pde = metrics["res_s_rmse"]["pde"]
    ax.annotate(
        f"-2.18% vs P0\n(Damping)",
        xy=(xr[3] + w, r_pde),
        xytext=(xr[3] + 0.1, r_pde * 1.15),
        fontsize=8,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ecfdf5", edgecolor="#10b981", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#047857", lw=0.8),
    )

    ax.set_xticks(xr)
    ax.set_xticklabels(res_cats)
    ax.set_ylabel("Physical Residual RMS / RMSE")
    ax.set_title("(b) Conservation & Residual Damping", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, max(p0_res) * 1.35)

    # ----------------------------------------------------
    # Panel C: Schmidt Subgroup Regimes (Sc=0.1 vs Sc=1.0)
    # ----------------------------------------------------
    ax = axes[2]
    sc_cats = ["VRMSE\n(Sc=0.1)", "VRMSE\n(Sc=1.0)", "Tracer r_s\n(Sc=0.1)", "Tracer r_s\n(Sc=1.0)"]
    sc01_m = subgroups["Sc_0.1"]["metrics"]
    sc10_m = subgroups["Sc_1.0"]["metrics"]

    d0_sc = [sc01_m["vrmse_standard"]["d0"], sc10_m["vrmse_standard"]["d0"], sc01_m["res_s_rmse"]["d0"], sc10_m["res_s_rmse"]["d0"]]
    p0_sc = [sc01_m["vrmse_standard"]["p0"], sc10_m["vrmse_standard"]["p0"], sc01_m["res_s_rmse"]["p0"], sc10_m["res_s_rmse"]["p0"]]
    pde_sc = [sc01_m["vrmse_standard"]["pde"], sc10_m["vrmse_standard"]["pde"], sc01_m["res_s_rmse"]["pde"], sc10_m["res_s_rmse"]["pde"]]

    xs = np.arange(len(sc_cats))
    ax.bar(xs - w, d0_sc, width=w, label="D0 (Baseline)", color=c_d0, edgecolor="black", linewidth=0.5)
    ax.bar(xs, p0_sc, width=w, label="P0 (Control)", color=c_p0, edgecolor="black", linewidth=0.5)
    ax.bar(xs + w, pde_sc, width=w, label="PDE (Step 50)", color=c_pde, edgecolor="black", linewidth=0.5)

    # Highlight Sc=0.1 tracer residual improvement
    ax.annotate(
        f"-3.26% vs P0",
        xy=(xs[2] + w, pde_sc[2]),
        xytext=(xs[2] + 0.1, pde_sc[2] * 1.25),
        fontsize=8,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ecfdf5", edgecolor="#10b981", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#047857", lw=0.8),
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(sc_cats)
    ax.set_ylabel("Metric Value")
    ax.set_title("(c) Regime Sensitivity (Diffusion vs Advection)", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, max(p0_sc[1], pde_sc[1]) * 1.2)

    plt.tight_layout()

    out_dir = PROJECT_ROOT / "outputs" / "figures" / "pde_controlled"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "figure_pde_controlled_comparison.png"
    out_pdf = out_dir / "figure_pde_controlled_comparison.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[SUCCESS] Exported PDE figure to {out_png} and {out_pdf}")

    # Copy to artifact directories
    for adir_str in [
        "/root/.gemini/antigravity-ide/brain/25604b97-21bd-42b9-814b-524c55a6667a",
        "/root/.gemini/antigravity-ide/brain/df6d6fdd-c45c-4927-91ce-a466b0537f1e",
    ]:
        adir = Path(adir_str)
        if adir.exists():
            import shutil
            shutil.copy2(out_png, adir / out_png.name)
            shutil.copy2(out_pdf, adir / out_pdf.name)
            print(f"[COPIED] Also copied to artifact directory: {adir}")


if __name__ == "__main__":
    plot_pde_evaluation_figure()
