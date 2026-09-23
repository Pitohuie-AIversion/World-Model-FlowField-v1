"""Generate publication-ready LaTeX tables for the manuscript.

Outputs:
- outputs/tables/table_1_representation.tex
- outputs/tables/table_2_multi_seed_ablation.tex
- outputs/tables/table_3_within_seed_paired.tex
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = "outputs/tables"
SUMMARY_JSON = "outputs/metrics/closure_r4_physics_ablation_tri_seed_summary.json"
REP_JSON = "outputs/metrics/representation_metrics.json"


def generate_representation_table(rep_path: str = REP_JSON, out_dir: str = OUTPUT_DIR) -> str:
    with open(rep_path, "r") as f:
        data = json.load(f)

    test = data["test_metrics"]
    val = data["valid_metrics"]

    tex = r"""\begin{table}[t]
\centering
\small
\caption{\textbf{Pretrained Spatial Autoencoder ($4 \times$ Downsampled Latent Space) Representation Performance.} Evaluated on The Well Shear Flow grouped split ($N_{\mathrm{val}}=45, N_{\mathrm{test}}=45$ trajectories).}
\label{tab:representation_performance}
\begin{tabular}{lcccc}
\toprule
\textbf{Flow Variable} & \textbf{Validation RMSE} & \textbf{Validation VRMSE} & \textbf{Test RMSE} & \textbf{Test VRMSE} \\
\midrule
Streamwise Velocity ($u$) & """ + f"{val['rmse_u']:.4f}" + r""" & """ + f"{val['vrmse_u']:.4f}" + r""" & """ + f"{test['rmse_u']:.4f}" + r""" & """ + f"{test['vrmse_u']:.4f}" + r""" \\
Cross-Stream Velocity ($v$) & """ + f"{val['rmse_v']:.4f}" + r""" & """ + f"{val['vrmse_v']:.4f}" + r""" & """ + f"{test['rmse_v']:.4f}" + r""" & """ + f"{test['vrmse_v']:.4f}" + r""" \\
Gauge Pressure ($p$) & """ + f"{val['rmse_p']:.4f}" + r""" & """ + f"{val['vrmse_p']:.4f}" + r""" & """ + f"{test['rmse_p']:.4f}" + r""" & """ + f"{test['vrmse_p']:.4f}" + r""" \\
Passive Tracer Concentration ($s$) & """ + f"{val['rmse_s']:.4f}" + r""" & """ + f"{val['vrmse_s']:.4f}" + r""" & """ + f"{test['rmse_s']:.4f}" + r""" & """ + f"{test['vrmse_s']:.4f}" + r""" \\
\midrule
\textbf{Field Mean} & \textbf{""" + f"{val['rmse_mean']:.4f}" + r"""} & \textbf{""" + f"{val['vrmse_mean']:.4f}" + r"""} & \textbf{""" + f"{test['rmse_mean']:.4f}" + r"""} & \textbf{""" + f"{test['vrmse_mean']:.4f}" + r"""} \\
Vorticity RMSE ($\|\omega - \omega^*\|$) & \multicolumn{2}{c}{""" + f"{val['vorticity_rmse']:.4f}" + r"""} & \multicolumn{2}{c}{""" + f"{test['vorticity_rmse']:.4f}" + r"""} \\
Zero-Mean Gauge Error ($|\bar{p}|$) & \multicolumn{2}{c}{""" + f"{val['pressure_mean_gauge_abs']:.2e}" + r"""} & \multicolumn{2}{c}{""" + f"{test['pressure_mean_gauge_abs']:.2e}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""
    out_path = os.path.join(out_dir, "table_1_representation.tex")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"Saved {out_path}")
    return tex


def generate_multi_seed_ablation_table(summary_path: str = SUMMARY_JSON, out_dir: str = OUTPUT_DIR) -> str:
    with open(summary_path, "r") as f:
        summary = json.load(f)

    gstats = summary["group_statistics"]
    horizons = [1, 5, 10, 20, 30]

    key_metrics = [
        ("vrmse_mean", "Field Mean VRMSE"),
        ("div_rmse", r"Divergence RMSE ($\|\nabla \cdot \mathbf{u}\|$)"),
        ("vort_rmse", r"Vorticity RMSE ($\|\omega - \omega^*\|$)"),
        ("energy_spectrum_mae", "Energy Spectrum MAE"),
        ("enstrophy_rel_err", r"Enstrophy Rel. Err. ($\frac{|\Omega - \Omega^*|}{\Omega^*}$)"),
        ("tracer_out_of_bounds_rate", "Tracer Particle Escape Rate"),
    ]

    group_names = [
        ("E0_single_step", r"E0: Single-Step Pure Field ($H=1$)$^\dagger$"),
        ("E1_rollout_field", r"E1: Rollout-Aware Field ($H=2$)"),
        ("E2_plus_L_div", r"E2: $+ L_{\mathrm{div}}$ (Divergence-Free)"),
        ("E3_plus_L_vort", r"E3: $+ L_\omega$ (Vorticity-Aware)"),
        ("E4_full_physics", r"E4: $+ L_{\mathrm{div}} + L_\omega$ (Full Physics)"),
    ]

    tex = r"""\begin{table*}[t]
\centering
\small
\caption{\textbf{Multi-Seed Autoregressive Rollout Benchmark under Closure-R4 Protocol.} Results report $\text{Mean} \pm \text{Sample Std}$ across $N=3$ independent training seeds (Seeds 42, 43, 44) on test trajectories ($N_{\mathrm{test}}=45$). $^\dagger$E0 evaluated on Seed 42.}
\label{tab:multi_seed_ablation}
\resizebox{\textwidth}{!}{
\begin{tabular}{llccccc}
\toprule
\textbf{Metric} & \textbf{Model Group} & \textbf{Step 1 ($t=1$)} & \textbf{Step 5 ($t=5$)} & \textbf{Step 10 ($t=10$)} & \textbf{Step 20 ($t=20$)} & \textbf{Step 30 ($t=30$)} \\
\midrule
"""

    for mk, mlabel in key_metrics:
        tex += r"\multirow{5}{*}{\shortstack[l]{" + mlabel + r"}}" + "\n"
        for grp_key, grp_label in group_names:
            row_vals = []
            for h in horizons:
                entry = gstats[grp_key][f"step_{h}"][mk]
                m = entry["mean"]
                s = entry["std"]
                if s is not None:
                    row_vals.append(f"{m:.4f} $\\pm$ {s:.4f}")
                else:
                    row_vals.append(f"{m:.4f}")
            tex += f" & {grp_label} & " + " & ".join(row_vals) + r" \\" + "\n"
        tex += r"\midrule" + "\n"

    # Remove trailing midrule and add bottomrule
    tex = tex.rstrip("\n")
    if tex.endswith(r"\midrule"):
        tex = tex[:-len(r"\midrule")]
    tex += r"""\bottomrule
\end{tabular}
}
\end{table*}
"""
    out_path = os.path.join(out_dir, "table_2_multi_seed_ablation.tex")
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"Saved {out_path}")
    return tex


def generate_paired_contrasts_table(summary_path: str = SUMMARY_JSON, out_dir: str = OUTPUT_DIR) -> str:
    with open(summary_path, "r") as f:
        summary = json.load(f)

    peffects = summary["paired_effects"]
    horizons = [1, 5, 10, 20, 30]

    key_metrics = [
        ("vrmse_mean", "Field Mean VRMSE"),
        ("div_rmse", r"Divergence RMSE ($\|\nabla \cdot \mathbf{u}\|$)"),
        ("vort_rmse", r"Vorticity RMSE ($\|\omega - \omega^*\|$)"),
        ("energy_spectrum_mae", "Energy Spectrum MAE"),
        ("enstrophy_rel_err", r"Enstrophy Rel. Err. ($\frac{|\Omega - \Omega^*|}{\Omega^*}$)"),
        ("tracer_out_of_bounds_rate", "Tracer Particle Escape Rate"),
    ]

    compare_groups = [
        ("E2_plus_L_div", r"E2 ($+ L_{\mathrm{div}}$) vs E1"),
        ("E3_plus_L_vort", r"E3 ($+ L_\omega$) vs E1"),
        ("E4_full_physics", r"E4 ($+ L_{\mathrm{div}} + L_\omega$) vs E1"),
    ]

    tex = r"""\begin{table*}[t]
\centering
\small
\caption{\textbf{Within-Seed Paired Difference Analysis Relative to Baseline E1: Rollout-Aware Field.} Each pair contrast is evaluated per seed ($\Delta_i = v_{E_x, \mathrm{seed}_i} - v_{E_1, \mathrm{seed}_i}$). Cells report $\text{Mean Delta} \pm \text{Std}$ and $[\text{Wins}/N_{\mathrm{seeds}}]$, where a win indicates within-seed improvement.}
\label{tab:within_seed_paired_contrasts}
\resizebox{\textwidth}{!}{
\begin{tabular}{llccccc}
\toprule
\textbf{Contrast Pair} & \textbf{Physical Metric} & \textbf{Step 1 ($t=1$)} & \textbf{Step 5 ($t=5$)} & \textbf{Step 10 ($t=10$)} & \textbf{Step 20 ($t=20$)} & \textbf{Step 30 ($t=30$)} \\
\midrule
"""

    for grp_key, grp_label in compare_groups:
        tex += r"\multirow{6}{*}{\shortstack[l]{" + grp_label + r"}}" + "\n"
        for mk, mlabel in key_metrics:
            row_vals = []
            for h in horizons:
                entry = peffects[grp_key][f"step_{h}"][mk]
                md = entry["mean_delta"]
                sd = entry["std_delta"]
                w = entry["win_count"]
                n = entry["n"]
                sd_str = f" $\\pm$ {sd:.4f}" if sd is not None else ""
                sign = "+" if md > 0 else ""
                # Bold 100% wins
                val_str = f"{sign}{md:.4f}{sd_str} [{w}/{n}]"
                if w == n:
                    val_str = r"\textbf{" + val_str + r"}"
                row_vals.append(val_str)
            tex += f" & {mlabel} & " + " & ".join(row_vals) + r" \\" + "\n"
        tex += r"\midrule" + "\n"

    tex = tex.rstrip("\n")
    if tex.endswith(r"\midrule"):
        tex = tex[:-len(r"\midrule")]
    tex += r"""\bottomrule
\end{tabular}
}
\end{table*}
"""
    out_path = os.path.join(out_dir, "table_3_within_seed_paired.tex")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"Saved {out_path}")
    return tex


def generate_architecture_ablation_table(
    bench_path: str = "outputs/metrics/rollout_benchmark.json",
    summary_path: str = SUMMARY_JSON,
    out_dir: str = OUTPUT_DIR,
) -> str:
    with open(bench_path, "r") as f:
        bench = json.load(f)
    with open(summary_path, "r") as f:
        summary = json.load(f)

    gstats = summary["group_statistics"]
    horizons = [1, 5, 10, 20, 30]

    key_metrics = [
        ("vrmse_mean", "Field Mean VRMSE"),
        ("div_rmse", r"Divergence RMSE ($\|\nabla \cdot \mathbf{u}\|$)"),
        ("vort_rmse", r"Vorticity RMSE ($\|\omega - \omega^*\|$)"),
        ("energy_spectrum_mae", "Energy Spectrum MAE"),
    ]

    models = [
        ("Persistence Baseline", "---", bench["persistence"], False),
        ("Direct ST-Transformer (Physical Grid)", "7.73M", bench["direct_transformer"], False),
        ("FNO-2D (Fourier Neural Operator)", "16.80M", bench["fno"], False),
        ("Latent World Model (E0: Single-Step, $H=1$)$^\\dagger$", "9.75M", gstats["E0_single_step"], True),
        ("Latent World Model (E1: Rollout Field, $H=2$)", "9.75M", gstats["E1_rollout_field"], True),
        ("Latent World Model (E4: Full Physics)", "9.75M", gstats["E4_full_physics"], True),
    ]

    tex = r"""\begin{table*}[t]
\centering
\small
\caption{\textbf{Architectural and Operator Baseline Comparison Across Autoregressive Rollout Horizons.} Compares direct physical-space forecasting, neural operator baselines, and our latent world model framework across horizons $t \in \{1, 5, 10, 20, 30\}$. Direct ST-Transformer operates on raw $128 \times 128$ physical fields and suffers from catastrophic derivative explosion ($\|\nabla \cdot \mathbf{u}\| = 89.73$ at $t=30$). FNO-2D maintains stability via Fourier mode truncation but incurs high initial field error ($\mathrm{VRMSE} = 0.6450$ at $t=1$). Our Latent World Model with physics regularization (E4) achieves high initial fidelity ($\mathrm{VRMSE} = 0.0532 \pm 0.0076$) while stabilizing vorticity and enstrophy over 30 autoregressive steps. $^\dagger$E0 evaluated on Seed 42.}
\label{tab:architecture_and_baselines}
\resizebox{\textwidth}{!}{
\begin{tabular}{llcccccc}
\toprule
\textbf{Model / Architecture} & \textbf{Param Count} & \textbf{Physical Metric} & \textbf{Step 1 ($t=1$)} & \textbf{Step 5 ($t=5$)} & \textbf{Step 10 ($t=10$)} & \textbf{Step 20 ($t=20$)} & \textbf{Step 30 ($t=30$)} \\
\midrule
"""

    for mlabel, pcount, mdata, is_summary in models:
        tex += r"\multirow{4}{*}{\shortstack[l]{" + mlabel + r"}} & \multirow{4}{*}{" + pcount + r"}" + "\n"
        for mk, metric_name in key_metrics:
            row_vals = []
            for h in horizons:
                step_key = f"step_{h}"
                entry = mdata[step_key]
                if is_summary:
                    m = entry[mk]["mean"]
                    s = entry[mk]["std"]
                    if s is not None and s > 0:
                        row_vals.append(f"{m:.4f} $\\pm$ {s:.4f}")
                    else:
                        row_vals.append(f"{m:.4f}")
                else:
                    v = entry[mk]
                    row_vals.append(f"{v:.4f}")
            tex += f" & {metric_name} & " + " & ".join(row_vals) + r" \\" + "\n"
        tex += r"\midrule" + "\n"

    tex = tex.rstrip("\n")
    if tex.endswith(r"\midrule"):
        tex = tex[:-len(r"\midrule")]
    tex += r"""\bottomrule
\end{tabular}
}
\end{table*}
"""
    out_path = os.path.join(out_dir, "table_4_architecture_ablation.tex")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"Saved {out_path}")
    return tex


def main():
    generate_representation_table()
    generate_multi_seed_ablation_table()
    generate_paired_contrasts_table()
    generate_architecture_ablation_table()
    print("All LaTeX tables successfully generated.")


if __name__ == "__main__":
    main()
