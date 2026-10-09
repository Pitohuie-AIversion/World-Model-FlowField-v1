#!/usr/bin/env python3
"""Generate publication-grade diagrams and evaluation plots for the first-author synthesis report.

Generated Materials:
1. outputs/figures/paper_synthesis/fig_data_generation_and_contract.png (Diagram 1)
2. outputs/figures/paper_synthesis/fig_model_architecture_detailed.png (Diagram 2)
3. outputs/figures/paper_synthesis/fig_rollout_and_training_mechanisms.png (Diagram 3)
4. outputs/figures/paper_synthesis/fig_single_step_prediction_eval.png (Figure 4, rendered from real npz arrays)
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = "outputs/figures/paper_synthesis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Standard publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "figure.titlesize": 11.0,
})


def draw_data_generation_and_contract():
    """Diagram 1: Data Generation Pipeline & Historical vs Future Contracts."""
    fig, ax = plt.subplots(figsize=(14, 8.5), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7, 8.6, "Data Generation & Sample Construction Contracts (Numerical Simulation vs World Model)",
            fontsize=13.5, weight='bold', ha='center', va='center', color="#1a252f")

    # Upper: DNS Solver Generation
    rect_upper = patches.FancyBboxPatch((0.5, 4.8), 13, 3.4, boxstyle="round,pad=0.2",
                                        facecolor="#f0f4f8", edgecolor="#2b5c8f", linewidth=1.8)
    ax.add_patch(rect_upper)
    ax.text(0.8, 7.9, "[Stage 1] Offline DNS Trajectory Generation (Navier-Stokes + Passive Scalar Transport)",
            fontsize=11, weight='bold', color="#2b5c8f")

    boxes_upper = [
        ("1. Initial Conditions\n(IC Specification)", "Shear layer u(y) = tanh(y/delta)\nTransverse perturbation v'(x,y)\nPassive tracer s(x,y)\nLayer thickness delta, disturbance amp", 1.0, 5.2, 2.5, 2.3, "#e1edf7"),
        ("2. Physical Parameters\n& Boundaries (BCs)", "Design: Re in [10^3, 10^5]\nEvaluated subset: Re = 10^4\nBi-periodic BCs on Omega\nDomain [0, 1.0] x [0, 2.0]", 4.0, 5.2, 2.7, 2.3, "#e1edf7"),
        ("3. High-Precision DNS\n(Pseudo-Spectral)", "Incompressible N-S solver\nAdaptive CFL micro-step dt_cfl\nPseudo-spectral spatial deriv\nPoisson solver: nabla^2 p = -div(u.grad u)", 7.2, 5.2, 2.8, 2.3, "#e1edf7"),
        ("4. Trajectory Archive\n(Saved Grid Fields)", "Uniform stride dt = 0.1\nTotal steps T = 100 ~ 200\nNx x Ny = 128 x 256\nLx=1.0, Ly=2.0, dx=dy=1/128\nPhysical states: q = [u, v, p, s]", 10.5, 5.2, 2.6, 2.3, "#d5e8d4")
    ]
    for title, desc, x, y, w, h, col in boxes_upper:
        p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                  facecolor=col, edgecolor="#4a7c59" if col=="#d5e8d4" else "#2b5c8f", linewidth=1.2)
        ax.add_patch(p)
        ax.text(x + w/2, y + h - 0.35, title, fontsize=9.2, weight='bold', ha='center', va='top', color="#111")
        ax.text(x + w/2, y + 0.25, desc, fontsize=7.8, ha='center', va='bottom', color="#333", linespacing=1.25)

    for i in range(len(boxes_upper)-1):
        x1 = boxes_upper[i][2] + boxes_upper[i][4]
        x2 = boxes_upper[i+1][2]
        y_c = 6.35
        ax.annotate("", xy=(x2, y_c), xytext=(x1, y_c),
                    arrowprops=dict(arrowstyle="->", color="#2b5c8f", lw=2))

    # Lower: Sample Construction
    rect_lower = patches.FancyBboxPatch((0.5, 0.4), 13, 3.8, boxstyle="round,pad=0.2",
                                        facecolor="#fcf8f2", edgecolor="#c07d32", linewidth=1.8)
    ax.add_patch(rect_lower)
    ax.text(0.8, 3.9, "[Stage 2] World Model Dataset Construction & Causal Isolation Firewall",
            fontsize=11, weight='bold', color="#c07d32")

    ax.annotate("", xy=(2.2, 3.5), xytext=(11.8, 5.2),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2, linestyle="--",
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(7.5, 4.4, "Grouped Trajectory Split (Train: 33 trajs / Val: 6 trajs / Test: 5 trajs across disjoint clusters)",
            fontsize=9.0, weight='bold', color="#555", ha='center')

    boxes_lower = [
        ("1. Sliding Window\n(Internal Slices)", "Arbitrary valid starting time t0\nHistory length L = 4 frames\nForecast horizon H in [1, 16]\nTrain: stride=1; Val: stride=2", 1.0, 0.7, 2.5, 2.7, "#faebd7"),
        ("2. Model Dynamic Inputs\n(q_{t-3:t}, Re, Sc)", "Input: 4 historical frames\nq_{t-3}, q_{t-2}, q_{t-1}, q_t\nPhysical condition c = [log10 Re, log10 Sc]\nSimulation IC at t=0 is NOT direct input", 4.0, 0.7, 2.7, 2.7, "#d4edda"),
        ("3. Latent Autoregressive\nRollout (Pure Latent)", "FIFO latent state queue\nNext latent z_{t+1} written back\nZero GT feedback during rollout\nDecoded only at forecast horizon", 7.2, 0.7, 2.8, 2.7, "#d1ecf1"),
        ("4. Loss Supervision &\nEvaluation Metric", "Future GT: q*_{t+1:t+H}\nUsed ONLY for loss or test metric\nStrictly quarantined from model history\nOne-way causal firewall", 10.5, 0.7, 2.6, 2.7, "#f8d7da")
    ]
    for title, desc, x, y, w, h, col in boxes_lower:
        border_col = "#28a745" if "d4edda" in col else ("#17a2b8" if "d1ecf1" in col else ("#dc3545" if "f8d7da" in col else "#c07d32"))
        p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                  facecolor=col, edgecolor=border_col, linewidth=1.2)
        ax.add_patch(p)
        ax.text(x + w/2, y + h - 0.35, title, fontsize=9.2, weight='bold', ha='center', va='top', color="#111")
        ax.text(x + w/2, y + 0.25, desc, fontsize=7.8, ha='center', va='bottom', color="#333", linespacing=1.25)

    ax.annotate("", xy=(4.0, 2.05), xytext=(3.5, 2.05), arrowprops=dict(arrowstyle="->", color="#c07d32", lw=2))
    ax.annotate("", xy=(7.2, 2.05), xytext=(6.7, 2.05), arrowprops=dict(arrowstyle="->", color="#28a745", lw=2))
    ax.annotate("", xy=(10.5, 2.05), xytext=(10.0, 2.05), arrowprops=dict(arrowstyle="->", color="#17a2b8", lw=2))
    ax.text(10.25, 2.25, "Forecast", fontsize=8.0, color="#17a2b8", ha='center')

    ax.plot([10.25, 10.25], [0.7, 3.4], color="#dc3545", linestyle=":", lw=2)
    ax.text(10.25, 0.55, "[Causal Firewall] Zero Reverse Flow", fontsize=7.8, weight='bold', color="#dc3545", ha='center')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_data_generation_and_contract.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def draw_model_architecture_detailed():
    """Diagram 2: Shared Backbone Architecture, Optional Experimental Branches & Parameter Status."""
    fig, ax = plt.subplots(figsize=(15, 9.2), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9.6)
    ax.axis("off")

    ax.text(7.5, 9.3, "World Model Framework: Shared Representation Backbone & Modular Research Branches",
            fontsize=13.5, weight='bold', ha='center', va='center', color="#1a252f")

    # ------------------ TOP SECTION: SHARED BACKBONE TOPOLOGY ------------------
    rect_top = patches.FancyBboxPatch((0.4, 4.8), 14.2, 4.2, boxstyle="round,pad=0.15",
                                      facecolor="#f8fafc", edgecolor="#1e3a8a", linewidth=1.6)
    ax.add_patch(rect_top)
    ax.text(0.7, 8.7, "A. Shared Spatiotemporal Backbone Topology (Modular Connectivity)",
            fontsize=11.0, weight='bold', color="#1e3a8a")

    # 1. Historical Inputs
    ax.add_patch(patches.FancyBboxPatch((0.7, 5.2), 2.5, 3.2, boxstyle="round,pad=0.1",
                                       facecolor="#eef2f7", edgecolor="#2b5c8f", linewidth=1.4))
    ax.text(1.95, 8.0, "Physical History Input\nq_{t-3:t}", fontsize=9.5, weight='bold', ha='center', color="#2b5c8f")
    ax.text(1.95, 7.1, "(B, 4, 4, 128, 256)\n[B, L, C, Nx, Ny]", fontsize=8.2, ha='center', color="#c0392b", weight='bold')
    ax.text(1.95, 5.8, "Channels: [u, v, p, s]\nL = 4 frames\nDomain: [Lx=1.0, Ly=2.0]", fontsize=7.6, ha='center', color="#444")

    # Arrow 1 -> Encoder
    ax.annotate("", xy=(3.6, 6.8), xytext=(3.2, 6.8), arrowprops=dict(arrowstyle="->", color="#2b5c8f", lw=1.8))

    # 2. Encoder2D
    ax.add_patch(patches.FancyBboxPatch((3.6, 5.2), 2.6, 3.2, boxstyle="round,pad=0.1",
                                       facecolor="#ffffff", edgecolor="#475569", linewidth=1.4))
    ax.text(4.9, 8.0, "Spatial Encoder\nEncoder2D", fontsize=9.5, weight='bold', ha='center', color="#475569")
    ax.text(4.9, 7.1, "3 Residual Conv stages\nstride = 2 circular padding\nSpatial downsample: 8x\nChannels: 4 -> 64", fontsize=7.8, ha='center', color="#333")
    ax.text(4.9, 5.7, "Output: Z_{t-3:t}\n(B, 4, 64, 16, 32)\n[B, L, C, Nx_lat, Ny_lat]", fontsize=7.6, ha='center', color="#c0392b", weight='bold')

    # Arrow 2 -> LatentSTTransformer
    ax.annotate("", xy=(6.6, 6.8), xytext=(6.2, 6.8), arrowprops=dict(arrowstyle="->", color="#2b5c8f", lw=1.8))

    # 3. LatentSTTransformer
    ax.add_patch(patches.FancyBboxPatch((6.6, 5.0), 4.3, 3.5, boxstyle="round,pad=0.12",
                                       facecolor="#f0fdf4", edgecolor="#166534", linewidth=1.6))
    ax.text(8.75, 8.2, "Spatiotemporal Dynamics Backbone\nLatentSTTransformer (Factorized ST)", fontsize=9.8, weight='bold', ha='center', color="#166534")

    ax.add_patch(patches.Rectangle((6.8, 6.5), 3.9, 1.1, facecolor="#ffffff", edgecolor="#166534", lw=0.9))
    ax.text(8.75, 7.05, "Factorized Spatiotemporal Attention:\nSpatial Self-Attention (over 16x32 grid)\n+ Temporal Self-Attention (across L=4 history)",
            fontsize=7.6, ha='center', color="#222")

    ax.add_patch(patches.Rectangle((6.8, 5.2), 3.9, 1.1, facecolor="#ffffff", edgecolor="#ea580c", lw=0.9))
    ax.text(8.75, 5.75, "Adaptive LayerNorm (AdaLN):\nCondition c = [log10 Re, log10 Sc] -> MLP\nModulation: (1 + gamma)*LN(x) + beta (Zero-init)",
            fontsize=7.6, ha='center', color="#c0392b")

    # Arrow 3 -> Latent State
    ax.annotate("", xy=(11.3, 6.8), xytext=(10.9, 6.8), arrowprops=dict(arrowstyle="->", color="#2b5c8f", lw=1.8))
    ax.text(11.1, 7.05, "z_{t+1}", fontsize=8.2, ha='center', color="#2b5c8f", weight='bold')

    # 4. Decoder2D
    ax.add_patch(patches.FancyBboxPatch((11.3, 5.2), 3.1, 3.2, boxstyle="round,pad=0.1",
                                       facecolor="#ffffff", edgecolor="#475569", linewidth=1.4))
    ax.text(12.85, 8.0, "Spatial Decoder\nDecoder2D", fontsize=9.5, weight='bold', ha='center', color="#475569")
    ax.text(12.85, 6.9, "UpBlock2D (3 stages):\nBilinear Interpolation (2x)\n+ Circular Conv + ResBlock\n8x cumulative upsampling\n(project_pressure=False)", fontsize=7.6, ha='center', color="#333")
    ax.text(12.85, 5.6, "Physical Field q̂_{t+1}\n(B, 1, 4, 128, 256)\n[B, 1, C, Nx, Ny]", fontsize=7.6, ha='center', color="#c0392b", weight='bold')

    # Post-denorm gauge note
    ax.annotate("", xy=(12.85, 5.0), xytext=(12.85, 5.3), arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2))
    ax.text(12.85, 4.88, "[Eval Protocol: Zero-mean gauge applied post-denorm on p]", fontsize=6.8, ha='center', color="#64748b", style='italic')

    # ------------------ BOTTOM SECTION: BRANCHES & TRAINING STATUS ------------------
    rect_bot = patches.FancyBboxPatch((0.4, 0.4), 14.2, 4.2, boxstyle="round,pad=0.15",
                                      facecolor="#fefce8", edgecolor="#ca8a04", linewidth=1.6)
    ax.add_patch(rect_bot)
    ax.text(0.7, 4.25, "B. Modular Research Branches & Training Parameter Isolation Matrix",
            fontsize=11.0, weight='bold', color="#854d0e")

    branches = [
        ("Branch 1: Closure-R4 Physics Regularization",
         "• Objective: L_base + lambda_div*L_div + lambda_vort*L_vort\n"
         "• Trainable: LatentSTTransformer | Frozen: Encoder, Decoder\n"
         "• Spectral divergence & vorticity calculated via pseudo-spectral FFT\n"
         "• Outcome: Stabilizes enstrophy/vorticity structures; trade-offs on field error",
         0.6, 2.3, 6.8, 1.8, "#f0fdf4", "#15803d"),

        ("Branch 2: PDE-Controlled Equation Residual Fine-tuning",
         "• Objective: L_base + lambda_mom*||R_mom||^2 + lambda_tr*||R_tracer||^2\n"
         "• Trainable: LatentSTTransformer | Frozen: Encoder, Decoder\n"
         "• Permitted with multi-step rollout (H >= 1); requires pushforward_steps = 0\n"
         "• Outcome: Reduces residual vs continued training; does NOT surpass frozen D0",
         7.6, 2.3, 6.8, 1.8, "#eff6ff", "#1d4ed8"),

        ("Branch 3: ProbLatent G1 Gaussian Uncertainty Quantification",
         "• Latent distribution: z_{t+1} | z_hist, c ~ N(mu_t, diag(sigma_t^2))\n"
         "• Physical decoding: q̂_{t+1}^{(k)} = D_psi(z_{t+1}^{(k)}) (Nonlinear decoder)\n"
         "• Trainable: VarianceHead2D ONLY | Frozen: Encoder, Decoder, Transformer\n"
         "• Outcome: NLL improves by -0.1974 nats; 50% interval severely undercovers (26.5%)",
         0.6, 0.5, 6.8, 1.7, "#fdf2f8", "#be185d"),

        ("Branch 4: FM-R2 Residual Latent Flow Matching",
         "• Continuous ODE in latent space: dZ/dtau = v_theta(Z, tau; Q, c), tau in [0, 1]\n"
         "• Trainable: Latent Vector Field Net v_theta ONLY | Frozen: Backbone, AE\n"
         "• Rollout-aware self-conditioning eliminates autoregressive exposure bias\n"
         "• Outcome: Secondary metrics 4/4 improved; Primary 2 shows Seed 45 reversal (+11.99%)",
         7.6, 0.5, 6.8, 1.7, "#faf5ff", "#7e22ce"),
    ]

    for title, desc, bx, by, bw, bh, bg_c, border_c in branches:
        p = patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.08",
                                  facecolor=bg_c, edgecolor=border_c, linewidth=1.2)
        ax.add_patch(p)
        ax.text(bx + 0.15, by + bh - 0.22, title, fontsize=8.6, weight='bold', color=border_c, va='top')
        ax.text(bx + 0.15, by + 0.15, desc, fontsize=7.4, color="#1e293b", va='bottom', linespacing=1.25)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_model_architecture_detailed.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def draw_rollout_and_training_mechanisms():
    """Diagram 3: Rollout Mechanisms (FIFO queue, Teacher Forcing vs Free Rollout, Ensemble Axis)."""
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7.5, 8.6, "Autoregressive Rollout Mechanisms & Training Regimes Contrast",
            fontsize=13.5, weight='bold', ha='center', va='center', color="#1a252f")

    # Left: FIFO Latent Queue
    ax.add_patch(patches.FancyBboxPatch((0.5, 4.4), 6.5, 3.8, boxstyle="round,pad=0.1",
                                       facecolor="#edf2f7", edgecolor="#2b5c8f", linewidth=1.5))
    ax.text(3.75, 7.8, "Pure Latent Autoregressive FIFO Queue Rollout", fontsize=10.5, weight='bold', ha='center', color="#2b5c8f")

    steps_text = (
        "Step 1:  [z_{t-3},  z_{t-2},  z_{t-1},  z_t]       ──Forecaster──>  ẑ_{t+1}\n\n"
        "Step 2:  [z_{t-2},  z_{t-1},  z_t,      ẑ_{t+1}]    ──Forecaster──>  ẑ_{t+2}\n\n"
        "Step 3:  [z_{t-1},  z_t,      ẑ_{t+1},  ẑ_{t+2}]    ──Forecaster──>  ẑ_{t+3}\n\n"
        "Key Property: Encoded once at origin t0; zero intermediate decoding/re-encoding;\n"
        "All rollouts proceed autoregressively in compact 64-channel latent space."
    )
    ax.text(3.75, 5.8, steps_text, fontsize=8.6, ha='center', va='center', color="#222", linespacing=1.2, family='monospace')

    # Right: Training Condition Contrast
    ax.add_patch(patches.FancyBboxPatch((7.8, 4.4), 6.7, 3.8, boxstyle="round,pad=0.1",
                                       facecolor="#fef9e7", edgecolor="#f39c12", linewidth=1.5))
    ax.text(11.15, 7.8, "Historical Conditioning Regimes (Teacher-Forcing vs Rollout-Aware)", fontsize=10.5, weight='bold', ha='center', color="#d35400")

    regimes_text = (
        "Regime A: Ground-Truth History Conditioning (Teacher-Forcing / C1)\n"
        "  • At step 2, model input is forced with ground truth: [z_{t-2}, z_{t-1}, z_t, z*_{t+1}]\n"
        "  • Limitation: Never experiences its own errors during training;\n"
        "    suffers catastrophic Exposure Bias during autonomous test rollouts.\n\n"
        "Regime B: Rollout-Aware Self-Conditioning (Self-Generated / C2 & R2-A)\n"
        "  • At step 2, model input uses self-generated output: [z_{t-2}, z_{t-1}, z_t, ẑ_{t+1}]\n"
        "  • Benefit: Teaches network to tolerate and correct internal sub-pixel errors,\n"
        "    significantly suppressing long-horizon error dispersion."
    )
    ax.text(11.15, 5.8, regimes_text, fontsize=8.2, ha='center', va='center', color="#333", linespacing=1.25)

    # Lower: Probabilistic & Continuous Ensemble Space
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.5), 14.0, 3.4, boxstyle="round,pad=0.1",
                                       facecolor="#f4fbf7", edgecolor="#27ae60", linewidth=1.5))
    ax.text(7.5, 3.5, "Probabilistic Uncertainty Quantification & Continuous Generative Trajectories", fontsize=11.0, weight='bold', ha='center', color="#27ae60")

    uq_text = (
        "1. Heteroscedastic Gaussian Latent World Model (ProbLatent Phase 3):\n"
        "   • Diagonal Gaussian in latent space: z_{t+1} | z_hist, c ~ N(mu_t, diag(sigma_t^2)). Independent latent FIFO queues per ensemble member.\n"
        "   • Mean Parity: mu_t strictly matches deterministic baseline. Non-linear decoder D_psi maps latent samples to physical space.\n"
        "   • Calibration Reality: 50% interval severely undercovers (PICP = 26.51%); 90% interval overcovers (PICP = 94.94%).\n\n"
        "2. Continuous Residual Flow Matching (FM-R2):\n"
        "   • Explicit separation of physical time t and generative flow time tau in [0, 1].\n"
        "   • Solves continuous neural ODE: dZ/dtau = v_theta(Z, tau; Q, c) to generate distribution of future latent states."
    )
    ax.text(7.5, 1.9, uq_text, fontsize=8.4, ha='center', va='center', color="#222", linespacing=1.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_rollout_and_training_mechanisms.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def draw_single_step_prediction_eval():
    """Figure 4: Render Single-Step Prediction Evaluation directly from verified npz arrays."""
    from scripts.run_genuine_single_step_eval import verify_prediction_arrays

    npz_path = os.path.join(OUTPUT_DIR, "single_step_real_prediction_arrays.npz")
    prov_path = os.path.join(OUTPUT_DIR, "single_step_real_prediction_provenance.json")

    if not os.path.exists(npz_path) or not os.path.exists(prov_path):
        print("npz or provenance missing. Running genuine single-step evaluation...")
        from scripts.run_genuine_single_step_eval import run_genuine_single_step_evaluation
        run_genuine_single_step_evaluation()

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Cannot find {npz_path}")
    if not os.path.exists(prov_path):
        raise FileNotFoundError(f"Cannot find {prov_path}")

    # Mandatory upfront verification: Fail closed before reading or rendering
    is_valid = verify_prediction_arrays(npz_path, prov_path, atol=1e-5)
    if not is_valid:
        raise ValueError(f"Array verification failed for {npz_path} against {prov_path}")

    data = np.load(npz_path)
    with open(prov_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Strictly require identity fields - no fallback defaults permitted
    for required_id_key in ("traj_idx", "cluster_id", "source_file_relative", "sample_metrics"):
        if required_id_key not in meta:
            raise KeyError(f"Provenance missing mandatory identity key '{required_id_key}'")

    rel_source_path = meta["source_file_relative"]
    traj_idx = meta["traj_idx"]
    cluster_id = meta["cluster_id"]
    sample_metrics = meta["sample_metrics"]

    gt_arr = np.stack([data["gt_u"], data["gt_v"], data["gt_p"], data["gt_s"]], axis=0)
    pred_arr = np.stack([data["pred_u"], data["pred_v"], data["pred_p"], data["pred_s"]], axis=0)
    gt_vort = data["gt_vort"]
    pred_vort = data["pred_vort"]

    # Recompute error fields directly from ground-truth and prediction
    err_arr = np.abs(gt_arr - pred_arr)
    err_vort = np.abs(gt_vort - pred_vort)

    fig, axes = plt.subplots(5, 3, figsize=(12, 10), dpi=300)
    channel_data = [
        ("Streamwise Velocity u", gt_arr[0], pred_arr[0], err_arr[0], "RdBu_r"),
        ("Cross-stream Velocity v", gt_arr[1], pred_arr[1], err_arr[1], "RdBu_r"),
        ("Gauge Pressure p", gt_arr[2], pred_arr[2], err_arr[2], "viridis"),
        ("Passive Tracer s", gt_arr[3], pred_arr[3], err_arr[3], "inferno"),
        ("Vorticity w", gt_vort, pred_vort, err_vort, "bwr"),
    ]

    col_titles = ["Ground Truth (t+1)", "Model Prediction (t+1)", "Absolute Error |GT - Pred|"]

    for row_idx, (name, gt, pred, err, cmap) in enumerate(channel_data):
        vmin = min(np.min(gt), np.min(pred))
        vmax = max(np.max(gt), np.max(pred))
        if "RdBu" in cmap or "bwr" in cmap:
            bound = max(abs(vmin), abs(vmax))
            vmin, vmax = -bound, bound

        im0 = axes[row_idx, 0].imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
        axes[row_idx, 0].set_ylabel(name, fontsize=8.5, weight='bold')
        fig.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        im1 = axes[row_idx, 1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
        fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        im2 = axes[row_idx, 2].imshow(err, cmap="magma", aspect='auto', origin='lower')
        fig.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10.0, weight='bold', pad=8)

    title_str = (
        f"Single-Step Prediction Evaluation on Unseen Test Sample\n"
        f"[Source: {rel_source_path} (traj_idx={traj_idx}, cluster_id={cluster_id}) | "
        f"Sample Mean VRMSE = {sample_metrics['vrmse_mean']:.4f} | Vorticity RMSE = {sample_metrics['vorticity_rmse']:.4f}]"
    )
    plt.suptitle(title_str, fontsize=11.5, weight='bold', y=0.99)
    plt.tight_layout()
    out_fig = os.path.join(OUTPUT_DIR, "fig_single_step_prediction_eval.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Generated verified figure: {out_fig}")


if __name__ == "__main__":
    print("Generating official synthesis figures via v2 generator...")
    draw_data_generation_and_contract()
    draw_model_architecture_detailed()
    draw_rollout_and_training_mechanisms()
    draw_single_step_prediction_eval()
    print("All synthesis figures generated and verified!")
