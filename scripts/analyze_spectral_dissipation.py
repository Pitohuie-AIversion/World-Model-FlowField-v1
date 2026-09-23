"""Directional Spectral Dissipation Analysis for Physics Ablation Study (Issue #4).

Computes shell-integrated radial spectrum E(k), streamwise spectrum E(k_x),
and cross-stream spectrum E(k_y), along with spectral ratios E_pred / E_target,
to rigorously distinguish between high-frequency spurious noise accumulation
versus numerical over-dissipation across autoregressive rollout horizons.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.pipeline import create_flow_dataloaders
from src.metrics.spectral import compute_radial_energy_spectrum
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.utils.physics_contract import (
    PHYSICS_PROTOCOL,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
    validate_ablation_checkpoint_semantics,
)
from src.utils.provenance import (
    compute_file_sha256,
    compute_normalizer_hash,
    compute_split_hash_from_file,
    get_git_commit,
    is_git_dirty,
    resolve_checkpoint_provenance,
    validate_evaluation_provenance,
)
from src.utils.reproducibility import seed_everything


def compute_directional_energy_spectra(
    u: torch.Tensor,
    v: torch.Tensor,
    domain_size: Tuple[float, float] = (1.0, 2.0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 1D streamwise E_x(k_x) and cross-stream E_y(k_y) kinetic energy spectra.

    Args:
        u: Horizontal velocity, shape (..., Nx, Ny).
        v: Vertical velocity, shape (..., Nx, Ny).
        domain_size: (Lx, Ly).

    Returns:
        kx_pos: 1D positive streamwise wavenumbers, shape (Nx // 2 + 1,).
        e_kx: 1D streamwise kinetic energy spectrum, shape (Nx // 2 + 1,).
        ky_pos: 1D cross-stream wavenumbers, shape (Ny // 2 + 1,).
        e_ky: 1D cross-stream kinetic energy spectrum, shape (Ny // 2 + 1,).
    """
    nx, ny = u.shape[-2], u.shape[-1]
    lx, ly = domain_size
    device = u.device

    u_hat = torch.fft.rfft2(u, dim=(-2, -1), norm="forward")
    v_hat = torch.fft.rfft2(v, dim=(-2, -1), norm="forward")

    # 2D energy density (Parseval-consistent)
    energy_2d = 0.5 * (torch.abs(u_hat) ** 2 + torch.abs(v_hat) ** 2)
    # Double count positive non-zero, non-Nyquist frequencies along rfft dim (-1)
    if ny > 2:
        energy_2d[..., :, 1:-1] *= 2.0

    # Mean over batch/history dimensions if present
    while energy_2d.ndim > 2:
        energy_2d = energy_2d.mean(dim=0)

    # 1D along streamwise x (dim -2): sum over ky (dim -1)
    e_kx_full = energy_2d.sum(dim=-1)  # shape (nx,)
    half_nx = nx // 2
    kx = torch.fft.fftfreq(nx, d=lx / nx, device=device) * 2.0 * torch.pi
    kx_pos = kx[:half_nx + 1]

    e_kx = torch.zeros(half_nx + 1, device=device)
    e_kx[0] = e_kx_full[0]
    e_kx[half_nx] = e_kx_full[half_nx]
    for i in range(1, half_nx):
        e_kx[i] = e_kx_full[i] + e_kx_full[-i]

    # 1D along cross-stream y (dim -1): sum over kx (dim -2)
    e_ky = energy_2d.sum(dim=-2)  # shape (ny // 2 + 1,)
    ky_pos = torch.fft.rfftfreq(ny, d=ly / ny, device=device) * 2.0 * torch.pi

    return kx_pos, e_kx, ky_pos, e_ky


def diagnose_dissipation_mode(
    k_bins: np.ndarray,
    ratio_e: np.ndarray,
    cutoff_ratio: float = 0.5,
) -> Dict[str, Any]:
    """Diagnose whether model suffers from high-frequency spurious noise or over-dissipation.

    Args:
        k_bins: 1D wavenumber bins.
        ratio_e: E_pred(k) / E_target(k).
        cutoff_ratio: Fraction of max wavenumber to define high-frequency band.

    Returns:
        Dict with mean_high_k_ratio and diagnosis string.
    """
    k_max = k_bins[-1]
    k_cutoff = k_max * cutoff_ratio
    high_mask = k_bins >= k_cutoff

    if not np.any(high_mask):
        high_mask = np.ones_like(k_bins, dtype=bool)

    mean_high_ratio = float(np.mean(ratio_e[high_mask]))

    if mean_high_ratio > 1.10:
        diagnosis = "spurious_high_frequency_accumulation"
        description = f"High-frequency ratio {mean_high_ratio:.3f} > 1.10: energy piles up at small scales exceeding the +10% diagnostic tolerance."
    elif mean_high_ratio < 0.90:
        diagnosis = "numerical_over_dissipation"
        description = f"High-frequency ratio {mean_high_ratio:.3f} < 0.90: excessive dissipation damping exceeds the -10% diagnostic tolerance."
    else:
        diagnosis = "balanced_scale_preservation"
        description = f"High-frequency ratio {mean_high_ratio:.3f} is within the predefined ±10% diagnostic tolerance band [0.90, 1.10]."

    return {
        "mean_high_k_ratio": mean_high_ratio,
        "diagnosis": diagnosis,
        "description": description,
    }


def resolve_group_checkpoint_path(grp: str, seed: int) -> str:
    """Resolve checkpoint path for a group and seed with strict fail-closed check."""
    if grp == "E0_single_step":
        candidate_paths = [
            "outputs/checkpoints/dynamics/closure_r4/ablation_E0_single_step/latent_transformer/best_vrmse_mean.pt",
            "outputs/checkpoints/dynamics/closure_r4/ablation_E0_single_step/best_vrmse_mean.pt",
        ]
    else:
        candidate_paths = [
            f"outputs/checkpoints/dynamics/closure_r4/seed_{seed}/ablation_{grp}/latent_transformer/best_vrmse_mean.pt",
            f"outputs/checkpoints/dynamics/closure_r4/seed_{seed}/ablation_{grp}/best_vrmse_mean.pt",
            f"outputs/checkpoints/dynamics/closure_r4/seed_{seed}/{grp}/latent_transformer/best_vrmse_mean.pt",
            f"outputs/checkpoints/dynamics/closure_r4/seed_{seed}/{grp}/best_vrmse_mean.pt",
        ]
        if seed == 42:
            candidate_paths.extend([
                f"outputs/checkpoints/dynamics/closure_r4/ablation_{grp}/latent_transformer/best_vrmse_mean.pt",
                f"outputs/checkpoints/dynamics/closure_r4/ablation_{grp}/best_vrmse_mean.pt",
            ])

    for p in candidate_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No valid checkpoint found for group '{grp}' under seed {seed}. Looked in: {candidate_paths}"
    )


def load_and_validate_forecaster(
    grp: str,
    ckpt_path: str,
    seed: int,
    eval_split_hash: str,
    eval_normalizer_hash: str,
    manifest_path: Optional[str],
    device: torch.device,
) -> Tuple[LatentForecaster, Dict[str, Any]]:
    """Load checkpoint, execute fail-closed semantic and provenance checks, and return model."""
    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt_data.get("config", {})

    # 1. Semantics validation: verify physics protocol and architecture contracts
    validate_ablation_checkpoint_semantics(grp, cfg, is_legacy=False)

    # 2. Checkpoint provenance resolution
    ckpt_prov = resolve_checkpoint_provenance(
        ckpt_path=ckpt_path,
        ckpt_data=ckpt_data,
        manifest_path=manifest_path,
    )

    # 3. Fail-closed evaluation provenance match
    expected_seed = 42 if grp == "E0_single_step" else seed
    validate_evaluation_provenance(
        ckpt_provenance=ckpt_prov,
        eval_split_hash=eval_split_hash,
        eval_normalizer_hash=eval_normalizer_hash,
        expected_seed=expected_seed,
        fail_closed=True,
    )

    ckpt_sha256 = compute_file_sha256(ckpt_path)
    provenance_info = {
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": ckpt_sha256,
        "seed": expected_seed,
        "training_git_commit": ckpt_prov.get("training_git_commit"),
        "training_git_dirty": ckpt_prov.get("training_git_dirty", False),
        "split_hash": ckpt_prov.get("split_hash"),
        "normalizer_hash": ckpt_prov.get("normalizer_hash"),
        "protocol": PHYSICS_PROTOCOL,
    }

    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=cfg.get("embed_dim", 256),
        cond_dim=128,
        depth=cfg.get("depth", 6),
        num_heads=cfg.get("num_heads", 8),
        history_length=4,
        prediction_mode=cfg.get("prediction_mode", "direct"),
    )
    forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder).to(device)

    if "model_state_dict" in ckpt_data:
        forecaster.load_state_dict(ckpt_data["model_state_dict"])
    elif "encoder_state_dict" in ckpt_data:
        forecaster.encoder.load_state_dict(ckpt_data["encoder_state_dict"])
        forecaster.transformer.load_state_dict(ckpt_data["transformer_state_dict"])
        forecaster.decoder.load_state_dict(ckpt_data["decoder_state_dict"])

    forecaster.eval()
    return forecaster, provenance_info


def analyze_spectral_dissipation_for_groups(
    data_dir: str = "/root/autodl-tmp/datasets/shear_flow",
    split_file: str = "outputs/splits/grouped_split.json",
    seeds: Optional[List[int]] = None,
    seed: Optional[int] = None,
    groups: Optional[List[str]] = None,
    horizons: List[int] = [1, 5, 10, 20, 30],
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_json: str = "outputs/metrics/directional_spectral_analysis.json",
    output_fig: str = "outputs/figures/directional_spectral_ratio_curves.png",
    manifest_path: str = "outputs/manifests/closure_r4_seed42.json",
    allow_dirty: bool = False,
) -> Dict[str, Any]:
    """Run rollout and directional spectral dissipation analysis with fail-closed provenance."""
    seed_everything(42)
    device = torch.device(device_str)

    git_commit = get_git_commit(PROJECT_ROOT)
    git_dirty = is_git_dirty(PROJECT_ROOT)
    if git_dirty and not allow_dirty:
        raise RuntimeError(
            "Working tree is dirty. Formal spectral dissipation analysis requires a clean git working tree. "
            "Please commit all changes before running or pass allow_dirty=True for exploratory testing."
        )

    if seeds is None:
        if seed is not None:
            seeds = [seed]
        else:
            seeds = [42, 43, 44]
    else:
        seeds = sorted(list(seeds))

    if groups is None:
        groups = [
            "E0_single_step",
            "E1_rollout_field",
            "E2_plus_L_div",
            "E3_plus_L_vort",
            "E4_full_physics",
        ]

    max_h = max(horizons)

    print(f"Loading test set from: {split_file} (max horizon: {max_h})...")
    _, _, test_loader, normalizer = create_flow_dataloaders(
        split_type="grouped",
        split_file=split_file,
        data_root=data_dir,
        history_length=4,
        horizon=max_h,
        stride=20,
        downsample_factor=2,
        batch_size=2,
        num_workers=0,
        normalize=True,
    )

    eval_split_hash = (
        compute_split_hash_from_file(split_file)
        if split_file and os.path.exists(split_file)
        else "UNKNOWN_SPLIT"
    )
    eval_normalizer_hash = compute_normalizer_hash(normalizer)

    print(f"Loaded {len(test_loader.dataset)} test trajectories.")
    print(f"Evaluation split_hash:      {eval_split_hash}")
    print(f"Evaluation normalizer_hash: {eval_normalizer_hash}")
    print(f"Evaluating seeds:           {seeds}")

    results = {
        "__metadata__": {
            "analysis": "Directional Spectral Dissipation Analysis (Issue #4)",
            "protocol": PHYSICS_PROTOCOL,
            "seeds": seeds,
            "domain_size": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
            "split_hash": eval_split_hash,
            "normalizer_hash": eval_normalizer_hash,
            "horizons": horizons,
            "groups": groups,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "provenance_per_group": {},
        },
        "target_spectra": {},
        "group_spectra": {},
        "spectral_ratios": {},
        "dissipation_diagnoses": {},
    }

    # First pass: collect ground truth target spectra across horizons
    print("\n--- Computing Ground Truth Target Energy Spectra ---")
    gt_radial_accum = {h: [] for h in horizons}
    gt_stream_accum = {h: [] for h in horizons}
    gt_cross_accum = {h: [] for h in horizons}
    k_radial_bins = None
    kx_bins = None
    ky_bins = None

    with torch.no_grad():
        for batch in test_loader:
            q_future = batch["future"].to(device)
            future_phys = normalizer.denormalize(q_future) if normalizer else q_future

            for h in horizons:
                step_idx = h - 1
                u_gt = future_phys[:, step_idx, 0]
                v_gt = future_phys[:, step_idx, 1]

                kb, e_rad = compute_radial_energy_spectrum(u_gt, v_gt, domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)
                k_radial_bins = kb.cpu().numpy()
                gt_radial_accum[h].append(e_rad.cpu().numpy())

                kxb, e_kx, kyb, e_ky = compute_directional_energy_spectra(u_gt, v_gt, domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)
                kx_bins = kxb.cpu().numpy()
                ky_bins = kyb.cpu().numpy()
                gt_stream_accum[h].append(e_kx.cpu().numpy())
                gt_cross_accum[h].append(e_ky.cpu().numpy())

    # Average target spectra
    for h in horizons:
        results["target_spectra"][f"step_{h}"] = {
            "radial": {
                "k": k_radial_bins.tolist(),
                "e_k": np.mean(gt_radial_accum[h], axis=0).tolist(),
            },
            "streamwise": {
                "kx": kx_bins.tolist(),
                "e_kx": np.mean(gt_stream_accum[h], axis=0).tolist(),
            },
            "cross_stream": {
                "ky": ky_bins.tolist(),
                "e_ky": np.mean(gt_cross_accum[h], axis=0).tolist(),
            },
        }

    # Second pass: evaluate each model group across requested seeds
    eps = 1e-10

    for grp in groups:
        results["group_spectra"][grp] = {}
        results["spectral_ratios"][grp] = {}
        results["dissipation_diagnoses"][grp] = {}
        results["__metadata__"]["provenance_per_group"][grp] = {}

        # E0 was only trained for single-step on seed 42
        grp_seeds = [42] if grp == "E0_single_step" else seeds

        seed_radial_ratios = {h: [] for h in horizons}
        seed_stream_ratios = {h: [] for h in horizons}
        seed_cross_ratios = {h: [] for h in horizons}
        seed_high_k_ratios = {h: [] for h in horizons}
        seed_diagnoses = {h: {} for h in horizons}
        seed_mean_radial_spectra = {h: [] for h in horizons}
        seed_mean_stream_spectra = {h: [] for h in horizons}
        seed_mean_cross_spectra = {h: [] for h in horizons}

        for s in grp_seeds:
            ckpt_path = resolve_group_checkpoint_path(grp, s)
            print(f"\n--- Analyzing Directional Spectra for [{grp}] Seed {s} ({ckpt_path}) ---")

            forecaster, prov_info = load_and_validate_forecaster(
                grp=grp,
                ckpt_path=ckpt_path,
                seed=s,
                eval_split_hash=eval_split_hash,
                eval_normalizer_hash=eval_normalizer_hash,
                manifest_path=manifest_path,
                device=device,
            )
            results["__metadata__"]["provenance_per_group"][grp][str(s)] = prov_info

            pred_radial_accum = {h: [] for h in horizons}
            pred_stream_accum = {h: [] for h in horizons}
            pred_cross_accum = {h: [] for h in horizons}

            with torch.no_grad():
                for batch in test_loader:
                    q_hist = batch["history"].to(device)
                    re = batch.get("re", None)
                    if re is not None:
                        re = re.to(device)
                    sc = batch.get("sc", None)
                    if sc is not None:
                        sc = sc.to(device)

                    pred_traj = forecaster.forward_rollout(q_hist, re, sc, horizon=max_h)
                    pred_phys = normalizer.denormalize(pred_traj) if normalizer else pred_traj

                    for h in horizons:
                        step_idx = h - 1
                        u_pred = pred_phys[:, step_idx, 0]
                        v_pred = pred_phys[:, step_idx, 1]

                        _, e_rad = compute_radial_energy_spectrum(u_pred, v_pred, domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)
                        pred_radial_accum[h].append(e_rad.cpu().numpy())

                        _, e_kx, _, e_ky = compute_directional_energy_spectra(u_pred, v_pred, domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)
                        pred_stream_accum[h].append(e_kx.cpu().numpy())
                        pred_cross_accum[h].append(e_ky.cpu().numpy())

            for h in horizons:
                skey = f"step_{h}"
                mean_rad = np.mean(pred_radial_accum[h], axis=0)
                mean_kx = np.mean(pred_stream_accum[h], axis=0)
                mean_ky = np.mean(pred_cross_accum[h], axis=0)

                seed_mean_radial_spectra[h].append(mean_rad)
                seed_mean_stream_spectra[h].append(mean_kx)
                seed_mean_cross_spectra[h].append(mean_ky)

                tgt_rad = np.array(results["target_spectra"][skey]["radial"]["e_k"])
                tgt_kx = np.array(results["target_spectra"][skey]["streamwise"]["e_kx"])
                tgt_ky = np.array(results["target_spectra"][skey]["cross_stream"]["e_ky"])

                ratio_rad = (mean_rad + eps) / (tgt_rad + eps)
                ratio_kx = (mean_kx + eps) / (tgt_kx + eps)
                ratio_ky = (mean_ky + eps) / (tgt_ky + eps)

                seed_radial_ratios[h].append(ratio_rad)
                seed_stream_ratios[h].append(ratio_kx)
                seed_cross_ratios[h].append(ratio_ky)

                diag = diagnose_dissipation_mode(k_radial_bins, ratio_rad, cutoff_ratio=0.5)
                seed_high_k_ratios[h].append(diag["mean_high_k_ratio"])
                seed_diagnoses[h][str(s)] = diag
                print(f"  Seed {s} Step {h:2d} -> High-k Ratio: {diag['mean_high_k_ratio']:.3f} | Diagnosis: {diag['diagnosis']}")

        # Aggregate across seeds for this group
        for h in horizons:
            skey = f"step_{h}"
            avg_rad_e = np.mean(seed_mean_radial_spectra[h], axis=0)
            avg_kx_e = np.mean(seed_mean_stream_spectra[h], axis=0)
            avg_ky_e = np.mean(seed_mean_cross_spectra[h], axis=0)

            results["group_spectra"][grp][skey] = {
                "radial_e_k": avg_rad_e.tolist(),
                "streamwise_e_kx": avg_kx_e.tolist(),
                "cross_stream_e_ky": avg_ky_e.tolist(),
            }

            avg_rad_ratio = np.mean(seed_radial_ratios[h], axis=0)
            avg_kx_ratio = np.mean(seed_stream_ratios[h], axis=0)
            avg_ky_ratio = np.mean(seed_cross_ratios[h], axis=0)

            std_rad_ratio = (
                np.std(seed_radial_ratios[h], axis=0, ddof=1).tolist()
                if len(grp_seeds) >= 2
                else None
            )

            results["spectral_ratios"][grp][skey] = {
                "radial_ratio": avg_rad_ratio.tolist(),
                "radial_ratio_std": std_rad_ratio,
                "streamwise_ratio": avg_kx_ratio.tolist(),
                "cross_stream_ratio": avg_ky_ratio.tolist(),
                "per_seed_radial_ratios": {
                    str(s): seed_radial_ratios[h][idx].tolist()
                    for idx, s in enumerate(grp_seeds)
                },
            }

            mean_hk = float(np.mean(seed_high_k_ratios[h]))
            std_hk = (
                float(np.std(seed_high_k_ratios[h], ddof=1))
                if len(grp_seeds) >= 2
                else None
            )
            overall_diag = diagnose_dissipation_mode(k_radial_bins, avg_rad_ratio, cutoff_ratio=0.5)

            results["dissipation_diagnoses"][grp][skey] = {
                "mean_high_k_ratio": mean_hk,
                "std_high_k_ratio": std_hk,
                "n_seeds": len(grp_seeds),
                "diagnosis": overall_diag["diagnosis"],
                "description": overall_diag["description"],
                "per_seed": seed_diagnoses[h],
            }

    # Save JSON metrics
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Provenance Verified] Saved directional spectral analysis to: {output_json}")

    # Generate Publication Figure
    plot_directional_spectral_figures(results, output_fig)
    return results


def plot_directional_spectral_figures(results: Dict[str, Any], save_path: str):
    """Plot publication-grade multi-panel figure for directional spectral analysis."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    k_rad = np.array(results["target_spectra"]["step_30"]["radial"]["k"])
    kx = np.array(results["target_spectra"]["step_30"]["streamwise"]["kx"])
    ky = np.array(results["target_spectra"]["step_30"]["cross_stream"]["ky"])

    groups = [g for g in results["group_spectra"].keys()]
    styles = {
        "E0_single_step": {"label": r"E0: Single-Step ($H=1$)", "color": "#7f7f7f", "linestyle": ":", "marker": "o"},
        "E1_rollout_field": {"label": r"E1: Rollout Field ($H=2$)", "color": "#1f77b4", "linestyle": "--", "marker": "s"},
        "E2_plus_L_div": {"label": r"E2: $+ L_{\mathrm{div}}$", "color": "#2ca02c", "linestyle": "-.", "marker": "^"},
        "E3_plus_L_vort": {"label": r"E3: $+ L_\omega$", "color": "#ff7f0e", "linestyle": "-.", "marker": "v"},
        "E4_full_physics": {"label": r"E4: $+ L_{\mathrm{div}} + L_\omega$", "color": "#d62728", "linestyle": "-", "marker": "D"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()

    target_rad_30 = np.array(results["target_spectra"]["step_30"]["radial"]["e_k"])

    # Panel (a): Radial Spectral Ratio at Step 30
    ax = axes[0]
    for grp in groups:
        r_rad = np.array(results["spectral_ratios"][grp]["step_30"]["radial_ratio"])
        st = styles.get(grp, {"label": grp, "color": "blue", "linestyle": "-", "marker": "o"})
        ax.plot(k_rad, r_rad, label=st["label"], color=st["color"], linestyle=st["linestyle"], marker=st["marker"], markersize=5, linewidth=2.0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.8, label="Ideal Conservation ($1.0$)")
    ax.fill_between(k_rad, 0.90, 1.10, color="green", alpha=0.08, label=r"$\pm 10\%$ Diagnostic Tolerance Band")
    ax.fill_between(k_rad, 0.0, 0.90, color="gray", alpha=0.06, label="Over-Dissipation ($<0.90$)")
    ax.fill_between(k_rad, 1.10, 3.0, color="red", alpha=0.05, label="Spurious Noise ($>1.10$)")
    ax.set_title(r"(a) Radial Spectral Ratio $E_{\mathrm{pred}}(k) / E_{\mathrm{target}}(k)$ at $h=30$", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Radial Wavenumber $k = \sqrt{k_x^2 + k_y^2}$", fontsize=10)
    ax.set_ylabel(r"Energy Ratio $R(k)$", fontsize=10)
    ax.set_ylim(0.0, 2.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")

    # Panel (b): Directional Streamwise Spectral Ratio at Step 30
    ax = axes[1]
    for grp in groups:
        r_kx = np.array(results["spectral_ratios"][grp]["step_30"]["streamwise_ratio"])
        st = styles.get(grp, {"label": grp, "color": "blue", "linestyle": "-", "marker": "o"})
        ax.plot(kx, r_kx, label=st["label"], color=st["color"], linestyle=st["linestyle"], marker=st["marker"], markersize=5, linewidth=2.0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_title(r"(b) Streamwise Spectral Ratio $E_{\mathrm{pred}}(k_x) / E_{\mathrm{target}}(k_x)$ at $h=30$", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Streamwise Wavenumber $k_x$", fontsize=10)
    ax.set_ylabel(r"Streamwise Ratio $R_x(k_x)$", fontsize=10)
    ax.set_ylim(0.0, 2.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")

    # Panel (c): Directional Cross-Stream Spectral Ratio at Step 30
    ax = axes[2]
    for grp in groups:
        r_ky = np.array(results["spectral_ratios"][grp]["step_30"]["cross_stream_ratio"])
        st = styles.get(grp, {"label": grp, "color": "blue", "linestyle": "-", "marker": "o"})
        ax.plot(ky, r_ky, label=st["label"], color=st["color"], linestyle=st["linestyle"], marker=st["marker"], markersize=5, linewidth=2.0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_title(r"(c) Cross-Stream Spectral Ratio $E_{\mathrm{pred}}(k_y) / E_{\mathrm{target}}(k_y)$ at $h=30$", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Cross-Stream Wavenumber $k_y$", fontsize=10)
    ax.set_ylabel(r"Cross-Stream Ratio $R_y(k_y)$", fontsize=10)
    ax.set_ylim(0.0, 2.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")

    # Panel (d): Raw Radial Kinetic Energy Spectra E(k) at Step 30
    ax = axes[3]
    valid_k = k_rad[1:]
    valid_tgt = target_rad_30[1:]
    ax.loglog(valid_k, valid_tgt, "k-", linewidth=2.8, label="Ground Truth Target")
    for grp in groups:
        e_rad = np.array(results["group_spectra"][grp]["step_30"]["radial_e_k"])[1:]
        st = styles.get(grp, {"label": grp, "color": "blue", "linestyle": "-", "marker": "o"})
        ax.loglog(valid_k, e_rad, label=st["label"], color=st["color"], linestyle=st["linestyle"], marker=st["marker"], markersize=5, linewidth=1.8)

    ax.set_title("(d) Raw Kinetic Energy Spectra $E(k)$ at $h=30$ (Log-Log)", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Radial Wavenumber $k$", fontsize=10)
    ax.set_ylabel(r"Energy Density $E(k)$", fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "The Well Shear Flow V1: Directional Spectral Dissipation Analysis\n"
        "Distinguishing High-Frequency Spurious Energy vs Numerical Over-Dissipation across Scale and Direction",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved directional spectral figure to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run directional spectral dissipation analysis.")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--split_file", type=str, default="outputs/splits/grouped_split.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Seed checkpoints to analyze")
    parser.add_argument("--seed", type=int, default=None, help="Single seed override")
    parser.add_argument("--groups", type=str, nargs="+", default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20, 30])
    parser.add_argument("--output_json", type=str, default="outputs/metrics/directional_spectral_analysis.json")
    parser.add_argument("--output_fig", type=str, default="outputs/figures/directional_spectral_ratio_curves.png")
    parser.add_argument("--manifest", type=str, default="outputs/manifests/closure_r4_seed42.json")
    parser.add_argument("--allow_dirty", action="store_true", default=False, help="Allow dirty working tree for testing")
    args = parser.parse_args()

    analyze_spectral_dissipation_for_groups(
        data_dir=args.data_dir,
        split_file=args.split_file,
        seeds=args.seeds,
        seed=args.seed,
        groups=args.groups,
        horizons=args.horizons,
        output_json=args.output_json,
        output_fig=args.output_fig,
        manifest_path=args.manifest,
        allow_dirty=args.allow_dirty,
    )


if __name__ == "__main__":
    main()
