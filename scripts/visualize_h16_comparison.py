#!/usr/bin/env python3
"""Unified Publication-Grade Qualitative Comparison: H8 vs H16-Short vs H16-Long vs GT.

Generates comprehensive side-by-side rollout comparisons on canonical test samples:
  - Sample 0: Fixed reference sample (for longitudinal benchmark consistency)
  - Sample 23: Median error sample (representative behavior across test set)
  - Sample 36: High error sample (85th percentile stress case)

Variables visualized:
  - u-velocity (horizontal velocity field)
  - Vorticity (omega = dv/dx - du/dy)
  - Passive tracer (scalar concentration)

Across rollout horizons: h in [1, 10, 20, 30].

Layout per figure (4 rows x 7 columns):
  - Row h in {1, 10, 20, 30}:
      Col 0: Ground Truth (GT)
      Col 1: Parent H8 Long-Best (Pred)
      Col 2: Parent H8 (|Error|)
      Col 3: H16 Short-Best (Pred)
      Col 4: H16 Short-Best (|Error|)
      Col 5: H16 Long-Best (Pred)
      Col 6: H16 Long-Best (|Error|)

Contract guarantees:
  - Shared colormap & [vmin, vmax] across GT and all Predictions per step.
  - Shared colormap & [0, emax] across all Absolute Errors per step.
  - Companion metadata JSON saved alongside each generated figure.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.evaluate_h16_benchmark import (
    BENCHMARK_TARGETS,
    load_model_from_checkpoint,
    validate_benchmark_checkpoint,
)
from src.data.pipeline import create_flow_dataloaders, FieldNormalizer
from src.metrics.field import evaluate_field_metrics
from src.utils.fft_derivatives import compute_vorticity
from src.utils.physics_contract import (
    PHYSICS_PROTOCOL,
    SPATIAL_AXIS_CONTRACT,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
    zero_mean_pressure_gauge,
)
from src.utils.provenance import (
    compute_file_sha256,
    compute_normalizer_hash,
    compute_split_hash_from_file,
    get_git_commit,
    is_git_dirty,
)
from src.utils.reproducibility import seed_everything

TARGET_SAMPLES = [
    {
        "index": 0,
        "tag": "fixed_ref",
        "desc": "Fixed Reference Sample (Index 0)",
        "selection_basis": "Fixed anchor window index 0 for longitudinal benchmark consistency",
        "source_file": "shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5",
        "source_relative_path": "data/test/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5",
        "sim_idx": 1,
        "start_t": 0,
        "end_t": 34,
    },
    {
        "index": 23,
        "tag": "median_err",
        "desc": "Median Error Sample (Index 23)",
        "selection_basis": "Median window when 45 test windows are sorted by Parent H8 J_long VRMSE error (rank 23/45)",
        "source_file": "shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5",
        "source_relative_path": "data/train/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5",
        "sim_idx": 5,
        "start_t": 100,
        "end_t": 134,
    },
    {
        "index": 36,
        "tag": "high_err",
        "desc": "High Error Sample (Index 36, 85th percentile)",
        "selection_basis": "High-error window when 45 test windows are sorted by Parent H8 J_long VRMSE error (rank 39/45, ~85th percentile)",
        "source_file": "shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5",
        "source_relative_path": "data/train/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5",
        "sim_idx": 16,
        "start_t": 0,
        "end_t": 34,
    },
]

TARGET_VARS = ["u", "vorticity", "tracer"]
HORIZONS = [1, 10, 20, 30]


def extract_relative_data_path(file_path: str) -> str:
    """Extract relative dataset path starting from 'data/' if present, else basename."""
    norm_path = os.path.normpath(file_path)
    parts = norm_path.split(os.sep)
    if "data" in parts:
        data_idx = parts.index("data")
        return "/".join(parts[data_idx:])
    return os.path.basename(file_path)


def validate_target_samples_provenance(
    dataset,
    target_samples: list = TARGET_SAMPLES,
) -> None:
    """Fail-closed validation: cross-check hardcoded TARGET_SAMPLES metadata
    against the actual dataset.samples index at runtime.

    This prevents silent provenance drift if the data split, file ordering,
    stride, or horizon configuration changes.

    Args:
        dataset: A ShearFlowDataset instance with `.samples` and `.file_paths`.
        target_samples: The TARGET_SAMPLES list of dicts to validate.

    Raises:
        IndexError: If a target index exceeds dataset size.
        ValueError: If any source_file, sim_idx, start_t, or end_t mismatch.
    """
    for spec in target_samples:
        idx = spec["index"]
        if idx >= len(dataset.samples):
            raise IndexError(
                f"TARGET_SAMPLES specifies index={idx} but dataset has only "
                f"{len(dataset.samples)} samples. Data split or windowing "
                f"configuration may have changed."
            )

        # Unpack: (f_idx, sim_idx, start_t, split_t, end_t, re_val, sc_val)
        f_idx, sim_idx, start_t, _split_t, end_t, _re, _sc = dataset.samples[idx]
        actual_file = os.path.basename(dataset.file_paths[f_idx])
        actual_rel_path = extract_relative_data_path(dataset.file_paths[f_idx])

        # Dynamically record relative source path
        if "source_relative_path" not in spec:
            spec["source_relative_path"] = actual_rel_path

        mismatches = []
        if "source_file" in spec and actual_file != spec["source_file"]:
            mismatches.append(
                f"source_file: expected '{spec['source_file']}', actual '{actual_file}'"
            )
        if "source_relative_path" in spec and actual_rel_path != spec["source_relative_path"]:
            mismatches.append(
                f"source_relative_path: expected '{spec['source_relative_path']}', actual '{actual_rel_path}'"
            )
        if "sim_idx" in spec and sim_idx != spec["sim_idx"]:
            mismatches.append(
                f"sim_idx: expected {spec['sim_idx']}, actual {sim_idx}"
            )
        if "start_t" in spec and start_t != spec["start_t"]:
            mismatches.append(
                f"start_t: expected {spec['start_t']}, actual {start_t}"
            )
        if "end_t" in spec and end_t != spec["end_t"]:
            mismatches.append(
                f"end_t: expected {spec['end_t']}, actual {end_t}"
            )

        if mismatches:
            raise ValueError(
                f"TARGET_SAMPLES provenance mismatch at index={idx} "
                f"(tag='{spec.get('tag', '?')}'): "
                + "; ".join(mismatches)
                + ". The data split, file ordering, or windowing "
                "configuration has changed since TARGET_SAMPLES was authored. "
                "Update TARGET_SAMPLES or investigate the data pipeline."
            )

    print(f"  ✓ TARGET_SAMPLES provenance validated against dataset ({len(target_samples)} samples)")



def extract_field_slice(
    tensor_4d: torch.Tensor,
    var_name: str,
    domain_size: tuple = SHEAR_FLOW_DOMAIN_SIZE_XY,
) -> np.ndarray:
    """Extract scalar 2D numpy field from (4, Nx, Ny) physical tensor."""
    # tensor_4d: (4, Nx, Ny) where 0=u, 1=v, 2=p, 3=tracer
    if var_name == "u":
        return tensor_4d[0].detach().cpu().numpy()
    elif var_name == "v":
        return tensor_4d[1].detach().cpu().numpy()
    elif var_name == "p":
        return tensor_4d[2].detach().cpu().numpy()
    elif var_name in ("tracer", "s"):
        return tensor_4d[3].detach().cpu().numpy()
    elif var_name == "vorticity":
        u = tensor_4d[0:1]  # (1, Nx, Ny)
        v = tensor_4d[1:2]
        vort = compute_vorticity(u, v, domain_size=domain_size)
        return vort[0].detach().cpu().numpy()
    else:
        raise ValueError(f"Unknown variable name: {var_name}")


def generate_comparison_grid(
    sample_info: dict,
    var_name: str,
    gt_trajectory: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    models_meta: Dict[str, dict],
    out_dir: Path,
    re_val: float,
    sc_val: float,
    git_commit: str,
    git_dirty: bool,
    split_hash: str,
    norm_hash: str,
) -> Tuple[str, dict]:
    """Generates 4x7 comparative grid figure with row-shared colorbars."""
    n_rows = len(HORIZONS)
    n_cols = 7  # GT, H8 Pred, H8 Err, H16-S Pred, H16-S Err, H16-L Pred, H16-L Err

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(24, 14),
        gridspec_kw={"wspace": 0.08, "hspace": 0.25},
    )

    models_order = ["parent_h8_ep11", "h16_short_best_ep8", "h16_long_best_ep12"]
    model_labels = {
        "parent_h8_ep11": "Parent H8 Long-Best",
        "h16_short_best_ep8": "H16 Short-Best (Ep8)",
        "h16_long_best_ep12": "H16 Long-Best (Ep12)",
    }

    # Colormap selection
    if var_name in ("vorticity", "v"):
        field_cmap = "RdBu_r"
    elif var_name == "u":
        field_cmap = "coolwarm"
    else:
        field_cmap = "viridis"
    err_cmap = "inferno"

    norm_metadata = {}

    for row_idx, h in enumerate(HORIZONS):
        step_idx = h - 1

        gt_field = extract_field_slice(gt_trajectory[step_idx], var_name)
        preds_field = {
            m: extract_field_slice(predictions[m][step_idx], var_name)
            for m in models_order
        }
        errs_field = {
            m: np.abs(preds_field[m] - gt_field)
            for m in models_order
        }

        # Compute shared field normalization [vmin, vmax] across GT and all models
        all_field_vals = np.concatenate([gt_field.flatten()] + [preds_field[m].flatten() for m in models_order])
        if var_name in ("vorticity", "v"):
            abs_max = float(np.percentile(np.abs(all_field_vals), 99.5))
            vmin, vmax = -abs_max, abs_max
        else:
            vmin = float(np.percentile(all_field_vals, 0.5))
            vmax = float(np.percentile(all_field_vals, 99.5))

        # Compute shared error normalization [0, emax] across all models
        all_err_vals = np.concatenate([errs_field[m].flatten() for m in models_order])
        emax = float(np.percentile(all_err_vals, 99.5))
        if emax <= 1e-6:
            emax = 1e-4

        norm_metadata[f"h{h}"] = {
            "field_vmin": vmin,
            "field_vmax": vmax,
            "error_vmax": emax,
        }

        # Col 0: Ground Truth
        ax_gt = axes[row_idx, 0]
        im_gt = ax_gt.imshow(gt_field.T, origin="lower", cmap=field_cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax_gt.set_title(f"GT (h={h})", fontsize=11, fontweight="bold")
        ax_gt.set_ylabel(f"Step h={h}", fontsize=12, fontweight="bold")
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])

        col_counter = 1
        for m in models_order:
            # Model prediction
            ax_pred = axes[row_idx, col_counter]
            ax_pred.imshow(preds_field[m].T, origin="lower", cmap=field_cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax_pred.set_title(f"{model_labels[m]}\nPred", fontsize=10)
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            col_counter += 1

            # Model error
            ax_err = axes[row_idx, col_counter]
            im_err = ax_err.imshow(errs_field[m].T, origin="lower", cmap=err_cmap, vmin=0, vmax=emax, aspect="auto")
            m_err_mean = np.mean(errs_field[m])
            ax_err.set_title(f"|Error|\nMAE={m_err_mean:.3e}", fontsize=9, color="#990000")
            ax_err.set_xticks([])
            ax_err.set_yticks([])
            col_counter += 1

        # Add row colorbars at the ends
        # Field colorbar on GT column
        cbar_field = fig.colorbar(im_gt, ax=axes[row_idx, 0], orientation="horizontal", pad=0.08, fraction=0.046)
        cbar_field.ax.tick_params(labelsize=8)

        # Error colorbar on the last error column
        cbar_err = fig.colorbar(im_err, ax=axes[row_idx, -1], orientation="horizontal", pad=0.08, fraction=0.046)
        cbar_err.ax.tick_params(labelsize=8)

    sample_title = f"{sample_info['desc']} | Var: {var_name.upper()} | Re={re_val:.0f}, Sc={sc_val:.2f}"
    fig.suptitle(sample_title, fontsize=15, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.955,
        "Row-shared colorbars (per-horizon independent scaling: field 0.5%-99.5%, error 99.5%) | Do not compare color intensities across rows",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    # File output
    filename = f"comparison_{sample_info['tag']}_{var_name}.png"
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    meta_payload = {
        "figure_name": filename,
        "sample_index": sample_info["index"],
        "sample_tag": sample_info["tag"],
        "sample_description": sample_info["desc"],
        "sample_selection_basis": sample_info.get("selection_basis"),
        "source_file": sample_info.get("source_file"),
        "source_relative_path": sample_info.get("source_relative_path"),
        "simulation_index": sample_info.get("sim_idx"),
        "start_t": sample_info.get("start_t"),
        "end_t": sample_info.get("end_t"),
        "variable": var_name,
        "re": re_val,
        "sc": sc_val,
        "evaluation_git_commit": git_commit,
        "evaluation_git_dirty": git_dirty,
        "split_hash": split_hash,
        "normalizer_hash": norm_hash,
        "horizons": HORIZONS,
        "colorbar_scaling_policy": "同一时间步共享色条；不同时间步独立缩放；显示范围采用 0.5%–99.5%（场）与 99.5%（误差）百分位截断。不可直接跨时间步（行）比较颜色深浅。",
        "colorbar_norms": norm_metadata,
        "models_evaluated": models_meta,
    }
    meta_path = out_dir / f"comparison_{sample_info['tag']}_{var_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_payload, f, indent=2)

    return str(out_path), meta_payload


def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade qualitative comparisons across H8 and H16")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--out_dir", type=str, default="outputs/figures/h16_comparison_v2")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--formal", action="store_true", help="Fail closed on dirty git state")
    args = parser.parse_args()

    # Capture git state at startup BEFORE creating any output directories or files
    git_commit = get_git_commit(PROJECT_ROOT)
    git_dirty = is_git_dirty(PROJECT_ROOT)
    if args.formal and git_dirty:
        raise RuntimeError("Formal visualization failed-closed: working tree is dirty at startup.")

    seed_everything(42)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating qualitative comparisons on {device} (Git Commit: {git_commit}, Dirty: {git_dirty})...")

    # 1. Dataset & Normalizer
    split_file = "outputs/splits/grouped_split.json"
    stats_path = "outputs/normalization/stats_grouped.pt"
    normalizer = FieldNormalizer()
    normalizer.load_state_dict(torch.load(stats_path, weights_only=True, map_location="cpu"))
    split_hash = compute_split_hash_from_file(split_file)
    norm_hash = compute_normalizer_hash(normalizer)

    _, _, test_loader, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file=split_file,
        data_root=args.data_dir,
        history_length=4,
        horizon=max(HORIZONS),
        stride=20,
        downsample_factor=2,
        batch_size=1,
        num_workers=0,
        normalize=True,
        normalizer=normalizer,
        preload_to_memory=False,
        seed=42,
    )

    # Validate TARGET_SAMPLES provenance against live dataset
    validate_target_samples_provenance(test_loader.dataset)

    # 2. Load all 3 models with strict contract checks
    models = {}
    models_meta = {}
    for m_key, m_info in BENCHMARK_TARGETS.items():
        ckpt_path = m_info["path"]
        print(f"Loading and validating: {m_info['title']} ({ckpt_path})")
        model, cfg, ckpt_data = load_model_from_checkpoint(ckpt_path, device=device)
        validate_benchmark_checkpoint(
            ckpt_path=ckpt_path,
            ckpt_data=ckpt_data,
            cfg=cfg,
            target_key=m_key,
            target_info=m_info,
            eval_split_hash=split_hash,
            eval_normalizer_hash=norm_hash,
            formal=args.formal,
        )
        models[m_key] = model
        models_meta[m_key] = {
            "title": m_info["title"],
            "checkpoint_path": ckpt_path,
            "full_sha256": compute_file_sha256(ckpt_path),
            "expected_horizon": m_info["expected_horizon"],
            "selection_criterion": m_info["selection_criterion"],
        }

    # 3. Cache trajectories for target samples
    target_indices = {s["index"]: s for s in TARGET_SAMPLES}
    max_idx = max(target_indices.keys())

    print(f"\nCollecting trajectories for target samples: {list(target_indices.keys())}...")
    sample_trajectories = {}

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx in target_indices:
                q_hist = batch["history"].to(device)
                q_fut = batch["future"].to(device)
                re = batch["re"].to(device)
                sc = batch["sc"].to(device)

                # Denormalize ground truth and apply robust zero-mean pressure gauge
                t_phys = normalizer.denormalize(q_fut[0, :max(HORIZONS)]).clone()
                t_phys = zero_mean_pressure_gauge(t_phys)

                preds = {}
                for m_key, m in models.items():
                    pred_traj = m.forward_rollout(q_hist, re, sc, horizon=max(HORIZONS))
                    p_phys = normalizer.denormalize(pred_traj[0, :max(HORIZONS)]).clone()
                    p_phys = zero_mean_pressure_gauge(p_phys)
                    preds[m_key] = p_phys

                sample_trajectories[idx] = {
                    "sample_info": target_indices[idx],
                    "gt": t_phys,
                    "preds": preds,
                    "re": float(re[0].item()),
                    "sc": float(sc[0].item()),
                }

            if idx >= max_idx:
                break

    # 4. Generate comparison plots
    print(f"\nGenerating comparative visualization panels in {out_dir}...")
    manifest = []
    for s_idx, s_data in sample_trajectories.items():
        s_info = s_data["sample_info"]
        print(f"--- Processing {s_info['desc']} ---")
        for var in TARGET_VARS:
            fig_path, meta = generate_comparison_grid(
                sample_info=s_info,
                var_name=var,
                gt_trajectory=s_data["gt"],
                predictions=s_data["preds"],
                models_meta=models_meta,
                out_dir=out_dir,
                re_val=s_data["re"],
                sc_val=s_data["sc"],
                git_commit=git_commit,
                git_dirty=git_dirty,
                split_hash=split_hash,
                norm_hash=norm_hash,
            )
            print(f"  Saved: {fig_path}")
            manifest.append(meta)

    # Save index manifest
    with open(out_dir / "comparison_figures_index.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nAll qualitative figures and metadata generated successfully in {out_dir}")


if __name__ == "__main__":
    main()
