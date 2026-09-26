#!/usr/bin/env python3
"""Horizon-R1 Unified Evaluation: Parent baseline + H2/H4/H8 on test set at h=1,5,10,20,30.

This script provides the formal, unified evaluation for the Horizon-R1 ablation study.
All checkpoints are evaluated on the SAME test set with IDENTICAL rollout horizons,
producing directly comparable metrics.

Key features:
1. Epoch-0 parent baseline on both validation and test sets
2. Dual checkpoint selection: short-best (val VRMSE) and long-best (J_long)
3. Unified h=1,5,10,20,30 autoregressive rollout evaluation
4. Full provenance tracking (checkpoint SHA256, git commit, etc.)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from src.data.pipeline import create_flow_dataloaders, FieldNormalizer
from src.metrics.field import evaluate_field_metrics
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_transformer import LatentSTTransformer
from src.models.latent_forecaster import LatentForecaster
from src.utils.checkpoint import load_checkpoint, resolve_spatial_pos_config
from src.utils.reproducibility import seed_everything
from src.utils.physics_contract import (
    PHYSICS_PROTOCOL,
    SPATIAL_AXIS_CONTRACT,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
)
from src.utils.provenance import (
    get_git_commit,
    is_git_dirty,
    git_commit_exists,
    compute_file_sha256 as compute_full_sha256,
    compute_split_hash_from_file,
    compute_normalizer_hash,
    resolve_checkpoint_provenance,
    validate_evaluation_provenance,
    validate_formal_provenance_bundle,
)


# ============================================================
# Constants
# ============================================================
PARENT_CKPT = "outputs/checkpoints/dynamics/closure_r4/ablation_E4_full_physics/latent_transformer/best_vrmse_mean.pt"
HORIZON_R1_BASE = "outputs/checkpoints/dynamics/horizon_r1/seed_42"

EVAL_HORIZONS = [1, 5, 10, 20, 30]
MAX_HORIZON = 30

# Checkpoint registry: (label, path, selection_criterion)
# short-best = saved best_vrmse_mean.pt
# long-best = checkpoint with lowest J_long among saved checkpoints
CHECKPOINT_REGISTRY = {
    "parent": PARENT_CKPT,
    "H2_short": f"{HORIZON_R1_BASE}/E4_H2_control/latent_transformer/best_vrmse_mean.pt",
    "H4_short": f"{HORIZON_R1_BASE}/E4_H4/latent_transformer/best_vrmse_mean.pt",
    "H8_short": f"{HORIZON_R1_BASE}/E4_H8/latent_transformer/best_vrmse_mean.pt",
}

# Long-best checkpoints: determined by J_long = mean(h10, h20, h30) from training logs
# These are populated by analyze_horizon_ablation.py or can be set manually
LONG_BEST_REGISTRY = {
    "H2_long": f"{HORIZON_R1_BASE}/E4_H2_control/latent_transformer/checkpoint_step_9_vrmse_mean_0.1394.pt",
    "H4_long": f"{HORIZON_R1_BASE}/E4_H4/latent_transformer/checkpoint_step_7_vrmse_mean_0.1594.pt",
    "H8_long": f"{HORIZON_R1_BASE}/E4_H8/latent_transformer/checkpoint_step_11_vrmse_mean_0.2186.pt",
}


def compute_file_sha256(path: str, prefix_len: Optional[int] = 16) -> str:
    """Compute SHA256 hash of a file, return first prefix_len hex chars (or full if None)."""
    full = compute_full_sha256(path)
    return full[:prefix_len] if prefix_len else full


def load_forecaster(ckpt_path: str, device: torch.device) -> Tuple[LatentForecaster, dict]:
    """Load a LatentForecaster from checkpoint, return (model, config)."""
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt_data.get("config", {})

    pred_mode = cfg.get("prediction_mode", ckpt_data.get("prediction_mode", "direct"))
    emb_dim = cfg.get("embed_dim", 256)
    depth = cfg.get("depth", 6)
    num_heads = cfg.get("num_heads", 8)

    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    use_spatial_pos = resolve_spatial_pos_config(ckpt_data)
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=emb_dim,
        cond_dim=128,
        depth=depth,
        num_heads=num_heads,
        history_length=4,
        prediction_mode=pred_mode,
        use_spatial_pos=use_spatial_pos,
    )
    forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder).to(device)

    if "model_state_dict" in ckpt_data:
        forecaster.load_state_dict(ckpt_data["model_state_dict"])
    print(f"  Loaded checkpoint: {ckpt_path}")
    print(f"  Config: prediction_mode={pred_mode}, embed_dim={emb_dim}, depth={depth}, num_heads={num_heads}")

    return forecaster, cfg


def evaluate_at_horizons(
    model: LatentForecaster,
    data_loader: DataLoader,
    device: torch.device,
    eval_horizons: List[int],
    normalizer=None,
    use_condition: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model at specified horizons via autoregressive rollout.

    Returns dict keyed by 'h{horizon}' with field metrics at each horizon.
    """
    model.eval()
    max_h = max(eval_horizons)

    # Accumulate metrics per horizon
    accumulated = {f"h{h}": {} for h in eval_horizons}
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            q_hist = batch["history"].to(device)
            q_future = batch["future"].to(device)
            re = batch["re"].to(device) if use_condition and "re" in batch else None
            sc = batch["sc"].to(device) if use_condition and "sc" in batch else None
            b = len(q_hist)
            total_samples += b

            # Full rollout to max horizon
            pred_traj = model.forward_rollout(q_hist, re, sc, horizon=max_h)

            # Denormalize
            if normalizer is not None:
                pred_eval = normalizer.denormalize(pred_traj)
                future_eval = normalizer.denormalize(q_future[:, :max_h])
            else:
                pred_eval = pred_traj
                future_eval = q_future[:, :max_h]

            # Zero-mean pressure gauge
            pred_eval[:, :, 2:3, :, :] = pred_eval[:, :, 2:3, :, :] - pred_eval[:, :, 2:3, :, :].mean(dim=(-2, -1), keepdim=True)
            future_eval[:, :, 2:3, :, :] = future_eval[:, :, 2:3, :, :] - future_eval[:, :, 2:3, :, :].mean(dim=(-2, -1), keepdim=True)

            # Extract metrics at each evaluation horizon
            for h in eval_horizons:
                if h > pred_eval.shape[1]:
                    continue
                pred_h = pred_eval[:, h - 1:h, :, :, :]  # (B, 1, 4, Ny, Nx)
                gt_h = future_eval[:, h - 1:h, :, :, :]

                metrics = evaluate_field_metrics(pred_h[:, 0], gt_h[:, 0])
                for k, v in metrics.items():
                    accumulated[f"h{h}"][k] = accumulated[f"h{h}"].get(k, 0.0) + v * b

    # Average
    results = {}
    for h_key, h_data in accumulated.items():
        results[h_key] = {k: v / total_samples for k, v in h_data.items()}

    return results


def parse_training_logs(log_dir: str = "outputs") -> Dict[str, list]:
    """Parse Horizon-R1 training logs to extract per-epoch diagnostics."""
    log_files = {
        "H2-control": os.path.join(log_dir, "train_horizon_r1_seed_42_E4_H2_control.log"),
        "H4": os.path.join(log_dir, "train_horizon_r1_seed_42_E4_H4.log"),
        "H8": os.path.join(log_dir, "train_horizon_r1_seed_42_E4_H8.log"),
    }

    pattern = re.compile(
        r"Epoch \[(\d+)/(\d+)\] \| Train Loss: ([\d.e+-]+) \| "
        r"Val Rollout Mean VRMSE: ([\d.]+) \| "
        r"Step 1 VRMSE: ([\d.]+) \(u: ([\d.]+), v: ([\d.]+), p: ([\d.]+), s: ([\d.]+)\) \| "
        r"Diag \[h10: ([\d.]+), h20: ([\d.]+), h30: ([\d.]+)\]"
    )

    all_data = {}
    for name, path in log_files.items():
        if not os.path.exists(path):
            print(f"  Warning: Log file not found: {path}")
            continue
        epochs = []
        with open(path) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    h10, h20, h30 = float(m.group(10)), float(m.group(11)), float(m.group(12))
                    epochs.append({
                        "epoch": int(m.group(1)),
                        "train_loss": float(m.group(3)),
                        "val_vrmse": float(m.group(4)),
                        "step1_vrmse": float(m.group(5)),
                        "step1_u": float(m.group(6)),
                        "step1_v": float(m.group(7)),
                        "step1_p": float(m.group(8)),
                        "step1_s": float(m.group(9)),
                        "diag_h10": h10,
                        "diag_h20": h20,
                        "diag_h30": h30,
                        "j_long": (h10 + h20 + h30) / 3.0,
                    })
        all_data[name] = epochs
    return all_data


def find_long_best_checkpoints(
    training_data: Dict[str, list],
    ckpt_base: str = HORIZON_R1_BASE,
) -> Dict[str, Optional[str]]:
    """Find the best J_long checkpoint among saved checkpoints for each group.

    Returns dict mapping group label to checkpoint path (or None if not found).
    """
    group_dirs = {
        "H2-control": f"{ckpt_base}/E4_H2_control/latent_transformer",
        "H4": f"{ckpt_base}/E4_H4/latent_transformer",
        "H8": f"{ckpt_base}/E4_H8/latent_transformer",
    }

    long_best = {}
    for group, epochs in training_data.items():
        ckpt_dir = group_dirs.get(group)
        if not ckpt_dir or not os.path.isdir(ckpt_dir):
            long_best[group] = None
            continue

        # List saved checkpoints and extract their epoch numbers
        saved_epochs = {}
        for fname in os.listdir(ckpt_dir):
            m = re.match(r"checkpoint_step_(\d+)_vrmse_mean_([\d.]+)\.pt", fname)
            if m:
                saved_epochs[int(m.group(1))] = os.path.join(ckpt_dir, fname)

        # Find epoch with best J_long among saved checkpoints
        best_j = float("inf")
        best_path = None
        best_epoch = None
        for ep_data in epochs:
            ep_num = ep_data["epoch"]
            if ep_num in saved_epochs:
                if ep_data["j_long"] < best_j:
                    best_j = ep_data["j_long"]
                    best_path = saved_epochs[ep_num]
                    best_epoch = ep_num

        # Also report the overall best J_long epoch (even if not saved)
        if epochs:
            overall_best = min(epochs, key=lambda e: e["j_long"])
            if best_epoch != overall_best["epoch"]:
                print(f"  Note [{group}]: Best J_long epoch={overall_best['epoch']} (J={overall_best['j_long']:.4f}) "
                      f"NOT saved. Using saved epoch={best_epoch} (J={best_j:.4f}) instead.")

        long_best[group] = best_path
        if best_path:
            print(f"  {group} long-best: epoch {best_epoch}, J_long={best_j:.4f} -> {best_path}")

    return long_best


def run_evaluation(
    data_dir: str = "/root/autodl-tmp/datasets/shear_flow",
    output_file: str = "outputs/metrics/horizon_r1_test_evaluation.json",
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    include_validation: bool = True,
    formal: bool = False,
):
    """Run the full Horizon-R1 unified evaluation."""
    seed_everything(42)
    device = torch.device(device_str)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("=" * 90)
    print("HORIZON-R1 UNIFIED TEST-SET EVALUATION" + (" [FORMAL MODE]" if formal else ""))
    print("=" * 90)

    # ────────────────────────────────────────────────
    # 0. Formal mode preflight: Fail-closed git cleanliness
    # ────────────────────────────────────────────────
    if formal:
        dirty = is_git_dirty()
        if dirty:
            raise RuntimeError(
                "Formal evaluation failed-closed: git working tree is dirty (git_dirty=True). "
                "Publication-grade formal artifacts require a clean git working tree."
            )
        print("  [Formal Preflight] Working tree clean (git_dirty=False): PASSED")

    # ────────────────────────────────────────────────
    # 1. Parse training logs and resolve long-best checkpoints
    # ────────────────────────────────────────────────
    print("\n[1/5] Resolving long-best checkpoints...")
    training_data = parse_training_logs()
    long_best_paths = find_long_best_checkpoints(training_data)

    # Priority 1: Check for true best_long_vrmse.pt saved by training long_tracker
    for group_dirname, label in [("E4_H2_control", "H2_long"), ("E4_H4", "H4_long"), ("E4_H8", "H8_long")]:
        direct_long_path = f"{HORIZON_R1_BASE}/{group_dirname}/latent_transformer/best_long_vrmse.pt"
        if os.path.exists(direct_long_path):
            print(f"  Found direct long-tracker checkpoint for {label}: {direct_long_path}")
            LONG_BEST_REGISTRY[label] = direct_long_path

    # Priority 2: Fallback to saved checkpoint with lowest J_long from logs
    label_map = {"H2-control": "H2_long", "H4": "H4_long", "H8": "H8_long"}
    for group, label in label_map.items():
        if label not in LONG_BEST_REGISTRY or not os.path.exists(LONG_BEST_REGISTRY[label]):
            if group in long_best_paths and long_best_paths[group]:
                LONG_BEST_REGISTRY[label] = long_best_paths[group]

    # ────────────────────────────────────────────────
    # 2. Load test dataset with strictly read-only normalizer
    # ────────────────────────────────────────────────
    print("\n[2/5] Loading test dataset with read-only normalizer...")
    split_file = "outputs/splits/grouped_split.json"
    if not os.path.exists(split_file):
        split_file = "outputs/splits/grouped.json"

    # Strictly load fitted normalizer in read-only mode so outputs/normalization/ is NEVER touched
    stats_path = "outputs/normalization/stats_grouped.pt"
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Required normalizer statistics not found: {stats_path}")
    normalizer = FieldNormalizer()
    normalizer.load_state_dict(torch.load(stats_path, weights_only=True, map_location="cpu"))
    eval_normalizer_hash = compute_normalizer_hash(normalizer)
    eval_split_hash = compute_split_hash_from_file(split_file)
    print(f"  Eval Normalizer SHA256: {eval_normalizer_hash[:12]}...")
    print(f"  Eval Split SHA256:      {eval_split_hash[:12]}...")

    _, val_loader, test_loader, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file=split_file,
        data_root=data_dir,
        history_length=4,
        horizon=MAX_HORIZON,
        stride=20,
        downsample_factor=2,
        batch_size=2,
        num_workers=0,
        normalize=True,
        normalizer=normalizer,  # Preloaded normalizer: NEVER touches outputs/normalization/
    )
    print(f"  Test samples: {len(test_loader.dataset)}")
    if val_loader:
        print(f"  Val samples: {len(val_loader.dataset)}")

    def validate_and_record_ckpt(ckpt_path: str, label: str) -> Tuple[str, str, Dict[str, Any]]:
        sha_16 = compute_file_sha256(ckpt_path, prefix_len=16)
        sha_full = compute_file_sha256(ckpt_path, prefix_len=None)
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        prov = resolve_checkpoint_provenance(ckpt_path, ckpt_data)

        # Enforce fail-closed protocol identity validation
        validate_evaluation_provenance(
            ckpt_provenance=prov,
            eval_split_hash=eval_split_hash,
            eval_normalizer_hash=eval_normalizer_hash,
            expected_seed=42,
            fail_closed=formal,
        )

        commit = prov.get("training_git_commit")
        commit_valid = git_commit_exists(commit)
        legacy_attestation = prov.get("legacy_attestation")

        if formal:
            validate_formal_provenance_bundle(prov, label=f"{label} ({ckpt_path})", fail_closed=True)
            if legacy_attestation:
                print(f"  [Formal] Note: {label} accepted via legacy attestation: {legacy_attestation.get('note', '')[:70]}...")

        prov_bundle = {
            "checkpoint_path": ckpt_path,
            "sha256": sha_16,
            "full_sha256": sha_full,
            "training_git_commit": commit,
            "training_git_commit_valid": commit_valid,
            "training_git_dirty": prov.get("training_git_dirty"),
            "seed": prov.get("seed"),
            "split_hash": prov.get("split_hash"),
            "normalizer_hash": prov.get("normalizer_hash"),
            "physics_protocol": prov.get("physics_protocol"),
            "spatial_axis_contract": prov.get("spatial_axis_contract"),
            "physics_domain_size_xy": prov.get("physics_domain_size_xy"),
            "legacy_attestation": legacy_attestation,
        }

        return sha_16, sha_full, prov_bundle

    # ────────────────────────────────────────────────
    # 3. Evaluate parent baseline
    # ────────────────────────────────────────────────
    print("\n[3/5] Evaluating parent (epoch 0) baseline...")
    results = {"__meta__": {
        "eval_horizons": EVAL_HORIZONS,
        "git_commit": get_git_commit(),
        "git_dirty": is_git_dirty(),
        "formal_evaluation": formal,
        "eval_split_hash": eval_split_hash,
        "eval_normalizer_hash": eval_normalizer_hash,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }}

    if os.path.exists(PARENT_CKPT):
        parent_sha16, parent_sha_full, parent_prov = validate_and_record_ckpt(PARENT_CKPT, "parent")
        print(f"  Parent checkpoint SHA256: {parent_sha16}")

        forecaster, cfg = load_forecaster(PARENT_CKPT, device)

        # Test set
        print("  Evaluating on TEST set...")
        parent_test = evaluate_at_horizons(forecaster, test_loader, device, EVAL_HORIZONS, normalizer)
        results["parent_test"] = parent_test
        results["parent_test"]["__ckpt__"] = PARENT_CKPT
        results["parent_test"]["__sha256__"] = parent_sha16
        results["parent_test"]["__full_sha256__"] = parent_sha_full
        results["parent_test"]["__provenance__"] = parent_prov

        # Validation set (for epoch-0 baseline comparison)
        if include_validation and val_loader:
            print("  Evaluating on VALIDATION set (epoch 0 baseline)...")
            parent_val = evaluate_at_horizons(forecaster, val_loader, device, EVAL_HORIZONS, normalizer)
            results["parent_val"] = parent_val
            results["parent_val"]["__ckpt__"] = PARENT_CKPT

        del forecaster
        torch.cuda.empty_cache()
    else:
        print(f"  WARNING: Parent checkpoint not found: {PARENT_CKPT}")

    # ────────────────────────────────────────────────
    # 4. Evaluate short-best checkpoints (H2/H4/H8)
    # ────────────────────────────────────────────────
    print("\n[4/5] Evaluating short-best checkpoints on TEST set...")
    for label, ckpt_path in CHECKPOINT_REGISTRY.items():
        if label == "parent":
            continue  # Already done
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {label}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n  --- {label} ---")
        sha16, sha_full, prov_bundle = validate_and_record_ckpt(ckpt_path, label)
        forecaster, cfg = load_forecaster(ckpt_path, device)
        test_metrics = evaluate_at_horizons(forecaster, test_loader, device, EVAL_HORIZONS, normalizer)
        test_metrics["__ckpt__"] = ckpt_path
        test_metrics["__sha256__"] = sha16
        test_metrics["__full_sha256__"] = sha_full
        test_metrics["__selection__"] = "short-best (val VRMSE)"
        test_metrics["__training_horizon__"] = int(label.split("_")[0].replace("H", ""))
        test_metrics["__provenance__"] = prov_bundle
        results[label] = test_metrics

        del forecaster
        torch.cuda.empty_cache()

    # ────────────────────────────────────────────────
    # 5. Evaluate long-best checkpoints (H2/H4/H8)
    # ────────────────────────────────────────────────
    print("\n[5/5] Evaluating long-best (J_long) checkpoints on TEST set...")
    for label, ckpt_path in LONG_BEST_REGISTRY.items():
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"  SKIP {label}: checkpoint not found at {ckpt_path}")
            continue

        # Skip if same as short-best
        short_label = label.replace("_long", "_short")
        if short_label in CHECKPOINT_REGISTRY:
            short_path = CHECKPOINT_REGISTRY[short_label]
            if os.path.exists(short_path) and os.path.realpath(ckpt_path) == os.path.realpath(short_path):
                print(f"  {label}: Same as {short_label}, skipping duplicate evaluation.")
                results[label] = results.get(short_label, {}).copy()
                results[label]["__selection__"] = "long-best (J_long) [same as short-best]"
                continue

        print(f"\n  --- {label} ---")
        sha16, sha_full, prov_bundle = validate_and_record_ckpt(ckpt_path, label)
        forecaster, cfg = load_forecaster(ckpt_path, device)
        test_metrics = evaluate_at_horizons(forecaster, test_loader, device, EVAL_HORIZONS, normalizer)
        test_metrics["__ckpt__"] = ckpt_path
        test_metrics["__sha256__"] = sha16
        test_metrics["__full_sha256__"] = sha_full
        test_metrics["__selection__"] = "long-best (J_long)"
        test_metrics["__provenance__"] = prov_bundle
        results[label] = test_metrics

        del forecaster
        torch.cuda.empty_cache()

    # ────────────────────────────────────────────────
    # Print summary table
    # ────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("HORIZON-R1 UNIFIED EVALUATION SUMMARY (Test Set, h=1,5,10,20,30)")
    print("=" * 100)

    # Collect model labels (excluding meta)
    model_labels = [k for k in results if not k.startswith("__") and not k.endswith("_val")]

    header = f"{'Model':<16} | {'h=1':>8} | {'h=5':>8} | {'h=10':>8} | {'h=20':>8} | {'h=30':>8} | {'J_long':>8}"
    print(header)
    print("-" * len(header))

    for label in model_labels:
        if label.endswith("_val"):
            continue
        metrics = results[label]
        vals = []
        for h in EVAL_HORIZONS:
            h_key = f"h{h}"
            v = metrics.get(h_key, {}).get("vrmse_mean", float("nan"))
            vals.append(v)

        # J_long from test metrics
        h10 = metrics.get("h10", {}).get("vrmse_mean", float("nan"))
        h20 = metrics.get("h20", {}).get("vrmse_mean", float("nan"))
        h30 = metrics.get("h30", {}).get("vrmse_mean", float("nan"))
        j_long = (h10 + h20 + h30) / 3.0

        print(f"{label:<16} | {vals[0]:>8.4f} | {vals[1]:>8.4f} | {vals[2]:>8.4f} | {vals[3]:>8.4f} | {vals[4]:>8.4f} | {j_long:>8.4f}")

    # Also print parent val baseline if available
    if "parent_val" in results:
        print("\n--- Parent Validation Baseline (epoch 0 reference) ---")
        pv = results["parent_val"]
        vals = [pv.get(f"h{h}", {}).get("vrmse_mean", float("nan")) for h in EVAL_HORIZONS]
        h10 = pv.get("h10", {}).get("vrmse_mean", float("nan"))
        h20 = pv.get("h20", {}).get("vrmse_mean", float("nan"))
        h30 = pv.get("h30", {}).get("vrmse_mean", float("nan"))
        j_long = (h10 + h20 + h30) / 3.0
        print(f"{'parent_val':<16} | {vals[0]:>8.4f} | {vals[1]:>8.4f} | {vals[2]:>8.4f} | {vals[3]:>8.4f} | {vals[4]:>8.4f} | {j_long:>8.4f}")

    # Save
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to: {output_file}")

    # Also save epoch trajectories with J_long
    traj_file = output_file.replace("test_evaluation", "epoch_trajectories")
    with open(traj_file, "w") as f:
        json.dump(training_data, f, indent=2)
    print(f"Saved epoch trajectories to: {traj_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Horizon-R1 Unified Test-Set Evaluation")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--output_file", type=str, default="outputs/metrics/horizon_r1_test_evaluation.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_validation", action="store_true", help="Skip parent validation baseline")
    parser.add_argument("--formal", action="store_true", help="Enforce fail-closed provenance validation and clean git state")
    args = parser.parse_args()

    run_evaluation(
        data_dir=args.data_dir,
        output_file=args.output_file,
        device_str=args.device,
        include_validation=not args.no_validation,
        formal=args.formal,
    )
