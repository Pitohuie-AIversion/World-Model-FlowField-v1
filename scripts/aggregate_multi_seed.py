"""Multi-seed statistical aggregation and paired effect analysis for physics ablations.

Ingests multi-seed evaluation JSON outputs (e.g. seeds 42, 43, 44), strictly enforces
fail-closed provenance, split_hash, normalizer_hash, and training_git_dirty validations,
computes sample statistics (mean, sample std with ddof=1) without fabricating uncertainty
for N < 2, and calculates paired effect deltas (E_x - E_1) across seeds.
"""

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.physics_contract import PHYSICS_PROTOCOL
from src.utils.provenance import compute_file_sha256

DEFAULT_HORIZONS = [1, 5, 10, 20, 30]
DEFAULT_GROUPS = [
    "E1_rollout_field",
    "E2_plus_L_div",
    "E3_plus_L_vort",
    "E4_full_physics",
]
DEFAULT_METRICS = [
    ("vrmse_mean", "Field Mean VRMSE", "minimize"),
    ("rmse_mean", "Field Mean RMSE", "minimize"),
    ("div_rmse", "Divergence RMSE", "minimize"),
    ("vort_rmse", "Vorticity RMSE", "minimize"),
    ("energy_spectrum_mae", "Energy Spectrum MAE", "minimize"),
    ("ke_rel_err", "Kinetic Energy Rel Err", "minimize"),
    ("enstrophy_rel_err", "Enstrophy Rel Err", "minimize"),
    ("tracer_var_retention", "Tracer Var Retention", "target_1.0"),
    ("tracer_out_of_bounds_rate", "Tracer OOB Rate", "minimize"),
    ("tracer_mass_error", "Tracer Mass Error (L1)", "minimize"),
    ("tracer_mean_err", "Tracer Mean Error", "minimize"),
]


def resolve_seed_metrics_file(metrics_dir: str, seed: int) -> str:
    """Resolve file path for a given seed with strict fallback checks."""
    c1 = os.path.join(metrics_dir, f"closure_r4_physics_ablation_seed{seed}_v2.json")
    if os.path.exists(c1):
        return c1
    if seed == 42:
        c2 = os.path.join(metrics_dir, "closure_r4_physics_ablation_v2.json")
        if os.path.exists(c2):
            return c2
    raise FileNotFoundError(
        f"Metrics file for seed {seed} not found in '{metrics_dir}'. "
        f"Expected: {c1}"
    )


def load_and_validate_seed_metrics(
    file_path: str,
    expected_seed: Optional[int] = None,
    expected_split_hash: Optional[str] = None,
    expected_normalizer_hash: Optional[str] = None,
    require_clean: bool = True,
    manifest_path: Optional[str] = "outputs/manifests/closure_r4_seed42.json",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load evaluation file and strictly validate provenance contracts.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If seed, protocol, hashes, or clean training contracts are violated.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation metrics file not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    # Load backup manifest for seed 42 if needed
    manifest_data = {}
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)

    extracted_provenance = {}

    for grp_key, grp_data in data.items():
        if grp_key.startswith("__") or not isinstance(grp_data, dict):
            continue

        meta = grp_data.get("__metadata__", {})
        # Check manifest fallback if fields missing (e.g. early seed 42)
        if manifest_data and (not meta.get("seed") or not meta.get("split_hash")):
            man_groups = manifest_data.get("groups", {})
            mg_info = man_groups.get(grp_key) or man_groups.get(f"ablation_{grp_key}")
            if mg_info:
                meta.setdefault("seed", mg_info.get("seed"))
                meta.setdefault("split_hash", mg_info.get("split_hash"))
                meta.setdefault("normalizer_hash", mg_info.get("normalizer_hash"))
                meta.setdefault("training_git_dirty", False)

        seed = meta.get("seed")
        protocol = meta.get("evaluation_protocol", meta.get("checkpoint_training_protocol"))
        split_hash = meta.get("split_hash")
        norm_hash = meta.get("normalizer_hash")
        is_dirty = meta.get("training_git_dirty", False)

        # 1. Seed verification
        if expected_seed is not None and seed != expected_seed:
            raise ValueError(
                f"Seed mismatch in {file_path} for group {grp_key}: "
                f"found seed={seed}, expected {expected_seed}"
            )

        # 2. Protocol verification
        if protocol != PHYSICS_PROTOCOL:
            raise ValueError(
                f"Protocol mismatch in {file_path} for group {grp_key}: "
                f"found '{protocol}', expected '{PHYSICS_PROTOCOL}'"
            )

        # 3. Split hash verification
        if not split_hash:
            raise ValueError(f"Missing required 'split_hash' in {file_path} for group {grp_key}")
        if expected_split_hash is not None and split_hash != expected_split_hash:
            raise ValueError(
                f"Split hash mismatch across seeds for group {grp_key}: "
                f"{split_hash} != {expected_split_hash}"
            )

        # 4. Normalizer hash verification
        if not norm_hash:
            raise ValueError(f"Missing required 'normalizer_hash' in {file_path} for group {grp_key}")
        if expected_normalizer_hash is not None and norm_hash != expected_normalizer_hash:
            raise ValueError(
                f"Normalizer hash mismatch across seeds for group {grp_key}: "
                f"{norm_hash} != {expected_normalizer_hash}"
            )

        # 5. Clean training verification
        if require_clean and is_dirty:
            raise ValueError(
                f"Dirty training state detected in {file_path} for group {grp_key}: "
                f"training_git_dirty=True is strictly forbidden for formal publication artifacts."
            )

        if not extracted_provenance:
            extracted_provenance = {
                "seed": seed,
                "split_hash": split_hash,
                "normalizer_hash": norm_hash,
                "protocol": protocol,
                "file_path": file_path,
                "training_git_commit": meta.get("training_git_commit"),
                "training_git_dirty": is_dirty,
            }

    return data, extracted_provenance


def compute_sample_statistics(values: List[float]) -> Dict[str, Any]:
    """Compute sample mean and unbiased sample standard deviation (ddof=1).

    If len(values) < 2, does NOT fabricate an uncertainty; returns None for std.
    """
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    mean_val = float(np.mean(values))
    if n >= 2:
        std_val = float(np.std(values, ddof=1))
    else:
        std_val = None
    return {"mean": mean_val, "std": std_val, "n": n}


def compute_paired_deltas(
    seed_metrics: Dict[int, Dict[str, Any]],
    baseline_group: str = "E1_rollout_field",
    compare_groups: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute paired per-seed deltas (E_x - E_1) across all seeds and horizons.

    Returns:
        Structured paired effects mapping group -> horizon -> metric -> stats.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    if compare_groups is None:
        compare_groups = [g for g in DEFAULT_GROUPS if g != baseline_group]
    if metric_keys is None:
        metric_keys = [m[0] for m in DEFAULT_METRICS]

    # Map metric to objective: "minimize", "maximize", or "target_1.0"
    objective_map = {}
    for m in DEFAULT_METRICS:
        k = m[0]
        obj = m[2]
        if isinstance(obj, bool):
            objective_map[k] = "minimize" if obj else "maximize"
        else:
            objective_map[k] = str(obj).lower()

    seeds = sorted(seed_metrics.keys())
    paired_effects = {}

    for cmp_grp in compare_groups:
        paired_effects[cmp_grp] = {}
        for h in horizons:
            skey = f"step_{h}"
            paired_effects[cmp_grp][skey] = {}
            for mk in metric_keys:
                per_seed_deltas = {}
                delta_vals = []
                wins = 0
                obj = objective_map.get(mk, "minimize")

                for s in seeds:
                    base_val = seed_metrics[s].get(baseline_group, {}).get(skey, {}).get(mk)
                    cmp_val = seed_metrics[s].get(cmp_grp, {}).get(skey, {}).get(mk)
                    if base_val is not None and cmp_val is not None:
                        d = float(cmp_val - base_val)
                        per_seed_deltas[str(s)] = d
                        delta_vals.append(d)

                        # Scientific win condition
                        if obj == "target_1.0":
                            # Closer to 1.0 is an improvement
                            err_cmp = abs(cmp_val - 1.0)
                            err_base = abs(base_val - 1.0)
                            if err_cmp < err_base:
                                wins += 1
                        elif obj == "maximize":
                            if d > 0:
                                wins += 1
                        else:  # minimize
                            if d < 0:
                                wins += 1

                if delta_vals:
                    stats = compute_sample_statistics(delta_vals)
                    # Baseline mean for relative delta
                    base_vals = [
                        seed_metrics[s][baseline_group][skey][mk]
                        for s in seeds
                        if mk in seed_metrics[s].get(baseline_group, {}).get(skey, {})
                    ]
                    base_mean = float(np.mean(base_vals)) if base_vals else 1.0
                    rel_delta_pct = (
                        (stats["mean"] / (base_mean + 1e-12)) * 100.0
                        if abs(base_mean) > 1e-12
                        else None
                    )

                    win_rate = float(wins / len(delta_vals))

                    paired_effects[cmp_grp][skey][mk] = {
                        "mean_delta": stats["mean"],
                        "std_delta": stats["std"],
                        "n": stats["n"],
                        "rel_delta_pct": rel_delta_pct,
                        "win_count": wins,
                        "win_rate": win_rate,
                        "per_seed_deltas": per_seed_deltas,
                    }

    return paired_effects


def aggregate_multi_seed_metrics(
    seed_files: Dict[int, str],
    groups: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
    baseline_group: str = "E1_rollout_field",
    representation_ckpt_path: Optional[str] = "outputs/checkpoints/representation/best_vrmse_mean.pt",
    require_clean: bool = True,
    require_complete: bool = True,
    manifest_path: Optional[str] = "outputs/manifests/closure_r4_seed42.json",
) -> Dict[str, Any]:
    """Full pipeline for aggregating multi-seed ablation metrics."""
    if groups is None:
        groups = DEFAULT_GROUPS
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    seeds = sorted(seed_files.keys())
    seed_data = {}
    provenance_per_seed = {}

    expected_split_hash = None
    expected_norm_hash = None

    for s in seeds:
        fpath = seed_files[s]
        data, prov = load_and_validate_seed_metrics(
            file_path=fpath,
            expected_seed=s,
            expected_split_hash=expected_split_hash,
            expected_normalizer_hash=expected_norm_hash,
            require_clean=require_clean,
            manifest_path=manifest_path,
        )
        seed_data[s] = data
        provenance_per_seed[str(s)] = prov

        if expected_split_hash is None:
            expected_split_hash = prov["split_hash"]
        if expected_norm_hash is None:
            expected_norm_hash = prov["normalizer_hash"]

    # Compute representation checkpoint SHA-256
    rep_sha256 = None
    if representation_ckpt_path and os.path.exists(representation_ckpt_path):
        rep_sha256 = compute_file_sha256(representation_ckpt_path)

    # 1. Direct Group Statistics
    group_stats = {}
    # Include E0 if present in seed 42
    all_groups_to_process = list(groups)
    if "E0_single_step" not in all_groups_to_process and 42 in seed_data and "E0_single_step" in seed_data[42]:
        all_groups_to_process = ["E0_single_step"] + all_groups_to_process

    for grp in all_groups_to_process:
        group_stats[grp] = {}
        for h in horizons:
            skey = f"step_{h}"
            group_stats[grp][skey] = {}
            for mk, _, _ in DEFAULT_METRICS:
                vals = []
                for s in seeds:
                    val = seed_data[s].get(grp, {}).get(skey, {}).get(mk)
                    if val is not None:
                        vals.append(float(val))

                # Formal completeness check: all requested seeds must be present for multi-seed ablation groups
                if require_complete and grp in groups:
                    if len(vals) != len(seeds):
                        missing_seeds = [
                            s for s in seeds
                            if seed_data[s].get(grp, {}).get(skey, {}).get(mk) is None
                        ]
                        raise ValueError(
                            f"Formal completeness violation: missing metric '{mk}' for group '{grp}' "
                            f"at horizon step_{h} for seed(s): {missing_seeds}. "
                            f"Expected N={len(seeds)} observations, got N={len(vals)}."
                        )

                if vals:
                    st = compute_sample_statistics(vals)
                    group_stats[grp][skey][mk] = {
                        "mean": st["mean"],
                        "std": st["std"],
                        "n": st["n"],
                        "values": {str(s): seed_data[s][grp][skey][mk] for s in seeds if grp in seed_data[s] and mk in seed_data[s][grp][skey]},
                    }

    # 2. Paired Effect Statistics
    paired_effects = compute_paired_deltas(
        seed_metrics=seed_data,
        baseline_group=baseline_group,
        compare_groups=[g for g in groups if g != baseline_group],
        horizons=horizons,
    )

    summary_bundle = {
        "__metadata__": {
            "aggregation_script": "scripts/aggregate_multi_seed.py",
            "protocol": PHYSICS_PROTOCOL,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "split_hash": expected_split_hash,
            "normalizer_hash": expected_norm_hash,
            "representation_checkpoint": representation_ckpt_path,
            "representation_checkpoint_sha256": rep_sha256,
            "baseline_group": baseline_group,
            "provenance_per_seed": provenance_per_seed,
        },
        "group_statistics": group_stats,
        "paired_effects": paired_effects,
    }

    # Also maintain top-level compatibility mapping for existing readers:
    for grp, grp_h in group_stats.items():
        summary_bundle[grp] = grp_h

    return summary_bundle


def print_summary_tables(summary_bundle: Dict[str, Any]):
    """Pretty-print group statistics and paired effects to stdout."""
    group_stats = summary_bundle["group_statistics"]
    paired_effects = summary_bundle["paired_effects"]
    seeds = summary_bundle["__metadata__"]["seeds"]
    n_seeds = len(seeds)

    print("\n" + "=" * 118)
    print(f"MULTI-SEED ABLATION SUMMARY (N={n_seeds} Seeds: {seeds}, Protocol: {PHYSICS_PROTOCOL})")
    print("=" * 118)

    key_metrics = [
        ("vrmse_mean", "Field Mean VRMSE"),
        ("div_rmse", "Divergence RMSE"),
        ("vort_rmse", "Vorticity RMSE"),
        ("energy_spectrum_mae", "Energy Spectrum MAE"),
        ("enstrophy_rel_err", "Enstrophy Rel Err"),
        ("tracer_oob_rate", "Tracer OOB Rate"),
    ]

    for mk, mtitle in key_metrics:
        # map name if needed
        actual_mk = "tracer_out_of_bounds_rate" if mk == "tracer_oob_rate" else mk
        print(f"\n### {mtitle}")
        print(f"{'Group':<18} | {'Step 1':<18} | {'Step 5':<18} | {'Step 10':<18} | {'Step 20':<18} | {'Step 30':<18}")
        print("-" * 118)
        for grp, grp_data in group_stats.items():
            row = []
            for h in DEFAULT_HORIZONS:
                skey = f"step_{h}"
                m_info = grp_data.get(skey, {}).get(actual_mk)
                if m_info and m_info["mean"] is not None:
                    if m_info["std"] is not None:
                        row.append(f"{m_info['mean']:.4f}±{m_info['std']:.4f}")
                    else:
                        row.append(f"{m_info['mean']:.4f}")
                else:
                    row.append("N/A")
            print(f"{grp:<18} | {row[0]:<18} | {row[1]:<18} | {row[2]:<18} | {row[3]:<18} | {row[4]:<18}")

    print("\n" + "=" * 118)
    print("WITHIN-SEED PAIRED DIFFERENCE ANALYSIS vs BASELINE E1: Rollout-Aware Field (Mean Delta ± Std, [Win Rate])")
    print("Format: Mean Delta (Rel %) ± Std [Wins/Total]")
    print("=" * 118)

    for cmp_grp, cmp_data in paired_effects.items():
        print(f"\n>>> Paired Effect: {cmp_grp} minus E1_rollout_field")
        print(f"{'Metric':<24} | {'Step 1':<18} | {'Step 5':<18} | {'Step 10':<18} | {'Step 20':<18} | {'Step 30':<18}")
        print("-" * 118)
        for mk, mtitle in key_metrics:
            actual_mk = "tracer_out_of_bounds_rate" if mk == "tracer_oob_rate" else mk
            row = []
            for h in DEFAULT_HORIZONS:
                skey = f"step_{h}"
                p_info = cmp_data.get(skey, {}).get(actual_mk)
                if p_info and p_info["mean_delta"] is not None:
                    md = p_info["mean_delta"]
                    sd = p_info["std_delta"]
                    w = p_info["win_count"]
                    n = p_info["n"]
                    sd_str = f"±{sd:.4f}" if sd is not None else ""
                    row.append(f"{md:+.4f}{sd_str} [{w}/{n}]")
                else:
                    row.append("N/A")
            print(f"{mtitle:<24} | {row[0]:<18} | {row[1]:<18} | {row[2]:<18} | {row[3]:<18} | {row[4]:<18}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed physics ablation metrics.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="List of seeds to aggregate")
    parser.add_argument("--groups", type=str, nargs="+", default=DEFAULT_GROUPS, help="Groups to aggregate")
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS, help="Evaluation horizons")
    parser.add_argument("--metrics_dir", type=str, default="outputs/metrics", help="Directory containing per-seed metrics")
    parser.add_argument("--output_file", type=str, default="outputs/metrics/closure_r4_physics_ablation_tri_seed_summary.json", help="Path to save summary JSON")
    parser.add_argument("--baseline_group", type=str, default="E1_rollout_field", help="Baseline group for paired deltas")
    parser.add_argument("--representation_checkpoint", type=str, default="outputs/checkpoints/representation/best_vrmse_mean.pt", help="Path to representation checkpoint")
    parser.add_argument("--manifest", type=str, default="outputs/manifests/closure_r4_seed42.json", help="Path to seed 42 manifest")
    parser.add_argument("--allow_dirty", action="store_true", default=False, help="Allow dirty training git status (not recommended)")
    parser.add_argument("--allow_incomplete", action="store_true", default=False, help="Allow incomplete seed observations (not recommended for formal benchmark)")
    args = parser.parse_args()

    seed_files = {}
    for s in args.seeds:
        seed_files[s] = resolve_seed_metrics_file(args.metrics_dir, s)

    print(f"Aggregating {len(args.seeds)} seeds: {args.seeds}")
    for s, fp in seed_files.items():
        print(f"  Seed {s} -> {fp}")

    summary = aggregate_multi_seed_metrics(
        seed_files=seed_files,
        groups=args.groups,
        horizons=args.horizons,
        baseline_group=args.baseline_group,
        representation_ckpt_path=args.representation_checkpoint,
        require_clean=not args.allow_dirty,
        require_complete=not args.allow_incomplete,
        manifest_path=args.manifest,
    )

    print_summary_tables(summary)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Provenance Complete] Saved verified multi-seed summary to: {args.output_file}")


if __name__ == "__main__":
    main()
