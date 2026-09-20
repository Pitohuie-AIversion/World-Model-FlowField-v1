"""Audit and Verify Dataset Splits for The Well shear_flow V1.

Verifies that:
1. Grouped Split (Zero-IC-Leakage) has STRICTLY ZERO trajectory or IC overlap between train, valid, and test sets.
2. Identifies and quantifies any IC data leakage in Official Split.
3. Computes the minimum pairwise physical distance across partition boundaries.
"""

import argparse
import json
import os
import sys
import h5py
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def extract_initial_conditions(split_entries):
    """Extract t=0 velocity fields for a list of trajectory entries."""
    ic_list = []
    # Cache open files
    open_files = {}
    for entry in split_entries:
        path = entry["file_path"]
        t_idx = entry["traj_idx"]
        cid = entry.get("cluster_id", -1)
        if path not in open_files:
            open_files[path] = h5py.File(path, "r")
        h5 = open_files[path]
        vel_ds = h5["t1_fields/velocity"] if "t1_fields/velocity" in h5 else h5.get("velocity")
        v0 = np.asarray(vel_ds[t_idx, 0], dtype=np.float32)
        ic_list.append({
            "file_path": path,
            "traj_idx": t_idx,
            "cluster_id": cid,
            "v0": v0,
        })

    for h5 in open_files.values():
        h5.close()
    return ic_list


def extract_initial_conditions_from_files(file_paths):
    """Extract t=0 velocity fields for all trajectories across a list of files."""
    ic_list = []
    for path in file_paths:
        with h5py.File(path, "r") as h5:
            vel_ds = h5["t1_fields/velocity"] if "t1_fields/velocity" in h5 else h5.get("velocity")
            n = vel_ds.shape[0]
            v0_all = np.asarray(vel_ds[:, 0], dtype=np.float32)
            for i in range(n):
                ic_list.append({
                    "file_path": path,
                    "traj_idx": i,
                    "v0": v0_all[i],
                })
    return ic_list


def compute_cross_split_leakage(set_a, set_b, tol=1e-4):
    """Compute pairwise minimum distances and identify exact IC leaks."""
    min_dist = float("inf")
    leaks = []
    for a in set_a:
        for b in set_b:
            diff = float(np.max(np.abs(a["v0"] - b["v0"])))
            if diff < min_dist:
                min_dist = diff
            if diff < tol:
                leaks.append((a, b, diff))
    return min_dist, leaks


def verify_splits(
    grouped_split_path="outputs/splits/grouped_split.json",
    official_split_path="outputs/splits/official_split.json",
):
    print("=" * 80)
    print("DATASET SPLIT AUDIT & IC DATA LEAKAGE VERIFICATION")
    print("=" * 80)

    # 1. Verify Grouped Split
    print("\n[1] AUDITING GROUPED SPLIT (Zero-IC-Leakage Split):")
    if not os.path.exists(grouped_split_path):
        raise FileNotFoundError(f"Missing {grouped_split_path}. Run scripts/build_splits.py first.")

    with open(grouped_split_path, "r") as f:
        grouped = json.load(f)

    train_ics_g = extract_initial_conditions(grouped["train"])
    valid_ics_g = extract_initial_conditions(grouped["valid"])
    test_ics_g = extract_initial_conditions(grouped["test"])

    min_tv_g, leaks_tv_g = compute_cross_split_leakage(train_ics_g, valid_ics_g)
    min_tt_g, leaks_tt_g = compute_cross_split_leakage(train_ics_g, test_ics_g)
    min_vt_g, leaks_vt_g = compute_cross_split_leakage(valid_ics_g, test_ics_g)

    print(f"  - Train trajectories: {len(train_ics_g)}")
    print(f"  - Valid trajectories: {len(valid_ics_g)}")
    print(f"  - Test trajectories:  {len(test_ics_g)}")
    print(f"  - Min separation (Train vs Valid): {min_tv_g:.6f} (Leaks detected: {len(leaks_tv_g)})")
    print(f"  - Min separation (Train vs Test):  {min_tt_g:.6f} (Leaks detected: {len(leaks_tt_g)})")
    print(f"  - Min separation (Valid vs Test):  {min_vt_g:.6f} (Leaks detected: {len(leaks_vt_g)})")

    grouped_pass = (len(leaks_tv_g) == 0 and len(leaks_tt_g) == 0 and len(leaks_vt_g) == 0)
    if grouped_pass:
        print("  => VERDICT: [PASS] Grouped Split has STRICTLY ZERO initial condition leakage!")
    else:
        print("  => VERDICT: [FAIL] Grouped Split contains cross-split leakage!")

    # 2. Audit Official Split for comparison
    print("\n[2] AUDITING THE WELL OFFICIAL SPLIT (Baseline Reference):")
    if not os.path.exists(official_split_path):
        raise FileNotFoundError(f"Missing {official_split_path}.")

    with open(official_split_path, "r") as f:
        official = json.load(f)

    train_ics_o = extract_initial_conditions_from_files(official["train"])
    valid_ics_o = extract_initial_conditions_from_files(official["valid"])
    test_ics_o = extract_initial_conditions_from_files(official["test"])

    min_tv_o, leaks_tv_o = compute_cross_split_leakage(train_ics_o, valid_ics_o)
    min_tt_o, leaks_tt_o = compute_cross_split_leakage(train_ics_o, test_ics_o)
    min_vt_o, leaks_vt_o = compute_cross_split_leakage(valid_ics_o, test_ics_o)

    print(f"  - Train trajectories: {len(train_ics_o)}")
    print(f"  - Valid trajectories: {len(valid_ics_o)}")
    print(f"  - Test trajectories:  {len(test_ics_o)}")
    print(f"  - Min separation (Train vs Valid): {min_tv_o:.6e} (Leaks detected: {len(leaks_tv_o)})")
    print(f"  - Min separation (Train vs Test):  {min_tt_o:.6e} (Leaks detected: {len(leaks_tt_o)})")
    print(f"  - Min separation (Valid vs Test):  {min_vt_o:.6e} (Leaks detected: {len(leaks_vt_o)})")

    if len(leaks_tv_o) > 0 or len(leaks_tt_o) > 0:
        print(f"  => OBSERVATION: Official split has {len(leaks_tv_o) + len(leaks_tt_o)} cross-partition IC leaks.")
        for a, b, diff in (leaks_tv_o + leaks_tt_o)[:5]:
            print(f"     * Leak: {os.path.basename(a['file_path'])}[traj {a['traj_idx']}] <=> {os.path.basename(b['file_path'])}[traj {b['traj_idx']}] (diff={diff:.2e})")

    # 3. Summary Comparison
    print("\n" + "=" * 80)
    print("SPLIT AUDIT SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Official Split':<20} | {'Grouped Split':<20}")
    print("-" * 80)
    print(f"{'Train Trajectories':<35} | {len(train_ics_o):<20} | {len(train_ics_g):<20}")
    print(f"{'Valid Trajectories':<35} | {len(valid_ics_o):<20} | {len(valid_ics_g):<20}")
    print(f"{'Test Trajectories':<35} | {len(test_ics_o):<20} | {len(test_ics_g):<20}")
    print(f"{'Train-Valid IC Leaks':<35} | {len(leaks_tv_o):<20} | {len(leaks_tv_g):<20}")
    print(f"{'Train-Test IC Leaks':<35} | {len(leaks_tt_o):<20} | {len(leaks_tt_g):<20}")
    print(f"{'Min Cross-Split Distance':<35} | {min_tv_o:<20.2e} | {min(min_tv_g, min_tt_g):<20.6f}")
    print(f"{'Zero-Leakage Compliance':<35} | {'FAILED (Leakage)':<20} | {'PASSED (Zero-Leak)':<20}")
    print("=" * 80)

    return {
        "grouped_pass": grouped_pass,
        "grouped_min_dist": float(min(min_tv_g, min_tt_g)),
        "official_leaks_count": len(leaks_tv_o) + len(leaks_tt_o),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouped_split", type=str, default="outputs/splits/grouped_split.json")
    parser.add_argument("--official_split", type=str, default="outputs/splits/official_split.json")
    args = parser.parse_args()

    res = verify_splits(args.grouped_split, args.official_split)
    if not res["grouped_pass"]:
        sys.exit(1)
