"""Script to build, verify, and export the three dataset split protocols for shear_flow V1."""

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.splits import SplitManager, parse_shear_flow_filename


def build_and_save_splits(
    data_dir: str = "/root/autodl-tmp/datasets/shear_flow",
    output_dir: str = "outputs/splits",
    holdout_re: float = 1e5,
    holdout_sc: float = 10.0,
):
    """Scan available HDF5 files and construct reproducible split JSON registries."""
    os.makedirs(output_dir, exist_ok=True)

    import h5py

    def is_valid_hdf5(path: str) -> bool:
        if os.path.exists(path + ".aria2"):
            return False
        try:
            with h5py.File(path, "r") as h5:
                return "t0_fields" in h5 or "pressure" in h5
        except Exception:
            return False

    train_files = sorted([f for f in glob.glob(os.path.join(data_dir, "**/train/*.hdf5"), recursive=True) if is_valid_hdf5(f)])
    valid_files = sorted([f for f in glob.glob(os.path.join(data_dir, "**/valid/*.hdf5"), recursive=True) if is_valid_hdf5(f)])
    test_files = sorted([f for f in glob.glob(os.path.join(data_dir, "**/test/*.hdf5"), recursive=True) if is_valid_hdf5(f)])

    all_files = train_files + valid_files + test_files
    print(f"Total verified HDF5 files located: {len(all_files)}")
    print(f"  Train partition files: {len(train_files)}")
    print(f"  Valid partition files: {len(valid_files)}")
    print(f"  Test partition files:  {len(test_files)}")

    if not all_files:
        print(f"Warning: No HDF5 files found in {data_dir}. Ensure data is downloaded first.")
        return

    rel_train_files = sorted([os.path.relpath(f, data_dir) for f in train_files])
    rel_valid_files = sorted([os.path.relpath(f, data_dir) for f in valid_files])
    rel_test_files = sorted([os.path.relpath(f, data_dir) for f in test_files])
    rel_all_files = rel_train_files + rel_valid_files + rel_test_files

    # 1. Official Split
    official = SplitManager.get_official_split(rel_train_files, rel_valid_files, rel_test_files)
    off_path = os.path.join(output_dir, "official_split.json")
    with open(off_path, "w") as f:
        json.dump(official, f, indent=2)
    print(f"Exported Official Split to: {off_path}")

    # 2. Grouped Split (Zero-IC-Leakage Split)
    grouped = SplitManager.get_grouped_split(
        all_files=rel_all_files,
        data_root=data_dir,
        train_ratio=0.77,
        valid_ratio=0.11,
        seed=42,
    )
    grp_path = os.path.join(output_dir, "grouped_split.json")
    with open(grp_path, "w") as f:
        json.dump(grouped, f, indent=2)
    print(f"Exported Grouped Split (Zero-IC-Leakage) to: {grp_path}")
    print(f"  Grouped Train trajectories: {len(grouped['train'])} across {grouped['metadata']['train_clusters']} clusters")
    print(f"  Grouped Valid trajectories: {len(grouped['valid'])} across {grouped['metadata']['valid_clusters']} clusters")
    print(f"  Grouped Test trajectories:  {len(grouped['test'])} across {grouped['metadata']['test_clusters']} clusters")

    # 3. Parameter Holdout Splits (Separated by physical mechanism: Re for Dynamics, Sc for Transport)
    available_res = {parse_shear_flow_filename(f)["re"] for f in rel_all_files}
    available_scs = {parse_shear_flow_filename(f)["sc"] for f in rel_all_files}

    # 3A. Reynolds Holdout Split (OOD Dynamics)
    actual_holdout_re = holdout_re if holdout_re in available_res else (max(available_res) if len(available_res) > 1 else None)
    if actual_holdout_re is not None:
        holdout_re_split = SplitManager.get_parameter_holdout_re_split(
            all_files=rel_all_files,
            holdout_re=actual_holdout_re,
            valid_ratio=0.1,
        )
    else:
        holdout_re_split = SplitManager.get_parameter_holdout_re_split(rel_all_files, holdout_re=1e5, valid_ratio=0.1)
    re_path = os.path.join(output_dir, "parameter_holdout_re.json")
    with open(re_path, "w") as f:
        json.dump(holdout_re_split, f, indent=2)
    print(f"Exported Re Parameter Holdout Split (Holdout Re={actual_holdout_re}) to: {re_path}")
    print(f"  Holdout Re Train: {len(holdout_re_split['train'])}, Valid: {len(holdout_re_split['valid'])}, Test: {len(holdout_re_split['test'])}")

    # 3B. Schmidt Holdout Split (OOD Scalar Transport)
    actual_holdout_sc = holdout_sc if holdout_sc in available_scs else (max(available_scs) if len(available_scs) > 1 else None)
    if actual_holdout_sc is not None:
        holdout_sc_split = SplitManager.get_parameter_holdout_sc_split(
            all_files=rel_all_files,
            holdout_sc=actual_holdout_sc,
            valid_ratio=0.1,
        )
    else:
        holdout_sc_split = SplitManager.get_parameter_holdout_sc_split(rel_all_files, holdout_sc=1.0, valid_ratio=0.1)
    sc_path = os.path.join(output_dir, "parameter_holdout_sc.json")
    with open(sc_path, "w") as f:
        json.dump(holdout_sc_split, f, indent=2)
    print(f"Exported Sc Parameter Holdout Split (Holdout Sc={actual_holdout_sc}) to: {sc_path}")
    print(f"  Holdout Sc Train: {len(holdout_sc_split['train'])}, Valid: {len(holdout_sc_split['valid'])}, Test: {len(holdout_sc_split['test'])}")

    # 3C. General Holdout Split (Backward compatibility)
    holdout = SplitManager.get_parameter_holdout_split(
        all_files=rel_all_files,
        holdout_re=actual_holdout_re if len(available_res) > 1 else None,
        holdout_sc=actual_holdout_sc,
        valid_ratio=0.1,
    )
    hold_path = os.path.join(output_dir, "parameter_holdout_split.json")
    with open(hold_path, "w") as f:
        json.dump(holdout, f, indent=2)
    print(f"Exported Combined Parameter Holdout Split to: {hold_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and export dataset splits.")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--output_dir", type=str, default="outputs/splits")
    parser.add_argument("--holdout_re", type=float, default=1e5)
    parser.add_argument("--holdout_sc", type=float, default=1.0)
    args = parser.parse_args()

    build_and_save_splits(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        holdout_re=args.holdout_re,
        holdout_sc=args.holdout_sc,
    )
