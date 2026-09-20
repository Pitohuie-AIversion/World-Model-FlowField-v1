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

    # 1. Official Split
    official = SplitManager.get_official_split(train_files, valid_files, test_files)
    off_path = os.path.join(output_dir, "official_split.json")
    with open(off_path, "w") as f:
        json.dump(official, f, indent=2)
    print(f"Exported Official Split to: {off_path}")

    # 2. Parameter Holdout Split
    holdout = SplitManager.get_parameter_holdout_split(
        all_files=all_files,
        holdout_re=holdout_re,
        holdout_sc=holdout_sc,
        valid_ratio=0.1,
    )
    hold_path = os.path.join(output_dir, "parameter_holdout_split.json")
    with open(hold_path, "w") as f:
        json.dump(holdout, f, indent=2)
    print(f"Exported Parameter Holdout Split (Holdout Re={holdout_re}, Sc={holdout_sc}) to: {hold_path}")
    print(f"  Holdout Train files: {len(holdout['train'])}")
    print(f"  Holdout Valid files: {len(holdout['valid'])}")
    print(f"  Holdout Test files:  {len(holdout['test'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and export dataset splits.")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--output_dir", type=str, default="outputs/splits")
    parser.add_argument("--holdout_re", type=float, default=1e5)
    parser.add_argument("--holdout_sc", type=float, default=10.0)
    args = parser.parse_args()

    build_and_save_splits(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        holdout_re=args.holdout_re,
        holdout_sc=args.holdout_sc,
    )
