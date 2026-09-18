"""Script to download selected subsets of The Well's shear_flow dataset from Hugging Face.

Prevents disk exhaustion on /root/autodl-tmp by selectively downloading targeted
(Reynolds, Schmidt) combinations instead of the entire 547 GB repository.
"""

import argparse
import os
import sys

# Ensure local project root is at the very front of sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from huggingface_hub import HfApi, hf_hub_download
from src.data.splits import parse_shear_flow_filename


def download_shear_flow_subset(
    target_dir: str = "/root/autodl-tmp/datasets/shear_flow",
    splits: tuple = ("valid", "test", "train"),
    reynolds_subset: tuple = (1e4, 1e5),
    schmidt_subset: tuple = (0.1, 1.0, 5.0, 10.0),
    single_sample: bool = False,
):
    """Download matching HDF5 files from polymathic-ai/shear_flow."""
    os.makedirs(target_dir, exist_ok=True)
    repo_id = "polymathic-ai/shear_flow"
    api = HfApi()

    print(f"Querying repository files for {repo_id}...")
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    hdf5_files = [f for f in all_files if f.endswith(".hdf5")]
    print(f"Found {len(hdf5_files)} HDF5 files in repo.")

    matching_files = []
    for f in hdf5_files:
        parts = f.split("/")
        if len(parts) < 3:
            continue
        split_name = parts[1]
        filename = parts[2]

        if split_name not in splits:
            continue

        try:
            params = parse_shear_flow_filename(filename)
        except ValueError:
            continue

        # Check filter
        match_re = any(abs(params["re"] - target_re) / target_re < 1e-4 for target_re in reynolds_subset)
        match_sc = any(abs(params["sc"] - target_sc) / max(target_sc, 1e-4) < 1e-4 for target_sc in schmidt_subset)

        if match_re and match_sc:
            matching_files.append((split_name, filename, f))

    print(f"Matched {len(matching_files)} files corresponding to parameter subset.")

    if single_sample and matching_files:
        # Pick the smallest validation sample for immediate audit
        matching_files = [f for f in matching_files if f[0] == "valid"][:1]
        print(f"Single sample mode selected: {matching_files[0][1]}")

    for idx, (split_name, filename, remote_path) in enumerate(matching_files, 1):
        dest_dir = os.path.join(target_dir, split_name)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)

        if os.path.exists(dest_path):
            print(f"[{idx}/{len(matching_files)}] File already exists: {dest_path}")
            continue

        print(f"[{idx}/{len(matching_files)}] Downloading {remote_path} to {dest_path}...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Successfully downloaded to: {downloaded}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download The Well shear_flow subset.")
    parser.add_argument("--target_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--single_sample", action="store_true", help="Download just one validation file for audit.")
    parser.add_argument("--splits", nargs="+", default=["valid", "test", "train"])
    args = parser.parse_args()

    download_shear_flow_subset(
        target_dir=args.target_dir,
        splits=tuple(args.splits),
        single_sample=args.single_sample,
    )
