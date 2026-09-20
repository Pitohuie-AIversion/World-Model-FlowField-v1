"""Script to inspect and audit actual HDF5 data files from The Well shear_flow.

Prints complete structural metadata, tensor shapes, dtypes, coordinate ranges,
channel statistics (mean, std, min, max), and verifies incompressibility.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import h5py
import numpy as np
import torch
from src.utils.fft_derivatives import compute_divergence, compute_vorticity


def audit_hdf5_file(file_path: str):
    """Perform comprehensive data audit on a single shear_flow HDF5 file."""
    print("=" * 80)
    print(f"AUDITING DATA FILE: {file_path}")
    print("=" * 80)

    if not os.path.exists(file_path):
        print(f"Error: File does not exist: {file_path}")
        return

    file_size_mb = os.path.getsize(file_path) / (1024**2)
    print(f"File size: {file_size_mb:.2f} MB")

    with h5py.File(file_path, "r") as h5:
        print("\n--- HDF5 Root Attributes ---")
        for k, v in h5.attrs.items():
            print(f"  {k}: {v}")

        print("\n--- Internal Datasets and Shapes ---")

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: '{name}', Shape: {obj.shape}, Dtype: {obj.dtype}, Chunks: {obj.chunks}")
                for ak, av in obj.attrs.items():
                    print(f"    Attr: {ak} = {av}")

        h5.visititems(visitor)

        # Field analysis
        vel_ds = h5.get("t1_fields/velocity") or h5.get("velocity")
        p_ds = h5.get("t0_fields/pressure") or h5.get("pressure")
        s_ds = h5.get("t0_fields/tracer") or h5.get("tracer")

        print("\n--- Physical Fields Analysis ---")
        if vel_ds is not None:
            sample_vel = np.asarray(vel_ds[0, 0], dtype=np.float32)  # First sim, t=0
            print(f"Sample velocity shape at (sim=0, t=0): {sample_vel.shape}")
            if sample_vel.shape[-1] == 2:
                u = sample_vel[..., 0]
                v = sample_vel[..., 1]
            else:
                u = sample_vel[0]
                v = sample_vel[1]
            print(f"  u: min={u.min():.4e}, max={u.max():.4e}, mean={u.mean():.4e}, std={u.std():.4e}")
            print(f"  v: min={v.min():.4e}, max={v.max():.4e}, mean={v.mean():.4e}, std={v.std():.4e}")

            # Divergence check on t=0
            u_t = torch.from_numpy(u).unsqueeze(0)
            v_t = torch.from_numpy(v).unsqueeze(0)
            div = compute_divergence(u_t, v_t, domain_size=(2.0, 1.0))
            max_div = torch.max(torch.abs(div)).item()
            mean_div = torch.mean(torch.abs(div)).item()
            print(f"  Divergence check at t=0: max |div| = {max_div:.4e}, mean |div| = {mean_div:.4e}")

        if p_ds is not None:
            sample_p = np.asarray(p_ds[0, 0], dtype=np.float32)
            print(f"Sample pressure shape at (sim=0, t=0): {sample_p.shape}")
            print(f"  p: min={sample_p.min():.4e}, max={sample_p.max():.4e}, mean={sample_p.mean():.4e}, std={sample_p.std():.4e}")

        if s_ds is not None:
            sample_s = np.asarray(s_ds[0, 0], dtype=np.float32)
            print(f"Sample tracer shape at (sim=0, t=0): {sample_s.shape}")
            print(f"  s: min={sample_s.min():.4e}, max={sample_s.max():.4e}, mean={sample_s.mean():.4e}, std={sample_s.std():.4e}")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect HDF5 dataset structure.")
    parser.add_argument("file_path", type=str, help="Path to HDF5 file.")
    args = parser.parse_args()

    audit_hdf5_file(args.file_path)
