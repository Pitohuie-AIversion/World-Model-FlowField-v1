"""Physical Consistency Verification: Velocity/Pressure invariance across different Schmidt numbers.

Physical Principle:
In incompressible Navier-Stokes with passive scalar transport, the tracer s
does not exert buoyancy or feedback on the velocity field.
Therefore, for identical initial conditions and Reynolds number, trajectories
with different Schmidt numbers (e.g. Sc=0.1 vs Sc=1.0) MUST have IDENTICAL
velocity fields (u, v) and pressure fields (p) within numerical precision,
while the passive tracer (s) field MUST evolve differently due to distinct diffusion coefficients.
"""

import argparse
import os
import sys
import h5py
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def verify_schmidt_invariance(
    file_sc01: str = "/root/autodl-tmp/datasets/shear_flow/data/valid/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5",
    file_sc10: str = "/root/autodl-tmp/datasets/shear_flow/data/valid/shear_flow_Reynolds_1e4_Schmidt_1e0.hdf5",
    traj_idx: int = 0,
):
    print("=" * 80)
    print("PHYSICAL CONSISTENCY VERIFICATION: SCHMIDT NUMBER INVARIANCE AUDIT")
    print("=" * 80)
    print(f"File A (Sc=0.1): {file_sc01}")
    print(f"File B (Sc=1.0): {file_sc10}")
    print(f"Auditing Trajectory Index: {traj_idx}")

    if not os.path.exists(file_sc01) or not os.path.exists(file_sc10):
        print(f"Error: One or both files do not exist.")
        return

    with h5py.File(file_sc01, "r") as h5_a, h5py.File(file_sc10, "r") as h5_b:
        u_a = h5_a["t1_fields/velocity"][traj_idx, :, :, :, 0]
        v_a = h5_a["t1_fields/velocity"][traj_idx, :, :, :, 1]
        p_a = h5_a["t0_fields/pressure"][traj_idx, :, :, :]
        s_a = h5_a["t0_fields/tracer"][traj_idx, :, :, :]

        u_b = h5_b["t1_fields/velocity"][traj_idx, :, :, :, 0]
        v_b = h5_b["t1_fields/velocity"][traj_idx, :, :, :, 1]
        p_b = h5_b["t0_fields/pressure"][traj_idx, :, :, :]
        s_b = h5_b["t0_fields/tracer"][traj_idx, :, :, :]

    # Compute differences
    diff_u_max = np.max(np.abs(u_a - u_b))
    diff_u_mean = np.mean(np.abs(u_a - u_b))

    diff_v_max = np.max(np.abs(v_a - v_b))
    diff_v_mean = np.mean(np.abs(v_a - v_b))

    diff_p_max = np.max(np.abs(p_a - p_b))
    diff_p_mean = np.mean(np.abs(p_a - p_b))

    diff_s_max = np.max(np.abs(s_a - s_b))
    diff_s_mean = np.mean(np.abs(s_a - s_b))

    # Initial condition check at t=0
    diff_u0 = np.max(np.abs(u_a[0] - u_b[0]))
    diff_v0 = np.max(np.abs(v_a[0] - v_b[0]))
    diff_p0 = np.max(np.abs(p_a[0] - p_b[0]))
    diff_s0 = np.max(np.abs(s_a[0] - s_b[0]))

    print("\n1. Initial Condition (t=0) Consistency:")
    print(f"   Max |u_A - u_B| at t=0: {diff_u0:.6e}")
    print(f"   Max |v_A - v_B| at t=0: {diff_v0:.6e}")
    print(f"   Max |p_A - p_B| at t=0: {diff_p0:.6e}")
    print(f"   Max |s_A - s_B| at t=0: {diff_s0:.6e}")

    print("\n2. Full Trajectory (200 Time Steps) Difference:")
    print(f"   Streamwise Velocity u:  Max Diff = {diff_u_max:.6e}, Mean Diff = {diff_u_mean:.6e}")
    print(f"   Cross-stream Velocity v: Max Diff = {diff_v_max:.6e}, Mean Diff = {diff_v_mean:.6e}")
    print(f"   Pressure Field p:        Max Diff = {diff_p_max:.6e}, Mean Diff = {diff_p_mean:.6e}")
    print(f"   Passive Tracer s:        Max Diff = {diff_s_max:.6e}, Mean Diff = {diff_s_mean:.6e}")

    print("\n3. Physics Verification Conclusion:")
    is_uvp_identical = (diff_u_max < 1e-5) and (diff_v_max < 1e-5) and (diff_p_max < 1e-5)
    is_s_distinct = diff_s_max > 1e-2

    if is_uvp_identical:
        print("   [PASS] u, v, p fields are STRICTLY IDENTICAL across different Sc numbers.")
    else:
        print(f"   [FAIL/DEVIATION] u, v, p fields differ across Sc (max diff={max(diff_u_max, diff_v_max, diff_p_max):.6e}).")

    if is_s_distinct:
        print(f"   [PASS] Tracer field s exhibits expected distinct physical diffusion dynamics (Max Diff = {diff_s_max:.4f}).")
    else:
        print("   [WARNING] Tracer field s is identical, Schmidt number had no effect.")

    return {
        "diff_u_max": float(diff_u_max),
        "diff_v_max": float(diff_v_max),
        "diff_p_max": float(diff_p_max),
        "diff_s_max": float(diff_s_max),
        "is_uvp_identical": bool(is_uvp_identical),
        "is_s_distinct": bool(is_s_distinct),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_sc01", type=str, default="/root/autodl-tmp/datasets/shear_flow/data/valid/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5")
    parser.add_argument("--file_sc10", type=str, default="/root/autodl-tmp/datasets/shear_flow/data/valid/shear_flow_Reynolds_1e4_Schmidt_1e0.hdf5")
    parser.add_argument("--traj_idx", type=int, default=0)
    args = parser.parse_args()

    verify_schmidt_invariance(args.file_sc01, args.file_sc10, args.traj_idx)
