"""Visualization tool for 2D flow fields [u, v, p, s], vorticity, and kinetic energy spectra."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
from src.data.shear_flow_dataset import ShearFlowDataset
from src.metrics.spectral import compute_radial_energy_spectrum
from src.utils.fft_derivatives import compute_vorticity


def plot_flow_state(
    sample_dict: dict,
    time_idx: int = 0,
    save_path: str = "outputs/figures/flow_state_t0.png",
):
    """Plot u, v, p, s, vorticity omega, and radial energy spectrum."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # history shape: (L, 4, Ny, Nx)
    fields = sample_dict["history"][time_idx]  # (4, Ny, Nx)
    u = fields[0]
    v = fields[1]
    p = fields[2]
    s = fields[3]

    # Compute vorticity
    omega = compute_vorticity(u.unsqueeze(0), v.unsqueeze(0), domain_size=(1.0, 1.0))[0]

    # Compute spectrum
    k_bins, e_k = compute_radial_energy_spectrum(u, v, domain_size=(1.0, 1.0))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150)

    # 1. Streamwise velocity u
    im0 = axes[0, 0].imshow(u.numpy(), cmap="RdBu_r", origin="lower", aspect="auto")
    axes[0, 0].set_title(f"Streamwise Velocity u (range: [{u.min():.2f}, {u.max():.2f}])")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 2. Cross-stream velocity v
    im1 = axes[0, 1].imshow(v.numpy(), cmap="RdBu_r", origin="lower", aspect="auto")
    axes[0, 1].set_title(f"Cross-stream Velocity v (range: [{v.min():.3f}, {v.max():.3f}])")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. Pressure p
    im2 = axes[0, 2].imshow(p.numpy(), cmap="viridis", origin="lower", aspect="auto")
    axes[0, 2].set_title(f"Pressure p (range: [{p.min():.4f}, {p.max():.4f}])")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 4. Passive Tracer s
    im3 = axes[1, 0].imshow(s.numpy(), cmap="inferno", origin="lower", aspect="auto")
    axes[1, 0].set_title(f"Passive Tracer s (range: [{s.min():.2f}, {s.max():.2f}])")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 5. Vorticity omega
    im4 = axes[1, 1].imshow(omega.numpy(), cmap="seismic", origin="lower", aspect="auto")
    axes[1, 1].set_title(f"Vorticity $\\omega = \\partial_x v - \\partial_y u$")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 6. Radial Energy Spectrum E(k)
    k_np = k_bins.cpu().numpy()
    e_np = e_k.cpu().numpy()
    valid = (k_np > 0) & (e_np > 0)
    axes[1, 2].loglog(k_np[valid], e_np[valid], "b-", lw=2, label="$E(k)$")
    # Reference Kolmogorov -5/3 or 2D turbulence -3 slope
    if valid.sum() > 5:
        ref_k = k_np[valid][1:15]
        ref_e = ref_k ** (-3.0) * (e_np[valid][1] / (ref_k[0] ** -3.0))
        axes[1, 2].loglog(ref_k, ref_e, "k--", alpha=0.7, label="$k^{-3}$ (2D Enstrophy Cascade)")
    axes[1, 2].set_title("Kinetic Energy Spectrum $E(k)$")
    axes[1, 2].set_xlabel("Wavenumber $k$")
    axes[1, 2].set_ylabel("Energy Density")
    axes[1, 2].legend()
    axes[1, 2].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved flow state visualization to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="/root/autodl-tmp/datasets/shear_flow/data/valid/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5",
    )
    parser.add_argument("--save_path", type=str, default="outputs/figures/flow_state_real.png")
    parser.add_argument("--time_idx", type=int, default=0)
    args = parser.parse_args()

    if os.path.exists(args.file):
        ds = ShearFlowDataset([args.file], history_length=1, horizon=1)
        sample = ds[0]
        plot_flow_state(sample, time_idx=0, save_path=args.save_path)
    else:
        print(f"File not found: {args.file}")
