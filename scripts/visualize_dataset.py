"""Comprehensive visualization of the shear_flow dataset.

Generates the following figures:
1. Field snapshots (u, v, p, s) at multiple timesteps for a sample trajectory
2. Channel statistics (mean, std, min, max) across time
3. Temporal evolution strip for each channel
4. Vorticity field derived from (u, v)
5. Energy spectrum (spatial FFT) at selected timesteps
6. Parameter space coverage and dataset split summary
"""

import os
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as ticker

# ── Config ──────────────────────────────────────────────────────────────────
DATA_ROOT = "/root/autodl-tmp/datasets/shear_flow/data"
OUT_DIR = "/root/mzy/Flow Field Prediction in World Models/World-Model-FlowField-v1/outputs/dataset_viz"
os.makedirs(OUT_DIR, exist_ok=True)

CHANNEL_NAMES = ["u (velocity-x)", "v (velocity-y)", "p (pressure)", "s (tracer)"]
CHANNEL_SHORT = ["u", "v", "p", "s"]
CHANNEL_CMAPS = ["RdBu_r", "RdBu_r", "viridis", "magma"]

# Nice plot style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def load_fields(h5path, sim_idx=0):
    """Load (u, v, p, s) from a single HDF5 file for one trajectory.
    
    Returns: np.ndarray of shape (T, 4, Ny, Nx), float32
    """
    with h5py.File(h5path, "r") as h5:
        vel = h5["t1_fields/velocity"][sim_idx]  # (T, Ny, Nx, 2)
        u = vel[..., 0].astype(np.float32)
        v = vel[..., 1].astype(np.float32)

        p = h5["t0_fields/pressure"][sim_idx].astype(np.float32)  # (T, Ny, Nx)
        s = h5["t0_fields/tracer"][sim_idx].astype(np.float32)

        time = h5["dimensions/time"][:].astype(np.float64)
        x = h5["dimensions/x"][:].astype(np.float32)
        y = h5["dimensions/y"][:].astype(np.float32)
        re = float(h5.attrs["Reynolds"])
        sc = float(h5.attrs["Schmidt"])

    fields = np.stack([u, v, p, s], axis=1)  # (T, 4, Ny, Nx)
    return fields, time, x, y, re, sc


# ── 1. Select a representative file ────────────────────────────────────────
train_file = os.path.join(DATA_ROOT, "train", "shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5")
valid_file_sc01 = os.path.join(DATA_ROOT, "valid", "shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5")
valid_file_sc10 = os.path.join(DATA_ROOT, "valid", "shear_flow_Reynolds_1e4_Schmidt_1e0.hdf5")

print("Loading train data ...")
fields, time_arr, x, y, re_val, sc_val = load_fields(train_file, sim_idx=0)
T, C, Ny, Nx = fields.shape
print(f"  Shape: T={T}, C={C}, Ny={Ny}, Nx={Nx}")
print(f"  Re={re_val}, Sc={sc_val}")
print(f"  Time range: [{time_arr[0]:.3f}, {time_arr[-1]:.3f}]")
print(f"  Spatial: x∈[{x.min():.2f}, {x.max():.2f}], y∈[{y.min():.2f}, {y.max():.2f}]")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Field snapshots at selected timesteps
# ══════════════════════════════════════════════════════════════════════════════
def plot_field_snapshots(fields, time_arr, x, y, re_val, sc_val,
                         timesteps=(0, 49, 99, 149, 199), save_name="field_snapshots.png"):
    """4 channels × N timesteps grid."""
    n_t = len(timesteps)
    fig, axes = plt.subplots(4, n_t, figsize=(4 * n_t, 3.5 * 4))

    for col, t_idx in enumerate(timesteps):
        for row in range(4):
            ax = axes[row, col]
            data = fields[t_idx, row]

            if row < 2:  # velocity: diverging colormap centered at 0
                vmax = max(abs(data.min()), abs(data.max()))
                norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                im = ax.imshow(data, origin="lower", cmap=CHANNEL_CMAPS[row],
                               norm=norm, extent=[x.min(), x.max(), y.min(), y.max()],
                               aspect="auto")
            else:
                im = ax.imshow(data, origin="lower", cmap=CHANNEL_CMAPS[row],
                               extent=[x.min(), x.max(), y.min(), y.max()],
                               aspect="auto")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if row == 0:
                ax.set_title(f"t = {time_arr[t_idx]:.1f}", fontweight="bold")
            if col == 0:
                ax.set_ylabel(CHANNEL_NAMES[row], fontweight="bold")
            else:
                ax.set_ylabel("")

            if row < 3:
                ax.set_xticklabels([])

    fig.suptitle(f"Shear Flow Field Snapshots — Re={re_val:.0f}, Sc={sc_val}", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, save_name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


print("\n[1/6] Generating field snapshots ...")
plot_field_snapshots(fields, time_arr, x, y, re_val, sc_val)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Per-channel statistics over time
# ══════════════════════════════════════════════════════════════════════════════
def plot_channel_statistics(fields, time_arr, save_name="channel_statistics.png"):
    """Mean, std, min, max per channel vs. time."""
    T, C, Ny, Nx = fields.shape
    means = fields.mean(axis=(2, 3))   # (T, C)
    stds  = fields.std(axis=(2, 3))
    mins  = fields.min(axis=(2, 3))
    maxs  = fields.max(axis=(2, 3))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]

    for ch in range(4):
        ax = axes[ch // 2, ch % 2]
        ax.fill_between(time_arr, mins[:, ch], maxs[:, ch], alpha=0.15, color=colors[ch], label="min–max")
        ax.fill_between(time_arr, means[:, ch] - stds[:, ch], means[:, ch] + stds[:, ch],
                        alpha=0.35, color=colors[ch], label="mean ± std")
        ax.plot(time_arr, means[:, ch], color=colors[ch], linewidth=1.8, label="mean")
        ax.set_title(CHANNEL_NAMES[ch], fontweight="bold")
        ax.set_xlabel("Time")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Channel Statistics Over Time", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, save_name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


print("[2/6] Generating channel statistics ...")
plot_channel_statistics(fields, time_arr)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Temporal evolution strips (Hovmöller-style)
# ══════════════════════════════════════════════════════════════════════════════
def plot_temporal_strips(fields, time_arr, y_coord, save_name="temporal_evolution.png"):
    """For each channel, show a time-y strip at mid-x location."""
    T, C, Ny, Nx = fields.shape
    mid_x = Nx // 2

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

    for ch in range(4):
        ax = axes[ch]
        strip = fields[:, ch, :, mid_x]  # (T, Ny)
        
        if ch < 2:
            vmax = max(abs(strip.min()), abs(strip.max()))
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(strip.T, origin="lower", aspect="auto", cmap=CHANNEL_CMAPS[ch],
                           norm=norm,
                           extent=[time_arr[0], time_arr[-1], y_coord.min(), y_coord.max()])
        else:
            im = ax.imshow(strip.T, origin="lower", aspect="auto", cmap=CHANNEL_CMAPS[ch],
                           extent=[time_arr[0], time_arr[-1], y_coord.min(), y_coord.max()])

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(CHANNEL_NAMES[ch], fontweight="bold")
        ax.set_xlabel("Time")
        if ch == 0:
            ax.set_ylabel("y")

    fig.suptitle(f"Temporal Evolution (Hovmöller at x = mid)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, save_name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


print("[3/6] Generating temporal evolution strips ...")
plot_temporal_strips(fields, time_arr, y)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Vorticity field
# ══════════════════════════════════════════════════════════════════════════════
def plot_vorticity(fields, time_arr, x, y, timesteps=(0, 49, 99, 149, 199),
                   save_name="vorticity.png"):
    """Compute and visualize ω_z = ∂v/∂x − ∂u/∂y."""
    dx = (x[-1] - x[0]) / (len(x) - 1)
    dy = (y[-1] - y[0]) / (len(y) - 1)

    n_t = len(timesteps)
    fig, axes = plt.subplots(1, n_t, figsize=(4.5 * n_t, 4))
    if n_t == 1:
        axes = [axes]

    for i, t_idx in enumerate(timesteps):
        u_field = fields[t_idx, 0]
        v_field = fields[t_idx, 1]
        dvdx = np.gradient(v_field, dx, axis=1)
        dudy = np.gradient(u_field, dy, axis=0)
        omega = dvdx - dudy

        vmax = np.percentile(np.abs(omega), 99)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = axes[i].imshow(omega, origin="lower", cmap="RdBu_r", norm=norm,
                            extent=[x.min(), x.max(), y.min(), y.max()], aspect="auto")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        axes[i].set_title(f"t = {time_arr[t_idx]:.1f}", fontweight="bold")
        if i == 0:
            axes[i].set_ylabel("y")
        axes[i].set_xlabel("x")

    fig.suptitle("Vorticity ωz = ∂v/∂x − ∂u/∂y", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, save_name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


print("[4/6] Generating vorticity fields ...")
plot_vorticity(fields, time_arr, x, y)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: Energy spectrum (2D spatial FFT)
# ══════════════════════════════════════════════════════════════════════════════
def plot_energy_spectrum(fields, x, y, time_arr, timesteps=(0, 99, 199),
                         save_name="energy_spectrum.png"):
    """1D radially averaged energy spectrum E(k) from 2D velocity FFT."""
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]
    Ny, Nx = fields.shape[2], fields.shape[3]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors_t = plt.cm.plasma(np.linspace(0.15, 0.85, len(timesteps)))

    for ci, t_idx in enumerate(timesteps):
        u = fields[t_idx, 0]
        v = fields[t_idx, 1]

        # 2D FFT of velocity components
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)

        # Energy = |u_hat|^2 + |v_hat|^2
        energy_2d = (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) / (Nx * Ny) ** 2

        # Wavenumber grids
        kx = np.fft.fftfreq(Nx, d=Lx / Nx) * 2 * np.pi
        ky = np.fft.fftfreq(Ny, d=Ly / Ny) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX ** 2 + KY ** 2)

        # Radial binning
        k_max = min(kx.max(), ky.max())
        k_bins = np.linspace(0, k_max, 80)
        E_k = np.zeros(len(k_bins) - 1)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

        for bi in range(len(k_bins) - 1):
            mask = (K >= k_bins[bi]) & (K < k_bins[bi + 1])
            if mask.any():
                E_k[bi] = energy_2d[mask].sum()

        # Filter out zero bins
        valid = E_k > 0
        ax.loglog(k_centers[valid], E_k[valid], color=colors_t[ci], linewidth=1.5,
                  label=f"t = {time_arr[t_idx]:.1f}")

    # Reference slope
    k_ref = np.logspace(1.0, 2.2, 50)
    ax.loglog(k_ref, 1e-2 * k_ref ** (-5 / 3), "k--", alpha=0.5, label=r"$k^{-5/3}$")

    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title("Kinetic Energy Spectrum (Radially Averaged)", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, save_name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


print("[5/6] Generating energy spectrum ...")
plot_energy_spectrum(fields, x, y, time_arr)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6: Dataset summary & parameter space
# ══════════════════════════════════════════════════════════════════════════════
def plot_dataset_summary(save_name="dataset_summary.png"):
    """Show parameter coverage, trajectory counts, and data shapes across splits."""
    splits_info = []
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(DATA_ROOT, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".hdf5"):
                continue
            fpath = os.path.join(split_dir, fname)
            with h5py.File(fpath, "r") as h5:
                re = float(h5.attrs["Reynolds"])
                sc = float(h5.attrs["Schmidt"])
                n_traj = int(h5.attrs["n_trajectories"])
                T = h5["t0_fields/pressure"].shape[1]
                Ny, Nx = h5["t0_fields/pressure"].shape[2], h5["t0_fields/pressure"].shape[3]
            splits_info.append({
                "split": split, "file": fname, "Re": re, "Sc": sc,
                "n_traj": n_traj, "T": T, "Ny": Ny, "Nx": Nx,
            })

    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 2, width_ratios=[1, 1.6], wspace=0.3)

    # Left: Parameter space scatter
    ax1 = fig.add_subplot(gs[0])
    split_colors = {"train": "#2A9D8F", "valid": "#E9C46A", "test": "#E63946"}
    split_markers = {"train": "o", "valid": "s", "test": "^"}

    for info in splits_info:
        ax1.scatter(info["Re"], info["Sc"],
                    c=split_colors[info["split"]], marker=split_markers[info["split"]],
                    s=info["n_traj"] * 15 + 50, alpha=0.85, edgecolors="k", linewidths=0.5,
                    label=info["split"] if info["split"] not in [i["split"] for i in splits_info[:splits_info.index(info)]] else "")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Reynolds Number", fontweight="bold")
    ax1.set_ylabel("Schmidt Number", fontweight="bold")
    ax1.set_title("Parameter Space Coverage", fontweight="bold")
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")

    # Right: Summary table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")

    col_labels = ["Split", "File", "Re", "Sc", "#Traj", "T", "Ny×Nx"]
    cell_text = []
    for info in splits_info:
        cell_text.append([
            info["split"],
            info["file"].replace("shear_flow_", "").replace(".hdf5", ""),
            f"{info['Re']:.0e}",
            f"{info['Sc']:.1g}",
            str(info["n_traj"]),
            str(info["T"]),
            f"{info['Ny']}×{info['Nx']}",
        ])

    table = ax2.table(cellText=cell_text, colLabels=col_labels, loc="center",
                      cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Color header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#264653")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Color rows by split
    for i, info in enumerate(splits_info):
        for j in range(len(col_labels)):
            table[(i + 1, j)].set_facecolor(split_colors[info["split"]] + "30")

    ax2.set_title("Dataset Files Summary", fontweight="bold", pad=20)

    fig.suptitle("Shear Flow Dataset Overview", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, save_name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


print("[6/6] Generating dataset summary ...")
plot_dataset_summary()


# ══════════════════════════════════════════════════════════════════════════════
# Bonus: Compare Schmidt=0.1 vs Schmidt=1.0 tracer fields
# ══════════════════════════════════════════════════════════════════════════════
def plot_schmidt_comparison(save_name="schmidt_comparison.png"):
    """Side-by-side tracer fields for different Schmidt numbers."""
    fields_01, time_01, x_01, y_01, re_01, sc_01 = load_fields(valid_file_sc01, sim_idx=0)
    fields_10, time_10, x_10, y_10, re_10, sc_10 = load_fields(valid_file_sc10, sim_idx=0)

    timesteps = [0, 49, 99, 199]
    fig, axes = plt.subplots(2, len(timesteps), figsize=(4.5 * len(timesteps), 7))

    for col, t_idx in enumerate(timesteps):
        for row, (fld, sc_label) in enumerate([(fields_01, f"Sc={sc_01}"), (fields_10, f"Sc={sc_10}")]):
            ax = axes[row, col]
            tracer = fld[t_idx, 3]  # tracer channel
            im = ax.imshow(tracer, origin="lower", cmap="magma",
                           extent=[x_01.min(), x_01.max(), y_01.min(), y_01.max()],
                           aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(f"t = {time_01[t_idx]:.1f}", fontweight="bold")
            if col == 0:
                ax.set_ylabel(sc_label, fontweight="bold", fontsize=13)

    fig.suptitle("Tracer Field Comparison: Schmidt Number Effect (Re=10000)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, save_name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


print("\n[Bonus] Generating Schmidt number comparison ...")
plot_schmidt_comparison()


# ══════════════════════════════════════════════════════════════════════════════
# Bonus 2: Multi-trajectory ensemble at single timestep
# ══════════════════════════════════════════════════════════════════════════════
def plot_multi_trajectory(save_name="multi_trajectory.png"):
    """Show 8 different initial conditions' tracer field at t=100."""
    n_show = 8
    t_idx = 99

    fig, axes = plt.subplots(2, 4, figsize=(18, 7))

    with h5py.File(train_file, "r") as h5:
        tracer = h5["t0_fields/tracer"][:n_show, t_idx]  # (8, Ny, Nx)

    for i in range(n_show):
        ax = axes[i // 4, i % 4]
        im = ax.imshow(tracer[i], origin="lower", cmap="magma",
                       extent=[x.min(), x.max(), y.min(), y.max()], aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Trajectory #{i}", fontweight="bold")

    fig.suptitle(f"Tracer Field at t = {time_arr[t_idx]:.1f} — 8 Trajectories (Re=10000, Sc=0.1)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, save_name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


print("[Bonus 2] Generating multi-trajectory overview ...")
plot_multi_trajectory()

print(f"\n✅ All visualizations saved to: {OUT_DIR}")
