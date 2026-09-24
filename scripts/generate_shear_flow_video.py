"""High-Fidelity Temporal Video Generator for Periodic Shear Flow DNS.

Renders full-trajectory temporal animations (t=0 to t=199) capturing
the complete physical lifecycle of the Kelvin-Helmholtz shear layer:
laminar shear -> perturbation growth -> vortex roll-up & pairing -> turbulent dissipation.

Supports:
- Quad-panel layout: Vorticity omega, Streamwise u, Passive Tracer s, Velocity Magnitude |U|
- Triple-panel layout: Vorticity omega, Streamwise u, Passive Tracer s
- Single-panel focus: High-resolution focus on selected field (e.g. Vorticity)
- Fluid dynamics conventions: Fixed colorbars, physical domain (1.0, 2.0), x-streamwise / y-cross-stream
- In-memory pipe streaming to FFmpeg (H.264 / yuv420p) with fallback to OpenCV VideoWriter
- Companion metadata JSON generation for provenance and reproducibility
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.splits import parse_shear_flow_filename
from src.utils.fft_derivatives import compute_vorticity
from src.utils.physics_contract import SHEAR_FLOW_DOMAIN_SIZE_XY
from src.utils.provenance import compute_file_sha256, get_git_commit


FIELD_CONFIGS = {
    "vorticity": {
        "title": r"Vorticity $\omega = \partial_x v - \partial_y u$",
        "cmap": "seismic",
        "vmin": -2.0,
        "vmax": 2.0,
        "cbar_label": r"$\omega$ ($s^{-1}$)",
    },
    "u": {
        "title": "Streamwise Velocity $u$",
        "cmap": "RdBu_r",
        "vmin": -0.45,
        "vmax": 0.50,
        "cbar_label": "$u$ (m/s)",
    },
    "tracer": {
        "title": "Passive Tracer $s$",
        "cmap": "inferno",
        "vmin": -0.35,
        "vmax": 0.45,
        "cbar_label": "Concentration $s$",
    },
    "speed": {
        "title": r"Velocity Magnitude $|U| = \sqrt{u^2 + v^2}$",
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 0.50,
        "cbar_label": "$|U|$ (m/s)",
    },
    "v": {
        "title": "Cross-Stream Velocity $v$",
        "cmap": "PiYG",
        "vmin": -0.12,
        "vmax": 0.12,
        "cbar_label": "$v$ (m/s)",
    },
    "pressure": {
        "title": "Gauge Pressure $p$",
        "cmap": "coolwarm",
        "vmin": -0.025,
        "vmax": 0.020,
        "cbar_label": "$p$ (Pa)",
    },
}


def load_trajectory_data(
    hdf5_path: str,
    sim_idx: int = 0,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Load physical states from HDF5 file for the specified simulation index."""
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 dataset not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        # Check velocity
        vel_ds = f["t1_fields/velocity"] if "t1_fields/velocity" in f else f["velocity"]
        tracer_ds = f["t0_fields/tracer"] if "t0_fields/tracer" in f else f["tracer"]
        pressure_ds = f["t0_fields/pressure"] if "t0_fields/pressure" in f else f["pressure"]

        n_sims = vel_ds.shape[0]
        if sim_idx < 0 or sim_idx >= n_sims:
            raise ValueError(f"Requested sim_idx={sim_idx} out of range [0, {n_sims - 1}] in {hdf5_path}")

        total_timesteps = vel_ds.shape[1]
        n_frames = total_timesteps if max_frames is None else min(max_frames, total_timesteps)

        # Slice data: (n_frames, 256, 512, 2)
        vel = np.asarray(vel_ds[sim_idx, :n_frames], dtype=np.float32)
        tracer = np.asarray(tracer_ds[sim_idx, :n_frames], dtype=np.float32)
        pressure = np.asarray(pressure_ds[sim_idx, :n_frames], dtype=np.float32)

        # Time coordinate
        time_coords = np.asarray(f["dimensions/time"][:n_frames], dtype=np.float32) if "dimensions/time" in f else np.arange(n_frames, dtype=np.float32)

        # Physical parameters
        re = float(f["scalars/Reynolds"][()]) if "scalars/Reynolds" in f else 10000.0
        sc = float(f["scalars/Schmidt"][()]) if "scalars/Schmidt" in f else 0.1

    u = vel[..., 0]  # (n_frames, Nx, Ny)
    v = vel[..., 1]  # (n_frames, Nx, Ny)

    # Compute vorticity using spectral derivatives
    # Convert to torch tensor: (n_frames, 1, Nx, Ny)
    u_torch = torch.from_numpy(u).unsqueeze(1)
    v_torch = torch.from_numpy(v).unsqueeze(1)
    omega_torch = compute_vorticity(u_torch, v_torch, domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)
    omega = omega_torch.squeeze(1).numpy()  # (n_frames, Nx, Ny)

    speed = np.sqrt(u ** 2 + v ** 2)

    return {
        "n_frames": n_frames,
        "u": u,
        "v": v,
        "tracer": tracer,
        "pressure": pressure,
        "vorticity": omega,
        "speed": speed,
        "time_coords": time_coords,
        "re": re,
        "sc": sc,
        "sim_idx": sim_idx,
        "hdf5_path": hdf5_path,
        "shape": u.shape[1:],  # (Nx, Ny)
    }


def _create_ffmpeg_writer(output_path: str, width: int, height: int, fps: int) -> subprocess.Popen:
    """Create an FFmpeg subprocess that accepts raw RGBA frames via stdin."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgba",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def render_shear_flow_video(
    hdf5_path: str,
    output_video: str,
    sim_idx: int = 0,
    fps: int = 20,
    layout: str = "quad",
    max_frames: Optional[int] = None,
    dpi: int = 150,
    export_gif: bool = True,
) -> Dict[str, Any]:
    """Render full-trajectory shear flow video and accompanying metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
    ffmpeg_available = shutil.which("ffmpeg") is not None

    print(f"Loading trajectory data: sim_idx={sim_idx} from {hdf5_path}...")
    data = load_trajectory_data(hdf5_path, sim_idx=sim_idx, max_frames=max_frames)
    n_frames = data["n_frames"]
    re = data["re"]
    sc = data["sc"]
    time_coords = data["time_coords"]

    # Setup layout panels
    if layout == "quad":
        panel_keys = ["vorticity", "u", "tracer", "speed"]
        fig, axes_grid = plt.subplots(2, 2, figsize=(16, 9), dpi=dpi)
        axes_list = [axes_grid[0, 0], axes_grid[0, 1], axes_grid[1, 0], axes_grid[1, 1]]
    elif layout == "triple":
        panel_keys = ["vorticity", "u", "tracer"]
        fig, axes_list = plt.subplots(1, 3, figsize=(18, 5.5), dpi=dpi)
    elif layout in FIELD_CONFIGS:
        panel_keys = [layout]
        fig, ax = plt.subplots(1, 1, figsize=(12, 6.5), dpi=dpi)
        axes_list = [ax]
    else:
        raise ValueError(f"Unknown layout: {layout}. Choose 'quad', 'triple', or one of {list(FIELD_CONFIGS.keys())}")

    ims = []
    cbars = []
    for key, ax in zip(panel_keys, axes_list):
        cfg = FIELD_CONFIGS[key]
        # Frame 0 initial plot
        # Note: Transpose field so that x is horizontal (dim-2) and y is vertical (dim-1)
        initial_frame = data[key][0].T  # (Ny, Nx)
        im = ax.imshow(
            initial_frame,
            origin="lower",
            cmap=cfg["cmap"],
            vmin=cfg["vmin"],
            vmax=cfg["vmax"],
            aspect="auto",
        )
        ax.set_title(cfg["title"], fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Streamwise Coordinate $x$ ($L_x = 1.0$)")
        ax.set_ylabel("Cross-Stream Coordinate $y$ ($L_y = 2.0$)")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cfg["cbar_label"], fontsize=10)
        ims.append((key, im))
        cbars.append(cbar)

    # Super title and dynamic time annotation
    suptitle_obj = fig.suptitle(
        f"Periodic Shear Flow Direct Numerical Simulation (DNS) | $Re = {re:g}, Sc = {sc:g}$ | Trajectory #{sim_idx}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    time_text = fig.text(
        0.5,
        0.02,
        f"Frame: t = 000 / {n_frames - 1:03d}  |  Physical Time: {time_coords[0]:.2f} s  |  Status: Initial Perturbation",
        ha="center",
        fontsize=11,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#cccccc", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    height, width, _ = buf.shape

    # Initialize video output pipe or writer
    ffmpeg_proc = None
    cv2_writer = None

    if ffmpeg_available:
        ffmpeg_proc = _create_ffmpeg_writer(output_video, width, height, fps)
    else:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cv2_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"Rendering {n_frames} frames ({width}x{height} @ {fps} fps)...")
    try:
        for t in range(n_frames):
            # Update each panel image data
            for key, im in ims:
                frame_data = data[key][t].T
                im.set_data(frame_data)

            # Update status text
            if t < 30:
                stage = "Initial Perturbation Growth"
            elif t < 70:
                stage = "Kelvin-Helmholtz Shear Roll-Up"
            elif t < 120:
                stage = "Vortex Pairing & Core Interaction"
            else:
                stage = "Turbulent Dissipation & Entrainment"

            t_val = time_coords[t]
            time_text.set_text(
                f"Frame: t = {t:03d} / {n_frames - 1:03d}  |  Physical Time: {t_val:.2f} s  |  Stage: {stage}"
            )

            fig.canvas.draw()
            frame_buf = np.asarray(fig.canvas.buffer_rgba())

            if ffmpeg_proc is not None:
                ffmpeg_proc.stdin.write(frame_buf.tobytes())
            else:
                import cv2
                bgr_frame = cv2.cvtColor(frame_buf, cv2.COLOR_RGBA2BGR)
                cv2_writer.write(bgr_frame)

            if (t + 1) % 50 == 0 or (t + 1) == n_frames:
                print(f"  Processed [{t + 1}/{n_frames}] frames ({(t + 1) / n_frames * 100:.1f}%)")

    finally:
        plt.close(fig)
        if ffmpeg_proc is not None:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
            if ffmpeg_proc.returncode != 0:
                err = ffmpeg_proc.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"FFmpeg failed with return code {ffmpeg_proc.returncode}:\n{err}")
        elif cv2_writer is not None:
            cv2_writer.release()

    video_size_bytes = os.path.getsize(output_video)
    duration_sec = n_frames / float(fps)
    print(f"Video saved successfully: {output_video} ({video_size_bytes / 1024 / 1024:.2f} MB, {duration_sec:.1f} s)")

    # Optional GIF export for instant markdown preview
    gif_path = None
    if export_gif and ffmpeg_available:
        gif_path = output_video.rsplit(".", 1)[0] + ".gif"
        print(f"Generating optimized preview GIF: {gif_path}...")
        # Scale to max 640px width, 10 fps for compact markdown embedding
        gif_cmd = [
            "ffmpeg",
            "-y",
            "-i", output_video,
            "-vf", "fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            gif_path,
        ]
        res = subprocess.run(gif_cmd, capture_output=True)
        if res.returncode == 0:
            gif_size = os.path.getsize(gif_path)
            print(f"Preview GIF saved: {gif_path} ({gif_size / 1024 / 1024:.2f} MB)")
        else:
            print(f"GIF generation skipped (FFmpeg exited with {res.returncode})")
            gif_path = None

    # Write companion metadata JSON
    meta_json_path = output_video.rsplit(".", 1)[0] + "_metadata.json"
    metadata = {
        "video_path": output_video,
        "gif_path": gif_path,
        "dataset_path": hdf5_path,
        "dataset_sha256": compute_file_sha256(hdf5_path) if os.path.exists(hdf5_path) else None,
        "simulation_index": sim_idx,
        "reynolds": re,
        "schmidt": sc,
        "layout": layout,
        "panels": panel_keys,
        "total_frames": n_frames,
        "fps": fps,
        "duration_seconds": duration_sec,
        "resolution": {"width": width, "height": height},
        "file_size_bytes": video_size_bytes,
        "video_codec": "H.264 / yuv420p" if ffmpeg_available else "mp4v",
        "domain_size": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
        "spatial_grid": list(data["shape"]),
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit(),
    }
    with open(meta_json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {meta_json_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Render full temporal video of periodic shear flow DNS.")
    parser.add_argument(
        "--hdf5_path",
        type=str,
        default="/root/autodl-tmp/datasets/shear_flow/data/valid/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5",
        help="Path to HDF5 trajectory dataset",
    )
    parser.add_argument(
        "--sim_idx",
        type=int,
        default=0,
        help="Simulation trajectory index within HDF5 file (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video playback frame rate (default: 20 FPS)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="quad",
        choices=["quad", "triple", "vorticity", "u", "tracer", "speed", "v", "pressure"],
        help="Panel layout: 'quad' (2x2), 'triple' (1x3), or single field name",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="outputs/videos/shear_flow_dns_re1e4_sc0.1_sim0_quad.mp4",
        help="Destination path for MP4 video",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to render (default: full trajectory)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Plot rendering DPI (default: 150)",
    )
    parser.add_argument(
        "--no_gif",
        action="store_true",
        help="Disable automatic preview GIF generation",
    )

    args = parser.parse_args()
    render_shear_flow_video(
        hdf5_path=args.hdf5_path,
        output_video=args.output_video,
        sim_idx=args.sim_idx,
        fps=args.fps,
        layout=args.layout,
        max_frames=args.max_frames,
        dpi=args.dpi,
        export_gif=not args.no_gif,
    )


if __name__ == "__main__":
    main()
