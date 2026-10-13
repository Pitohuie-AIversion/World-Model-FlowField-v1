#!/usr/bin/env python3
"""Publication-grade qualitative reconstruction visualization for Autoencoder.

Compares Ground Truth vs Reconstructed fields vs Absolute Error
across [u, v, p, s, vorticity].
Outputs: outputs/figures/representation/sample_autoencoder_reconstruction.png
and companion metadata JSON.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.pipeline import create_flow_dataloaders
from src.data.normalization import FieldNormalizer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.utils.fft_derivatives import compute_vorticity


class Autoencoder(torch.nn.Module):
    def __init__(self, in_channels=4, latent_channels=64, base_channels=32, project_pressure=False):
        super().__init__()
        self.encoder = Encoder2D(in_channels=in_channels, latent_channels=latent_channels, base_channels=base_channels)
        self.decoder = Decoder2D(latent_channels=latent_channels, out_channels=in_channels, base_channels=base_channels, project_pressure=project_pressure)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("outputs/checkpoints/representation/best_autoencoder.pt")
    out_dir = Path("outputs/figures/representation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "sample_autoencoder_reconstruction.png"
    out_json = out_dir / "sample_autoencoder_reconstruction_metadata.json"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    model = Autoencoder(
        in_channels=cfg.get("in_channels", 4),
        latent_channels=cfg.get("latent_channels", 64),
        base_channels=cfg.get("base_channels", 32),
        project_pressure=cfg.get("project_pressure", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load dataloader
    train_loader, valid_loader, test_loader, normalizer = create_flow_dataloaders(
        data_root=cfg.get("data_root", "/root/autodl-tmp/datasets/shear_flow"),
        split_type="grouped",
        batch_size=1,
        downsample_factor=cfg.get("downsample_factor", 2),
        normalize=True,
    )

    batch = next(iter(test_loader))
    # Extract frame: sample has "history" shape (B, T_in, C, H, W)
    if "history" in batch:
        x_in = batch["history"][:, 0].to(device)
    elif "fields" in batch:
        x_in = batch["fields"][:, 0].to(device) if batch["fields"].ndim == 5 else batch["fields"].to(device)
    else:
        raise KeyError(f"Unexpected keys in batch: {list(batch.keys())}")

    with torch.no_grad():
        x_rec = model(x_in)

    # Denormalize if normalizer available
    if normalizer is None:
        normalizer = FieldNormalizer.load("outputs/normalization/channel_stats.pt")
    x_in_phys = normalizer.denormalize(x_in).cpu().numpy()[0]
    x_rec_phys = normalizer.denormalize(x_rec).cpu().numpy()[0]

    # Compute vorticity using fft
    u_gt, v_gt = torch.from_numpy(x_in_phys[0:1]), torch.from_numpy(x_in_phys[1:2])
    u_rec, v_rec = torch.from_numpy(x_rec_phys[0:1]), torch.from_numpy(x_rec_phys[1:2])
    vort_gt = compute_vorticity(u_gt, v_gt, domain_size=(1.0, 2.0)).numpy()[0]
    vort_rec = compute_vorticity(u_rec, v_rec, domain_size=(1.0, 2.0)).numpy()[0]

    gt_fields = [x_in_phys[0], x_in_phys[1], x_in_phys[2], x_in_phys[3], vort_gt]
    rec_fields = [x_rec_phys[0], x_rec_phys[1], x_rec_phys[2], x_rec_phys[3], vort_rec]
    err_fields = [np.abs(g - r) for g, r in zip(gt_fields, rec_fields)]

    var_names = ["Streamwise $u$", "Cross-stream $v$", "Pressure $p$", "Tracer $s$", r"Vorticity $\omega$"]
    cmaps = ["RdBu_r", "RdBu_r", "viridis", "inferno", "PRGn"]

    # Plot 3x5 publication figure
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, axes = plt.subplots(3, 5, figsize=(18, 10), constrained_layout=True)

    row_titles = ["Ground Truth ($q$)", r"Reconstructed ($\widetilde{q}$)", r"Absolute Error ($|q - \widetilde{q}|$)"]

    for col in range(5):
        vmin = min(gt_fields[col].min(), rec_fields[col].min())
        vmax = max(gt_fields[col].max(), rec_fields[col].max())
        err_max = err_fields[col].max()

        # Row 0: GT
        im0 = axes[0, col].imshow(gt_fields[col], cmap=cmaps[col], vmin=vmin, vmax=vmax, origin="lower")
        axes[0, col].set_title(var_names[col], fontsize=12, fontweight="bold", pad=8)
        fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

        # Row 1: Rec
        im1 = axes[1, col].imshow(rec_fields[col], cmap=cmaps[col], vmin=vmin, vmax=vmax, origin="lower")
        fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

        # Row 2: Error
        im2 = axes[2, col].imshow(err_fields[col], cmap="magma", vmin=0, vmax=err_max, origin="lower")
        fig.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04)

        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    for row in range(3):
        axes[row, 0].set_ylabel(row_titles[row], fontsize=12, fontweight="bold")

    fig.suptitle("Spatial Autoencoder Representation Learning: Closed-Loop Reconstruction vs Ground Truth", fontsize=15, fontweight="bold")
    plt.savefig(out_png, dpi=300)
    plt.close()

    metrics = {
        "u_rmse": float(np.sqrt(np.mean((gt_fields[0] - rec_fields[0])**2))),
        "v_rmse": float(np.sqrt(np.mean((gt_fields[1] - rec_fields[1])**2))),
        "p_rmse": float(np.sqrt(np.mean((gt_fields[2] - rec_fields[2])**2))),
        "s_rmse": float(np.sqrt(np.mean((gt_fields[3] - rec_fields[3])**2))),
        "vorticity_rmse": float(np.sqrt(np.mean((vort_gt - vort_rec)**2))),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "checkpoint": str(ckpt_path)}, f, indent=2)

    print(f"Successfully generated {out_png}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
