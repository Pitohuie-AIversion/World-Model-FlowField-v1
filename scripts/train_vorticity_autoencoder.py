"""Minimal training and resumption entrypoint for Single-Channel Vorticity Autoencoder.

Key engineering contracts:
1. Synthetic data only: Generates periodic vorticity fields on [0, Lx) x [0, Ly).
2. Explicit capacity logging: Explicitly records element compression ratio (1.0 for Cz=64).
3. Pure MSE training: Optimizes single MSE reconstruction loss; enstrophy spectrum is strictly evaluation-only.
4. Resumption safety & verification:
   - Fails immediately if --resume checkpoint or --config does not exist (no silent fallback to fresh training).
   - Validates protected configuration keys against checkpoint before resuming (prevents altered data/model definitions).
   - Atomic checkpoint writes (temp file + rename) to protect latest_checkpoint.pt from corruption.
   - Persists full enstrophy spectrum analysis (curves, ratios, spurious energy) in checkpoint and summary.
5. Strict isolation: Zero references to real StocBench data files.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.metrics.vorticity_representation import (
    compute_relative_l2_error,
    compute_enstrophy_spectrum_ratio,
)


class VorticityAutoencoder(nn.Module):
    """End-to-end Autoencoder for single-channel 2D scalar vorticity fields."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 64,
        base_channels: int = 32,
        project_pressure: bool = False,
    ):
        super().__init__()
        assert in_channels == 1 and out_channels == 1, "VorticityAutoencoder requires in/out channels == 1"
        assert not project_pressure, "Single-channel vorticity must have project_pressure=False (no pressure gauge)"

        self.encoder = Encoder2D(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
        )
        self.decoder = Decoder2D(
            latent_channels=latent_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            project_pressure=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def generate_synthetic_vorticity_dataset(
    num_samples: int,
    nx: int = 64,
    ny: int = 64,
    lx: float = 1.0,
    ly: float = 1.0,
    base_wavenumber: int = 1,
    perturbation_modes: Optional[List[Tuple[int, int]]] = None,
    perturbation_amplitude: float = 0.1,
    seed: int = 42,
) -> torch.Tensor:
    """Generate periodic scalar vorticity fields on [0, Lx) x [0, Ly) with non-repeating grid endpoints.

    Base field: omega_0(x, y) = 2.0 * cos(2*pi*k_0*x/Lx) * cos(2*pi*k_0*y/Ly)
    Perturbed field j: omega_j = omega_0 + sum_{m, n} eps * sin(2*pi*(m*x/Lx + n*y/Ly) + phi_j)

    Returns:
        Tensor of shape (num_samples, 1, nx, ny).
    """
    gen = torch.Generator().manual_seed(seed)

    # Periodic coordinates without repeating the right/top boundary
    x = torch.arange(nx, dtype=torch.float32) * (lx / nx)
    y = torch.arange(ny, dtype=torch.float32) * (ly / ny)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Base periodic quadrupole vorticity field
    omega_0 = 2.0 * torch.cos(2.0 * torch.pi * base_wavenumber * X / lx) * torch.cos(2.0 * torch.pi * base_wavenumber * Y / ly)

    modes = [tuple(m) for m in (perturbation_modes or [(1, 0), (0, 1), (1, 1), (2, 1)])]
    samples = []

    for _ in range(num_samples):
        field = omega_0.clone()
        for m, n in modes:
            phi = float(torch.rand(1, generator=gen).item() * 2.0 * torch.pi)
            field = field + perturbation_amplitude * torch.sin(
                2.0 * torch.pi * (m * X / lx + n * Y / ly) + phi
            )
        samples.append(field.unsqueeze(0))  # (1, nx, ny)

    dataset_tensor = torch.stack(samples, dim=0)  # (num_samples, 1, nx, ny)
    return dataset_tensor


def compute_capacity_metadata(
    nx: int = 64,
    ny: int = 64,
    in_channels: int = 1,
    latent_channels: int = 64,
    downsample_factor: int = 8,
) -> Dict[str, Any]:
    """Compute and document element compression ratio and dimensions."""
    hz = nx // downsample_factor
    wz = ny // downsample_factor
    in_elem = in_channels * nx * ny
    latent_elem = latent_channels * hz * wz
    ratio = float(in_elem / latent_elem)

    return {
        "input_shape": [in_channels, nx, ny],
        "latent_shape": [latent_channels, hz, wz],
        "input_elements": in_elem,
        "latent_elements": latent_elem,
        "element_compression_ratio": ratio,
        "capacity_interpretation": (
            "Element ratio 1.0 indicates spatial-to-channel representation rearrangement without capacity reduction."
            if abs(ratio - 1.0) < 1e-5
            else f"Element compression ratio: {ratio:.2f}x"
        ),
    }


def _validate_resumption_config(
    current_config: Dict[str, Any],
    checkpoint_config: Dict[str, Any],
) -> None:
    """Validate that protected dataset, model, and training definition parameters match between current config and checkpoint."""
    protected_keys = [
        ("model", "in_channels"),
        ("model", "out_channels"),
        ("model", "latent_channels"),
        ("model", "base_channels"),
        ("model", "project_pressure"),
        ("domain", "nx"),
        ("domain", "ny"),
        ("domain", "lx"),
        ("domain", "ly"),
        ("synthetic_data", "seed"),
        ("synthetic_data", "num_train_samples"),
        ("synthetic_data", "num_val_samples"),
        ("synthetic_data", "base_wavenumber"),
        ("synthetic_data", "perturbation_modes"),
        ("synthetic_data", "perturbation_amplitude"),
        ("training", "batch_size"),
        ("training", "lr"),
        ("training", "weight_decay"),
        ("training", "loss_type"),
    ]

    for section, key in protected_keys:
        curr_val = current_config.get(section, {}).get(key)
        ckpt_val = checkpoint_config.get(section, {}).get(key)

        # Normalize list/tuple comparisons
        if isinstance(curr_val, list) and isinstance(ckpt_val, list):
            norm_curr = [list(x) if isinstance(x, (list, tuple)) else x for x in curr_val]
            norm_ckpt = [list(x) if isinstance(x, (list, tuple)) else x for x in ckpt_val]
            match = (norm_curr == norm_ckpt)
        else:
            match = (curr_val == ckpt_val)

        if not match:
            raise ValueError(
                f"Resume config mismatch for protected key '{section}.{key}': "
                f"checkpoint has {ckpt_val}, but current config has {curr_val}. "
                f"Resuming with altered data or model definitions is prohibited."
            )


def train_vorticity_autoencoder(
    config: Optional[Dict[str, Any]] = None,
    resume_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    override_epochs: Optional[int] = None,
    override_batch_size: Optional[int] = None,
    override_lr: Optional[float] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run self-reconstruction training on synthetic vorticity fields with safe resumption support."""
    # Fail-fast: if resume_path is explicitly given, it must exist
    if resume_path:
        resume_file = Path(resume_path)
        if not resume_file.is_file():
            raise FileNotFoundError(f"Resume checkpoint file not found: {resume_path}")

    cfg = config or {}
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    synth_cfg = cfg.get("synthetic_data", {})
    domain_cfg = cfg.get("domain", {})

    epochs = override_epochs or train_cfg.get("epochs", 10)
    batch_size = override_batch_size or train_cfg.get("batch_size", 16)
    lr = override_lr or train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)

    nx = domain_cfg.get("nx", 64)
    ny = domain_cfg.get("ny", 64)
    lx = domain_cfg.get("lx", 1.0)
    ly = domain_cfg.get("ly", 1.0)
    domain_size = (lx, ly)

    latent_channels = model_cfg.get("latent_channels", 64)
    base_channels = model_cfg.get("base_channels", 32)

    seed = synth_cfg.get("seed", 42)
    n_train = synth_cfg.get("num_train_samples", 128)
    n_val = synth_cfg.get("num_val_samples", 32)
    base_wavenumber = synth_cfg.get("base_wavenumber", 1)
    perturbation_modes = synth_cfg.get("perturbation_modes", [[1, 0], [0, 1], [1, 1], [2, 1]])
    perturbation_amplitude = synth_cfg.get("perturbation_amplitude", 0.1)

    # Build canonical effective configuration snapshot
    effective_config = {
        "model": {
            "in_channels": 1,
            "out_channels": 1,
            "latent_channels": latent_channels,
            "base_channels": base_channels,
            "project_pressure": False,
        },
        "domain": {
            "nx": nx,
            "ny": ny,
            "lx": lx,
            "ly": ly,
        },
        "synthetic_data": {
            "num_train_samples": n_train,
            "num_val_samples": n_val,
            "base_wavenumber": base_wavenumber,
            "perturbation_modes": perturbation_modes,
            "perturbation_amplitude": perturbation_amplitude,
            "seed": seed,
        },
        "training": {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "loss_type": "mse",
        },
    }

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    if not resume_path:
        torch.manual_seed(seed)

    # Compute explicit representation capacity
    capacity_meta = compute_capacity_metadata(nx, ny, 1, latent_channels)

    # Initialize model
    model = VorticityAutoencoder(
        in_channels=1,
        out_channels=1,
        latent_channels=latent_channels,
        base_channels=base_channels,
        project_pressure=False,
    ).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    start_epoch = 1
    global_step = 0
    history = {"train_loss": [], "val_rel_l2": [], "val_abs_l2": []}

    out_path = Path(output_dir or cfg.get("checkpoint", {}).get("save_dir", "outputs/checkpoints/vorticity_ae_synthetic"))
    out_path.mkdir(parents=True, exist_ok=True)

    # Handle resumption if checkpoint provided
    if resume_path:
        ckpt = torch.load(resume_path, map_location=dev, weights_only=False)

        # Validate configuration compatibility
        if "config" in ckpt:
            _validate_resumption_config(effective_config, ckpt["config"])

        if epochs <= ckpt["epoch"]:
            raise ValueError(
                f"Requested target epochs ({epochs}) must be strictly greater than checkpoint completed epoch "
                f"({ckpt['epoch']}) to continue training."
            )

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # PyTorch requires CPU ByteTensor for torch.set_rng_state()
        if "rng_state_torch" in ckpt and ckpt["rng_state_torch"] is not None:
            torch.set_rng_state(ckpt["rng_state_torch"].cpu())
        if "rng_state_cuda" in ckpt and ckpt["rng_state_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt["rng_state_cuda"].cpu())

        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"[VorticityAutoencoder] Resumed safely from {resume_path} at epoch {start_epoch}, global_step {global_step}")

    # Synthesize data with validated parameters
    train_data = generate_synthetic_vorticity_dataset(
        num_samples=n_train,
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        base_wavenumber=base_wavenumber,
        perturbation_modes=perturbation_modes,
        perturbation_amplitude=perturbation_amplitude,
        seed=seed,
    )
    val_data = generate_synthetic_vorticity_dataset(
        num_samples=n_val,
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        base_wavenumber=base_wavenumber,
        perturbation_modes=perturbation_modes,
        perturbation_amplitude=perturbation_amplitude,
        seed=seed + 1000,
    )

    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=False)

    print(f"[VorticityAutoencoder] Starting training for epochs [{start_epoch} -> {epochs}] on {dev}")
    print(f"[Capacity] Input: {capacity_meta['input_shape']}, Latent: {capacity_meta['latent_shape']}, Element Ratio: {capacity_meta['element_compression_ratio']}")

    initial_loss: Optional[float] = None
    final_loss: Optional[float] = None
    last_spec_res: Optional[Dict[str, Any]] = None

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (batch_x,) in enumerate(train_loader):
            batch_x = batch_x.to(dev)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = nn.functional.mse_loss(recon, batch_x)
            loss.backward()
            optimizer.step()

            global_step += 1
            loss_val = float(loss.item())
            epoch_loss += loss_val
            if initial_loss is None:
                initial_loss = loss_val
            final_loss = loss_val

        avg_train_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_train_loss)

        # Validation evaluation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for (val_x,) in val_loader:
                val_x = val_x.to(dev)
                val_rec = model(val_x)
                val_preds.append(val_rec.cpu())
                val_targets.append(val_x.cpu())

        all_val_pred = torch.cat(val_preds, dim=0)
        all_val_targ = torch.cat(val_targets, dim=0)

        l2_res = compute_relative_l2_error(all_val_pred, all_val_targ)
        spec_res = compute_enstrophy_spectrum_ratio(all_val_pred, all_val_targ, domain_size=domain_size)
        last_spec_res = spec_res

        history["val_rel_l2"].append(l2_res["relative_l2"])
        history["val_abs_l2"].append(l2_res["absolute_l2"])

        # Atomic checkpoint write (save to temporary file then replace)
        ckpt_data = {
            "epoch": epoch,
            "global_step": global_step,
            "next_batch_idx": 0,  # epoch boundary resumption
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state_torch": torch.get_rng_state(),
            "rng_state_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "config": effective_config,
            "capacity_metadata": capacity_meta,
            "train_loss": avg_train_loss,
            "val_relative_l2": l2_res["relative_l2"],
            "val_absolute_l2": l2_res["absolute_l2"],
            "enstrophy_spectrum": {
                "k_bins": spec_res["k_bins"],
                "spectrum_target": spec_res["spectrum_target"],
                "spectrum_pred": spec_res["spectrum_pred"],
                "spectrum_ratio": spec_res["spectrum_ratio"],
                "valid_bins_count": spec_res["valid_bins_count"],
                "total_bins": spec_res["total_bins"],
                "spurious_energy_in_zero_bins": spec_res["spurious_energy_in_zero_bins"],
            },
        }
        latest_ckpt = out_path / "latest_checkpoint.pt"
        temp_ckpt = out_path / f"latest_checkpoint.pt.tmp_{epoch}"
        try:
            torch.save(ckpt_data, temp_ckpt)
            temp_ckpt.replace(latest_ckpt)
        except Exception as e:
            if temp_ckpt.exists():
                try:
                    temp_ckpt.unlink()
                except OSError:
                    pass
            raise IOError(f"Failed to atomic save checkpoint for epoch {epoch}: {e}") from e

    summary_file = out_path / "training_summary.json"
    summary_data = {
        "status": "COMPLETED",
        "epochs_completed": epochs,
        "global_steps": global_step,
        "initial_step_loss": initial_loss,
        "final_step_loss": final_loss,
        "loss_reduction_ratio": float((initial_loss - final_loss) / initial_loss) if initial_loss and initial_loss > 0 else 0.0,
        "capacity_metadata": capacity_meta,
        "effective_config": effective_config,
        "final_val_relative_l2": history["val_rel_l2"][-1] if history["val_rel_l2"] else None,
        "final_val_absolute_l2": history["val_abs_l2"][-1] if history["val_abs_l2"] else None,
        "enstrophy_spectrum_summary": {
            "k_bins": last_spec_res["k_bins"] if last_spec_res else [],
            "spectrum_target": last_spec_res["spectrum_target"] if last_spec_res else [],
            "spectrum_pred": last_spec_res["spectrum_pred"] if last_spec_res else [],
            "spectrum_ratio": last_spec_res["spectrum_ratio"] if last_spec_res else [],
            "valid_bins_count": last_spec_res["valid_bins_count"] if last_spec_res else 0,
            "spurious_energy_in_zero_bins": last_spec_res["spurious_energy_in_zero_bins"] if last_spec_res else 0.0,
        },
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return {
        "model": model,
        "optimizer": optimizer,
        "history": history,
        "summary": summary_data,
        "capacity_metadata": capacity_meta,
        "effective_config": effective_config,
        "checkpoint_path": str(out_path / "latest_checkpoint.pt"),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Single-Channel Vorticity Autoencoder on Synthetic Data")
    parser.add_argument("--config", type=str, default="configs/train/vorticity_autoencoder.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    res = train_vorticity_autoencoder(
        config=cfg,
        resume_path=args.resume,
        output_dir=args.output_dir,
        override_epochs=args.epochs,
        override_batch_size=args.batch_size,
        override_lr=args.lr,
        device=args.device,
    )
    print(f"Training completed successfully. Summary: {res['summary']}")


if __name__ == "__main__":
    main()
