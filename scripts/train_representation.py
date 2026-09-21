"""Stage 2 / 5: Spatial Representation Learning and Reconstruction Evaluation.

Reconstruction closed-loop: q -> Z -> q_tilde
- Encoder2D: 8x spatial downsampling to latent states Z in R^(H_z x W_z x 64)
- Decoder2D: 8x spatial upsampling back to physical fields q_tilde in R^(Ny x Nx x 4)
- Pressure Zero-Mean Gauge Projection: enforces integral(p) = 0
- Records detailed reconstruction errors across all four channels [u, v, p, s] + vorticity
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.pipeline import create_flow_dataloaders
from src.losses.field import FieldLoss
from src.metrics.field import compute_max_error, compute_nmse, compute_vrmse
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.utils.checkpoint import BestCheckpointTracker
from src.utils.fft_derivatives import compute_vorticity
from src.utils.reproducibility import seed_everything


class Autoencoder(nn.Module):
    """End-to-end Autoencoder combining spatial Encoder2D and Decoder2D."""

    def __init__(
        self,
        in_channels: int = 4,
        latent_channels: int = 64,
        base_channels: int = 32,
        project_pressure: bool = False,
    ):
        super().__init__()
        self.encoder = Encoder2D(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
        )
        self.decoder = Decoder2D(
            latent_channels=latent_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            project_pressure=project_pressure,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def evaluate_autoencoder(
    model: Autoencoder,
    data_loader,
    device: torch.device,
    normalizer=None,
) -> Dict[str, float]:
    """Comprehensive reconstruction evaluation across u, v, p, s and vorticity."""
    model.eval()
    mse_sum = {c: 0.0 for c in ["u", "v", "p", "s"]}
    rmse_sum = {c: 0.0 for c in ["u", "v", "p", "s"]}
    vrmse_sum = {c: 0.0 for c in ["u", "v", "p", "s"]}
    max_err_max = {c: 0.0 for c in ["u", "v", "p", "s"]}
    p_mean_abs_sum = 0.0
    vort_rmse_sum = 0.0
    total_samples = 0

    channel_names = ["u", "v", "p", "s"]

    with torch.no_grad():
        for batch in data_loader:
            # batch["history"]: (B, L=1, C=4, Ny, Nx)
            x = batch["history"][:, 0].to(device)
            recon = model(x)

            # If normalizer is present, convert back to physical units for physical metrics
            if normalizer is not None:
                x_phys = normalizer.denormalize(x)
                recon_phys = normalizer.denormalize(recon)
            else:
                x_phys = x
                recon_phys = recon

            # Enforce physical zero-mean pressure gauge on denormalized fields
            x_phys = x_phys.clone()
            recon_phys = recon_phys.clone()
            x_phys[:, 2] = x_phys[:, 2] - x_phys[:, 2].mean(dim=(-2, -1), keepdim=True)
            recon_phys[:, 2] = recon_phys[:, 2] - recon_phys[:, 2].mean(dim=(-2, -1), keepdim=True)

            b_size = x.shape[0]
            total_samples += b_size

            for c_idx, c_name in enumerate(channel_names):
                target_c = x_phys[:, c_idx]
                pred_c = recon_phys[:, c_idx]

                mse = torch.mean((pred_c - target_c) ** 2, dim=(-2, -1))
                rmse = torch.sqrt(mse)
                vrmse = compute_vrmse(pred_c, target_c)
                max_err = compute_max_error(pred_c, target_c)

                mse_sum[c_name] += float(mse.sum().item())
                rmse_sum[c_name] += float(rmse.sum().item())
                vrmse_sum[c_name] += float(vrmse.item() * b_size)
                max_err_max[c_name] = max(max_err_max[c_name], float(max_err.item()))

            # Check pressure spatial mean after gauge
            p_recon = recon_phys[:, 2]  # (B, Ny, Nx)
            p_mean = torch.mean(p_recon, dim=(-2, -1)).abs()
            p_mean_abs_sum += float(p_mean.sum().item())

            # Check vorticity reconstruction
            u_gt, v_gt = x_phys[:, 0], x_phys[:, 1]
            u_rec, v_rec = recon_phys[:, 0], recon_phys[:, 1]
            w_gt = compute_vorticity(u_gt, v_gt)
            w_rec = compute_vorticity(u_rec, v_rec)
            w_rmse = torch.sqrt(torch.mean((w_rec - w_gt) ** 2, dim=(-2, -1)))
            vort_rmse_sum += float(w_rmse.sum().item())

    results = {}
    for c_name in channel_names:
        results[f"mse_{c_name}"] = mse_sum[c_name] / total_samples
        results[f"rmse_{c_name}"] = rmse_sum[c_name] / total_samples
        results[f"vrmse_{c_name}"] = vrmse_sum[c_name] / total_samples
        results[f"max_err_{c_name}"] = max_err_max[c_name]

    results["vrmse_mean"] = sum(results[f"vrmse_{c}"] for c in channel_names) / len(channel_names)
    results["rmse_mean"] = sum(results[f"rmse_{c}"] for c in channel_names) / len(channel_names)
    results["pressure_mean_gauge_abs"] = p_mean_abs_sum / total_samples
    results["vorticity_rmse"] = vort_rmse_sum / total_samples

    return results


def run_representation_pipeline(
    data_dir: str = "/root/autodl-tmp/datasets/shear_flow",
    split_type: str = "grouped",
    split_file: Optional[str] = None,
    output_dir: str = "outputs/checkpoints/representation",
    metrics_dir: str = "outputs/metrics",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 2e-4,
    downsample_factor: int = 2,
    stride: int = 4,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    eval_only: bool = False,
    checkpoint_path: str = None,
):
    seed_everything(seed)
    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print("=" * 80)
    print("STAGE 2 / 5: SPATIAL AUTOENCODER (q -> Z -> q_tilde) RECONSTRUCTION")
    print("=" * 80)
    print(f"Device: {device} | Split: {split_type} | Downsample: {downsample_factor}x | Data Root: {data_dir}")

    if split_file is None:
        if split_type.endswith(".json"):
            split_file = split_type
        elif split_type.startswith("outputs/splits/"):
            split_file = split_type
        else:
            candidate = f"outputs/splits/{split_type}.json"
            if os.path.exists(candidate):
                split_file = candidate
            else:
                split_file = f"outputs/splits/{split_type}_split.json"

    train_loader, valid_loader, test_loader, normalizer = create_flow_dataloaders(
        split_type=split_type,
        split_file=split_file,
        data_root=data_dir,
        history_length=1,
        horizon=1,
        stride=stride,
        downsample_factor=downsample_factor,
        batch_size=batch_size,
        num_workers=2,
        normalize=True,
        seed=seed,
    )

    model = Autoencoder(
        in_channels=4,
        latent_channels=64,
        base_channels=32,
        project_pressure=False,
    ).to(device)

    best_ckpt_path = os.path.join(output_dir, "best_autoencoder.pt")

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    elif eval_only:
        load_path = best_ckpt_path if os.path.exists(best_ckpt_path) else os.path.join(output_dir, "best_vrmse_mean.pt")
        if os.path.exists(load_path):
            print(f"Loading best checkpoint from: {load_path}")
            ckpt = torch.load(load_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        else:
            print("Warning: No checkpoint found to evaluate!")

    if not eval_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        loss_fn = FieldLoss(loss_type="mse").to(device)
        tracker = BestCheckpointTracker(save_dir=output_dir, metric_name="vrmse_mean", mode="min", keep_top_k=2)

        print(f"Training Autoencoder for {epochs} epochs (Train Batches: {len(train_loader)})...")

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            t0 = time.time()

            for batch in train_loader:
                x = batch["history"][:, 0].to(device)
                optimizer.zero_grad()
                recon = model(x)
                loss = loss_fn(recon, x)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(x)

            scheduler.step()
            train_loss /= (len(train_loader) * batch_size)

            val_metrics = evaluate_autoencoder(model, valid_loader, device, normalizer)
            elapsed = time.time() - t0

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
                f"Train MSE: {train_loss:.4e} | "
                f"Val VRMSE Mean: {val_metrics['vrmse_mean']:.4f} | "
                f"RMSE: u={val_metrics['rmse_u']:.4f}, v={val_metrics['rmse_v']:.4f}, "
                f"p={val_metrics['rmse_p']:.4f}, s={val_metrics['rmse_s']:.4f}"
            )

            state = {
                "epoch": epoch,
                "config": {
                    "in_channels": 4,
                    "latent_channels": 64,
                    "base_channels": 32,
                    "downsample_factor": downsample_factor,
                    "split_type": split_type,
                    "data_root": data_dir,
                    "normalize": True,
                    "project_pressure": False,
                },
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "decoder_state_dict": model.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }
            tracker.update(val_metrics["vrmse_mean"], state, epoch)

        # Save the definitive best model checkpoint
        best_path = os.path.join(output_dir, f"best_{tracker.metric_name}.pt")
        if os.path.exists(best_path):
            best_data = torch.load(best_path, map_location=device, weights_only=False)
            torch.save(best_data, best_ckpt_path)
            print(f"Best Autoencoder saved to: {best_ckpt_path} (VRMSE: {tracker.best_score:.4f})")
            model.load_state_dict(best_data["model_state_dict"])

    # Final Evaluation on both Valid and Test splits
    print("\n" + "=" * 80)
    print("FINAL RECONSTRUCTION EVALUATION ON VALID AND TEST PARTITIONS")
    print("=" * 80)
    valid_results = evaluate_autoencoder(model, valid_loader, device, normalizer)
    test_results = evaluate_autoencoder(model, test_loader, device, normalizer)

    print("\n--- VALIDATION PARTITION METRICS ---")
    print(f"  u (streamwise):    RMSE = {valid_results['rmse_u']:.6f}, VRMSE = {valid_results['vrmse_u']:.6f}, Max Err = {valid_results['max_err_u']:.6f}")
    print(f"  v (cross-stream):  RMSE = {valid_results['rmse_v']:.6f}, VRMSE = {valid_results['vrmse_v']:.6f}, Max Err = {valid_results['max_err_v']:.6f}")
    print(f"  p (pressure):      RMSE = {valid_results['rmse_p']:.6f}, VRMSE = {valid_results['vrmse_p']:.6f}, Max Err = {valid_results['max_err_p']:.6f}")
    print(f"  s (tracer):        RMSE = {valid_results['rmse_s']:.6f}, VRMSE = {valid_results['vrmse_s']:.6f}, Max Err = {valid_results['max_err_s']:.6f}")
    print(f"  Overall VRMSE:     {valid_results['vrmse_mean']:.6f}")
    print(f"  Vorticity RMSE:    {valid_results['vorticity_rmse']:.6f}")
    print(f"  Pressure Mean (p): {valid_results['pressure_mean_gauge_abs']:.2e} (Zero-mean Gauge Enforcement)")

    print("\n--- TEST PARTITION METRICS ---")
    print(f"  u (streamwise):    RMSE = {test_results['rmse_u']:.6f}, VRMSE = {test_results['vrmse_u']:.6f}, Max Err = {test_results['max_err_u']:.6f}")
    print(f"  v (cross-stream):  RMSE = {test_results['rmse_v']:.6f}, VRMSE = {test_results['vrmse_v']:.6f}, Max Err = {test_results['max_err_v']:.6f}")
    print(f"  p (pressure):      RMSE = {test_results['rmse_p']:.6f}, VRMSE = {test_results['vrmse_p']:.6f}, Max Err = {test_results['max_err_p']:.6f}")
    print(f"  s (tracer):        RMSE = {test_results['rmse_s']:.6f}, VRMSE = {test_results['vrmse_s']:.6f}, Max Err = {test_results['max_err_s']:.6f}")
    print(f"  Overall VRMSE:     {test_results['vrmse_mean']:.6f}")
    print(f"  Vorticity RMSE:    {test_results['vorticity_rmse']:.6f}")
    print(f"  Pressure Mean (p): {test_results['pressure_mean_gauge_abs']:.2e}")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split_type": split_type,
        "downsample_factor": downsample_factor,
        "valid_metrics": valid_results,
        "test_metrics": test_results,
    }
    metrics_path = os.path.join(metrics_dir, "representation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved structured evaluation report to: {metrics_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Autoencoder representation.")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--split_type", type=str, default="grouped")
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints/representation")
    parser.add_argument("--metrics_dir", type=str, default="outputs/metrics")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--downsample_factor", type=int, default=2)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    run_representation_pipeline(
        data_dir=args.data_dir,
        split_type=args.split_type,
        split_file=args.split_file,
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        downsample_factor=args.downsample_factor,
        stride=args.stride,
        eval_only=args.eval_only,
        checkpoint_path=args.checkpoint,
    )
