"""Stage B: Spatial Representation Learning (Encoder-Decoder Reconstruction Training)."""

import argparse
import glob
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.shear_flow_dataset import ShearFlowDataset
from src.losses.field import FieldLoss
from src.metrics.field import evaluate_field_metrics
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.utils.checkpoint import BestCheckpointTracker, save_checkpoint
from src.utils.reproducibility import seed_everything


class Autoencoder(nn.Module):
    """End-to-end Autoencoder combining spatial Encoder2D and Decoder2D."""

    def __init__(self, in_channels: int = 4, latent_channels: int = 64, base_channels: int = 32):
        super().__init__()
        self.encoder = Encoder2D(in_channels=in_channels, latent_channels=latent_channels, base_channels=base_channels)
        self.decoder = Decoder2D(latent_channels=latent_channels, out_channels=in_channels, base_channels=base_channels, project_pressure=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def train_representation(
    data_dir: str = "/root/autodl-tmp/datasets/shear_flow",
    output_dir: str = "outputs/checkpoints/representation",
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
):
    seed_everything(seed)
    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)

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

    if not train_files:
        if valid_files and test_files:
            print(f"Notice: Train files downloading. Using valid partition ({len(valid_files)} file(s)) for training and test partition ({len(test_files)} file(s)) for validation.")
            train_files = valid_files
            valid_files = test_files
        elif valid_files:
            print(f"Notice: No train files found in {data_dir}. Using available valid files for representation learning.")
            train_files = valid_files
        else:
            print(f"No files found in {data_dir}. Please download data first.")
            return

    train_dataset = ShearFlowDataset(train_files, history_length=1, horizon=1, stride=4)
    valid_dataset = ShearFlowDataset(valid_files, history_length=1, horizon=1, stride=8)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = Autoencoder(in_channels=4, latent_channels=64, base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = FieldLoss(loss_type="mse").to(device)
    tracker = BestCheckpointTracker(save_dir=output_dir, metric_name="vrmse_mean", mode="min", keep_top_k=3)

    print(f"Starting Representation Training on {device} | Epochs: {epochs} | Train samples: {len(train_dataset)}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_t = time.time()

        for batch in train_loader:
            x = batch["history"][:, 0].to(device)  # (B, 4, Ny, Nx)
            optimizer.zero_grad()
            recon = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_metrics_sum = {}
        with torch.no_grad():
            for batch in valid_loader:
                x = batch["history"][:, 0].to(device)
                recon = model(x)
                batch_metrics = evaluate_field_metrics(recon, x)
                for k, v in batch_metrics.items():
                    val_metrics_sum[k] = val_metrics_sum.get(k, 0.0) + v * len(x)

        val_metrics = {k: v / len(valid_dataset) for k, v in val_metrics_sum.items()}
        elapsed = time.time() - start_t

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train MSE: {train_loss:.4e} | Val VRMSE Mean: {val_metrics['vrmse_mean']:.4f} "
            f"(u: {val_metrics['vrmse_u']:.4f}, v: {val_metrics['vrmse_v']:.4f}, "
            f"p: {val_metrics['vrmse_p']:.4f}, s: {val_metrics['vrmse_s']:.4f})"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "decoder_state_dict": model.decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
        }
        tracker.update(val_metrics["vrmse_mean"], state, epoch)

    print(f"Representation training complete. Best VRMSE: {tracker.best_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints/representation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_representation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
