"""Stage C & D: Dynamics Model Training (Single-Step & Rollout-Aware Training)."""

import argparse
import glob
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.baselines.fno import FNO2D
from src.data.shear_flow_dataset import ShearFlowDataset
from src.losses.divergence import DivergenceLoss
from src.losses.field import FieldLoss
from src.losses.rollout import RolloutLoss
from src.losses.vorticity import VorticityLoss
from src.metrics.field import evaluate_field_metrics
from src.models.decoder import Decoder2D
from src.models.direct_transformer import DirectSTTransformer
from src.models.encoder import Encoder2D
from src.models.history_buffer import HistoryBuffer
from src.models.latent_transformer import LatentSTTransformer
from src.utils.checkpoint import BestCheckpointTracker, load_checkpoint, save_checkpoint
from src.utils.reproducibility import seed_everything


class LatentForecasterWrapper(nn.Module):
    """Integrates Encoder, LatentSTTransformer, and Decoder for end-to-end training."""

    def __init__(
        self,
        encoder: Encoder2D,
        transformer: LatentSTTransformer,
        decoder: Decoder2D,
        freeze_representation: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.transformer = transformer
        self.decoder = decoder
        self.freeze_representation = freeze_representation

        if freeze_representation:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False

    def forward_single_step(
        self,
        q_hist: torch.Tensor,
        re: torch.Tensor,
        sc: torch.Tensor,
    ) -> torch.Tensor:
        """q_hist: (B, L, 4, Ny, Nx) -> q_pred: (B, 1, 4, Ny, Nx)."""
        b, l, c, ny, nx = q_hist.shape
        # Encode history frames into latent states
        if self.freeze_representation:
            with torch.no_grad():
                z_hist = self.encoder(q_hist)  # (B, L, C_z, H_z, W_z)
        else:
            z_hist = self.encoder(q_hist)

        # Latent Transformer prediction
        z_next = self.transformer(z_hist, re=re, sc=sc)  # (B, 1, C_z, H_z, W_z)

        # Decode back to physical space
        if self.freeze_representation:
            with torch.no_grad():
                q_next = self.decoder(z_next)
        else:
            q_next = self.decoder(z_next)

        return q_next

    def forward_rollout(
        self,
        q_hist: torch.Tensor,
        re: torch.Tensor,
        sc: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Roll out H steps entirely in latent space, then decode."""
        if self.freeze_representation:
            with torch.no_grad():
                z_hist = self.encoder(q_hist)
        else:
            z_hist = self.encoder(q_hist)

        buf = HistoryBuffer(history_length=q_hist.shape[1])
        buf.reset(z_hist)

        def step_fn(hist_z, _cond=None):
            return self.transformer(hist_z, re=re, sc=sc)

        # Rollout purely in latent space
        z_rollout = buf.rollout(step_fn, steps=horizon)  # (B, H, C_z, H_z, W_z)

        # Decode entire rollout trajectory
        if self.freeze_representation:
            with torch.no_grad():
                q_rollout = self.decoder(z_rollout)
        else:
            q_rollout = self.decoder(z_rollout)

        return q_rollout


def train_forecaster(
    model_type: str = "latent_transformer",
    data_dir: str = "/root/autodl-tmp/datasets/shear_flow",
    output_dir: str = "outputs/checkpoints/dynamics",
    repr_checkpoint: str = "outputs/checkpoints/representation/best_vrmse_mean.pt",
    freeze_representation: bool = True,
    prediction_mode: str = "direct",
    use_condition: bool = True,
    horizon: int = 1,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    lambda_div: float = 0.0,
    lambda_vort: float = 0.0,
    seed: int = 42,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    seed_everything(seed)
    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(data_dir, "train", "*.hdf5")))
    valid_files = sorted(glob.glob(os.path.join(data_dir, "valid", "*.hdf5")))

    if not train_files:
        print(f"No data found in {data_dir}/train. Please download data first.")
        return

    train_dataset = ShearFlowDataset(train_files, history_length=4, horizon=horizon, stride=2)
    valid_dataset = ShearFlowDataset(valid_files, history_length=4, horizon=horizon, stride=8)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model
    if model_type == "latent_transformer":
        encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
        decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=True)
        if os.path.exists(repr_checkpoint):
            print(f"Loading pretrained representation weights from {repr_checkpoint}")
            ckpt = torch.load(repr_checkpoint, map_location="cpu")
            if "encoder_state_dict" in ckpt:
                encoder.load_state_dict(ckpt["encoder_state_dict"])
                decoder.load_state_dict(ckpt["decoder_state_dict"])

        transformer = LatentSTTransformer(
            latent_channels=64,
            embed_dim=256,
            cond_dim=128,
            depth=6,
            num_heads=8,
            history_length=4,
            prediction_mode=prediction_mode,
        )
        model = LatentForecasterWrapper(
            encoder=encoder,
            transformer=transformer,
            decoder=decoder,
            freeze_representation=freeze_representation,
        ).to(device)

    elif model_type == "direct_transformer":
        model = DirectSTTransformer(
            in_channels=4,
            patch_size=(8, 8),
            embed_dim=256,
            cond_dim=128,
            depth=6,
            num_heads=8,
            history_length=4,
            prediction_mode=prediction_mode,
        ).to(device)

    elif model_type == "fno":
        model = FNO2D(
            in_channels=16,  # 4 frames * 4 channels
            out_channels=4,
            modes1=16,
            modes2=16,
            width=64,
            num_layers=4,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    field_loss_fn = FieldLoss(loss_type="mse")
    rollout_loss_fn = RolloutLoss(field_loss=field_loss_fn)
    div_loss_fn = DivergenceLoss()
    vort_loss_fn = VorticityLoss()

    tracker = BestCheckpointTracker(save_dir=output_dir, metric_name="vrmse_mean", mode="min", keep_top_k=3)

    print(f"Training {model_type} on {device} | Horizon: {horizon} | Epochs: {epochs}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            q_hist = batch["history"].to(device)  # (B, L, 4, Ny, Nx)
            q_future = batch["future"].to(device)  # (B, H, 4, Ny, Nx)
            re = batch["re"].to(device) if use_condition else None
            sc = batch["sc"].to(device) if use_condition else None

            optimizer.zero_grad()

            if model_type == "latent_transformer":
                if horizon == 1:
                    pred = model.forward_single_step(q_hist, re, sc)
                else:
                    pred = model.forward_rollout(q_hist, re, sc, horizon=horizon)
            elif model_type == "direct_transformer":
                pred = model(q_hist, re=re, sc=sc)
            elif model_type == "fno":
                pred = model(q_hist)

            loss = rollout_loss_fn(pred, q_future)

            if lambda_div > 0:
                loss = loss + lambda_div * div_loss_fn(pred)
            if lambda_vort > 0:
                loss = loss + lambda_vort * vort_loss_fn(pred, q_future)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(q_hist)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_metrics_sum = {}
        with torch.no_grad():
            for batch in valid_loader:
                q_hist = batch["history"].to(device)
                q_future = batch["future"].to(device)
                re = batch["re"].to(device) if use_condition else None
                sc = batch["sc"].to(device) if use_condition else None

                if model_type == "latent_transformer":
                    if horizon == 1:
                        pred = model.forward_single_step(q_hist, re, sc)
                    else:
                        pred = model.forward_rollout(q_hist, re, sc, horizon=horizon)
                elif model_type == "direct_transformer":
                    pred = model(q_hist, re=re, sc=sc)
                elif model_type == "fno":
                    pred = model(q_hist)

                step_metrics = evaluate_field_metrics(pred[:, 0], q_future[:, 0])
                for k, v in step_metrics.items():
                    val_metrics_sum[k] = val_metrics_sum.get(k, 0.0) + v * len(q_hist)

        val_metrics = {k: v / len(valid_dataset) for k, v in val_metrics_sum.items()}

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4e} | "
            f"Val VRMSE Mean: {val_metrics['vrmse_mean']:.4f} "
            f"(u: {val_metrics['vrmse_u']:.4f}, v: {val_metrics['vrmse_v']:.4f}, "
            f"p: {val_metrics['vrmse_p']:.4f}, s: {val_metrics['vrmse_s']:.4f})"
        )

        state = {
            "epoch": epoch,
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
        }
        tracker.update(val_metrics["vrmse_mean"], state, epoch)

    print(f"Training completed. Best VRMSE: {tracker.best_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="latent_transformer", choices=["latent_transformer", "direct_transformer", "fno"])
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints/dynamics")
    parser.add_argument("--repr_checkpoint", type=str, default="outputs/checkpoints/representation/best_vrmse_mean.pt")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_div", type=float, default=0.0)
    parser.add_argument("--lambda_vort", type=float, default=0.0)
    args = parser.parse_args()

    train_forecaster(
        model_type=args.model,
        data_dir=args.data_dir,
        output_dir=os.path.join(args.output_dir, args.model),
        repr_checkpoint=args.repr_checkpoint,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_div=args.lambda_div,
        lambda_vort=args.lambda_vort,
    )
