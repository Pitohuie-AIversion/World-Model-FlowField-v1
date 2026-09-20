"""Stage C & D: Dynamics Model Training (Single-Step & Rollout-Aware Training)."""

import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Optional
import h5py

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.baselines.fno import FNO2D
from src.data.pipeline import create_flow_dataloaders
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

        # Decode back to physical space (decoder weights are frozen if freeze_representation is True,
        # but gradient passes back to z_next)
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
        z_rollout = buf.rollout(step_fn, steps=horizon)
        # Decode entire rollout trajectory
        q_rollout = self.decoder(z_rollout)
        return q_rollout

    def forward(
        self,
        q_hist: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        sc: Optional[torch.Tensor] = None,
        horizon: int = 1,
    ) -> torch.Tensor:
        """Unified forward interface."""
        if horizon == 1:
            return self.forward_single_step(q_hist, re, sc)
        else:
            return self.forward_rollout(q_hist, re, sc, horizon=horizon)


def train_forecaster(
    model_type: str = "latent_transformer",
    data_dir: str = "/root/autodl-tmp/datasets/shear_flow",
    output_dir: str = "outputs/checkpoints/dynamics",
    repr_checkpoint: str = "outputs/checkpoints/representation/best_vrmse_mean.pt",
    split_type: str = "grouped",
    split_file: Optional[str] = None,
    train_stride: int = 2,
    valid_stride: int = 8,
    normalize: bool = True,
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
    preload_to_memory: bool = False,
    num_workers: int = 4,
    embed_dim: int = 256,
    depth: int = 6,
    num_heads: int = 8,
    use_amp: bool = False,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Distributed Data Parallel (DDP) detection
    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if is_distributed:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        device = torch.device(device_str)

    seed_everything(seed + global_rank)
    if global_rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    if split_file is None:
        split_file = f"outputs/splits/{split_type}_split.json"

    train_loader, valid_loader, test_loader, normalizer, train_sampler = create_flow_dataloaders(
        split_type=split_type,
        split_file=split_file,
        history_length=4,
        horizon=horizon,
        train_stride=train_stride,
        valid_stride=valid_stride,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize,
        preload_to_memory=preload_to_memory,
        is_distributed=is_distributed,
        rank=global_rank,
        world_size=world_size,
        seed=seed,
        return_sampler=True,
    )
    valid_dataset = valid_loader.dataset

    # Initialize model
    if model_type == "latent_transformer":
        encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
        decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=True)
        if os.path.exists(repr_checkpoint):
            if global_rank == 0:
                print(f"Loading pretrained representation weights from {repr_checkpoint}")
            ckpt = torch.load(repr_checkpoint, map_location="cpu")
            if "encoder_state_dict" in ckpt:
                encoder.load_state_dict(ckpt["encoder_state_dict"])
                decoder.load_state_dict(ckpt["decoder_state_dict"])

        transformer = LatentSTTransformer(
            latent_channels=64,
            embed_dim=embed_dim,
            cond_dim=128,
            depth=depth,
            num_heads=num_heads,
            history_length=4,
            prediction_mode=prediction_mode,
        )
        model = LatentForecasterWrapper(
            encoder=encoder,
            transformer=transformer,
            decoder=decoder,
            freeze_representation=freeze_representation,
        ).to(device)

    elif model_type in ("direct_transformer", "pde_transformer"):
        model = DirectSTTransformer(
            in_channels=4,
            patch_size=(8, 8),
            embed_dim=embed_dim,
            cond_dim=128,
            depth=depth,
            num_heads=num_heads,
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

    # Wrap model with DDP
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    field_loss_fn = FieldLoss(loss_type="mse").to(device)
    rollout_loss_fn = RolloutLoss(field_loss=field_loss_fn).to(device)
    div_loss_fn = DivergenceLoss().to(device)
    vort_loss_fn = VorticityLoss().to(device)

    tracker = None
    if global_rank == 0:
        tracker = BestCheckpointTracker(save_dir=output_dir, metric_name="vrmse_mean", mode="min", keep_top_k=3)
        print(f"Training {model_type} on {device} (Distributed: {is_distributed}, World Size: {world_size}) | Horizon: {horizon} | Epochs: {epochs}")

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(1, epochs + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0

        for batch in train_loader:
            q_hist = batch["history"].to(device)  # (B, L, 4, Ny, Nx)
            q_future = batch["future"].to(device)  # (B, H, 4, Ny, Nx)
            re = batch["re"].to(device) if use_condition else None
            sc = batch["sc"].to(device) if use_condition else None

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                if model_type == "latent_transformer":
                    if is_distributed:
                        pred = model(q_hist, re, sc, horizon=horizon)
                    else:
                        if horizon == 1:
                            pred = model.forward_single_step(q_hist, re, sc)
                        else:
                            pred = model.forward_rollout(q_hist, re, sc, horizon=horizon)
                elif model_type in ("direct_transformer", "pde_transformer"):
                    if horizon == 1:
                        pred = model(q_hist, re=re, sc=sc)
                    else:
                        buf = HistoryBuffer(history_length=q_hist.shape[1])
                        buf.reset(q_hist)
                        pred = buf.rollout(lambda hist, _c: model(hist, re=re, sc=sc), steps=horizon)
                elif model_type == "fno":
                    if horizon == 1:
                        pred = model(q_hist)  # (B, 1, C, Ny, Nx)
                    else:
                        pred_list = []
                        hist_window = q_hist  # (B, L, C, Ny, Nx)
                        for _ in range(horizon):
                            step_pred = model(hist_window)  # (B, 1, C, Ny, Nx)
                            pred_list.append(step_pred)
                            hist_window = torch.cat([hist_window[:, 1:], step_pred], dim=1)
                        pred = torch.cat(pred_list, dim=1)  # (B, H, C, Ny, Nx)

                loss = rollout_loss_fn(pred, q_future)

                if lambda_div > 0:
                    loss = loss + lambda_div * div_loss_fn(pred)
                if lambda_vort > 0:
                    loss = loss + lambda_vort * vort_loss_fn(pred, q_future)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(q_hist)

        train_loss /= len(train_loader.dataset)

        # Validation only on rank 0
        if global_rank == 0:
            model.eval()
            rollout_step_metrics_sum = {h: {} for h in range(horizon)}
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=use_amp):
                    for batch in valid_loader:
                        q_hist = batch["history"].to(device)
                        q_future = batch["future"].to(device)
                        re = batch["re"].to(device) if use_condition else None
                        sc = batch["sc"].to(device) if use_condition else None

                        eval_model = model.module if is_distributed else model
                        if model_type == "latent_transformer":
                            if horizon == 1:
                                pred = eval_model.forward_single_step(q_hist, re, sc)
                            else:
                                pred = eval_model.forward_rollout(q_hist, re, sc, horizon=horizon)
                        elif model_type in ("direct_transformer", "pde_transformer"):
                            if horizon == 1:
                                pred = eval_model(q_hist, re=re, sc=sc)
                            else:
                                buf = HistoryBuffer(history_length=q_hist.shape[1])
                                buf.reset(q_hist)
                                pred = buf.rollout(lambda hist, _c: eval_model(hist, re=re, sc=sc), steps=horizon)
                        elif model_type == "fno":
                            if horizon == 1:
                                pred = eval_model(q_hist)
                            else:
                                pred_list = []
                                hist_window = q_hist
                                for _ in range(horizon):
                                    step_pred = eval_model(hist_window)
                                    pred_list.append(step_pred)
                                    hist_window = torch.cat([hist_window[:, 1:], step_pred], dim=1)
                                pred = torch.cat(pred_list, dim=1)

                        # Denormalize to physical units for physical metric evaluation
                        if normalizer is not None:
                            pred_eval = normalizer.denormalize(pred)
                            target_eval = normalizer.denormalize(q_future)
                        else:
                            pred_eval = pred
                            target_eval = q_future

                        b_samples = len(q_hist)
                        for h in range(horizon):
                            step_m = evaluate_field_metrics(pred_eval[:, h], target_eval[:, h])
                            for k, v in step_m.items():
                                rollout_step_metrics_sum[h][k] = rollout_step_metrics_sum[h].get(k, 0.0) + v * b_samples

            n_val = len(valid_dataset)
            val_step_metrics = {
                h: {k: v / n_val for k, v in rollout_step_metrics_sum[h].items()}
                for h in range(horizon)
            }

            # Step 1 metrics
            val_metrics = dict(val_step_metrics[0])

            # In multi-step rollout, compute overall rollout trajectory mean metrics
            if horizon > 1:
                rollout_mean_vrmse = sum(val_step_metrics[h]["vrmse_mean"] for h in range(horizon)) / horizon
                rollout_mean_rmse = sum(val_step_metrics[h]["rmse_mean"] for h in range(horizon)) / horizon
                val_metrics["rollout_mean_vrmse"] = rollout_mean_vrmse
                val_metrics["rollout_mean_rmse"] = rollout_mean_rmse
                val_criterion = rollout_mean_vrmse
            else:
                val_criterion = val_metrics["vrmse_mean"]

            vram_gb = torch.cuda.max_memory_allocated(device=device) / (1024**3) if torch.cuda.is_available() else 0.0
            if horizon > 1:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4e} | "
                    f"Val Rollout Mean VRMSE: {val_metrics['rollout_mean_vrmse']:.4f} | "
                    f"Step 1 VRMSE: {val_metrics['vrmse_mean']:.4f} "
                    f"(u: {val_metrics['vrmse_u']:.4f}, v: {val_metrics['vrmse_v']:.4f}, "
                    f"p: {val_metrics['vrmse_p']:.4f}, s: {val_metrics['vrmse_s']:.4f}) | "
                    f"Max VRAM: {vram_gb:.2f} GB"
                )
            else:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4e} | "
                    f"Val VRMSE Mean: {val_metrics['vrmse_mean']:.4f} "
                    f"(u: {val_metrics['vrmse_u']:.4f}, v: {val_metrics['vrmse_v']:.4f}, "
                    f"p: {val_metrics['vrmse_p']:.4f}, s: {val_metrics['vrmse_s']:.4f}) | "
                    f"Max VRAM: {vram_gb:.2f} GB"
                )

            raw_model = model.module if is_distributed else model
            state = {
                "epoch": epoch,
                "model_type": model_type,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "val_step_metrics": val_step_metrics,
                "horizon": horizon,
            }
            tracker.update(val_criterion, state, epoch)

    if global_rank == 0 and tracker is not None:
        print(f"Training completed. Best VRMSE: {tracker.best_score:.4f}")

    if is_distributed:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage C & D: Dynamics Model Training.")
    parser.add_argument("--model", type=str, default="latent_transformer", choices=["latent_transformer", "direct_transformer", "fno", "pde_transformer"])
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/datasets/shear_flow")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints/dynamics")
    parser.add_argument("--repr_checkpoint", type=str, default="outputs/checkpoints/representation/best_vrmse_mean.pt")
    parser.add_argument("--split_type", type=str, default="grouped", choices=["grouped", "official"])
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--train_stride", type=int, default=2)
    parser.add_argument("--valid_stride", type=int, default=8)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze_representation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--joint", action="store_true", help="Jointly train representation and dynamics")
    parser.add_argument("--prediction_mode", type=str, default="direct", choices=["direct", "residual"])
    parser.add_argument("--use_condition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_div", type=float, default=0.0)
    parser.add_argument("--lambda_vort", type=float, default=0.0)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--preload_to_memory", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    args = parser.parse_args()

    freeze_rep = False if args.joint else args.freeze_representation
    out_dir = args.output_dir if args.output_dir.endswith(args.model) else os.path.join(args.output_dir, args.model)

    train_forecaster(
        model_type=args.model,
        data_dir=args.data_dir,
        output_dir=out_dir,
        repr_checkpoint=args.repr_checkpoint,
        split_type=args.split_type,
        split_file=args.split_file,
        train_stride=args.train_stride,
        valid_stride=args.valid_stride,
        normalize=args.normalize,
        freeze_representation=freeze_rep,
        prediction_mode=args.prediction_mode,
        use_condition=args.use_condition,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_div=args.lambda_div,
        lambda_vort=args.lambda_vort,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_workers=args.num_workers,
        preload_to_memory=args.preload_to_memory,
        use_amp=args.use_amp,
    )
