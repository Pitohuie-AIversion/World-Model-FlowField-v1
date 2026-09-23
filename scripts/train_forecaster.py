"""Stage C & D: Dynamics Model Training (Single-Step & Rollout-Aware Training)."""

import argparse
import glob
import hashlib
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
from src.utils.physics_contract import (
    PHYSICS_PROTOCOL,
    SPATIAL_AXIS_CONTRACT,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
)
from src.utils.provenance import (
    get_git_commit,
    is_git_dirty,
    compute_split_hash_from_file,
    compute_normalizer_hash,
    validate_init_checkpoint_contract,
)


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
    downsample_factor: int = 2,
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
    field_loss_space: str = "normalized",
    seed: int = 42,
    preload_to_memory: bool = False,
    num_workers: int = 4,
    embed_dim: int = 256,
    depth: int = 6,
    num_heads: int = 8,
    use_amp: bool = False,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    stats_dir: Optional[str] = None,
    init_checkpoint: Optional[str] = None,
    grad_accum_steps: int = 1,
    val_diagnostic_horizons: Optional[List[int]] = None,
    expected_init_horizon: Optional[int] = None,
):
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

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

    loader_kwargs = {}
    if stats_dir is not None:
        loader_kwargs["stats_dir"] = stats_dir

    effective_valid_horizon = horizon
    if val_diagnostic_horizons:
        effective_valid_horizon = max([horizon] + val_diagnostic_horizons)

    train_loader, valid_loader, test_loader, normalizer, train_sampler = create_flow_dataloaders(
        split_type=split_type,
        split_file=split_file,
        data_root=data_dir,
        history_length=4,
        horizon=horizon,
        valid_horizon=effective_valid_horizon,
        train_stride=train_stride,
        valid_stride=valid_stride,
        downsample_factor=downsample_factor,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize,
        preload_to_memory=preload_to_memory,
        is_distributed=is_distributed,
        rank=global_rank,
        world_size=world_size,
        seed=seed,
        return_sampler=True,
        **loader_kwargs,
    )
    valid_dataset = valid_loader.dataset

    # Experiment Provenance Fingerprinting
    split_hash = (
        compute_split_hash_from_file(split_file)
        if split_file and os.path.exists(split_file)
        else "UNKNOWN_SPLIT"
    )
    normalizer_hash = compute_normalizer_hash(normalizer)
    training_git_commit = get_git_commit(PROJECT_ROOT)
    training_git_dirty = is_git_dirty(PROJECT_ROOT)

    # Initialize model
    if model_type == "latent_transformer":
        encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
        # Pressure gauge projection is performed on denormalized physical fields, keep decoder raw
        decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
        if os.path.exists(repr_checkpoint):
            if global_rank == 0:
                print(f"Loading pretrained representation weights from {repr_checkpoint}")
            ckpt = torch.load(repr_checkpoint, map_location="cpu")
            required_keys = {"encoder_state_dict", "decoder_state_dict"}
            missing_keys = required_keys - set(ckpt.keys())
            if missing_keys:
                raise KeyError(
                    f"Representation checkpoint at '{repr_checkpoint}' is missing required keys: {sorted(list(missing_keys))}. "
                    f"Keys found: {list(ckpt.keys())}. Aborting to avoid training with uninitialized/random frozen weights!"
                )
            encoder.load_state_dict(ckpt["encoder_state_dict"])
            decoder.load_state_dict(ckpt["decoder_state_dict"])
        elif freeze_representation:
            raise FileNotFoundError(
                f"Representation checkpoint '{repr_checkpoint}' does not exist! "
                f"When freeze_representation=True, a valid pretrained autoencoder checkpoint "
                f"is strictly required to prevent training against random, frozen latent representations. "
                f"Please train Stage B representation first or specify --repr_checkpoint."
            )
        else:
            if global_rank == 0:
                print(f"Notice: repr_checkpoint '{repr_checkpoint}' not found, but --joint training is active. "
                      f"Initializing representation weights from scratch for joint training.")

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

    # Load weights from init_checkpoint if provided (warm-start dynamics model)
    parent_checkpoint_path = None
    parent_checkpoint_sha256 = None
    if init_checkpoint is not None:
        if not os.path.exists(init_checkpoint):
            raise FileNotFoundError(f"init_checkpoint not found at '{init_checkpoint}'")

        hasher = hashlib.sha256()
        with open(init_checkpoint, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        parent_checkpoint_sha256 = hasher.hexdigest()
        parent_checkpoint_path = init_checkpoint

        if global_rank == 0:
            print(
                f"Loading dynamics weights from init_checkpoint: {init_checkpoint} "
                f"(SHA256: {parent_checkpoint_sha256[:12]}...)"
            )

        init_ckpt = torch.load(init_checkpoint, map_location="cpu")

        # Enforce fail-closed semantic contract validation for warm-start parent checkpoint
        validate_init_checkpoint_contract(
            init_ckpt=init_ckpt,
            ckpt_path=init_checkpoint,
            requested_model_type=model_type,
            current_split_hash=split_hash,
            current_normalizer_hash=normalizer_hash,
            expected_seed=seed,
            expected_horizon=expected_init_horizon,
            expected_lambda_div=lambda_div,
            expected_lambda_vort=lambda_vort,
            expected_protocol=PHYSICS_PROTOCOL,
            expected_axis_contract=SPATIAL_AXIS_CONTRACT,
            expected_domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY,
            expected_prediction_mode=prediction_mode,
            expected_use_condition=use_condition,
            fail_closed=True,
        )

        # 4. Load weights strictly into model (without optimizer state)
        state_dict_to_load = init_ckpt.get("model_state_dict") or init_ckpt.get("state_dict") or init_ckpt
        load_msg = model.load_state_dict(state_dict_to_load, strict=True)
        if global_rank == 0:
            print(f"Successfully loaded dynamics model_state_dict: {load_msg}")

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
    div_loss_fn = DivergenceLoss(domain_size=(1.0, 2.0)).to(device)
    vort_loss_fn = VorticityLoss(domain_size=(1.0, 2.0)).to(device)

    tracker = None
    long_tracker = None
    if global_rank == 0:
        tracker = BestCheckpointTracker(save_dir=output_dir, metric_name="vrmse_mean", mode="min", keep_top_k=3)
        if val_diagnostic_horizons and {10, 20, 30}.issubset(set(val_diagnostic_horizons)):
            long_tracker = BestCheckpointTracker(save_dir=output_dir, metric_name="long_vrmse", mode="min", keep_top_k=3)
        effective_batch_size = batch_size * grad_accum_steps
        diag_info = f" | Diag Horizons: {val_diagnostic_horizons}" if val_diagnostic_horizons else ""
        warm_info = f" | Warm-Start from: {parent_checkpoint_path} (SHA256: {parent_checkpoint_sha256[:8]})" if parent_checkpoint_path else ""
        print(
            f"Training {model_type} on {device} (Distributed: {is_distributed}, World Size: {world_size}) | "
            f"Horizon: {horizon} | Epochs: {epochs} | Microbatch: {batch_size} | "
            f"Grad Accum: {grad_accum_steps} (Effective Batch: {effective_batch_size}) | LR: {lr}{diag_info}{warm_info}"
        )

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(1, epochs + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        accum_count = 0

        # Precompute microbatch sample counts for exact sample-weighted gradient accumulation
        num_batches = len(train_loader)
        is_drop_last = getattr(train_loader, "drop_last", False)
        sampler = getattr(train_loader, "sampler", None)
        total_samples = len(sampler) if sampler is not None else len(train_loader.dataset)

        if is_drop_last:
            batch_sample_counts = [batch_size] * num_batches
        else:
            remainder = total_samples % batch_size
            last_batch_size = remainder if remainder != 0 else batch_size
            batch_sample_counts = [batch_size] * (num_batches - 1) + [last_batch_size] if num_batches > 0 else []

        for batch_idx, batch in enumerate(train_loader):
            q_hist = batch["history"].to(device)  # (B, L, 4, Ny, Nx)
            q_future = batch["future"].to(device)  # (B, H, 4, Ny, Nx)
            re = batch["re"].to(device) if use_condition else None
            sc = batch["sc"].to(device) if use_condition else None

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

                if field_loss_space == "physical" and normalizer is not None:
                    pred_field_loss = normalizer.denormalize(pred)
                    target_field_loss = normalizer.denormalize(q_future)
                else:
                    pred_field_loss = pred
                    target_field_loss = q_future

                loss = rollout_loss_fn(pred_field_loss, target_field_loss)

                # Physics losses must strictly be computed in physical dimensional space
                if lambda_div > 0 or lambda_vort > 0:
                    if normalizer is not None:
                        pred_phys = normalizer.denormalize(pred)
                        target_phys = normalizer.denormalize(q_future)
                    else:
                        pred_phys = pred
                        target_phys = q_future

                    if lambda_div > 0:
                        loss = loss + lambda_div * div_loss_fn(pred_phys)
                    if lambda_vort > 0:
                        loss = loss + lambda_vort * vort_loss_fn(pred_phys, target_phys)

                # Exact sample-weighted accumulation window scaling
                # Guarantees mathematical equivalence at the sample level for tail microbatches
                window_start = (batch_idx // grad_accum_steps) * grad_accum_steps
                window_end = min(window_start + grad_accum_steps, num_batches)
                window_total_samples = sum(batch_sample_counts[window_start:window_end])
                sample_weight = len(q_hist) / window_total_samples
                scaled_loss = loss * sample_weight

            scaler.scale(scaled_loss).backward()
            accum_count += 1

            if accum_count == grad_accum_steps or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_count = 0

            train_loss += loss.item() * len(q_hist)

        train_loss /= len(train_loader.dataset)

        # Validation only on rank 0
        if global_rank == 0:
            model.eval()
            eval_horizon = effective_valid_horizon
            rollout_step_metrics_sum = {h: {} for h in range(eval_horizon)}
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=use_amp):
                    for batch in valid_loader:
                        q_hist = batch["history"].to(device)
                        q_future = batch["future"].to(device)
                        re = batch["re"].to(device) if use_condition else None
                        sc = batch["sc"].to(device) if use_condition else None

                        eval_model = model.module if is_distributed else model
                        if model_type == "latent_transformer":
                            if eval_horizon == 1:
                                pred = eval_model.forward_single_step(q_hist, re, sc)
                            else:
                                pred = eval_model.forward_rollout(q_hist, re, sc, horizon=eval_horizon)
                        elif model_type in ("direct_transformer", "pde_transformer"):
                            if eval_horizon == 1:
                                pred = eval_model(q_hist, re=re, sc=sc)
                            else:
                                buf = HistoryBuffer(history_length=q_hist.shape[1])
                                buf.reset(q_hist)
                                pred = buf.rollout(lambda hist, _c: eval_model(hist, re=re, sc=sc), steps=eval_horizon)
                        elif model_type == "fno":
                            if eval_horizon == 1:
                                pred = eval_model(q_hist)
                            else:
                                pred_list = []
                                hist_window = q_hist
                                for _ in range(eval_horizon):
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

                        # Enforce zero-mean pressure gauge in physical space
                        pred_eval[:, :, 2:3, :, :] = pred_eval[:, :, 2:3, :, :] - pred_eval[:, :, 2:3, :, :].mean(dim=(-2, -1), keepdim=True)
                        target_eval[:, :, 2:3, :, :] = target_eval[:, :, 2:3, :, :] - target_eval[:, :, 2:3, :, :].mean(dim=(-2, -1), keepdim=True)

                        b_samples = len(q_hist)
                        for h in range(eval_horizon):
                            step_m = evaluate_field_metrics(pred_eval[:, h], target_eval[:, h])
                            for k, v in step_m.items():
                                rollout_step_metrics_sum[h][k] = rollout_step_metrics_sum[h].get(k, 0.0) + v * b_samples

            n_val = len(valid_dataset)
            val_step_metrics = {
                h: {k: v / n_val for k, v in rollout_step_metrics_sum[h].items()}
                for h in range(eval_horizon)
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

            diag_strs = []
            if val_diagnostic_horizons:
                for dh in sorted(val_diagnostic_horizons):
                    if dh <= eval_horizon:
                        dh_vrmse = val_step_metrics[dh - 1]["vrmse_mean"]
                        val_metrics[f"val_h{dh}_vrmse"] = dh_vrmse
                        diag_strs.append(f"h{dh}: {dh_vrmse:.4f}")
            diag_suffix = f" | Diag [{', '.join(diag_strs)}]" if diag_strs else ""

            vram_gb = torch.cuda.max_memory_allocated(device=device) / (1024**3) if torch.cuda.is_available() else 0.0
            if horizon > 1:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4e} | "
                    f"Val Rollout Mean VRMSE: {val_metrics['rollout_mean_vrmse']:.4f} | "
                    f"Step 1 VRMSE: {val_metrics['vrmse_mean']:.4f} "
                    f"(u: {val_metrics['vrmse_u']:.4f}, v: {val_metrics['vrmse_v']:.4f}, "
                    f"p: {val_metrics['vrmse_p']:.4f}, s: {val_metrics['vrmse_s']:.4f})"
                    f"{diag_suffix} | Max VRAM: {vram_gb:.2f} GB"
                )
            else:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4e} | "
                    f"Val VRMSE Mean: {val_metrics['vrmse_mean']:.4f} "
                    f"(u: {val_metrics['vrmse_u']:.4f}, v: {val_metrics['vrmse_v']:.4f}, "
                    f"p: {val_metrics['vrmse_p']:.4f}, s: {val_metrics['vrmse_s']:.4f})"
                    f"{diag_suffix} | Max VRAM: {vram_gb:.2f} GB"
                )

            raw_model = model.module if is_distributed else model
            state = {
                "epoch": epoch,
                "model_type": model_type,
                "prediction_mode": prediction_mode,
                "use_condition": use_condition,
                "downsample_factor": downsample_factor,
                "physics_protocol": PHYSICS_PROTOCOL,
                "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
                "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
                "training_git_commit": training_git_commit,
                "training_git_dirty": training_git_dirty,
                "seed": seed,
                "split_type": split_type,
                "split_hash": split_hash,
                "normalizer_hash": normalizer_hash,
                "parent_checkpoint_path": parent_checkpoint_path,
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": batch_size * grad_accum_steps,
                "val_diagnostic_horizons": val_diagnostic_horizons,
                "config": {
                    "model_type": model_type,
                    "physics_protocol": PHYSICS_PROTOCOL,
                    "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
                    "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
                    "training_git_commit": training_git_commit,
                    "training_git_dirty": training_git_dirty,
                    "seed": seed,
                    "split_type": split_type,
                    "split_hash": split_hash,
                    "normalizer_hash": normalizer_hash,
                    "prediction_mode": prediction_mode,
                    "use_condition": use_condition,
                    "downsample_factor": downsample_factor,
                    "embed_dim": embed_dim,
                    "depth": depth,
                    "num_heads": num_heads,
                    "horizon": horizon,
                    "freeze_representation": freeze_representation,
                    "lambda_div": lambda_div,
                    "lambda_vort": lambda_vort,
                    "field_loss_space": field_loss_space,
                    "lr": lr,
                    "batch_size": batch_size,
                    "grad_accum_steps": grad_accum_steps,
                    "effective_batch_size": batch_size * grad_accum_steps,
                    "parent_checkpoint_path": parent_checkpoint_path,
                    "parent_checkpoint_sha256": parent_checkpoint_sha256,
                    "val_diagnostic_horizons": val_diagnostic_horizons,
                    "train_stride": train_stride,
                    "valid_stride": valid_stride,
                    "split_type": split_type,
                    "normalize": normalize,
                },
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "val_step_metrics": val_step_metrics,
                "horizon": horizon,
            }
            tracker.update(val_criterion, state, epoch)

            # Dual tracking: save true long-best checkpoint if diagnostic horizons are monitored
            if long_tracker is not None and all(f"val_h{dh}_vrmse" in val_metrics for dh in [10, 20, 30]):
                j_long = (val_metrics["val_h10_vrmse"] + val_metrics["val_h20_vrmse"] + val_metrics["val_h30_vrmse"]) / 3.0
                val_metrics["j_long"] = j_long
                long_state = dict(state)
                long_state["j_long"] = j_long
                long_state["selection_criterion"] = "j_long"
                is_best_long = long_tracker.update(j_long, long_state, epoch)
                if is_best_long:
                    print(f"  >>> New Best J_long: {j_long:.4f} (Saved to best_long_vrmse.pt)")

    if global_rank == 0:
        if tracker is not None:
            print(f"Training completed. Best VRMSE: {tracker.best_score:.4f}")
        if long_tracker is not None:
            print(f"Best J_long: {long_tracker.best_score:.4f} (Saved at best_long_vrmse.pt)")

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
    parser.add_argument(
        "--split_type",
        type=str,
        default="grouped",
        choices=["grouped", "official", "parameter_holdout_re", "parameter_holdout_sc", "parameter_holdout_split"],
    )
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--train_stride", type=int, default=2)
    parser.add_argument("--valid_stride", type=int, default=8)
    parser.add_argument("--downsample_factor", type=int, default=2, help="Spatial downsampling factor (default: 2 for 128x256)")
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
    parser.add_argument(
        "--field_loss_space",
        type=str,
        default="normalized",
        choices=["normalized", "physical"],
        help="Space to compute field prediction loss: 'normalized' (balanced channel variance) or 'physical'.",
    )
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--preload_to_memory", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training reproducibility")
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for warm-start dynamics model weights.",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (default: 1).",
    )
    parser.add_argument(
        "--val_diagnostic_horizons",
        type=int,
        nargs="+",
        default=None,
        help="Auxiliary rollout horizons to track on validation set (e.g. 10 20 30).",
    )
    parser.add_argument(
        "--expected_init_horizon",
        type=int,
        default=None,
        help="Expected prediction horizon of init_checkpoint for fail-closed lineage validation.",
    )
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
        downsample_factor=args.downsample_factor,
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
        field_loss_space=args.field_loss_space,
        seed=args.seed,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_workers=args.num_workers,
        preload_to_memory=args.preload_to_memory,
        use_amp=args.use_amp,
        init_checkpoint=args.init_checkpoint,
        grad_accum_steps=args.grad_accum_steps,
        val_diagnostic_horizons=args.val_diagnostic_horizons,
        expected_init_horizon=args.expected_init_horizon,
    )
