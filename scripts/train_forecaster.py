"""Stage C & D: Dynamics Model Training (Single-Step & Rollout-Aware Training)."""

import argparse
import glob
import hashlib
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
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
from src.losses.spectral import EnergySpectrumLoss
from src.metrics.field import evaluate_field_metrics
from src.metrics.spectral import compute_spectral_error
from src.models.decoder import Decoder2D
from src.models.direct_transformer import DirectSTTransformer
from src.models.encoder import Encoder2D
from src.models.history_buffer import HistoryBuffer
from src.models.latent_transformer import LatentSTTransformer
from src.training import CurriculumConfig, CurriculumRolloutScheduler
from src.utils.checkpoint import BestCheckpointTracker, load_checkpoint, save_checkpoint, strip_compiled_prefix
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
        pushforward_steps: int = 0,
        noise_std: float = 0.0,
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

        if pushforward_steps > 0:
            buf.pushforward(step_fn, steps=pushforward_steps, noise_std=noise_std)

        # Rollout purely in latent space
        z_rollout = buf.rollout(step_fn, steps=horizon, noise_std=noise_std)
        # Decode entire rollout trajectory
        q_rollout = self.decoder(z_rollout)
        return q_rollout

    def forward(
        self,
        q_hist: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        sc: Optional[torch.Tensor] = None,
        horizon: int = 1,
        pushforward_steps: int = 0,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Unified forward interface."""
        if horizon == 1 and pushforward_steps == 0:
            return self.forward_single_step(q_hist, re, sc)
        else:
            return self.forward_rollout(
                q_hist,
                re=re,
                sc=sc,
                horizon=horizon,
                pushforward_steps=pushforward_steps,
                noise_std=noise_std,
            )


def _init_distributed_context(
    device_str: str,
    seed: int,
) -> Tuple[bool, int, int, int, torch.device]:
    """Initialize Distributed Data Parallel (DDP) environment or fallback to single device."""
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
    return is_distributed, local_rank, global_rank, world_size, device


def _resolve_split_file(split_type: str, split_file: Optional[str]) -> Optional[str]:
    """Resolve data split file path from split_type or explicit path."""
    if split_file is not None:
        return split_file
    if split_type.endswith(".json") or split_type.startswith("outputs/splits/"):
        return split_type
    candidate = f"outputs/splits/{split_type}.json"
    if os.path.exists(candidate):
        return candidate
    return f"outputs/splits/{split_type}_split.json"


def _build_model(
    model_type: str,
    embed_dim: int,
    depth: int,
    num_heads: int,
    prediction_mode: str,
    freeze_representation: bool,
    repr_checkpoint: str,
    init_checkpoint: Optional[str],
    split_hash: str,
    normalizer_hash: str,
    seed: int,
    expected_init_horizon: Optional[int],
    lambda_div: float,
    lambda_vort: float,
    use_condition: bool,
    device: torch.device,
    is_distributed: bool,
    local_rank: int,
    global_rank: int,
    compile_model: bool = False,
) -> Tuple[nn.Module, Optional[str], Optional[str]]:
    """Construct neural model architecture, load representation/warm-start checkpoints."""
    if model_type == "latent_transformer":
        encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
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
                print(
                    f"Notice: repr_checkpoint '{repr_checkpoint}' not found, but --joint training is active. "
                    f"Initializing representation weights from scratch for joint training."
                )

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
            in_channels=16,
            out_channels=4,
            modes1=16,
            modes2=16,
            width=64,
            num_layers=4,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

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

        state_dict_to_load = init_ckpt.get("model_state_dict") or init_ckpt.get("state_dict") or init_ckpt
        state_dict_to_load = strip_compiled_prefix(state_dict_to_load)
        load_msg = model.load_state_dict(state_dict_to_load, strict=True)
        if global_rank == 0:
            print(f"Successfully loaded dynamics model_state_dict: {load_msg}")

    if compile_model:
        if hasattr(torch, "compile"):
            if global_rank == 0:
                print("Enabling torch.compile(dynamic=True) for model acceleration...")
            try:
                if model_type == "latent_transformer":
                    model.transformer = torch.compile(model.transformer, dynamic=True)
                    model.decoder = torch.compile(model.decoder, dynamic=True)
                else:
                    model = torch.compile(model, dynamic=True)
            except Exception as e:
                if global_rank == 0:
                    print(f"Warning: torch.compile failed with error: {e}. Falling back to uncompiled execution.")
        else:
            if global_rank == 0:
                print("Notice: torch.compile not available in this PyTorch version. Skipping compilation.")

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    return model, parent_checkpoint_path, parent_checkpoint_sha256


def _forward_model_prediction(
    model: nn.Module,
    model_type: str,
    q_hist: torch.Tensor,
    re: Optional[torch.Tensor],
    sc: Optional[torch.Tensor],
    horizon: int,
    is_distributed: bool,
    pushforward_steps: int = 0,
    noise_std: float = 0.0,
) -> torch.Tensor:
    """Compute model forecast prediction for a given horizon."""
    if model_type == "latent_transformer":
        if is_distributed:
            return model(q_hist, re, sc, horizon=horizon, pushforward_steps=pushforward_steps, noise_std=noise_std)
        if horizon == 1 and pushforward_steps == 0 and hasattr(model, "forward_single_step"):
            return model.forward_single_step(q_hist, re, sc)
        if hasattr(model, "forward_rollout"):
            return model.forward_rollout(
                q_hist, re, sc, horizon=horizon, pushforward_steps=pushforward_steps, noise_std=noise_std
            )
        return model(
            q_hist, re=re, sc=sc, horizon=horizon, pushforward_steps=pushforward_steps, noise_std=noise_std
        )

    if model_type in ("direct_transformer", "pde_transformer"):
        if horizon == 1 and pushforward_steps == 0:
            return model(q_hist, re=re, sc=sc)
        buf = HistoryBuffer(history_length=q_hist.shape[1])
        buf.reset(q_hist)
        step_fn = lambda hist, _c: model(hist, re=re, sc=sc)
        if pushforward_steps > 0:
            buf.pushforward(step_fn, steps=pushforward_steps, noise_std=noise_std)
        return buf.rollout(step_fn, steps=horizon, noise_std=noise_std)

    if model_type == "fno":
        if horizon == 1 and pushforward_steps == 0:
            return model(q_hist)
        buf = HistoryBuffer(history_length=q_hist.shape[1])
        buf.reset(q_hist)
        step_fn = lambda hist, _c: model(hist)
        if pushforward_steps > 0:
            buf.pushforward(step_fn, steps=pushforward_steps, noise_std=noise_std)
        return buf.rollout(step_fn, steps=horizon, noise_std=noise_std)

    raise ValueError(f"Unknown model_type: {model_type}")


def _compute_batch_loss(
    pred: torch.Tensor,
    q_future: torch.Tensor,
    normalizer: Optional[Any],
    field_loss_space: str,
    lambda_div: float,
    lambda_vort: float,
    rollout_loss_fn: nn.Module,
    div_loss_fn: Optional[nn.Module] = None,
    vort_loss_fn: Optional[nn.Module] = None,
    lambda_spec: float = 0.0,
    spec_loss_fn: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Compute field loss, physical conservation penalties, and multi-scale spectral loss."""
    if field_loss_space == "physical" and normalizer is not None:
        pred_field_loss = normalizer.denormalize(pred)
        target_field_loss = normalizer.denormalize(q_future)
    else:
        pred_field_loss = pred
        target_field_loss = q_future

    loss = rollout_loss_fn(pred_field_loss, target_field_loss)

    if lambda_div > 0 or lambda_vort > 0 or lambda_spec > 0:
        if normalizer is not None:
            pred_phys = normalizer.denormalize(pred)
            target_phys = normalizer.denormalize(q_future)
        else:
            pred_phys = pred
            target_phys = q_future

        if lambda_div > 0 and div_loss_fn is not None:
            loss = loss + lambda_div * div_loss_fn(pred_phys)
        if lambda_vort > 0 and vort_loss_fn is not None:
            loss = loss + lambda_vort * vort_loss_fn(pred_phys, target_phys)
        if lambda_spec > 0 and spec_loss_fn is not None:
            loss = loss + lambda_spec * spec_loss_fn(pred_phys, target_phys)

    return loss


def _train_epoch(
    model: nn.Module,
    model_type: str,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    normalizer: Optional[Any],
    rollout_loss_fn: nn.Module,
    div_loss_fn: Optional[nn.Module],
    vort_loss_fn: Optional[nn.Module],
    horizon: int,
    grad_accum_steps: int,
    field_loss_space: str,
    lambda_div: float,
    lambda_vort: float,
    use_condition: bool,
    use_amp: bool,
    device: torch.device,
    is_distributed: bool,
    train_sampler: Optional[Any],
    epoch: int,
    pushforward_steps: int = 0,
    pushforward_noise_std: float = 0.0,
    pushforward_mode: str = "future",
    lambda_spec: float = 0.0,
    spec_loss_fn: Optional[nn.Module] = None,
) -> float:
    """Execute one training epoch with exact sample-weighted gradient accumulation."""
    if is_distributed and train_sampler is not None:
        train_sampler.set_epoch(epoch)

    model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    accum_count = 0

    num_batches = len(train_loader)
    is_drop_last = getattr(train_loader, "drop_last", False)
    sampler = getattr(train_loader, "sampler", None)
    if sampler is not None:
        total_samples = len(sampler)
    elif hasattr(train_loader, "dataset"):
        total_samples = len(train_loader.dataset)
    else:
        total_samples = len(train_loader)
    batch_size = getattr(train_loader, "batch_size", 1) or 1

    if is_drop_last:
        batch_sample_counts = [batch_size] * num_batches
    else:
        remainder = total_samples % batch_size
        last_batch_size = remainder if remainder != 0 else batch_size
        batch_sample_counts = [batch_size] * (num_batches - 1) + [last_batch_size] if num_batches > 0 else []

    for batch_idx, batch in enumerate(train_loader):
        q_hist = batch["history"].to(device)
        q_future = batch["future"].to(device)
        re = batch["re"].to(device) if use_condition else None
        sc = batch["sc"].to(device) if use_condition else None

        avail_future = q_future.shape[1]
        if pushforward_mode == "future" and pushforward_steps > 0:
            eff_push = min(pushforward_steps, max(0, avail_future - horizon))
            target_future = q_future[:, eff_push : eff_push + horizon]
        else:
            eff_push = pushforward_steps if pushforward_mode == "history" else 0
            target_future = q_future[:, :horizon]

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred = _forward_model_prediction(
                model=model,
                model_type=model_type,
                q_hist=q_hist,
                re=re,
                sc=sc,
                horizon=horizon,
                is_distributed=is_distributed,
                pushforward_steps=eff_push,
                noise_std=pushforward_noise_std,
            )
            loss = _compute_batch_loss(
                pred=pred,
                q_future=target_future,
                normalizer=normalizer,
                field_loss_space=field_loss_space,
                lambda_div=lambda_div,
                lambda_vort=lambda_vort,
                rollout_loss_fn=rollout_loss_fn,
                div_loss_fn=div_loss_fn,
                vort_loss_fn=vort_loss_fn,
                lambda_spec=lambda_spec,
                spec_loss_fn=spec_loss_fn,
            )

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

    if is_distributed:
        import torch.distributed as dist
        loss_tensor = torch.tensor([train_loss, float(total_samples)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return loss_tensor[0].item() / max(loss_tensor[1].item(), 1.0)
    return train_loss / max(total_samples, 1)


def _validate_epoch(
    model: nn.Module,
    model_type: str,
    valid_loader: DataLoader,
    normalizer: Optional[Any],
    horizon: int,
    effective_valid_horizon: int,
    val_diagnostic_horizons: Optional[List[int]],
    use_condition: bool,
    use_amp: bool,
    device: torch.device,
    is_distributed: bool,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]], float]:
    """Evaluate model on validation set with physical metrics and diagnostics."""
    model.eval()
    eval_horizon = effective_valid_horizon
    rollout_step_metrics_sum = {h: {} for h in range(eval_horizon)}
    eval_model = model.module if is_distributed else model

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp):
            for batch in valid_loader:
                q_hist = batch["history"].to(device)
                q_future = batch["future"].to(device)
                re = batch["re"].to(device) if use_condition else None
                sc = batch["sc"].to(device) if use_condition else None

                pred = _forward_model_prediction(
                    model=eval_model,
                    model_type=model_type,
                    q_hist=q_hist,
                    re=re,
                    sc=sc,
                    horizon=eval_horizon,
                    is_distributed=False,
                )

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

                    # Compute spectral error metrics for physical velocity fields
                    pred_u, pred_v = pred_eval[:, h, 0], pred_eval[:, h, 1]
                    target_u, target_v = target_eval[:, h, 0], target_eval[:, h, 1]
                    spec_m = compute_spectral_error(pred_u, pred_v, target_u, target_v, domain_size=(1.0, 2.0))
                    for k, v in spec_m.items():
                        rollout_step_metrics_sum[h][k] = rollout_step_metrics_sum[h].get(k, 0.0) + v * b_samples

    n_val = len(valid_loader.dataset)
    val_step_metrics = {
        h: {k: v / n_val for k, v in rollout_step_metrics_sum[h].items()}
        for h in range(eval_horizon)
    }

    val_metrics = dict(val_step_metrics[0])
    if horizon > 1:
        rollout_mean_vrmse = sum(val_step_metrics[h]["vrmse_mean"] for h in range(horizon)) / horizon
        rollout_mean_rmse = sum(val_step_metrics[h]["rmse_mean"] for h in range(horizon)) / horizon
        val_metrics["rollout_mean_vrmse"] = rollout_mean_vrmse
        val_metrics["rollout_mean_rmse"] = rollout_mean_rmse
        val_metrics["rollout_mean_spec_err"] = sum(val_step_metrics[h]["spec_err_total"] for h in range(horizon)) / horizon
        val_criterion = rollout_mean_vrmse
    else:
        val_criterion = val_metrics["vrmse_mean"]

    if val_diagnostic_horizons:
        for dh in sorted(val_diagnostic_horizons):
            if dh <= eval_horizon:
                val_metrics[f"val_h{dh}_vrmse"] = val_step_metrics[dh - 1]["vrmse_mean"]

    return val_metrics, val_step_metrics, val_criterion


def _save_checkpoint_artifacts(
    tracker: BestCheckpointTracker,
    long_tracker: Optional[BestCheckpointTracker],
    val_criterion: float,
    val_metrics: Dict[str, float],
    val_step_metrics: Dict[int, Dict[str, float]],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config_dict: Dict[str, Any],
    metadata_dict: Dict[str, Any],
    is_distributed: bool,
):
    """Save epoch checkpoint and update best score trackers."""
    raw_model = model.module if is_distributed else model
    # Strip _orig_mod. prefixes from torch.compile before persisting,
    # so checkpoints are always loadable by uncompiled evaluation scripts.
    clean_state_dict = strip_compiled_prefix(raw_model.state_dict())
    state = {
        "epoch": epoch,
        **metadata_dict,
        "config": config_dict,
        "model_state_dict": clean_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "val_step_metrics": val_step_metrics,
        "horizon": config_dict["horizon"],
    }
    tracker.update(val_criterion, state, epoch)

    if long_tracker is not None and all(f"val_h{dh}_vrmse" in val_metrics for dh in [10, 20, 30]):
        j_long = (val_metrics["val_h10_vrmse"] + val_metrics["val_h20_vrmse"] + val_metrics["val_h30_vrmse"]) / 3.0
        val_metrics["j_long"] = j_long
        long_state = dict(state)
        long_state["j_long"] = j_long
        long_state["selection_criterion"] = "j_long"
        is_best_long = long_tracker.update(j_long, long_state, epoch)
        if is_best_long:
            print(f"  >>> New Best J_long: {j_long:.4f} (Saved to best_long_vrmse.pt)")


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
    compile_model: bool = False,
    curriculum_rollout: bool = False,
    curriculum_start_horizon: int = 2,
    curriculum_step_epochs: int = 3,
    curriculum_schedule: str = "doubling",
    pushforward_steps: int = 0,
    pushforward_noise_std: float = 0.0,
    pushforward_mode: str = "future",
    lambda_spec: float = 0.0,
    spec_loss_type: str = "log_l1",
    spec_high_freq_weight: float = 0.0,
):
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

    is_distributed, local_rank, global_rank, world_size, device = _init_distributed_context(device_str, seed)
    if global_rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    effective_start_horizon = (
        min(curriculum_start_horizon, horizon) if not curriculum_rollout else curriculum_start_horizon
    )
    curriculum_config = CurriculumConfig(
        enabled=curriculum_rollout,
        start_horizon=effective_start_horizon,
        target_horizon=horizon,
        step_epochs=curriculum_step_epochs,
        schedule=curriculum_schedule,
        pushforward_steps=pushforward_steps,
        pushforward_noise_std=pushforward_noise_std,
        pushforward_mode=pushforward_mode,
    )
    scheduler = CurriculumRolloutScheduler(curriculum_config)

    effective_train_horizon = horizon
    if curriculum_rollout:
        effective_train_horizon = max(effective_train_horizon, curriculum_config.target_horizon)
    if pushforward_steps > 0 and pushforward_mode == "future":
        effective_train_horizon += pushforward_steps

    split_file = _resolve_split_file(split_type, split_file)
    effective_valid_horizon = max([horizon] + val_diagnostic_horizons) if val_diagnostic_horizons else horizon

    train_loader, valid_loader, test_loader, normalizer, train_sampler = create_flow_dataloaders(
        split_type=split_type,
        split_file=split_file,
        data_root=data_dir,
        history_length=4,
        horizon=effective_train_horizon,
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
        **({"stats_dir": stats_dir} if stats_dir is not None else {}),
    )

    split_hash = (
        compute_split_hash_from_file(split_file)
        if split_file and os.path.exists(split_file)
        else "UNKNOWN_SPLIT"
    )
    normalizer_hash = compute_normalizer_hash(normalizer)
    training_git_commit = get_git_commit(PROJECT_ROOT)
    training_git_dirty = is_git_dirty(PROJECT_ROOT)

    model, parent_checkpoint_path, parent_checkpoint_sha256 = _build_model(
        model_type=model_type,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        prediction_mode=prediction_mode,
        freeze_representation=freeze_representation,
        repr_checkpoint=repr_checkpoint,
        init_checkpoint=init_checkpoint,
        split_hash=split_hash,
        normalizer_hash=normalizer_hash,
        seed=seed,
        expected_init_horizon=expected_init_horizon,
        lambda_div=lambda_div,
        lambda_vort=lambda_vort,
        use_condition=use_condition,
        device=device,
        is_distributed=is_distributed,
        local_rank=local_rank,
        global_rank=global_rank,
        compile_model=compile_model,
    )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    field_loss_fn = FieldLoss(loss_type="mse").to(device)
    rollout_loss_fn = RolloutLoss(field_loss=field_loss_fn).to(device)
    div_loss_fn = DivergenceLoss(domain_size=(1.0, 2.0)).to(device)
    vort_loss_fn = VorticityLoss(domain_size=(1.0, 2.0)).to(device)
    spec_loss_fn = (
        EnergySpectrumLoss(
            domain_size=(1.0, 2.0),
            loss_type=spec_loss_type,
            high_freq_weight=spec_high_freq_weight,
        ).to(device)
        if lambda_spec > 0
        else None
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    tracker = None
    long_tracker = None
    if global_rank == 0:
        tracker = BestCheckpointTracker(save_dir=output_dir, metric_name="vrmse_mean", mode="min", keep_top_k=3)
        if val_diagnostic_horizons and {10, 20, 30}.issubset(set(val_diagnostic_horizons)):
            long_tracker = BestCheckpointTracker(save_dir=output_dir, metric_name="long_vrmse", mode="min", keep_top_k=3)
        effective_batch_size = batch_size * grad_accum_steps * (world_size if is_distributed else 1)
        diag_info = f" | Diag Horizons: {val_diagnostic_horizons}" if val_diagnostic_horizons else ""
        warm_info = f" | Warm-Start from: {parent_checkpoint_path} (SHA256: {parent_checkpoint_sha256[:8]})" if parent_checkpoint_path else ""
        print(
            f"Training {model_type} on {device} (Distributed: {is_distributed}, World Size: {world_size}) | "
            f"Horizon: {horizon} | Epochs: {epochs} | Microbatch: {batch_size} | "
            f"Grad Accum: {grad_accum_steps} (Effective Batch: {effective_batch_size}) | LR: {lr}{diag_info}{warm_info}"
        )

    for epoch in range(1, epochs + 1):
        epoch_horizon = scheduler.get_horizon(epoch)
        epoch_pushforward = scheduler.get_pushforward_steps(epoch)
        epoch_noise = scheduler.get_noise_std(epoch)

        train_loss = _train_epoch(
            model=model,
            model_type=model_type,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            normalizer=normalizer,
            rollout_loss_fn=rollout_loss_fn,
            div_loss_fn=div_loss_fn,
            vort_loss_fn=vort_loss_fn,
            horizon=epoch_horizon,
            grad_accum_steps=grad_accum_steps,
            field_loss_space=field_loss_space,
            lambda_div=lambda_div,
            lambda_vort=lambda_vort,
            use_condition=use_condition,
            use_amp=use_amp,
            device=device,
            is_distributed=is_distributed,
            train_sampler=train_sampler,
            epoch=epoch,
            pushforward_steps=epoch_pushforward,
            pushforward_noise_std=epoch_noise,
            pushforward_mode=curriculum_config.pushforward_mode,
            lambda_spec=lambda_spec,
            spec_loss_fn=spec_loss_fn,
        )

        if global_rank == 0:
            val_metrics, val_step_metrics, val_criterion = _validate_epoch(
                model=model,
                model_type=model_type,
                valid_loader=valid_loader,
                normalizer=normalizer,
                horizon=horizon,
                effective_valid_horizon=effective_valid_horizon,
                val_diagnostic_horizons=val_diagnostic_horizons,
                use_condition=use_condition,
                use_amp=use_amp,
                device=device,
                is_distributed=is_distributed,
            )

            diag_strs = [
                f"h{dh}: {val_metrics[f'val_h{dh}_vrmse']:.4f}"
                for dh in sorted(val_diagnostic_horizons or [])
                if f"val_h{dh}_vrmse" in val_metrics
            ]
            diag_suffix = f" | Diag [{', '.join(diag_strs)}]" if diag_strs else ""
            spec_suffix = f" | Spec Err: {val_metrics['spec_err_total']:.4f}" if "spec_err_total" in val_metrics else ""
            vram_gb = torch.cuda.max_memory_allocated(device=device) / (1024**3) if torch.cuda.is_available() else 0.0

            curric_info = f" (Train H: {epoch_horizon}, Push: {epoch_pushforward})" if (curriculum_rollout or pushforward_steps > 0) else ""

            if horizon > 1:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}]{curric_info} | Train Loss: {train_loss:.4e} | "
                    f"Val Rollout Mean VRMSE: {val_metrics['rollout_mean_vrmse']:.4f} | "
                    f"Step 1 VRMSE: {val_metrics['vrmse_mean']:.4f} "
                    f"(u: {val_metrics['vrmse_u']:.4f}, v: {val_metrics['vrmse_v']:.4f}, "
                    f"p: {val_metrics['vrmse_p']:.4f}, s: {val_metrics['vrmse_s']:.4f})"
                    f"{diag_suffix}{spec_suffix} | Max VRAM: {vram_gb:.2f} GB"
                )
            else:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}]{curric_info} | Train Loss: {train_loss:.4e} | "
                    f"Val VRMSE Mean: {val_metrics['vrmse_mean']:.4f} "
                    f"(u: {val_metrics['vrmse_u']:.4f}, v: {val_metrics['vrmse_v']:.4f}, "
                    f"p: {val_metrics['vrmse_p']:.4f}, s: {val_metrics['vrmse_s']:.4f})"
                    f"{diag_suffix}{spec_suffix} | Max VRAM: {vram_gb:.2f} GB"
                )

            metadata_dict = {
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
                "effective_batch_size": batch_size * grad_accum_steps * (world_size if is_distributed else 1),
                "world_size": world_size,
                "val_diagnostic_horizons": val_diagnostic_horizons,
                "curriculum_rollout": curriculum_rollout,
                "curriculum_config": curriculum_config.to_dict(),
            }
            config_dict = {
                **metadata_dict,
                "embed_dim": embed_dim,
                "depth": depth,
                "num_heads": num_heads,
                "horizon": horizon,
                "freeze_representation": freeze_representation,
                "lambda_div": lambda_div,
                "lambda_vort": lambda_vort,
                "lambda_spec": lambda_spec,
                "spec_loss_type": spec_loss_type,
                "spec_high_freq_weight": spec_high_freq_weight,
                "field_loss_space": field_loss_space,
                "lr": lr,
                "batch_size": batch_size,
                "train_stride": train_stride,
                "valid_stride": valid_stride,
                "normalize": normalize,
            }

            _save_checkpoint_artifacts(
                tracker=tracker,
                long_tracker=long_tracker,
                val_criterion=val_criterion,
                val_metrics=val_metrics,
                val_step_metrics=val_step_metrics,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config_dict=config_dict,
                metadata_dict=metadata_dict,
                is_distributed=is_distributed,
            )

        if is_distributed:
            import torch.distributed as dist
            dist.barrier()

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
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Enable PyTorch 2.x torch.compile(dynamic=True) for model kernel fusion acceleration.",
    )
    parser.add_argument(
        "--curriculum_rollout",
        action="store_true",
        default=False,
        help="Enable curriculum multi-step training horizon progression.",
    )
    parser.add_argument(
        "--curriculum_start_horizon",
        type=int,
        default=2,
        help="Initial training horizon at epoch 1 (default: 2).",
    )
    parser.add_argument(
        "--curriculum_step_epochs",
        type=int,
        default=3,
        help="Epoch interval to advance horizon (default: 3).",
    )
    parser.add_argument(
        "--curriculum_schedule",
        type=str,
        default="doubling",
        choices=["doubling", "linear", "fixed"],
        help="Curriculum schedule progression: doubling (2->4->8->16) or linear.",
    )
    parser.add_argument(
        "--pushforward_steps",
        type=int,
        default=0,
        help="Number of stop-gradient pushforward warmup steps (default: 0).",
    )
    parser.add_argument(
        "--pushforward_noise_std",
        type=float,
        default=0.0,
        help="Standard deviation of Gaussian noise added during pushforward rollout (default: 0.0).",
    )
    parser.add_argument(
        "--pushforward_mode",
        type=str,
        default="future",
        choices=["future", "history"],
        help="Pushforward mode: future or history (default: future).",
    )
    parser.add_argument(
        "--lambda_spec",
        type=float,
        default=0.0,
        help="Weight for multi-scale kinetic energy spectrum loss (default: 0.0).",
    )
    parser.add_argument(
        "--spec_loss_type",
        type=str,
        default="log_l1",
        choices=["log_l1", "rel_l2", "linear_l1", "combined"],
        help="Loss formulation for energy spectrum loss (default: log_l1).",
    )
    parser.add_argument(
        "--spec_high_freq_weight",
        type=float,
        default=0.0,
        help="Linear wavenumber weighting alpha for high-frequency emphasis (default: 0.0).",
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
        compile_model=args.compile,
        curriculum_rollout=args.curriculum_rollout,
        curriculum_start_horizon=args.curriculum_start_horizon,
        curriculum_step_epochs=args.curriculum_step_epochs,
        curriculum_schedule=args.curriculum_schedule,
        pushforward_steps=args.pushforward_steps,
        pushforward_noise_std=args.pushforward_noise_std,
        pushforward_mode=args.pushforward_mode,
        lambda_spec=args.lambda_spec,
        spec_loss_type=args.spec_loss_type,
        spec_high_freq_weight=args.spec_high_freq_weight,
    )
