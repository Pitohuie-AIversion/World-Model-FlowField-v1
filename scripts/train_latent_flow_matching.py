"""Latent Flow Matching (OT-CFM) Training Entrypoint.

Trains the LatentFlowMatcher on top of a strictly frozen D0 LatentForecaster.

Governance Contract:
1. Strict pre-flight identity verification (D0 SHA-256, split_hash, normalizer_hash).
   Fails closed if checkpoints or data contracts are missing or invalid.
2. Parameter freeze: Encoder, Decoder, and Transformer backbone are frozen (requires_grad=False).
3. Optimizer: Updates exclusively LatentFlowMatcher parameters (requires_grad=True).
4. Epoch 0 audit: Evaluates and logs initial baseline CFM loss on validation set before training.
5. Fail-closed on non-finite values: Any non-finite loss, gradient, or latent representation
   fails closed immediately.
6. Safe artifact isolation and overwrite: Smoke tests and formal runs use distinct directory paths.
   Existing artifacts are checked early and safely backed up if overwrite=True.
7. Model selection: Best checkpoint chosen exclusively on validation CFM loss.
"""

from datetime import datetime, timezone
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.latent_flow_matching import LatentFlowMatcher
from src.data.pipeline import create_flow_dataloaders
from src.data.normalization import FieldNormalizer
from src.utils.checkpoint import resolve_spatial_pos_config, strip_compiled_prefix
from src.utils.provenance import (
    compute_file_sha256,
    compute_normalizer_hash,
    compute_split_hash_from_file,
    resolve_checkpoint_provenance,
    hash_matches,
    get_git_commit,
    is_git_dirty,
)


def clip_and_validate_gradients(
    model_or_parameters: Union[nn.Module, Any],
    max_norm: float = 1.0,
    epoch: Optional[int] = None,
    batch_idx: Optional[int] = None,
) -> torch.Tensor:
    """Clips parameter gradients and strictly asserts finite norm before optimizer.step().

    Fails closed immediately by raising FloatingPointError if gradient norm is non-finite.
    Shared production implementation invoked by both training loop and verification tests.
    """
    params = model_or_parameters.parameters() if isinstance(model_or_parameters, nn.Module) else model_or_parameters
    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    if not torch.isfinite(total_norm):
        ctx = f" at epoch {epoch}, batch {batch_idx}" if (epoch is not None and batch_idx is not None) else ""
        raise FloatingPointError(
            f"Non-finite gradient norm detected{ctx}: total_norm={total_norm.item()}"
        )
    return total_norm


def verify_flow_matching_preflight_contract(
    d0_checkpoint_path: str,
    normalizer_path: str,
    split_file: str,
    residual_stats_path: Optional[str] = None,
    expected_seed: Optional[int] = 42,
    manifest_path: Optional[str] = "outputs/manifests/closure_r4_seed42.json",
) -> Tuple[Dict[str, Any], FieldNormalizer, str, str, str, Optional[Dict[str, Any]]]:
    """Verify cryptographic bindings for D0 checkpoint, normalizer, dataset split, and residual stats.

    Fails closed if checkpoints or data contracts are missing, or if provenance hashes diverge.

    Returns:
        (ckpt_data, normalizer, d0_sha256, runtime_split_hash, runtime_norm_hash, stats_data)
    """
    if not os.path.exists(d0_checkpoint_path):
        raise FileNotFoundError(f"D0 checkpoint not found: {d0_checkpoint_path}")
    d0_sha256 = compute_file_sha256(d0_checkpoint_path)
    ckpt_data = torch.load(d0_checkpoint_path, map_location="cpu", weights_only=False)

    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(f"Normalizer file not found: {normalizer_path}")
    normalizer = FieldNormalizer()
    normalizer.load_state_dict(torch.load(normalizer_path, weights_only=True, map_location="cpu"))
    runtime_norm_hash = compute_normalizer_hash(normalizer)

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Dataset split file not found: {split_file}")
    runtime_split_hash = compute_split_hash_from_file(split_file)

    # 1. Resolve and verify D0 provenance identity binding
    prov = resolve_checkpoint_provenance(d0_checkpoint_path, ckpt_data, manifest_path=manifest_path)
    d0_split_hash = prov.get("split_hash")
    d0_norm_hash = prov.get("normalizer_hash")
    d0_seed = prov.get("seed")

    errors = []
    if not d0_split_hash:
        errors.append("D0 checkpoint/manifest is missing required 'split_hash'")
    elif not hash_matches(d0_split_hash, runtime_split_hash, min_prefix_len=16):
        errors.append(
            f"Split hash mismatch: D0 has {d0_split_hash[:16]}..., "
            f"but runtime split has {runtime_split_hash[:16]}..."
        )

    if not d0_norm_hash:
        errors.append("D0 checkpoint/manifest is missing required 'normalizer_hash'")
    elif not hash_matches(d0_norm_hash, runtime_norm_hash, min_prefix_len=16):
        errors.append(
            f"Normalizer hash mismatch: D0 has {d0_norm_hash[:16]}..., "
            f"but runtime normalizer has {runtime_norm_hash[:16]}..."
        )

    # Strict fail-closed check on D0 seed
    if expected_seed is not None:
        if d0_seed is None:
            errors.append("D0 checkpoint/manifest is missing required 'seed'")
        elif d0_seed != expected_seed:
            errors.append(f"Seed mismatch: D0 has seed={d0_seed}, expected {expected_seed}")

    # 2. Verify residual stats binding (D0 SHA-256, split_hash, normalizer_hash, statistics)
    stats_data = None
    if residual_stats_path is not None:
        if not os.path.exists(residual_stats_path):
            raise FileNotFoundError(f"Residual statistics file not found: {residual_stats_path}")
        with open(residual_stats_path, "r") as f:
            stats_data = json.load(f)

        if not isinstance(stats_data, dict):
            raise ValueError(f"Residual stats file must contain a JSON object, got {type(stats_data)}")

        # 2a. Cryptographic binding to generating D0 checkpoint
        stats_d0 = stats_data.get("d0_checkpoint")
        if not isinstance(stats_d0, dict) or not stats_d0.get("sha256"):
            errors.append("Residual stats missing required 'd0_checkpoint.sha256'")
        else:
            stats_d0_sha = stats_d0["sha256"]
            if not hash_matches(stats_d0_sha, d0_sha256, min_prefix_len=16):
                errors.append(
                    f"Residual stats D0 checkpoint SHA mismatch: stats generated from D0={stats_d0_sha[:16]}..., "
                    f"but current runtime D0 has SHA={d0_sha256[:16]}..."
                )

        # 2b. Cryptographic binding to data protocol (split and normalizer)
        stats_data_proto = stats_data.get("data_protocol")
        if not isinstance(stats_data_proto, dict):
            errors.append("Residual stats missing required 'data_protocol' dictionary")
        else:
            stats_split_hash = stats_data_proto.get("split_hash")
            if not stats_split_hash:
                errors.append("Residual stats missing required 'data_protocol.split_hash'")
            elif not hash_matches(stats_split_hash, runtime_split_hash, min_prefix_len=16):
                errors.append(
                    f"Residual stats split_hash mismatch: stats has {stats_split_hash[:16]}..., "
                    f"runtime has {runtime_split_hash[:16]}..."
                )

            stats_norm_hash = stats_data_proto.get("normalizer_hash")
            if not stats_norm_hash:
                errors.append("Residual stats missing required 'data_protocol.normalizer_hash'")
            elif not hash_matches(stats_norm_hash, runtime_norm_hash, min_prefix_len=16):
                errors.append(
                    f"Residual stats normalizer_hash mismatch: stats has {stats_norm_hash[:16]}..., "
                    f"runtime has {runtime_norm_hash[:16]}..."
                )

        # 2c. Verify statistical second moments presence
        stats_statistics = stats_data.get("statistics")
        if not isinstance(stats_statistics, dict) or "channel_residual_second_moment_g0" not in stats_statistics:
            errors.append("Residual stats missing required 'statistics.channel_residual_second_moment_g0'")

    if errors:
        raise ValueError(
            "Preflight cryptographic verification failed-closed due to protocol identity mismatches:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return ckpt_data, normalizer, d0_sha256, runtime_split_hash, runtime_norm_hash, stats_data


def build_and_freeze_flow_matching_model(
    ckpt_data: Dict[str, Any],
    device: torch.device,
    hidden_channels: int = 128,
    num_blocks: int = 4,
    target_mode: str = "residual",
    use_spatial_attn: bool = True,
    zero_init: bool = True,
    residual_scale: Optional[torch.Tensor] = None,
) -> LatentForecaster:
    """Instantiate LatentForecaster with attached LatentFlowMatcher and freeze representation backbone.

    Fails closed if parameter freeze requirements are violated.
    """
    cfg = ckpt_data.get("config", {})
    pred_mode = cfg.get("prediction_mode", ckpt_data.get("prediction_mode", "direct"))
    emb_dim = cfg.get("embed_dim", 256)
    depth = cfg.get("depth", 6)
    num_heads = cfg.get("num_heads", 8)
    use_spatial_pos = resolve_spatial_pos_config(ckpt_data)

    # 1. Instantiate Forecaster components
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=emb_dim,
        cond_dim=128,
        depth=depth,
        num_heads=num_heads,
        history_length=4,
        prediction_mode=pred_mode,
        use_spatial_pos=use_spatial_pos,
    )
    forecaster = LatentForecaster(encoder=encoder, transformer=transformer, decoder=decoder).to(device)

    # 2. Load D0 deterministic weights
    sd = ckpt_data.get("model_state_dict", ckpt_data)
    cleaned_sd = {
        (k[7:] if k.startswith("module.") else k): v for k, v in sd.items()
    }
    cleaned_sd = strip_compiled_prefix(cleaned_sd)
    forecaster.load_state_dict(cleaned_sd, strict=True)

    # 3. Instantiate and attach LatentFlowMatcher (with optional residual scale)
    flow_matcher = LatentFlowMatcher(
        latent_channels=64,
        cond_dim=128,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        use_spatial_attn=use_spatial_attn,
        target_mode=target_mode,
        zero_init=zero_init,
        residual_scale=residual_scale,
    ).to(device)
    forecaster.attach_flow_matcher(flow_matcher)

    # 4. Strictly freeze representation and Transformer backbone
    forecaster.freeze_for_flow_matching_training()

    # 5. Governance assertion: verify parameter gradients
    trainable_params = []
    frozen_params = []
    for name, p in forecaster.named_parameters():
        if "flow_matcher" in name:
            if not p.requires_grad:
                raise RuntimeError(f"Flow matcher param {name} must have requires_grad=True")
            trainable_params.append(name)
        else:
            if p.requires_grad:
                raise RuntimeError(f"Model param {name} must be frozen (requires_grad=False)")
            frozen_params.append(name)

    assert len(trainable_params) > 0, "No trainable flow matcher parameters found"
    assert len(frozen_params) > 0, "No frozen parameters found"

    return forecaster


def evaluate_flow_matching_loss(
    forecaster: LatentForecaster,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 0,
    fail_on_non_finite: bool = True,
    eval_seed: int = 1234,
) -> Dict[str, Any]:
    """Evaluate mean OT-CFM loss on a dataloader.

    Fails closed on empty dataloader or non-finite values.
    """
    if len(dataloader) == 0:
        raise ValueError("Validation dataloader is empty.")

    fm = forecaster.flow_matcher
    if fm is None:
        raise RuntimeError("Forecaster has no attached flow_matcher.")

    forecaster.eval()
    total_loss = 0.0
    total_windows = 0
    generator = torch.Generator(device=device if device.type != "mps" else "cpu")
    generator.manual_seed(eval_seed)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            q_hist = batch["history"].to(device)  # (B, L, 4, Ny, Nx)
            q_future = batch["future"].to(device)  # (B, 1, 4, Ny, Nx)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)
            batch_size = q_hist.shape[0]

            # Encode with frozen encoder
            z_hist = forecaster.encoder(q_hist)
            z_next = forecaster.encoder(q_future)

            # Predict deterministic mean with frozen transformer
            mu = forecaster.transformer(z_hist, re=re, sc=sc)

            loss_dict = fm.compute_loss(
                z_next=z_next,
                mu=mu,
                re=re,
                sc=sc,
                generator=generator,
            )
            loss_val = loss_dict["loss"].item()

            if not math.isfinite(loss_val):
                if fail_on_non_finite:
                    raise FloatingPointError(f"Non-finite validation CFM loss encountered: {loss_val}")
                return {"cfm_loss": float("nan"), "windows_evaluated": total_windows, "is_valid": False}

            total_loss += loss_val * batch_size
            total_windows += batch_size

    mean_loss = total_loss / max(1, total_windows)
    return {
        "cfm_loss": float(mean_loss),
        "windows_evaluated": int(total_windows),
        "is_valid": True,
    }


def train_latent_flow_matching(
    d0_checkpoint: str,
    normalizer_path: str,
    split_file: str,
    output_dir: str,
    residual_stats_path: Optional[str] = "outputs/normalization/latent_residual_stats.json",
    disable_residual_normalization: bool = False,
    data_root: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    hidden_channels: int = 128,
    num_blocks: int = 4,
    target_mode: str = "residual",
    use_spatial_attn: bool = True,
    zero_init: bool = True,
    grad_clip: float = 1.0,
    max_train_batches: int = 0,
    max_val_batches: int = 0,
    smoke_test: bool = False,
    overwrite: bool = False,
    device_str: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute Latent Flow Matching training pipeline with strict governance."""
    # 1. Deterministic Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"=== Latent Flow Matching Training ===")
    print(f"Device: {device} | Seed: {seed} | Target Mode: {target_mode}")

    # 2. Strict stats path resolution (fail-closed: never silently downgrade to unnormalized)
    if target_mode == "residual" and not disable_residual_normalization:
        if not residual_stats_path:
            raise ValueError(
                "residual_stats_path must be specified when target_mode='residual'. "
                "To train unnormalized residual flow matching, explicitly pass disable_residual_normalization=True."
            )
        if not os.path.isfile(residual_stats_path):
            raise FileNotFoundError(
                f"Residual statistics file not found at: {residual_stats_path}. "
                "To train unnormalized residual flow matching, explicitly pass disable_residual_normalization=True."
            )
        actual_stats_path = residual_stats_path
    else:
        actual_stats_path = None

    # 3. Pre-flight Verification FIRST (fail-closed before filesystem modifications)
    ckpt_data, normalizer, d0_sha256, runtime_split_hash, runtime_norm_hash, stats_data = verify_flow_matching_preflight_contract(
        d0_checkpoint_path=d0_checkpoint,
        normalizer_path=normalizer_path,
        split_file=split_file,
        residual_stats_path=actual_stats_path,
        expected_seed=seed,
    )
    print(f"Preflight Verified: D0={d0_sha256[:12]}..., Split={runtime_split_hash[:12]}..., Norm={runtime_norm_hash[:12]}...")

    # 4. Smoke Test Automatic Restrictions
    if smoke_test:
        print(">>> SMOKE TEST MODE ACTIVATED: Restricting to 1 epoch, 2 train batches, 2 val batches <<<")
        epochs = 1
        max_train_batches = 2 if max_train_batches == 0 else max_train_batches
        max_val_batches = 2 if max_val_batches == 0 else max_val_batches
        if output_dir == "outputs/checkpoints/probabilistic/flow_matching":
            output_dir = "outputs/checkpoints/probabilistic/flow_matching_smoke"

    # 5. Artifact Directory Protection (strictly after preflight success)
    out_path = Path(output_dir)
    if out_path.exists():
        existing_files = list(out_path.glob("*.pt")) + list(out_path.glob("*.json"))
        if existing_files and not overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} contains existing artifacts. Use --overwrite to replace."
            )
        elif existing_files and overwrite:
            backup_dir = out_path / f"backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            for f in existing_files:
                shutil.move(str(f), str(backup_dir / f.name))
            print(f"Backed up existing artifacts to {backup_dir}")
    out_path.mkdir(parents=True, exist_ok=True)

    # 6. Residual Scale Normalization Setup (ArchesWeatherGen paradigm)
    residual_scale = None
    if target_mode == "residual" and stats_data is not None and "statistics" in stats_data:
        sec_moments = stats_data["statistics"].get("channel_residual_second_moment_g0")
        if sec_moments is not None:
            residual_scale = torch.sqrt(torch.tensor(sec_moments, dtype=torch.float32) + 1e-6)
            print(
                f"Residual Scale Normalization: Active for {len(residual_scale)} channels "
                f"([min={residual_scale.min():.4f}, max={residual_scale.max():.4f}])"
            )

    # 7. Build Model & Freeze Backbone
    forecaster = build_and_freeze_flow_matching_model(
        ckpt_data=ckpt_data,
        device=device,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        target_mode=target_mode,
        use_spatial_attn=use_spatial_attn,
        zero_init=zero_init,
        residual_scale=residual_scale,
    )
    flow_matcher = forecaster.flow_matcher
    print(f"Model Built: LatentFlowMatcher attached. Hidden channels={hidden_channels}, blocks={num_blocks}")

    # 8. Dataloaders
    train_loader, valid_loader, _, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file=split_file,
        data_root=data_root,
        history_length=4,
        horizon=1,
        valid_horizon=1,
        train_stride=8,
        valid_stride=8,
        downsample_factor=2,
        batch_size=batch_size,
        num_workers=2,
        normalize=True,
        normalizer=normalizer,
        seed=seed,
    )
    print(f"DataLoaders: Train={len(train_loader.dataset)} windows, Valid={len(valid_loader.dataset)} windows")

    # 9. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        flow_matcher.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.05,
    )

    # 10. Epoch 0 Audit
    print("\n--- Evaluating Epoch 0 Baseline ---")
    epoch0_metrics = evaluate_flow_matching_loss(
        forecaster=forecaster,
        dataloader=valid_loader,
        device=device,
        max_batches=max_val_batches,
        fail_on_non_finite=True,
    )
    print(f"Epoch 0 Baseline CFM Loss: {epoch0_metrics['cfm_loss']:.6f}")

    epoch0_ckpt_path = os.path.join(output_dir, "epoch0_baseline.pt")
    torch.save(
        {
            "flow_matcher_state_dict": flow_matcher.state_dict(),
            "epoch": 0,
            "val_cfm_loss": epoch0_metrics["cfm_loss"],
            "d0_sha256": d0_sha256,
            "residual_scale_applied": residual_scale is not None,
        },
        epoch0_ckpt_path,
    )

    # 11. Training Loop
    history = []
    best_loss = epoch0_metrics["cfm_loss"]
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        forecaster.train()
        train_loss_total = 0.0
        train_windows = 0

        for batch_idx, batch in enumerate(train_loader):
            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break

            q_hist = batch["history"].to(device)
            q_future = batch["future"].to(device)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)
            b = q_hist.shape[0]

            optimizer.zero_grad()

            with torch.no_grad():
                z_hist = forecaster.encoder(q_hist)
                z_next = forecaster.encoder(q_future)
                mu = forecaster.transformer(z_hist, re=re, sc=sc)

            loss_dict = flow_matcher.compute_loss(
                z_next=z_next,
                mu=mu,
                re=re,
                sc=sc,
            )
            loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_idx}: {loss.item()}")

            loss.backward()

            # Fail-closed check: verify gradient norm before optimizer step via shared production function
            clip_and_validate_gradients(flow_matcher, max_norm=grad_clip, epoch=epoch, batch_idx=batch_idx)

            optimizer.step()

            train_loss_total += loss.item() * b
            train_windows += b

        scheduler.step()
        train_cfm_loss = train_loss_total / max(1, train_windows)

        # Validation
        val_metrics = evaluate_flow_matching_loss(
            forecaster=forecaster,
            dataloader=valid_loader,
            device=device,
            max_batches=max_val_batches,
            fail_on_non_finite=True,
        )
        val_cfm_loss = val_metrics["cfm_loss"]

        epoch_record = {
            "epoch": epoch,
            "train_cfm_loss": float(train_cfm_loss),
            "val_cfm_loss": float(val_cfm_loss),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(epoch_record)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train CFM Loss: {train_cfm_loss:.6f} | Val CFM Loss: {val_cfm_loss:.6f}")

        # Checkpoint if best
        if val_cfm_loss < best_loss:
            best_loss = val_cfm_loss
            best_epoch = epoch
            best_path = os.path.join(output_dir, "best_latent_flow_matcher.pt")

            residual_stats_info = None
            if residual_scale is not None and actual_stats_path:
                stats_sha = compute_file_sha256(actual_stats_path)
                residual_stats_info = {
                    "path": actual_stats_path,
                    "sha256": stats_sha,
                    "d0_sha256": d0_sha256,
                    "scale_definition": "sqrt(channel_residual_second_moment_g0 + 1e-6)",
                    "scale_min": float(residual_scale.min()),
                    "scale_max": float(residual_scale.max()),
                }

            torch.save(
                {
                    "flow_matcher_state_dict": flow_matcher.state_dict(),
                    "epoch": epoch,
                    "val_cfm_loss": val_cfm_loss,
                    "config": {
                        "hidden_channels": hidden_channels,
                        "num_blocks": num_blocks,
                        "target_mode": target_mode,
                        "use_spatial_attn": use_spatial_attn,
                        "sigma_min": flow_matcher.sigma_min,
                        "num_flow_steps": 10,
                        "solver": "midpoint",
                        "residual_scale_applied": residual_scale is not None,
                    },
                    "provenance": {
                        "d0_checkpoint": {"path": d0_checkpoint, "sha256": d0_sha256},
                        "data_protocol": {
                            "split_file": split_file,
                            "split_hash": runtime_split_hash,
                            "normalizer_file": normalizer_path,
                            "normalizer_hash": runtime_norm_hash,
                        },
                        "residual_statistics": residual_stats_info,
                        "git_commit": get_git_commit(),
                        "is_git_dirty": is_git_dirty(),
                        "seed": seed,
                        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                },
                best_path,
            )
            print(f"  --> Saved new best checkpoint to {best_path} (Val Loss: {val_cfm_loss:.6f})")

    # 12. Final summary with explicit governance status
    selection_status = "IMPROVED" if best_epoch > 0 else "NO_IMPROVEMENT_OVER_EPOCH0"
    selected_ckpt = "best_latent_flow_matcher.pt" if best_epoch > 0 else "epoch0_baseline.pt"
    summary = {
        "best_epoch": best_epoch,
        "best_val_cfm_loss": float(best_loss),
        "epoch0_val_cfm_loss": float(epoch0_metrics["cfm_loss"]),
        "selection_status": selection_status,
        "selected_checkpoint": selected_ckpt,
        "residual_scale_applied": residual_scale is not None,
        "history": history,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Latent Flow Matcher on frozen D0 forecaster.")
    parser.add_argument(
        "--d0-checkpoint",
        type=str,
        default="outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt",
        help="Path to D0 checkpoint (.pt)",
    )
    parser.add_argument("--normalizer-path", type=str, default="outputs/normalization/stats_grouped.pt")
    parser.add_argument("--split-file", type=str, default="outputs/splits/grouped_split.json")
    parser.add_argument(
        "--residual-stats-path",
        type=str,
        default="outputs/normalization/latent_residual_stats.json",
        help="Path to Phase 0 latent residual statistics JSON for scale normalization",
    )
    parser.add_argument(
        "--disable-residual-normalization",
        action="store_true",
        help="Explicitly disable residual scale normalization for ablation experiments",
    )
    default_data_root = os.environ.get(
        "SHEAR_FLOW_DATA_DIR",
        "/root/autodl-tmp/datasets/shear_flow" if os.path.exists("/root/autodl-tmp/datasets/shear_flow") else None,
    )
    parser.add_argument("--data-root", type=str, default=default_data_root)
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints/probabilistic/flow_matching")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--target-mode", type=str, default="residual", choices=["residual", "direct"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_latent_flow_matching(
        d0_checkpoint=args.d0_checkpoint,
        normalizer_path=args.normalizer_path,
        split_file=args.split_file,
        residual_stats_path=args.residual_stats_path,
        disable_residual_normalization=args.disable_residual_normalization,
        output_dir=args.output_dir,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        target_mode=args.target_mode,
        grad_clip=args.grad_clip,
        smoke_test=args.smoke_test,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
