"""Latent Flow Matching Rollout-Aware (FM-R2) Training Entrypoint.

Implements the strictly controlled Exposure-Bias experiment comparing:
1. Branch C1: Extended 1-Step CFM continuation (single-step reference).
2. Branch C2: Teacher-Forced 2-Step CFM Control (uses z_GT_{t+1} as Step 2 history condition).
3. Branch R2-A: Rollout-Aware 2-Step CFM Treatment (uses detached z_hat_{t+1} as Step 2 history condition).

Governance and Scientific Equivalence Contracts:
1. Byte-identical Initialization: Both C2 and R2-A start from identical model weights
   loaded from parent FM checkpoint (e.g. 3-epoch Pilot checkpoint).
2. Backbone Freeze: D0 Encoder, Decoder, and Latent Transformer backbone are strictly frozen.
3. Fresh Optimizer: Both branches initialize fresh AdamW optimizers with identical hyperparameters.
4. Identical Targets & Horizons: Both branches supervise t+1 and t+2 on identical horizon-2 windows.
5. Identical Loss Structure: Both C2 and R2-A compute L = 0.5 * (L_{t+1} + L_{t+2}).
6. Common Random Numbers: CFM base noise and flow time draws are generated using dedicated,
   identically seeded RNGs so CFM loss random draws in Step 1 and Step 2 are 100% bit-identical
   between C2 and R2-A.
7. Detached History in R2-A: In R2-A, z_hat_{t+1} is strictly detached from computation graph
   before rolling into Step 2 history condition (no backprop through ODE integrator).
8. Fail-closed on Non-finite Values: Any non-finite loss, latent, or gradient immediately raises
   FloatingPointError.
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
from typing import Dict, Any, Tuple, Optional, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from scripts.train_latent_flow_matching import clip_and_validate_gradients


def verify_rollout_aware_preflight_contract(
    d0_checkpoint_path: str,
    parent_fm_checkpoint_path: str,
    normalizer_path: str,
    split_file: str,
    residual_stats_path: Optional[str] = None,
    expected_seed: Optional[int] = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any], FieldNormalizer, str, str, str, str, Optional[Dict[str, Any]]]:
    """Verify cryptographic bindings for D0, parent FM, normalizer, split, and residual stats.

    Fails closed if checkpoints or contracts are missing, or if provenance hashes diverge.

    Returns:
        (d0_ckpt, parent_fm_ckpt, normalizer, d0_sha, fm_sha, split_hash, norm_hash, stats_data)
    """
    if not os.path.exists(d0_checkpoint_path):
        raise FileNotFoundError(f"D0 checkpoint not found: {d0_checkpoint_path}")
    d0_sha = compute_file_sha256(d0_checkpoint_path)
    d0_ckpt = torch.load(d0_checkpoint_path, map_location="cpu", weights_only=False)

    if not os.path.exists(parent_fm_checkpoint_path):
        raise FileNotFoundError(f"Parent FM checkpoint not found: {parent_fm_checkpoint_path}")
    parent_fm_sha = compute_file_sha256(parent_fm_checkpoint_path)
    parent_fm_ckpt = torch.load(parent_fm_checkpoint_path, map_location="cpu", weights_only=False)

    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(f"Normalizer file not found: {normalizer_path}")
    normalizer = FieldNormalizer()
    normalizer.load_state_dict(torch.load(normalizer_path, weights_only=True, map_location="cpu"))
    runtime_norm_hash = compute_normalizer_hash(normalizer)

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    runtime_split_hash = compute_split_hash_from_file(split_file)

    # Verify parent FM provenance binding to D0 (Fail-Closed)
    fm_prov = parent_fm_ckpt.get("provenance", {})
    fm_d0_info = fm_prov.get("d0_checkpoint", {})
    fm_d0_sha = fm_d0_info.get("sha256")
    if not fm_d0_sha:
        raise ValueError(
            "Parent FM checkpoint provenance is missing 'd0_checkpoint.sha256'. (Fail-Closed)"
        )
    if not hash_matches(fm_d0_sha, d0_sha):
        raise ValueError(
            f"Parent FM checkpoint was trained on D0 with SHA {fm_d0_sha[:12]}..., "
            f"which diverges from runtime D0 SHA {d0_sha[:12]}... (Fail-Closed)"
        )

    # Verify split and normalizer hashes in parent FM provenance (Fail-Closed)
    fm_data_proto = fm_prov.get("data_protocol", {})
    fm_split_hash = fm_data_proto.get("split_hash")
    if not fm_split_hash:
        raise ValueError(
            "Parent FM checkpoint provenance is missing 'data_protocol.split_hash'. (Fail-Closed)"
        )
    if not hash_matches(fm_split_hash, runtime_split_hash):
        raise ValueError(
            f"Parent FM checkpoint split hash {fm_split_hash[:12]}... diverges from "
            f"runtime split hash {runtime_split_hash[:12]}... (Fail-Closed)"
        )

    fm_norm_hash = fm_data_proto.get("normalizer_hash")
    if not fm_norm_hash:
        raise ValueError(
            "Parent FM checkpoint provenance is missing 'data_protocol.normalizer_hash'. (Fail-Closed)"
        )
    if not hash_matches(fm_norm_hash, runtime_norm_hash):
        raise ValueError(
            f"Parent FM checkpoint normalizer hash {fm_norm_hash[:12]}... diverges from "
            f"runtime normalizer hash {runtime_norm_hash[:12]}... (Fail-Closed)"
        )

    # Verify seed (Fail-Closed)
    fm_seed = fm_prov.get("seed")
    if fm_seed is None:
        raise ValueError(
            "Parent FM checkpoint provenance is missing 'seed'. (Fail-Closed)"
        )
    if expected_seed is not None and fm_seed != expected_seed:
        raise ValueError(
            f"Parent FM seed ({fm_seed}) does not match expected seed ({expected_seed}). (Fail-Closed)"
        )

    # Load and verify residual stats if provided or if parent FM recorded it
    stats_data = None
    parent_stats_prov = fm_prov.get("residual_statistics")
    if residual_stats_path:
        if not os.path.exists(residual_stats_path):
            raise FileNotFoundError(f"Residual stats not found: {residual_stats_path}")
        runtime_stats_sha = compute_file_sha256(residual_stats_path)
        with open(residual_stats_path, "r") as f:
            stats_data = json.load(f)

        if parent_stats_prov:
            parent_stats_sha = parent_stats_prov.get("sha256")
            if not parent_stats_sha:
                raise ValueError(
                    "Parent FM checkpoint provenance is missing 'residual_statistics.sha256'. (Fail-Closed)"
                )
            if not hash_matches(runtime_stats_sha, parent_stats_sha):
                raise ValueError(
                    f"Residual stats SHA {runtime_stats_sha[:12]}... diverges from parent FM "
                    f"recorded residual stats SHA {parent_stats_sha[:12]}... (Fail-Closed)"
                )

        # Check internal consistency of residual stats
        stats_d0 = stats_data.get("d0_checkpoint", {})
        stats_d0_sha = stats_d0.get("sha256")
        if not stats_d0_sha:
            raise ValueError("Residual stats file is missing 'd0_checkpoint.sha256'. (Fail-Closed)")
        if not hash_matches(stats_d0_sha, d0_sha):
            raise ValueError(
                f"Residual stats D0 SHA {stats_d0_sha[:12]}... diverges from runtime D0 SHA {d0_sha[:12]}... (Fail-Closed)"
            )

        stats_proto = stats_data.get("data_protocol", {})
        stats_split_hash = stats_proto.get("split_hash")
        if not stats_split_hash:
            raise ValueError("Residual stats file is missing 'data_protocol.split_hash'. (Fail-Closed)")
        if not hash_matches(stats_split_hash, runtime_split_hash):
            raise ValueError(
                f"Residual stats split hash {stats_split_hash[:12]}... diverges from runtime split hash {runtime_split_hash[:12]}... (Fail-Closed)"
            )

        stats_norm_hash = stats_proto.get("normalizer_hash")
        if not stats_norm_hash:
            raise ValueError("Residual stats file is missing 'data_protocol.normalizer_hash'. (Fail-Closed)")
        if not hash_matches(stats_norm_hash, runtime_norm_hash):
            raise ValueError(
                f"Residual stats normalizer hash {stats_norm_hash[:12]}... diverges from runtime normalizer hash {runtime_norm_hash[:12]}... (Fail-Closed)"
            )
    elif parent_stats_prov:
        raise ValueError(
            "Parent FM was trained with residual statistics, but residual_stats_path was not provided. (Fail-Closed)"
        )

    return (
        d0_ckpt,
        parent_fm_ckpt,
        normalizer,
        d0_sha,
        parent_fm_sha,
        runtime_split_hash,
        runtime_norm_hash,
        stats_data,
    )


def build_and_freeze_rollout_aware_model(
    d0_ckpt_data: Dict[str, Any],
    parent_fm_ckpt_data: Dict[str, Any],
    device: torch.device,
    residual_scale: Optional[torch.Tensor] = None,
) -> LatentForecaster:
    """Instantiate LatentForecaster with attached LatentFlowMatcher loaded from parent FM checkpoint.

    Strictly enforces parameter freeze on representation backbone and transformer.
    """
    cfg_d0 = d0_ckpt_data.get("config", {})
    pred_mode = cfg_d0.get("prediction_mode", d0_ckpt_data.get("prediction_mode", "direct"))
    emb_dim = cfg_d0.get("embed_dim", 256)
    depth = cfg_d0.get("depth", 6)
    num_heads = cfg_d0.get("num_heads", 8)
    use_spatial_pos = resolve_spatial_pos_config(d0_ckpt_data)

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
    sd = d0_ckpt_data.get("model_state_dict", d0_ckpt_data)
    cleaned_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    cleaned_sd = strip_compiled_prefix(cleaned_sd)
    forecaster.load_state_dict(cleaned_sd, strict=True)

    # 3. Instantiate LatentFlowMatcher from parent FM config
    fm_cfg = parent_fm_ckpt_data.get("config", {})
    hidden_channels = fm_cfg.get("hidden_channels", 128)
    num_blocks = fm_cfg.get("num_blocks", 4)
    target_mode = fm_cfg.get("target_mode", "residual")
    use_spatial_attn = fm_cfg.get("use_spatial_attn", True)
    sigma_min = fm_cfg.get("sigma_min", 1e-4)

    flow_matcher = LatentFlowMatcher(
        latent_channels=64,
        cond_dim=128,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        use_spatial_attn=use_spatial_attn,
        target_mode=target_mode,
        sigma_min=sigma_min,
        residual_scale=residual_scale,
    ).to(device)

    # 4. Strictly load parent FM weights
    fm_sd = parent_fm_ckpt_data.get("flow_matcher_state_dict", parent_fm_ckpt_data)
    flow_matcher.load_state_dict(fm_sd, strict=True)

    forecaster.attach_flow_matcher(flow_matcher)

    # 5. Strictly freeze backbone (Encoder, Decoder, Transformer)
    forecaster.freeze_for_flow_matching_training()

    # 6. Governance assertion: verify parameter gradients
    trainable_params = []
    frozen_params = []
    for name, p in forecaster.named_parameters():
        if "flow_matcher" in name:
            if not p.requires_grad:
                raise RuntimeError(f"Flow matcher parameter {name} must have requires_grad=True")
            trainable_params.append(name)
        else:
            if p.requires_grad:
                raise RuntimeError(f"Model backbone parameter {name} must be frozen (requires_grad=False)")
            frozen_params.append(name)

    assert len(trainable_params) > 0, "No trainable flow matcher parameters found"
    assert len(frozen_params) > 0, "No frozen parameters found"

    return forecaster


def compute_rollout_aware_step_loss(
    forecaster: LatentForecaster,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    branch: str,
    sample_noise_scale: Optional[float] = None,
    num_flow_steps: int = 10,
    solver: str = "midpoint",
    gen_loss_1: Optional[torch.Generator] = None,
    gen_loss_2: Optional[torch.Generator] = None,
    gen_sample: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Execute single multi-step training step for C1, C2, or R2-A with strict contract compliance.

    Args:
        forecaster: LatentForecaster with attached flow_matcher and frozen backbone.
        batch: DataLoader batch containing 'history', 'future', 're', 'sc'.
        device: Device tensor execution.
        branch: 'C1' (1-step continuation), 'C2' (Teacher-Forced 2-step), or 'R2_A' (Rollout-Aware 2-step).
        sample_noise_scale: Noise scale alpha_noise for Step 1 sampling in R2-A.
        num_flow_steps: ODE integration steps in R2-A.
        solver: ODE solver for R2-A sampling.
        gen_loss_1: Dedicated RNG for Step 1 CFM loss.
        gen_loss_2: Dedicated RNG for Step 2 CFM loss.
        gen_sample: Dedicated RNG for Step 1 sampling in R2-A.

    Returns:
        dict with 'loss', 'loss_step1', 'loss_step2', 'branch'.
    """
    if branch == "R2_A":
        if sample_noise_scale is None:
            raise ValueError(
                "Branch R2_A requires sample_noise_scale to be explicitly specified. (Fail-Closed)"
            )
        if sample_noise_scale <= 0:
            raise ValueError(f"sample_noise_scale must be positive, got {sample_noise_scale}")
    fm = forecaster.flow_matcher
    if fm is None:
        raise RuntimeError("Forecaster has no attached flow_matcher.")

    q_hist = batch["history"].to(device)  # (B, 4, 4, Ny, Nx)
    q_future = batch["future"].to(device)  # (B, H>=2, 4, Ny, Nx)
    re = batch["re"].to(device)
    sc = batch["sc"].to(device)

    # 1. Encode with frozen backbone
    with torch.no_grad():
        z_hist_1 = forecaster.encoder(q_hist)  # (B, 4, 64, Ny_lat, Nx_lat)
        z_target_1 = forecaster.encoder(q_future[:, 0:1])  # (B, 1, 64, Ny_lat, Nx_lat)
        mu_1 = forecaster.transformer(z_hist_1, re=re, sc=sc)  # (B, 1, 64, Ny_lat, Nx_lat)

    # 2. Step 1 CFM Loss (Conditioned on GT history)
    loss_1_dict = fm.compute_loss(
        z_next=z_target_1,
        mu=mu_1,
        re=re,
        sc=sc,
        generator=gen_loss_1,
    )
    loss_1 = loss_1_dict["loss"]
    if not torch.isfinite(loss_1):
        raise FloatingPointError(f"Non-finite Step 1 CFM loss: {loss_1.item()}")

    # Branch C1: Single-Step continuation
    if branch == "C1":
        return {
            "loss": loss_1,
            "loss_step1": loss_1.item(),
            "loss_step2": 0.0,
            "branch": branch,
        }

    # 3. Determine Step 2 Condition
    if branch == "C2":
        # Teacher-Forced: Step 2 history condition uses ground truth z_target_1
        z_step1_cond = z_target_1

    elif branch == "R2_A":
        # Rollout-Aware: Step 2 history condition uses generated z_hat_1, STRICTLY DETACHED
        with torch.no_grad():
            z_hat_1 = fm.sample_next_latent(
                mu=mu_1,
                re=re,
                sc=sc,
                num_steps=num_flow_steps,
                solver=solver,
                noise_scale=sample_noise_scale,
                generator=gen_sample,
            )
        # Defense-in-depth: explicit detach to guarantee no backprop through ODE
        z_step1_cond = z_hat_1.detach()
        if not torch.isfinite(z_step1_cond).all():
            raise FloatingPointError("Non-finite values encountered in sampled Step 1 latent state.")

    else:
        raise ValueError(f"Unknown branch: {branch}. Must be one of ['C1', 'C2', 'R2_A'].")

    # 4. Roll history condition: drop oldest frame (dim 1 index 0), append step 1 latent
    z_hist_2 = torch.cat([z_hist_1[:, 1:], z_step1_cond], dim=1)  # (B, 4, 64, Ny_lat, Nx_lat)

    with torch.no_grad():
        z_target_2 = forecaster.encoder(q_future[:, 1:2])  # (B, 1, 64, Ny_lat, Nx_lat)
        mu_2 = forecaster.transformer(z_hist_2, re=re, sc=sc)  # (B, 1, 64, Ny_lat, Nx_lat)

    # 5. Step 2 CFM Loss
    loss_2_dict = fm.compute_loss(
        z_next=z_target_2,
        mu=mu_2,
        re=re,
        sc=sc,
        generator=gen_loss_2,
    )
    loss_2 = loss_2_dict["loss"]
    if not torch.isfinite(loss_2):
        raise FloatingPointError(f"Non-finite Step 2 CFM loss: {loss_2.item()}")

    # 6. Equal-weight composite loss
    total_loss = 0.5 * (loss_1 + loss_2)

    return {
        "loss": total_loss,
        "loss_step1": loss_1.item(),
        "loss_step2": loss_2.item(),
        "branch": branch,
    }


def evaluate_rollout_aware_validation(
    forecaster: LatentForecaster,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    branch: str,
    sample_noise_scale: Optional[float] = None,
    num_flow_steps: int = 10,
    solver: str = "midpoint",
    max_batches: int = 0,
    eval_seed: int = 4242,
) -> Dict[str, Any]:
    """Evaluate mean multi-step CFM loss on validation dataloader.

    Uses deterministic generators to guarantee reproducible evaluation across branches.
    """
    if branch == "R2_A":
        if sample_noise_scale is None:
            raise ValueError(
                "Branch R2_A requires sample_noise_scale to be explicitly specified. (Fail-Closed)"
            )
        if sample_noise_scale <= 0:
            raise ValueError(f"sample_noise_scale must be positive, got {sample_noise_scale}")

    if len(dataloader) == 0:
        raise ValueError("Validation dataloader is empty.")

    forecaster.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_l2 = 0.0
    total_windows = 0

    gen_loss_1 = torch.Generator(device=device if device.type != "mps" else "cpu")
    gen_loss_1.manual_seed(eval_seed)
    gen_loss_2 = torch.Generator(device=device if device.type != "mps" else "cpu")
    gen_loss_2.manual_seed(eval_seed + 1)
    gen_sample = torch.Generator(device=device if device.type != "mps" else "cpu")
    gen_sample.manual_seed(eval_seed + 2)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            step_res = compute_rollout_aware_step_loss(
                forecaster=forecaster,
                batch=batch,
                device=device,
                branch=branch,
                sample_noise_scale=sample_noise_scale,
                num_flow_steps=num_flow_steps,
                solver=solver,
                gen_loss_1=gen_loss_1,
                gen_loss_2=gen_loss_2,
                gen_sample=gen_sample,
            )
            b = batch["history"].shape[0]
            total_loss += step_res["loss"].item() * b
            total_l1 += step_res["loss_step1"] * b
            total_l2 += step_res["loss_step2"] * b
            total_windows += b

    return {
        "val_total_cfm_loss": float(total_loss / max(1, total_windows)),
        "val_step1_cfm_loss": float(total_l1 / max(1, total_windows)),
        "val_step2_cfm_loss": float(total_l2 / max(1, total_windows)),
        "windows_evaluated": int(total_windows),
        "branch": branch,
    }


def build_rollout_aware_checkpoint_payload(
    flow_matcher: LatentFlowMatcher,
    branch: str,
    epoch: int,
    val_loss_dict: Dict[str, Any],
    hidden_channels: int,
    num_blocks: int,
    target_mode: str,
    use_spatial_attn: bool,
    residual_scale: Optional[torch.Tensor],
    sample_noise_scale: Optional[float],
    d0_checkpoint: str,
    d0_sha256: str,
    parent_fm_checkpoint: str,
    parent_fm_sha256: str,
    split_file: str,
    runtime_split_hash: str,
    normalizer_path: str,
    runtime_norm_hash: str,
    actual_stats_path: Optional[str],
    seed: int,
    num_flow_steps: int = 10,
    solver: str = "midpoint",
) -> Dict[str, Any]:
    """Construct unified governed checkpoint payload for FM-R2 rollout-aware models."""
    residual_stats_info = None
    if residual_scale is not None and actual_stats_path:
        stats_sha = compute_file_sha256(actual_stats_path)
        residual_stats_info = {
            "path": actual_stats_path,
            "sha256": stats_sha,
            "d0_sha256": d0_sha256,
            "scale_min": float(residual_scale.min()),
            "scale_max": float(residual_scale.max()),
        }

    return {
        "flow_matcher_state_dict": flow_matcher.state_dict(),
        "branch": str(branch),
        "epoch": int(epoch),
        "val_loss_dict": val_loss_dict,
        "config": {
            "hidden_channels": hidden_channels,
            "num_blocks": num_blocks,
            "target_mode": target_mode,
            "use_spatial_attn": use_spatial_attn,
            "sigma_min": flow_matcher.sigma_min,
            "sample_noise_scale": sample_noise_scale,
            "num_flow_steps": int(num_flow_steps),
            "solver": str(solver),
            "residual_scale_applied": residual_scale is not None,
        },
        "provenance": {
            "d0_checkpoint": {"path": d0_checkpoint, "sha256": d0_sha256},
            "parent_fm_checkpoint": {"path": parent_fm_checkpoint, "sha256": parent_fm_sha256},
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
    }


def train_latent_flow_matching_rollout_aware(
    branch: str,
    d0_checkpoint: str,
    parent_fm_checkpoint: str,
    normalizer_path: str,
    split_file: str,
    output_dir: str,
    residual_stats_path: Optional[str] = "outputs/normalization/latent_residual_stats.json",
    data_root: Optional[str] = None,
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    sample_noise_scale: Optional[float] = None,
    num_flow_steps: int = 10,
    solver: str = "midpoint",
    grad_clip: float = 1.0,
    max_train_batches: int = 0,
    max_val_batches: int = 0,
    smoke_test: bool = False,
    overwrite: bool = False,
    device_str: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute Rollout-Aware (FM-R2) training pipeline with rigorous experimental controls."""
    if branch not in ("C1", "C2", "R2_A"):
        raise ValueError(f"Invalid branch '{branch}'. Must be one of ['C1', 'C2', 'R2_A'].")

    if branch == "R2_A" and sample_noise_scale is None:
        raise ValueError(
            "Branch R2_A requires --sample-noise-scale to be explicitly specified (e.g. 0.5). "
            "Implicit default of 1.0 is forbidden to avoid distribution mismatch. (Fail-Closed)"
        )
    if sample_noise_scale is not None and sample_noise_scale <= 0:
        raise ValueError(f"sample_noise_scale must be positive, got {sample_noise_scale}")

    # 1. Deterministic Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"=== Latent Flow Matching Rollout-Aware Training (FM-R2) ===")
    print(f"Branch: {branch} | Device: {device} | Seed: {seed} | Epochs: {epochs}")

    # 2. Strict Pre-flight Verification
    (
        d0_ckpt_data,
        parent_fm_ckpt_data,
        normalizer,
        d0_sha256,
        parent_fm_sha256,
        runtime_split_hash,
        runtime_norm_hash,
        stats_data,
    ) = verify_rollout_aware_preflight_contract(
        d0_checkpoint_path=d0_checkpoint,
        parent_fm_checkpoint_path=parent_fm_checkpoint,
        normalizer_path=normalizer_path,
        split_file=split_file,
        residual_stats_path=residual_stats_path,
        expected_seed=seed,
    )
    print(f"Preflight Verified: D0={d0_sha256[:12]}..., Parent FM={parent_fm_sha256[:12]}...")

    # 3. Smoke Test Automatic Restrictions
    if smoke_test:
        print(">>> SMOKE TEST MODE ACTIVATED: Restricting to 1 epoch, 2 train batches, 2 val batches <<<")
        epochs = 1
        max_train_batches = 2 if max_train_batches == 0 else max_train_batches
        max_val_batches = 2 if max_val_batches == 0 else max_val_batches

    # 4. Artifact Directory Protection
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

    # 5. Residual Scale Setup
    residual_scale = None
    if stats_data is not None and "statistics" in stats_data:
        sec_moments = stats_data["statistics"].get("channel_residual_second_moment_g0")
        if sec_moments is not None:
            residual_scale = torch.sqrt(torch.tensor(sec_moments, dtype=torch.float32) + 1e-6)

    # 6. Build Model and Strictly Load Parent FM Weights
    forecaster = build_and_freeze_rollout_aware_model(
        d0_ckpt_data=d0_ckpt_data,
        parent_fm_ckpt_data=parent_fm_ckpt_data,
        device=device,
        residual_scale=residual_scale,
    )
    flow_matcher = forecaster.flow_matcher
    fm_cfg = parent_fm_ckpt_data.get("config", {})
    hidden_channels = fm_cfg.get("hidden_channels", 128)
    num_blocks = fm_cfg.get("num_blocks", 4)
    target_mode = fm_cfg.get("target_mode", "residual")
    use_spatial_attn = fm_cfg.get("use_spatial_attn", True)

    # 7. Dataloaders: Horizon H=2 for all branches to guarantee identical windows
    train_loader, valid_loader, _, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file=split_file,
        data_root=data_root,
        history_length=4,
        horizon=2,
        valid_horizon=2,
        train_stride=8,
        valid_stride=8,
        downsample_factor=2,
        batch_size=batch_size,
        num_workers=2,
        normalize=True,
        normalizer=normalizer,
        seed=seed,
    )
    print(f"DataLoaders (H=2): Train={len(train_loader.dataset)} windows, Valid={len(valid_loader.dataset)} windows")

    # 8. Fresh Optimizer & Scheduler (Equally initialized across C1, C2, R2-A)
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

    # 9. Epoch 0 Baseline Audit
    print(f"\n--- Evaluating Epoch 0 Baseline ({branch}) ---")
    epoch0_val = evaluate_rollout_aware_validation(
        forecaster=forecaster,
        dataloader=valid_loader,
        device=device,
        branch=branch,
        sample_noise_scale=sample_noise_scale,
        num_flow_steps=num_flow_steps,
        solver=solver,
        max_batches=max_val_batches,
    )
    print(
        f"Epoch 0 Baseline: Total CFM={epoch0_val['val_total_cfm_loss']:.6f} "
        f"(Step 1={epoch0_val['val_step1_cfm_loss']:.6f}, Step 2={epoch0_val['val_step2_cfm_loss']:.6f})"
    )

    epoch0_payload = build_rollout_aware_checkpoint_payload(
        flow_matcher=flow_matcher,
        branch=branch,
        epoch=0,
        val_loss_dict=epoch0_val,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        target_mode=target_mode,
        use_spatial_attn=use_spatial_attn,
        residual_scale=residual_scale,
        sample_noise_scale=sample_noise_scale,
        d0_checkpoint=d0_checkpoint,
        d0_sha256=d0_sha256,
        parent_fm_checkpoint=parent_fm_checkpoint,
        parent_fm_sha256=parent_fm_sha256,
        split_file=split_file,
        runtime_split_hash=runtime_split_hash,
        normalizer_path=normalizer_path,
        runtime_norm_hash=runtime_norm_hash,
        actual_stats_path=residual_stats_path,
        seed=seed,
        num_flow_steps=num_flow_steps,
        solver=solver,
    )
    torch.save(epoch0_payload, os.path.join(output_dir, "epoch0_baseline.pt"))

    # Dedicated RNGs for CFM loss draws ensuring identical random sequences between C2 and R2-A
    gen_loss_1 = torch.Generator(device=device if device.type != "mps" else "cpu")
    gen_loss_1.manual_seed(seed + 100)
    gen_loss_2 = torch.Generator(device=device if device.type != "mps" else "cpu")
    gen_loss_2.manual_seed(seed + 200)
    gen_sample = torch.Generator(device=device if device.type != "mps" else "cpu")
    gen_sample.manual_seed(seed + 300)

    # 10. Training Loop
    history = []
    best_loss = epoch0_val["val_total_cfm_loss"]
    best_epoch = 0
    total_optimizer_updates = 0

    for epoch in range(1, epochs + 1):
        forecaster.train()
        train_loss_total = 0.0
        train_l1_total = 0.0
        train_l2_total = 0.0
        train_windows = 0

        for batch_idx, batch in enumerate(train_loader):
            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break

            optimizer.zero_grad()

            step_res = compute_rollout_aware_step_loss(
                forecaster=forecaster,
                batch=batch,
                device=device,
                branch=branch,
                sample_noise_scale=sample_noise_scale,
                num_flow_steps=num_flow_steps,
                solver=solver,
                gen_loss_1=gen_loss_1,
                gen_loss_2=gen_loss_2,
                gen_sample=gen_sample,
            )
            loss = step_res["loss"]

            loss.backward()

            # Production gradient clipping and finite norm assertion
            grad_norm = clip_and_validate_gradients(
                flow_matcher,
                max_norm=grad_clip,
                epoch=epoch,
                batch_idx=batch_idx,
            )

            optimizer.step()
            total_optimizer_updates += 1

            b = batch["history"].shape[0]
            train_loss_total += loss.item() * b
            train_l1_total += step_res["loss_step1"] * b
            train_l2_total += step_res["loss_step2"] * b
            train_windows += b

        scheduler.step()

        mean_train_loss = train_loss_total / max(1, train_windows)
        mean_train_l1 = train_l1_total / max(1, train_windows)
        mean_train_l2 = train_l2_total / max(1, train_windows)

        # Validation Audit
        val_res = evaluate_rollout_aware_validation(
            forecaster=forecaster,
            dataloader=valid_loader,
            device=device,
            branch=branch,
            sample_noise_scale=sample_noise_scale,
            num_flow_steps=num_flow_steps,
            solver=solver,
            max_batches=max_val_batches,
        )

        val_total = val_res["val_total_cfm_loss"]
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Total={mean_train_loss:.6f} "
            f"(L1={mean_train_l1:.6f}, L2={mean_train_l2:.6f}) | "
            f"Val Total={val_total:.6f} (L1={val_res['val_step1_cfm_loss']:.6f}, L2={val_res['val_step2_cfm_loss']:.6f})"
        )

        history.append({
            "epoch": epoch,
            "train_total_cfm_loss": mean_train_loss,
            "train_step1_cfm_loss": mean_train_l1,
            "train_step2_cfm_loss": mean_train_l2,
            "val_total_cfm_loss": val_total,
            "val_step1_cfm_loss": val_res["val_step1_cfm_loss"],
            "val_step2_cfm_loss": val_res["val_step2_cfm_loss"],
            "lr": float(scheduler.get_last_lr()[0]),
        })

        if val_total < best_loss:
            best_loss = val_total
            best_epoch = epoch
            best_payload = build_rollout_aware_checkpoint_payload(
                flow_matcher=flow_matcher,
                branch=branch,
                epoch=epoch,
                val_loss_dict=val_res,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
                target_mode=target_mode,
                use_spatial_attn=use_spatial_attn,
                residual_scale=residual_scale,
                sample_noise_scale=sample_noise_scale,
                d0_checkpoint=d0_checkpoint,
                d0_sha256=d0_sha256,
                parent_fm_checkpoint=parent_fm_checkpoint,
                parent_fm_sha256=parent_fm_sha256,
                split_file=split_file,
                runtime_split_hash=runtime_split_hash,
                normalizer_path=normalizer_path,
                runtime_norm_hash=runtime_norm_hash,
                actual_stats_path=residual_stats_path,
                seed=seed,
                num_flow_steps=num_flow_steps,
                solver=solver,
            )
            torch.save(best_payload, os.path.join(output_dir, "best_latent_flow_matcher.pt"))

    # If epoch 0 was not beaten, ensure best_latent_flow_matcher.pt exists
    best_ckpt_path = os.path.join(output_dir, "best_latent_flow_matcher.pt")
    if best_epoch == 0 and not os.path.exists(best_ckpt_path):
        torch.save(epoch0_payload, best_ckpt_path)

    # Save final checkpoint and summary
    final_payload = build_rollout_aware_checkpoint_payload(
        flow_matcher=flow_matcher,
        branch=branch,
        epoch=epochs,
        val_loss_dict=val_res,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        target_mode=target_mode,
        use_spatial_attn=use_spatial_attn,
        residual_scale=residual_scale,
        sample_noise_scale=sample_noise_scale,
        d0_checkpoint=d0_checkpoint,
        d0_sha256=d0_sha256,
        parent_fm_checkpoint=parent_fm_checkpoint,
        parent_fm_sha256=parent_fm_sha256,
        split_file=split_file,
        runtime_split_hash=runtime_split_hash,
        normalizer_path=normalizer_path,
        runtime_norm_hash=runtime_norm_hash,
        actual_stats_path=residual_stats_path,
        seed=seed,
        num_flow_steps=num_flow_steps,
        solver=solver,
    )
    torch.save(final_payload, os.path.join(output_dir, "final_latent_flow_matcher.pt"))

    best_ckpt_sha = compute_file_sha256(best_ckpt_path) if os.path.exists(best_ckpt_path) else None

    summary = {
        "branch": branch,
        "best_epoch": best_epoch,
        "best_val_cfm_loss": best_loss,
        "epoch0_val_cfm_loss": epoch0_val["val_total_cfm_loss"],
        "epoch0_val_step1_cfm_loss": epoch0_val["val_step1_cfm_loss"],
        "epoch0_val_step2_cfm_loss": epoch0_val["val_step2_cfm_loss"],
        "selected_checkpoint_path": best_ckpt_path,
        "selected_checkpoint_sha256": best_ckpt_sha,
        "experiment_config": {
            "branch": branch,
            "epochs": epochs,
            "sample_noise_scale": sample_noise_scale,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "num_flow_steps": num_flow_steps,
            "solver": solver,
            "grad_clip": grad_clip,
            "train_windows": len(train_loader.dataset),
            "val_windows": len(valid_loader.dataset),
            "num_optimizer_updates": total_optimizer_updates,
            "seed": seed,
        },
        "history": history,
        "provenance": final_payload["provenance"],
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Latent Flow Matching Rollout-Aware (FM-R2) Training")
    parser.add_argument("--branch", type=str, required=True, choices=["C1", "C2", "R2_A"],
                        help="Experimental branch: C1 (1-step continuation), C2 (Teacher-Forced 2-step), R2_A (Rollout-Aware 2-step)")
    parser.add_argument("--d0-checkpoint", type=str,
                        default="outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt")
    parser.add_argument("--parent-fm-checkpoint", type=str,
                        default="outputs/checkpoints/probabilistic/flow_matching_pilot3ep/best_latent_flow_matcher.pt")
    parser.add_argument("--normalizer-path", type=str, default="outputs/normalization/stats_grouped.pt")
    parser.add_argument("--split-file", type=str, default="outputs/splits/grouped_split.json")
    parser.add_argument("--residual-stats-path", type=str, default="outputs/normalization/latent_residual_stats.json")
    default_data_root = os.environ.get("SHEAR_FLOW_DATA_DIR", "/root/autodl-tmp/datasets/shear_flow" if os.path.exists("/root/autodl-tmp/datasets/shear_flow") else None)
    parser.add_argument("--data-root", type=str, default=default_data_root)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to outputs/checkpoints/probabilistic/flow_matching_r2/<branch>)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sample-noise-scale", type=float, default=None,
                        help="Sampling noise scale alpha_noise for Step 1 in R2_A (explicit value required for R2_A)")
    parser.add_argument("--num-flow-steps", type=int, default=10)
    parser.add_argument("--solver", type=str, default="midpoint")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output_dir is None:
        out_dir = f"outputs/checkpoints/probabilistic/flow_matching_r2/{args.branch}"
    else:
        out_dir = args.output_dir

    train_latent_flow_matching_rollout_aware(
        branch=args.branch,
        d0_checkpoint=args.d0_checkpoint,
        parent_fm_checkpoint=args.parent_fm_checkpoint,
        normalizer_path=args.normalizer_path,
        split_file=args.split_file,
        output_dir=out_dir,
        residual_stats_path=args.residual_stats_path,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        sample_noise_scale=args.sample_noise_scale,
        num_flow_steps=args.num_flow_steps,
        solver=args.solver,
        grad_clip=args.grad_clip,
        smoke_test=args.smoke_test,
        overwrite=args.overwrite,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
