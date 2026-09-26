"""ProbLatent-R1 Phase 2: Isolated Latent Variance Head Training Entrypoint.

Trains ONLY the conditional VarianceHead2D on top of a strictly frozen D0 LatentForecaster.
Governance Contract:
1. Strict pre-flight identity verification (D0 SHA-256, G0 stats SHA-256 verified against audit record,
   split_hash, normalizer_hash). Rejects missing D0 binding in G0 stats.
2. Parameter freeze: encoder, decoder, and Transformer backbone are frozen (requires_grad=False).
3. Optimizer: Adam updates exclusively VarianceHead2D parameters (requires_grad=True).
4. G0 alignment: Variance head is initialized with G0 second-moment biases and zero weights.
   Element-wise max difference against broadcast G0 variances is strictly asserted < 1e-5.
5. Epoch 0 audit: Evaluates and logs baseline G0 NLL on validation set before training steps,
   and saves g0_baseline_initialization.pt.
6. Fail-closed on non-finite values: Empty validation dataloader raises ValueError. Any non-finite
   loss, prediction, variance, or gradient fails closed immediately.
7. Artifact isolation: Smoke tests and formal runs use distinct directory paths. Reusing an existing
   output directory without --overwrite fails closed (FileExistsError).
8. Model selection & no-improvement handling: Best checkpoint chosen exclusively on validation NLL.
   If no training epoch beats Epoch 0 G0 baseline, outcome is recorded as NO_IMPROVEMENT_OVER_G0
   and no spurious best_g1_variance_head.pt is written.
"""

from datetime import datetime, timezone
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.models.probabilistic_latent_dynamics import (
    VarianceHead2D,
    compute_g1_bias_init_from_g0,
    gaussian_nll_latent_loss,
)
from src.data.pipeline import create_flow_dataloaders
from src.data.normalization import FieldNormalizer
from src.utils.checkpoint import resolve_spatial_pos_config, strip_compiled_prefix
from src.utils.provenance import (
    compute_file_sha256,
    compute_normalizer_hash,
    compute_split_hash_from_file,
    get_git_commit,
    hash_matches,
    is_git_dirty,
)


def verify_phase2_preflight_contract(
    d0_checkpoint_path: str,
    stats_path: str,
    normalizer_path: str,
    split_file: str,
    verification_record_path: Optional[str] = "outputs/normalization/latent_audit_verification_record.json",
    expected_stats_sha256: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], FieldNormalizer, str, str, str, str]:
    """Verify cryptographic bindings across D0 checkpoint, G0 stats, normalizer, split, and audit record.

    Returns:
        (ckpt_data, stats_data, normalizer, d0_sha256, stats_sha256, runtime_split_hash, runtime_norm_hash)
    """
    # 1. D0 checkpoint
    if not os.path.exists(d0_checkpoint_path):
        raise FileNotFoundError(f"D0 checkpoint not found: {d0_checkpoint_path}")
    d0_sha256 = compute_file_sha256(d0_checkpoint_path)
    ckpt_data = torch.load(d0_checkpoint_path, map_location="cpu", weights_only=False)

    # 2. G0 stats file
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"G0 latent residual stats file not found: {stats_path}")
    stats_sha256 = compute_file_sha256(stats_path)
    with open(stats_path, "r") as f:
        stats_data = json.load(f)

    # Cryptographic binding of G0 stats identity against frozen audit record
    if expected_stats_sha256 is not None:
        if stats_sha256 != expected_stats_sha256:
            raise ValueError(
                f"G0 stats SHA-256 mismatch against explicit expectation!\n"
                f"Expected: {expected_stats_sha256}\nActual:   {stats_sha256}"
            )
    elif verification_record_path and os.path.exists(verification_record_path):
        with open(verification_record_path, "r") as f:
            v_record = json.load(f)
        record_stats_sha = v_record.get("stats_file", {}).get("sha256")
        if not record_stats_sha:
            raise ValueError(f"Audit record '{verification_record_path}' is missing 'stats_file.sha256'.")
        if stats_sha256 != record_stats_sha:
            raise ValueError(
                f"G0 stats SHA-256 mismatch against verified audit record!\n"
                f"Expected: {record_stats_sha}\nActual:   {stats_sha256}"
            )

    # 3. Normalizer file
    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(f"Normalizer statistics file not found: {normalizer_path}")
    normalizer = FieldNormalizer()
    normalizer.load_state_dict(torch.load(normalizer_path, weights_only=True, map_location="cpu"))
    runtime_norm_hash = compute_normalizer_hash(normalizer)

    # 4. Split file
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    runtime_split_hash = compute_split_hash_from_file(split_file)

    # Cryptographic contract validations
    ckpt_split_hash = ckpt_data.get("split_hash") or ckpt_data.get("config", {}).get("split_hash")
    if not ckpt_split_hash:
        raise ValueError("D0 checkpoint is missing required 'split_hash'. Fails closed.")
    if not hash_matches(ckpt_split_hash, runtime_split_hash, min_prefix_len=16):
        raise ValueError(
            f"Split hash mismatch between D0 ({ckpt_split_hash}) and runtime split ({runtime_split_hash})."
        )

    ckpt_norm_hash = ckpt_data.get("normalizer_hash") or ckpt_data.get("config", {}).get("normalizer_hash")
    if not ckpt_norm_hash:
        raise ValueError("D0 checkpoint is missing required 'normalizer_hash'. Fails closed.")
    if not hash_matches(ckpt_norm_hash, runtime_norm_hash, min_prefix_len=16):
        raise ValueError(
            f"Normalizer hash mismatch between D0 ({ckpt_norm_hash}) and runtime normalizer ({runtime_norm_hash})."
        )

    stats_split_hash = stats_data.get("data_protocol", {}).get("split_hash")
    if not stats_split_hash:
        raise ValueError("G0 statistics file is missing required 'data_protocol.split_hash'. Fails closed.")
    if not hash_matches(stats_split_hash, runtime_split_hash, min_prefix_len=16):
        raise ValueError(
            f"Split hash mismatch between G0 stats ({stats_split_hash}) and runtime split ({runtime_split_hash})."
        )

    stats_norm_hash = stats_data.get("data_protocol", {}).get("normalizer_hash")
    if not stats_norm_hash:
        raise ValueError("G0 statistics file is missing required 'data_protocol.normalizer_hash'. Fails closed.")
    if not hash_matches(stats_norm_hash, runtime_norm_hash, min_prefix_len=16):
        raise ValueError(
            f"Normalizer hash mismatch between G0 stats ({stats_norm_hash}) and runtime normalizer ({runtime_norm_hash})."
        )

    # Reject missing D0 binding in G0 statistics file
    expected_d0_sha = stats_data.get("d0_checkpoint", {}).get("sha256")
    if not expected_d0_sha:
        raise ValueError("G0 statistics file is missing required 'd0_checkpoint.sha256'. Fails closed.")
    if expected_d0_sha != d0_sha256:
        raise ValueError(
            f"D0 SHA-256 bound in G0 stats ({expected_d0_sha}) does not match actual D0 file ({d0_sha256})."
        )

    return (
        ckpt_data,
        stats_data,
        normalizer,
        d0_sha256,
        stats_sha256,
        runtime_split_hash,
        runtime_norm_hash,
    )


def build_and_freeze_probabilistic_model(
    ckpt_data: Dict[str, Any],
    stats_data: Dict[str, Any],
    device: torch.device,
    variance_floor: float = 1e-4,
) -> Tuple[LatentForecaster, torch.Tensor]:
    """Build LatentForecaster with G0 initialized VarianceHead2D and strictly freeze D0.

    Returns:
        (forecaster, effective_g0)
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

    # 3. Initialize G0 aligned VarianceHead2D
    v_g0_list = stats_data["statistics"]["channel_residual_second_moment_g0"]
    assert len(v_g0_list) == 64, f"Expected 64 channel variances, got {len(v_g0_list)}"
    v_g0_tensor = torch.tensor(v_g0_list, dtype=torch.float32)

    b_init, effective_g0 = compute_g1_bias_init_from_g0(v_g0_tensor, variance_floor=variance_floor)

    v_head = VarianceHead2D(
        embed_dim=emb_dim,
        latent_channels=64,
        variance_floor=variance_floor,
    ).to(device)

    with torch.no_grad():
        v_head.linear.weight.zero_()
        v_head.linear.bias.copy_(b_init.to(device))

    forecaster.transformer.attach_variance_head(v_head)

    # 4. Strictly freeze representation and Transformer backbone
    forecaster.freeze_for_variance_training()

    # 5. Governance assertion: verify parameter gradients
    trainable_params = []
    frozen_params = []
    for name, p in forecaster.named_parameters():
        if "variance_head" in name:
            if not p.requires_grad:
                raise RuntimeError(f"Variance head param {name} must have requires_grad=True")
            trainable_params.append(name)
        else:
            if p.requires_grad:
                raise RuntimeError(f"Model param {name} must be frozen (requires_grad=False)")
            frozen_params.append(name)

    assert len(trainable_params) == 2, f"Expected 2 trainable parameters, found: {trainable_params}"
    assert len(frozen_params) > 0, "No frozen parameters found"

    return forecaster, effective_g0


def evaluate_variance_nll(
    forecaster: LatentForecaster,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 0,
    fail_on_non_finite: bool = True,
) -> Dict[str, Any]:
    """Evaluate one-step Gaussian NLL and variance statistics on a dataloader.

    Fails closed on empty dataloader or non-finite values.
    """
    forecaster.eval()
    total_loss = 0.0
    total_tokens = 0
    min_var = float("inf")
    max_var = float("-inf")
    var_sum = 0.0
    non_finite_count = 0
    batches_evaluated = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if 0 < max_batches <= batch_idx:
                break
            batches_evaluated += 1

            q_hist = batch["history"].to(device)  # (B, L, 4, Ny, Nx)
            q_next = batch["future"][:, 0:1].to(device)  # (B, 1, 4, Ny, Nx)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)

            mu, var = forecaster.predict_distribution_single_step(q_hist, re=re, sc=sc)
            target_z = forecaster.encoder(q_next)

            # Check predictions for finiteness
            if (
                torch.isnan(mu).any()
                or torch.isinf(mu).any()
                or torch.isnan(var).any()
                or torch.isinf(var).any()
            ):
                if fail_on_non_finite:
                    raise FloatingPointError(
                        f"Non-finite mu or var predicted at batch {batch_idx}."
                    )
                non_finite_count += 1
                continue

            loss = gaussian_nll_latent_loss(mu=mu, target=target_z, variance=var)

            if torch.isnan(loss) or torch.isinf(loss):
                if fail_on_non_finite:
                    raise FloatingPointError(
                        f"Non-finite NLL loss computed at batch {batch_idx}."
                    )
                non_finite_count += 1
                continue

            b_tokens = target_z.numel()
            total_loss += loss.item() * b_tokens
            total_tokens += b_tokens

            min_var = min(min_var, var.min().item())
            max_var = max(max_var, var.max().item())
            var_sum += var.mean().item() * b_tokens

    if batches_evaluated == 0:
        raise ValueError("Validation dataloader is empty; cannot evaluate NLL.")

    is_valid = (non_finite_count == 0) and (total_tokens > 0)
    if not is_valid:
        if fail_on_non_finite:
            raise FloatingPointError(
                f"Evaluation failed with {non_finite_count} non-finite batches out of {batches_evaluated}."
            )
        mean_nll = float("nan")
        mean_var = float("nan")
    else:
        mean_nll = total_loss / total_tokens
        mean_var = var_sum / total_tokens

    return {
        "nll": float(mean_nll),
        "mean_variance": float(mean_var),
        "min_variance": float(min_var) if min_var != float("inf") else float("nan"),
        "max_variance": float(max_var) if max_var != float("-inf") else float("nan"),
        "non_finite_batches": non_finite_count,
        "batches_evaluated": batches_evaluated,
        "tokens_evaluated": total_tokens,
        "is_valid": is_valid,
    }


def train_prob_latent_variance(
    d0_checkpoint: str,
    stats_path: str,
    normalizer_path: str,
    split_file: str,
    data_root: str,
    output_dir: str,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 16,
    max_train_batches: int = 0,
    max_val_batches: int = 0,
    variance_floor: float = 1e-4,
    device: Optional[torch.device] = None,
    smoke_test: bool = False,
    overwrite: bool = False,
    verification_record_path: Optional[str] = "outputs/normalization/latent_audit_verification_record.json",
) -> Dict[str, Any]:
    """Execute complete Phase 2 variance head training workflow."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_mode = "smoke_test" if smoke_test else "formal"

    print("=================================================================")
    print(f"ProbLatent-R1 Phase 2: Latent Variance Head Training [{run_mode.upper()}]")
    print("=================================================================")
    print(f"Target device:     {device}")
    print(f"D0 checkpoint:     {d0_checkpoint}")
    print(f"G0 stats file:     {stats_path}")
    print(f"Normalizer:        {normalizer_path}")
    print(f"Split file:        {split_file}")
    print(f"Output directory:  {output_dir}")

    # Output directory collision and reuse check
    if os.path.exists(output_dir):
        existing_artifacts = [
            f for f in ["best_g1_variance_head.pt", "variance_training_history.json", "g0_baseline_initialization.pt"]
            if os.path.exists(os.path.join(output_dir, f))
        ]
        if existing_artifacts and not overwrite:
            raise FileExistsError(
                f"Output directory '{output_dir}' already contains artifacts from a prior run: {existing_artifacts}. "
                f"Use a different output_dir or pass overwrite=True."
            )
        if overwrite:
            print(f"[Warning] Overwrite specified: clearing existing artifacts in {output_dir}")
            for f in existing_artifacts:
                os.remove(os.path.join(output_dir, f))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Pre-flight verification
    (
        ckpt_data,
        stats_data,
        normalizer,
        d0_sha256,
        stats_sha256,
        runtime_split_hash,
        runtime_norm_hash,
    ) = verify_phase2_preflight_contract(
        d0_checkpoint_path=d0_checkpoint,
        stats_path=stats_path,
        normalizer_path=normalizer_path,
        split_file=split_file,
        verification_record_path=verification_record_path,
    )
    print("\n[Contract Preflight Verified]")
    print(f"  D0 SHA-256:        {d0_sha256}")
    print(f"  G0 stats SHA-256:  {stats_sha256}")
    print(f"  Split hash:        {runtime_split_hash}")
    print(f"  Normalizer hash:   {runtime_norm_hash}")

    # 2. Build model and freeze D0
    forecaster, effective_g0 = build_and_freeze_probabilistic_model(
        ckpt_data=ckpt_data,
        stats_data=stats_data,
        device=device,
        variance_floor=variance_floor,
    )
    print("\n[Model Governance Ready]")
    print("  Representation & Transformer backbone frozen successfully.")
    print("  VarianceHead2D attached and initialized from G0 second moments.")

    # 3. Create DataLoaders
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
    )
    print(f"\n[DataLoaders Ready] Total dataset windows: Train={len(train_loader.dataset)}, Valid={len(valid_loader.dataset)}")
    if smoke_test:
        print(f"  Smoke test mode active: capping train batches to {max_train_batches}, val batches to {max_val_batches}.")

    # 4. Rigorous G0 broadcast vs G1 initial element-wise check on actual input
    forecaster.eval()
    first_val_batch = next(iter(valid_loader))
    with torch.no_grad():
        _, init_val_var = forecaster.predict_distribution_single_step(
            first_val_batch["history"].to(device),
            re=first_val_batch["re"].to(device),
            sc=first_val_batch["sc"].to(device),
        )
    # effective_g0 shape (C_z,) -> (1, 1, C_z, 1, 1)
    g0_broadcast = effective_g0.view(1, 1, -1, 1, 1).to(device)
    max_elem_diff = (init_val_var - g0_broadcast).abs().max().item()
    print(f"[G0/G1 Invariant] Max element-wise difference across tokens: {max_elem_diff:.8e}")
    if max_elem_diff >= 1e-5:
        raise AssertionError(
            f"G1 initial variance diverges from effective G0 variance! Max error: {max_elem_diff:.8e}"
        )

    # 5. Strict Optimizer Setup (VarianceHead2D only)
    optimizer = torch.optim.Adam(
        forecaster.transformer.variance_head.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # 6. Epoch 0 Baseline Evaluation and Checkpoint Saving
    print("\n--- Evaluating Epoch 0 (G0 Baseline on Validation Set) ---")
    epoch0_metrics = evaluate_variance_nll(
        forecaster=forecaster,
        dataloader=valid_loader,
        device=device,
        max_batches=max_val_batches,
        fail_on_non_finite=True,
    )
    print(f"Epoch 0 Validation NLL (G0 baseline): {epoch0_metrics['nll']:.6f}")
    print(
        f"  Mean variance: {epoch0_metrics['mean_variance']:.6f} "
        f"(min: {epoch0_metrics['min_variance']:.6f}, max: {epoch0_metrics['max_variance']:.6f})"
    )

    # Save initial baseline checkpoint
    g0_baseline_path = os.path.join(output_dir, "g0_baseline_initialization.pt")
    torch.save(
        {
            "variance_head_state_dict": forecaster.transformer.variance_head.state_dict(),
            "provenance": {
                "d0_checkpoint": {"path": d0_checkpoint, "sha256": d0_sha256},
                "stats_file": {"path": stats_path, "sha256": stats_sha256},
                "data_protocol": {
                    "split_file": split_file,
                    "split_hash": runtime_split_hash,
                    "normalizer_file": normalizer_path,
                    "normalizer_hash": runtime_norm_hash,
                },
                "run_mode": run_mode,
                "epoch": 0,
                "val_nll": epoch0_metrics["nll"],
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        },
        g0_baseline_path,
    )
    print(f"  --> Saved G0 baseline initialization state to {g0_baseline_path}")

    # 7. Training Loop
    history = []
    best_val_nll = epoch0_metrics["nll"]
    best_epoch = 0
    selected_ckpt_path: Optional[str] = None
    total_train_windows_processed = 0
    total_val_windows_processed = 0

    for epoch in range(1, epochs + 1):
        forecaster.encoder.eval()
        forecaster.decoder.eval()
        forecaster.transformer.eval()
        forecaster.transformer.variance_head.train()

        train_loss_total = 0.0
        train_tokens_total = 0
        train_batches_count = 0

        for batch_idx, batch in enumerate(train_loader):
            if 0 < max_train_batches <= batch_idx:
                break
            train_batches_count += 1

            q_hist = batch["history"].to(device)
            q_next = batch["future"][:, 0:1].to(device)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)

            optimizer.zero_grad()

            mu, var = forecaster.predict_distribution_single_step(q_hist, re=re, sc=sc)
            with torch.no_grad():
                target_z = forecaster.encoder(q_next)

            loss = gaussian_nll_latent_loss(mu=mu, target=target_z, variance=var)

            # Fail closed on non-finite loss during training
            if torch.isnan(loss) or torch.isinf(loss):
                raise FloatingPointError(
                    f"Training loss is non-finite (loss={loss.item()}) at epoch {epoch}, batch {batch_idx}."
                )

            loss.backward()

            # Fail closed on non-finite gradients
            for p_name, p in forecaster.transformer.variance_head.named_parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    raise FloatingPointError(
                        f"Non-finite gradients detected in parameter {p_name} at epoch {epoch}, batch {batch_idx}."
                    )

            torch.nn.utils.clip_grad_norm_(forecaster.transformer.variance_head.parameters(), max_norm=1.0)
            optimizer.step()

            b_tokens = target_z.numel()
            train_loss_total += loss.item() * b_tokens
            train_tokens_total += b_tokens
            total_train_windows_processed += q_hist.shape[0]

        train_nll = train_loss_total / max(train_tokens_total, 1)

        # Validation step: fail closed if any batch is non-finite
        val_metrics = evaluate_variance_nll(
            forecaster=forecaster,
            dataloader=valid_loader,
            device=device,
            max_batches=max_val_batches,
            fail_on_non_finite=True,
        )
        total_val_windows_processed += val_metrics["batches_evaluated"] * batch_size

        epoch_record = {
            "epoch": epoch,
            "train_nll": float(train_nll),
            "train_batches_evaluated": train_batches_count,
            "val_nll": val_metrics["nll"],
            "val_mean_variance": val_metrics["mean_variance"],
            "val_min_variance": val_metrics["min_variance"],
            "val_max_variance": val_metrics["max_variance"],
            "val_batches_evaluated": val_metrics["batches_evaluated"],
            "val_is_valid": val_metrics["is_valid"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train NLL: {train_nll:.6f} | "
            f"Val NLL: {val_metrics['nll']:.6f} | "
            f"Var [min, mean, max]: [{val_metrics['min_variance']:.4f}, {val_metrics['mean_variance']:.4f}, {val_metrics['max_variance']:.4f}]"
        )

        # Model selection: strictly valid validation NLL and strictly better than baseline
        if val_metrics["is_valid"] and not math.isnan(val_metrics["nll"]) and val_metrics["nll"] < best_val_nll:
            best_val_nll = val_metrics["nll"]
            best_epoch = epoch
            best_ckpt_path = os.path.join(output_dir, "best_g1_variance_head.pt")
            selected_ckpt_path = best_ckpt_path

            provenance = {
                "d0_checkpoint": {"path": d0_checkpoint, "sha256": d0_sha256},
                "stats_file": {"path": stats_path, "sha256": stats_sha256},
                "data_protocol": {
                    "split_file": split_file,
                    "split_hash": runtime_split_hash,
                    "normalizer_file": normalizer_path,
                    "normalizer_hash": runtime_norm_hash,
                },
                "run_mode": run_mode,
                "best_epoch": best_epoch,
                "best_val_nll": best_val_nll,
                "baseline_g0_nll": epoch0_metrics["nll"],
                "git_commit": get_git_commit(PROJECT_ROOT),
                "git_dirty": is_git_dirty(PROJECT_ROOT),
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            }

            torch.save(
                {
                    "variance_head_state_dict": forecaster.transformer.variance_head.state_dict(),
                    "provenance": provenance,
                },
                best_ckpt_path,
            )
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Val NLL: {best_val_nll:.6f})")

    # 8. Final Report Formulation
    if best_epoch > 0 and selected_ckpt_path is not None:
        training_outcome = "IMPROVEMENT_FOUND"
        training_status = "PHASE2_VARIANCE_TRAINING_COMPLETE"
    else:
        training_outcome = "NO_IMPROVEMENT_OVER_G0"
        training_status = "PHASE2_VARIANCE_TRAINING_NO_IMPROVEMENT"
        selected_ckpt_path = g0_baseline_path
        print(f"\n[Notice] No training epoch outperformed G0 baseline ({epoch0_metrics['nll']:.6f}).")
        print(f"  Referencing baseline checkpoint: {g0_baseline_path}")

    summary = {
        "status": training_status,
        "outcome": training_outcome,
        "run_mode": run_mode,
        "d0_sha256": d0_sha256,
        "stats_sha256": stats_sha256,
        "runtime_split_hash": runtime_split_hash,
        "runtime_norm_hash": runtime_norm_hash,
        "baseline_g0_nll": epoch0_metrics["nll"],
        "best_epoch": best_epoch,
        "best_val_nll": best_val_nll,
        "selected_checkpoint_path": selected_ckpt_path,
        "training_configuration": {
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "variance_floor": variance_floor,
            "max_train_batches": max_train_batches,
            "max_val_batches": max_val_batches,
            "total_train_windows_processed": total_train_windows_processed,
            "total_val_windows_processed": total_val_windows_processed,
        },
        "history": history,
    }

    history_file = os.path.join(output_dir, "variance_training_history.json")
    with open(history_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Training Complete] Summary report saved to: {history_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train isolated latent variance head for ProbLatent-R1 Phase 2.")
    parser.add_argument(
        "--d0_checkpoint",
        type=str,
        default="outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt",
        help="Path to full D0 checkpoint.",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="outputs/normalization/latent_residual_stats.json",
        help="Path to G0 latent residual statistics JSON.",
    )
    parser.add_argument(
        "--normalizer_path",
        type=str,
        default="outputs/normalization/stats_grouped.pt",
        help="Path to fitted normalizer state dict.",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="outputs/splits/grouped_split.json",
        help="Path to dataset split JSON.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/datasets/shear_flow",
        help="Path to HDF5 shear flow dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store variance head checkpoints and logs. Defaults to isolated smoke or formal directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tracking and directory organization.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for variance head Adam optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for Adam optimizer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run 1 epoch with small number of batches for connectivity verification.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it contains previous artifacts.",
    )
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="Limit number of train batches per epoch (0 for full train set).",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=0,
        help="Limit number of validation batches per epoch (0 for full validation set).",
    )
    args = parser.parse_args()

    epochs = 1 if args.smoke_test else args.epochs
    max_train_batches = 2 if args.smoke_test else args.max_train_batches
    max_val_batches = 2 if args.smoke_test else args.max_val_batches

    # Distinct directory isolation between smoke_test and formal runs
    if args.output_dir is not None:
        target_output_dir = args.output_dir
    else:
        if args.smoke_test:
            target_output_dir = f"outputs/checkpoints/probabilistic/variance_head/smoke_test/seed_{args.seed}"
        else:
            target_output_dir = f"outputs/checkpoints/probabilistic/variance_head/seed_{args.seed}"

    train_prob_latent_variance(
        d0_checkpoint=args.d0_checkpoint,
        stats_path=args.stats_path,
        normalizer_path=args.normalizer_path,
        split_file=args.split_file,
        data_root=args.data_root,
        output_dir=target_output_dir,
        epochs=epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        smoke_test=args.smoke_test,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
