"""ProbLatent-R1 Phase 2: Isolated Latent Variance Head Training Entrypoint.

Trains ONLY the conditional VarianceHead2D on top of a strictly frozen D0 LatentForecaster.
Governance Contract:
1. Strict pre-flight identity verification (D0 SHA-256, G0 stats SHA-256, split_hash, normalizer_hash).
2. Parameter freeze: encoder, decoder, and Transformer backbone are frozen (requires_grad=False).
3. Optimizer: Adam updates exclusively VarianceHead2D parameters (requires_grad=True).
4. G0 alignment: Variance head is initialized with G0 second-moment biases and zero weights.
5. Epoch 0 audit: Evaluates and logs baseline G0 NLL on validation set before training steps.
6. Target supervision: Target latent z^{GT} is encoded from true next frame q_{t+1}^{GT} using frozen D0 encoder.
   No truth backfilling during autoregressive rollout is used during training or inference.
7. Model selection: Best checkpoint chosen exclusively on validation NLL (test set untouched).
"""

from datetime import datetime, timezone
import argparse
import json
import math
import os
from pathlib import Path
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
) -> Tuple[Dict[str, Any], Dict[str, Any], FieldNormalizer, str, str, str, str]:
    """Verify cryptographic bindings across D0 checkpoint, G0 stats, normalizer, and split.

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
    if not hash_matches(stats_split_hash, runtime_split_hash, min_prefix_len=16):
        raise ValueError(
            f"Split hash mismatch between G0 stats ({stats_split_hash}) and runtime split ({runtime_split_hash})."
        )

    stats_norm_hash = stats_data.get("data_protocol", {}).get("normalizer_hash")
    if not hash_matches(stats_norm_hash, runtime_norm_hash, min_prefix_len=16):
        raise ValueError(
            f"Normalizer hash mismatch between G0 stats ({stats_norm_hash}) and runtime normalizer ({runtime_norm_hash})."
        )

    expected_d0_sha = stats_data.get("d0_checkpoint", {}).get("sha256")
    if expected_d0_sha and expected_d0_sha != d0_sha256:
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
) -> LatentForecaster:
    """Build LatentForecaster with G0 initialized VarianceHead2D and strictly freeze D0."""
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

    return forecaster


def evaluate_variance_nll(
    forecaster: LatentForecaster,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> Dict[str, float]:
    """Evaluate one-step Gaussian NLL and variance statistics on a dataloader."""
    forecaster.eval()
    total_loss = 0.0
    total_tokens = 0
    min_var = float("inf")
    max_var = float("-inf")
    var_sum = 0.0
    non_finite_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if 0 < max_batches <= batch_idx:
                break

            q_hist = batch["history"].to(device)  # (B, L, 4, Ny, Nx)
            q_next = batch["future"][:, 0:1].to(device)  # (B, 1, 4, Ny, Nx)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)

            mu, var = forecaster.predict_distribution_single_step(q_hist, re=re, sc=sc)
            target_z = forecaster.encoder(q_next)

            loss = gaussian_nll_latent_loss(mu=mu, target=target_z, variance=var)

            if torch.isnan(loss) or torch.isinf(loss):
                non_finite_count += 1
                continue

            b_tokens = target_z.numel()
            total_loss += loss.item() * b_tokens
            total_tokens += b_tokens

            min_var = min(min_var, var.min().item())
            max_var = max(max_var, var.max().item())
            var_sum += var.mean().item() * b_tokens

    mean_nll = total_loss / max(total_tokens, 1)
    mean_var = var_sum / max(total_tokens, 1)

    return {
        "nll": float(mean_nll),
        "mean_variance": float(mean_var),
        "min_variance": float(min_var) if min_var != float("inf") else 0.0,
        "max_variance": float(max_var) if max_var != float("-inf") else 0.0,
        "non_finite_batches": non_finite_count,
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
) -> Dict[str, Any]:
    """Execute complete Phase 2 variance head training workflow."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=================================================================")
    print("ProbLatent-R1 Phase 2: Latent Variance Head Training Entrypoint")
    print("=================================================================")
    print(f"Target device:     {device}")
    print(f"D0 checkpoint:     {d0_checkpoint}")
    print(f"G0 stats file:     {stats_path}")
    print(f"Normalizer:        {normalizer_path}")
    print(f"Split file:        {split_file}")

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
    )
    print("\n[Contract Preflight Verified]")
    print(f"  D0 SHA-256:        {d0_sha256}")
    print(f"  G0 stats SHA-256:  {stats_sha256}")
    print(f"  Split hash:        {runtime_split_hash}")
    print(f"  Normalizer hash:   {runtime_norm_hash}")

    # 2. Build model and freeze D0
    forecaster = build_and_freeze_probabilistic_model(
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
    print(f"\n[DataLoaders Ready] Train windows: {len(train_loader.dataset)}, Valid windows: {len(valid_loader.dataset)}")

    # 4. Strict Optimizer Setup (VarianceHead2D only)
    optimizer = torch.optim.Adam(
        forecaster.transformer.variance_head.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # 5. Epoch 0 Baseline Evaluation (G0 state before any updates)
    print("\n--- Evaluating Epoch 0 (G0 Baseline on Validation Set) ---")
    epoch0_metrics = evaluate_variance_nll(
        forecaster=forecaster,
        dataloader=valid_loader,
        device=device,
        max_batches=max_val_batches,
    )
    print(f"Epoch 0 Validation NLL (G0 baseline): {epoch0_metrics['nll']:.6f}")
    print(f"  Mean variance: {epoch0_metrics['mean_variance']:.6f} (min: {epoch0_metrics['min_variance']:.6f}, max: {epoch0_metrics['max_variance']:.6f})")

    # 6. Training Loop
    history = []
    best_val_nll = epoch0_metrics["nll"]
    best_epoch = 0
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        forecaster.encoder.eval()
        forecaster.decoder.eval()
        forecaster.transformer.eval()
        forecaster.transformer.variance_head.train()

        train_loss_total = 0.0
        train_tokens_total = 0
        train_non_finite = 0

        for batch_idx, batch in enumerate(train_loader):
            if 0 < max_train_batches <= batch_idx:
                break

            q_hist = batch["history"].to(device)
            q_next = batch["future"][:, 0:1].to(device)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)

            optimizer.zero_grad()

            mu, var = forecaster.predict_distribution_single_step(q_hist, re=re, sc=sc)
            with torch.no_grad():
                target_z = forecaster.encoder(q_next)

            loss = gaussian_nll_latent_loss(mu=mu, target=target_z, variance=var)

            if torch.isnan(loss) or torch.isinf(loss):
                train_non_finite += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(forecaster.transformer.variance_head.parameters(), max_norm=1.0)
            optimizer.step()

            b_tokens = target_z.numel()
            train_loss_total += loss.item() * b_tokens
            train_tokens_total += b_tokens

        train_nll = train_loss_total / max(train_tokens_total, 1)

        # Validation step
        val_metrics = evaluate_variance_nll(
            forecaster=forecaster,
            dataloader=valid_loader,
            device=device,
            max_batches=max_val_batches,
        )

        epoch_record = {
            "epoch": epoch,
            "train_nll": float(train_nll),
            "train_non_finite_batches": train_non_finite,
            "val_nll": val_metrics["nll"],
            "val_mean_variance": val_metrics["mean_variance"],
            "val_min_variance": val_metrics["min_variance"],
            "val_max_variance": val_metrics["max_variance"],
            "val_non_finite_batches": val_metrics["non_finite_batches"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train NLL: {train_nll:.6f} | "
            f"Val NLL: {val_metrics['nll']:.6f} | "
            f"Var [min, mean, max]: [{val_metrics['min_variance']:.4f}, {val_metrics['mean_variance']:.4f}, {val_metrics['max_variance']:.4f}]"
        )

        # Model selection: best checkpoint strictly on validation NLL
        if val_metrics["nll"] < best_val_nll:
            best_val_nll = val_metrics["nll"]
            best_epoch = epoch
            best_ckpt_path = os.path.join(output_dir, "best_g1_variance_head.pt")

            provenance = {
                "d0_checkpoint": {
                    "path": d0_checkpoint,
                    "sha256": d0_sha256,
                },
                "stats_file": {
                    "path": stats_path,
                    "sha256": stats_sha256,
                },
                "data_protocol": {
                    "split_file": split_file,
                    "split_hash": runtime_split_hash,
                    "normalizer_file": normalizer_path,
                    "normalizer_hash": runtime_norm_hash,
                },
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

    # 7. Write final training report
    summary = {
        "status": "PHASE2_VARIANCE_TRAINING_COMPLETE",
        "d0_sha256": d0_sha256,
        "stats_sha256": stats_sha256,
        "runtime_split_hash": runtime_split_hash,
        "runtime_norm_hash": runtime_norm_hash,
        "baseline_g0_nll": epoch0_metrics["nll"],
        "best_epoch": best_epoch,
        "best_val_nll": best_val_nll,
        "epochs_trained": epochs,
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
        default="outputs/checkpoints/probabilistic/variance_head/seed_42",
        help="Directory to store variance head checkpoints and logs.",
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

    train_prob_latent_variance(
        d0_checkpoint=args.d0_checkpoint,
        stats_path=args.stats_path,
        normalizer_path=args.normalizer_path,
        split_file=args.split_file,
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )


if __name__ == "__main__":
    main()
