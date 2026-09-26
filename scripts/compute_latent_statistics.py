"""Rigorous Latent Space Residual Statistics and Decoder Sensitivity Computation.

Fulfills Phase 0 audit contract for ProbLatent-R1:
1. Atomically loads full D0 checkpoint (Encoder, Transformer, Decoder from single artifact).
2. Verifies bit-wise parity against standalone representation checkpoint.
3. Restores exact training-time spatial positional embedding semantics via resolve_spatial_pos_config.
4. Computes per-channel:
   - residual_mean: E[r_c]
   - residual_centered_variance: Var(r_c)
   - residual_second_moment: E[r_c^2] (the mathematically exact variance for fixed-mean Gaussian NLL)
   - sample_count: N
5. Decomposes physical error: ||D(z_GT) - q_GT|| vs ||D(mu) - q_GT|| vs ||z_GT - mu||.
6. Measures empirical Decoder perturbation amplification ratio at real trajectory states.
7. Emits self-describing JSON bound to split_hash, normalizer_hash, and checkpoint SHA-256.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import create_flow_dataloaders
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.utils.checkpoint import resolve_spatial_pos_config, strip_compiled_prefix
from src.utils.physics_contract import (
    PHYSICS_PROTOCOL,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
    SPATIAL_AXIS_CONTRACT,
)
from src.utils.provenance import compute_file_sha256, get_git_commit, is_git_dirty


def verify_representation_parity(
    forecaster: LatentForecaster,
    standalone_repr_path: str,
) -> Tuple[bool, bool, float]:
    """Verify whether forecaster representation weights are bit-wise identical to standalone AE.

    Returns:
        (encoder_identical, decoder_identical, max_param_diff)
    """
    if not os.path.exists(standalone_repr_path):
        return False, False, float("inf")

    ae_ckpt = torch.load(standalone_repr_path, map_location="cpu", weights_only=False)
    enc_sd = ae_ckpt.get("encoder_state_dict", {})
    dec_sd = ae_ckpt.get("decoder_state_dict", {})

    model_sd = forecaster.state_dict()
    max_diff = 0.0
    enc_match = True
    dec_match = True

    for k, v in enc_sd.items():
        fk = f"encoder.{k}"
        if fk not in model_sd:
            enc_match = False
            continue
        diff = (v - model_sd[fk].cpu()).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > 0:
            enc_match = False

    for k, v in dec_sd.items():
        fk = f"decoder.{k}"
        if fk not in model_sd:
            dec_match = False
            continue
        diff = (v - model_sd[fk].cpu()).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > 0:
            dec_match = False

    return enc_match, dec_match, max_diff


def load_d0_forecaster(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[LatentForecaster, Dict[str, Any], str]:
    """Load full D0 LatentForecaster atomically from single checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"D0 checkpoint not found at: {checkpoint_path}")

    sha256 = compute_file_sha256(checkpoint_path)
    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt_data.get("config", {})

    pred_mode = cfg.get("prediction_mode", ckpt_data.get("prediction_mode", "direct"))
    emb_dim = cfg.get("embed_dim", 256)
    depth = cfg.get("depth", 6)
    num_heads = cfg.get("num_heads", 8)
    use_spatial_pos = resolve_spatial_pos_config(ckpt_data)

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

    sd = ckpt_data.get("model_state_dict", ckpt_data)
    cleaned_sd = {
        (k[7:] if k.startswith("module.") else k): v for k, v in sd.items()
    }
    cleaned_sd = strip_compiled_prefix(cleaned_sd)
    forecaster.load_state_dict(cleaned_sd, strict=True)
    forecaster.eval()

    return forecaster, ckpt_data, sha256


def compute_latent_statistics_and_diagnostics(
    forecaster: LatentForecaster,
    dataloader: DataLoader,
    normalizer: Any,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute per-channel residual statistics, error decomposition, and decoder sensitivity."""
    forecaster.eval()
    num_channels = 64

    sum_r = torch.zeros(num_channels, dtype=torch.float64)
    sum_r2 = torch.zeros(num_channels, dtype=torch.float64)
    total_tokens_per_channel = 0

    ae_recon_error_sum = 0.0
    forecaster_phys_error_sum = 0.0
    latent_diff_error_sum = 0.0
    total_samples = 0

    perturbation_scales = [0.01, 0.05, 0.1, 0.2]
    sensitivity_ratios = {scale: [] for scale in perturbation_scales}

    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            if max_batches is not None and b_idx >= max_batches:
                break

            q_hist = batch["history"].to(device)       # (B, L, 4, Ny, Nx)
            q_next = batch["future"][:, 0:1].to(device) # (B, 1, 4, Ny, Nx)
            re = batch["re"].to(device)
            sc = batch["sc"].to(device)
            b = q_hist.shape[0]

            # 1. Forward latent predictions
            z_hist = forecaster.encoder(q_hist)
            z_gt = forecaster.encoder(q_next)          # (B, 1, 64, Hz, Wz)
            z_pred = forecaster.transformer(z_hist, re=re, sc=sc) # (B, 1, 64, Hz, Wz)

            # Residual r = z_gt - z_pred
            r = (z_gt - z_pred).squeeze(1)             # (B, 64, Hz, Wz)
            r_flat = r.permute(1, 0, 2, 3).reshape(num_channels, -1).to(torch.float64).cpu()
            n_tokens = r_flat.shape[1]

            sum_r += r_flat.sum(dim=1)
            sum_r2 += (r_flat ** 2).sum(dim=1)
            total_tokens_per_channel += n_tokens

            # 2. Physical error decomposition
            q_rec_gt = forecaster.decoder(z_gt)        # D(E(q_gt))
            q_rec_pred = forecaster.decoder(z_pred)    # D(mu)

            # Physical space unnormalization
            q_next_phys = normalizer.denormalize(q_next)
            q_rec_gt_phys = normalizer.denormalize(q_rec_gt)
            q_rec_pred_phys = normalizer.denormalize(q_rec_pred)

            ae_diff = (q_rec_gt_phys - q_next_phys).pow(2).mean(dim=(-2, -1)).sqrt().mean().item()
            pred_diff = (q_rec_pred_phys - q_next_phys).pow(2).mean(dim=(-2, -1)).sqrt().mean().item()
            latent_diff = (z_gt - z_pred).pow(2).mean().sqrt().item()

            ae_recon_error_sum += ae_diff * b
            forecaster_phys_error_sum += pred_diff * b
            latent_diff_error_sum += latent_diff * b
            total_samples += b

            # 3. Local decoder sensitivity around predicted state mu
            for scale in perturbation_scales:
                delta = torch.randn_like(z_pred) * scale
                q_pert = forecaster.decoder(z_pred + delta)
                q_pert_phys = normalizer.denormalize(q_pert)
                
                delta_rms = delta.pow(2).mean().sqrt().item()
                dq_rms = (q_pert_phys - q_rec_pred_phys).pow(2).mean().sqrt().item()
                ratio = dq_rms / (delta_rms + 1e-8)
                sensitivity_ratios[scale].append(ratio)

    mean_r = (sum_r / total_tokens_per_channel).tolist()
    second_moment_r = (sum_r2 / total_tokens_per_channel).tolist()
    var_r = [(m2 - m ** 2) for m, m2 in zip(mean_r, second_moment_r)]

    mean_sensitivity = {
        f"scale_{scale}": float(sum(ratios) / len(ratios)) if ratios else 0.0
        for scale, ratios in sensitivity_ratios.items()
    }

    return {
        "sample_count": total_samples,
        "token_count_per_channel": total_tokens_per_channel,
        "channel_residual_mean": mean_r,
        "channel_residual_centered_variance": var_r,
        "channel_residual_second_moment": second_moment_r,
        "summary": {
            "global_residual_mean": float(sum(mean_r) / num_channels),
            "global_centered_variance": float(sum(var_r) / num_channels),
            "global_second_moment_g0": float(sum(second_moment_r) / num_channels),
            "min_channel_second_moment": float(min(second_moment_r)),
            "max_channel_second_moment": float(max(second_moment_r)),
        },
        "physical_error_decomposition": {
            "ae_reconstruction_rmse": float(ae_recon_error_sum / total_samples),
            "latent_predicted_physical_rmse": float(forecaster_phys_error_sum / total_samples),
            "latent_space_rms_error": float(latent_diff_error_sum / total_samples),
        },
        "decoder_local_amplification": mean_sensitivity,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute rigorous latent statistics for ProbLatent-R1.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt",
        help="Path to full D0 checkpoint.",
    )
    parser.add_argument(
        "--standalone_repr_path",
        type=str,
        default="outputs/checkpoints/representation/best_autoencoder.pt",
        help="Path to standalone representation checkpoint for parity verification.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/datasets/shear_flow",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default="grouped",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=25,
        help="Number of training batches to audit (default: 25 for fast representative audit, None for full set).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/normalization/latent_residual_stats.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Latent Audit] Using device: {device}")
    print(f"[Latent Audit] Loading D0: {args.checkpoint_path}")

    forecaster, ckpt_data, d0_sha256 = load_d0_forecaster(args.checkpoint_path, device)
    enc_match, dec_match, max_diff = verify_representation_parity(forecaster, args.standalone_repr_path)
    repr_sha256 = compute_file_sha256(args.standalone_repr_path) if os.path.exists(args.standalone_repr_path) else None

    print(f"[Parity] Standalone representation vs D0 internal weights:")
    print(f"  Encoder bit-wise match: {enc_match}")
    print(f"  Decoder bit-wise match: {dec_match}")
    print(f"  Max absolute difference: {max_diff:.8e}")

    train_loader, _, _, normalizer = create_flow_dataloaders(
        data_root=args.data_root,
        split_type=args.split_type,
        history_length=4,
        horizon=1,
        batch_size=args.batch_size,
        train_stride=8,
        downsample_factor=2,
        normalize=True,
        num_workers=0,
    )

    stats = compute_latent_statistics_and_diagnostics(
        forecaster=forecaster,
        dataloader=train_loader,
        normalizer=normalizer,
        device=device,
        max_batches=args.max_batches,
    )

    result_bundle = {
        "audit_version": "ProbLatent-R1-Phase0",
        "d0_checkpoint": {
            "path": args.checkpoint_path,
            "sha256": d0_sha256,
            "epoch": ckpt_data.get("epoch"),
            "training_git_commit": ckpt_data.get("training_git_commit"),
            "use_spatial_pos": resolve_spatial_pos_config(ckpt_data),
        },
        "representation_checkpoint": {
            "path": args.standalone_repr_path,
            "sha256": repr_sha256,
            "encoder_identical_to_d0": enc_match,
            "decoder_identical_to_d0": dec_match,
            "max_param_diff": max_diff,
        },
        "data_protocol": {
            "split_type": args.split_type,
            "split_hash": ckpt_data.get("split_hash"),
            "normalizer_hash": ckpt_data.get("normalizer_hash"),
            "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
            "physics_protocol": PHYSICS_PROTOCOL,
            "domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
        },
        "statistics": stats,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result_bundle, f, indent=2)
    print(f"[Latent Audit] Successfully wrote statistics to: {args.output_path}")


if __name__ == "__main__":
    main()
