"""Deterministic Latent Space Forecaster for 2D Periodic Vorticity Dynamics (Phase 2 A).

Integrates frozen VorticityAutoencoder (Encoder2D/Decoder2D) with LatentSTTransformer
for autoregressive physical world modeling on periodic scalar advection-diffusion fields.

Engineering & Scientific Contracts:
1. Frozen Representation: Encoder and Decoder are strictly frozen (requires_grad = False).
   Checkpoint SHA-256 is verified before and after execution to guarantee zero mutation.
2. Latent Forecaster: LatentSTTransformer operates exclusively in latent space R^(B, L, C_z, H_z, W_z).
3. HistoryBuffer: Autoregressive rollout maintains a fixed historical window L via FIFO rolling.
4. Triple-Trajectory Reporting: Every evaluation reports:
   - Analytical Ground Truth: omega_{t+h}
   - Frozen Autoencoder Reference: D(E(omega_{t+h}))
   - World Model Dynamic Prediction: D(hat{z}_{t+h})
   - Persistence Baseline: omega_t
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.audit_temporal_vorticity_representation import (
    compute_file_sha256,
    load_frozen_autoencoder,
)
from scripts.train_vorticity_autoencoder import VorticityAutoencoder
from src.models.history_buffer import HistoryBuffer
from src.models.latent_transformer import LatentSTTransformer


class VorticityLatentForecaster(nn.Module):
    """Integrates a frozen VorticityAutoencoder and a trainable LatentSTTransformer."""

    def __init__(
        self,
        autoencoder: VorticityAutoencoder,
        transformer: LatentSTTransformer,
        checkpoint_path: Optional[Union[str, Path]] = None,
        checkpoint_sha256: Optional[str] = None,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.transformer = transformer
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_sha256 = checkpoint_sha256

        # Enforce strict freeze on autoencoder
        self.autoencoder.eval()
        for p in self.autoencoder.parameters():
            p.requires_grad = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        latent_channels: int = 64,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> "VorticityLatentForecaster":
        """Load frozen autoencoder from checkpoint and construct forecaster."""
        ckpt_p = Path(checkpoint_path)
        ae, _, sha_before = load_frozen_autoencoder(
            checkpoint_path=ckpt_p,
            latent_channels=latent_channels,
            device=device,
        )

        tf_kwargs = {
            "latent_channels": latent_channels,
            "embed_dim": 128,
            "cond_dim": 64,
            "depth": 4,
            "num_heads": 4,
            "history_length": 4,
            "prediction_mode": "direct",
        }
        if transformer_kwargs:
            tf_kwargs.update(transformer_kwargs)

        transformer = LatentSTTransformer(**tf_kwargs)
        transformer.to(device)

        return cls(
            autoencoder=ae,
            transformer=transformer,
            checkpoint_path=ckpt_p,
            checkpoint_sha256=sha_before,
        )

    @classmethod
    def load_trained(
        cls,
        transformer_checkpoint_path: Union[str, Path],
        ae_checkpoint_path: Optional[Union[str, Path]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> "VorticityLatentForecaster":
        """Load trained forecaster from Phase 2 A checkpoint containing architecture metadata.

        Backward compatible with legacy pilot checkpoints by resolving AE references from
        sidecar metadata or standard experiment paths, and reconstructing transformer architecture
        via state_dict / sidecar inspection.
        """
        tf_ckpt_p = Path(transformer_checkpoint_path)
        payload = torch.load(tf_ckpt_p, map_location=device, weights_only=False)

        # 1. Resolve AE checkpoint path
        resolved_ae_path = None
        if ae_checkpoint_path is not None:
            resolved_ae_path = Path(ae_checkpoint_path)
        elif "ae_reference" in payload and "ae_checkpoint_path" in payload["ae_reference"]:
            resolved_ae_path = Path(payload["ae_reference"]["ae_checkpoint_path"])
        elif "ae_checkpoint_path" in payload:
            resolved_ae_path = Path(payload["ae_checkpoint_path"])
        else:
            # Fallback 1: check sidecar metadata or summary json in the checkpoint directory
            sidecar_candidates = [
                tf_ckpt_p.with_suffix(".metadata.json"),
                tf_ckpt_p.with_name(f"{tf_ckpt_p.stem}_metadata.json"),
                tf_ckpt_p.parent / "latent_forecaster_pilot_summary.json",
            ]
            for sc in sidecar_candidates:
                if sc.is_file():
                    try:
                        with open(sc, "r", encoding="utf-8") as f:
                            sc_meta = json.load(f)
                        cand_ae = (
                            sc_meta.get("ae_reference", {}).get("ae_checkpoint_path")
                            or sc_meta.get("checkpoint_integrity", {}).get("ae_checkpoint_path")
                        )
                        if cand_ae and Path(cand_ae).is_file():
                            resolved_ae_path = Path(cand_ae)
                            break
                    except Exception:
                        pass

            # Fallback 2: check standard capacity experiment path based on latent_channels
            if resolved_ae_path is None:
                latent_ch = payload.get("transformer_config", {}).get(
                    "latent_channels", payload.get("latent_channels", 64)
                )
                default_cand = Path(f"outputs/experiments/vorticity_capacity/cz_{latent_ch}/latest_checkpoint.pt")
                if default_cand.is_file():
                    resolved_ae_path = default_cand

        if resolved_ae_path is None or not resolved_ae_path.is_file():
            raise FileNotFoundError(
                f"Could not determine valid ae_checkpoint_path from payload, sidecar, or arguments for {tf_ckpt_p}."
            )

        latent_channels = payload.get("transformer_config", {}).get(
            "latent_channels", payload.get("latent_channels", 64)
        )

        ae, _, sha_before = load_frozen_autoencoder(
            checkpoint_path=resolved_ae_path,
            latent_channels=latent_channels,
            device=device,
        )

        # Verify AE SHA if recorded in checkpoint
        expected_ae_sha = (
            payload.get("ae_reference", {}).get("ae_checkpoint_sha256")
            or payload.get("ae_checkpoint_sha256")
        )
        if expected_ae_sha and sha_before != expected_ae_sha:
            raise RuntimeError(
                f"Autoencoder SHA mismatch: expected {expected_ae_sha}, got {sha_before}"
            )

        # 2. Reconstruct Transformer from config or state dict
        if "transformer_config" in payload:
            tf_kwargs = dict(payload["transformer_config"])
        else:
            # Check sidecar metadata
            sidecar_cfg = None
            sidecar_candidates = [
                tf_ckpt_p.with_suffix(".metadata.json"),
                tf_ckpt_p.with_name(f"{tf_ckpt_p.stem}_metadata.json"),
                tf_ckpt_p.parent / "latent_forecaster_pilot_summary.json",
            ]
            for sc in sidecar_candidates:
                if sc.is_file():
                    try:
                        with open(sc, "r", encoding="utf-8") as f:
                            sc_meta = json.load(f)
                        if "transformer_config" in sc_meta:
                            sidecar_cfg = sc_meta["transformer_config"]
                            break
                    except Exception:
                        pass
            if sidecar_cfg:
                tf_kwargs = dict(sidecar_cfg)
            else:
                sd = payload.get("transformer_state_dict", {})
                block_keys = [k for k in sd.keys() if k.endswith(".spatial_attn.qkv.weight")]
                inferred_depth = len(block_keys) if block_keys else int(payload.get("depth", 4))
                cond_w = sd.get("cond_embed.mlp.0.weight")
                inferred_cond = (cond_w.shape[0] // 2) if cond_w is not None else int(payload.get("cond_dim", 64))
                tf_kwargs = {
                    "latent_channels": latent_channels,
                    "embed_dim": int(payload.get("embed_dim", 128)),
                    "cond_dim": inferred_cond,
                    "depth": inferred_depth,
                    "num_heads": int(payload.get("num_heads", 4)),
                    "history_length": int(payload.get("history_length", 4)),
                    "prediction_mode": str(payload.get("prediction_mode", "direct")),
                }

        transformer = LatentSTTransformer(**tf_kwargs)
        transformer.load_state_dict(payload["transformer_state_dict"])
        transformer.to(device)

        return cls(
            autoencoder=ae,
            transformer=transformer,
            checkpoint_path=resolved_ae_path,
            checkpoint_sha256=sha_before,
        )

    def forward(self, q_hist: torch.Tensor) -> torch.Tensor:
        """Standard nn.Module forward pass: returns predicted physical field q_{t+1}."""
        q_pred, _, _ = self.forward_single_step(q_hist)
        return q_pred

    def verify_checkpoint_immutability(self) -> bool:
        """Verify that the underlying autoencoder checkpoint on disk was not modified."""
        if self.checkpoint_path and self.checkpoint_path.is_file() and self.checkpoint_sha256:
            current_sha = compute_file_sha256(self.checkpoint_path)
            if current_sha != self.checkpoint_sha256:
                raise RuntimeError(
                    f"Autoencoder checkpoint mutated! Original SHA: {self.checkpoint_sha256}, "
                    f"Current SHA: {current_sha}"
                )
            return True
        return True

    def encode_history(self, q_hist: torch.Tensor) -> torch.Tensor:
        """Encode physical history frames into latent states with gradient truncation.

        Args:
            q_hist: Tensor of shape (B, L, 1, H, W).

        Returns:
            z_hist: Latent tensor of shape (B, L, C_z, H_z, W_z).
        """
        with torch.no_grad():
            z_hist = self.autoencoder.encoder(q_hist)
        return z_hist

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent states back to physical space.

        Args:
            z: Latent tensor of shape (B, 1, C_z, H_z, W_z) or (B, T, C_z, H_z, W_z).

        Returns:
            q: Physical field of shape (B, 1, 1, H, W) or (B, T, 1, H, W).
        """
        # Decoder parameters are frozen (requires_grad=False), but gradients flow through to z
        return self.autoencoder.decoder(z)

    def forward_latent(self, z_hist: torch.Tensor) -> torch.Tensor:
        """Single-step latent prediction: Z_{t-L+1:t} -> hat{Z}_{t+1}."""
        return self.transformer(z_hist)

    def forward_single_step(
        self,
        q_hist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform full single-step physical prediction from physical history.

        Args:
            q_hist: Tensor of shape (B, L, 1, H, W).

        Returns:
            q_pred: Predicted physical field at t+1 of shape (B, 1, 1, H, W).
            z_pred: Predicted latent state at t+1 of shape (B, 1, C_z, H_z, W_z).
            z_hist: Encoded history latent states of shape (B, L, C_z, H_z, W_z).
        """
        z_hist = self.encode_history(q_hist)
        z_pred = self.forward_latent(z_hist)
        q_pred = self.decode(z_pred)
        return q_pred, z_pred, z_hist

    def rollout_latent(
        self,
        z_hist: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Autoregressively roll out H steps purely in latent space using HistoryBuffer.

        Args:
            z_hist: Initial history latent sequence (B, L, C_z, H_z, W_z).
            horizon: Rollout horizon H.

        Returns:
            z_rollout: Latent trajectory of shape (B, H, C_z, H_z, W_z).
        """
        buf = HistoryBuffer(history_length=z_hist.shape[1])
        buf.reset(z_hist)

        pred_latents = []
        for _ in range(horizon):
            z_next = self.transformer(buf.current)  # (B, 1, C_z, H_z, W_z)
            pred_latents.append(z_next)
            buf.push(z_next)

        return torch.cat(pred_latents, dim=1)  # (B, H, C_z, H_z, W_z)

    def rollout(
        self,
        q_hist: torch.Tensor,
        horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively roll out H steps from physical history and decode.

        Args:
            q_hist: Tensor of shape (B, L, 1, H, W).
            horizon: Rollout horizon H.

        Returns:
            q_rollout: Predicted physical fields (B, H, 1, H, W).
            z_rollout: Predicted latent states (B, H, C_z, H_z, W_z).
        """
        z_hist = self.encode_history(q_hist)
        z_rollout = self.rollout_latent(z_hist, horizon=horizon)
        q_rollout = self.decode(z_rollout)
        return q_rollout, z_rollout
