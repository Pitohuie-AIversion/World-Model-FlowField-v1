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
