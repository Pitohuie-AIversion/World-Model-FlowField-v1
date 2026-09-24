"""End-to-end Latent Forecaster integrating Encoder, LatentSTTransformer, and Decoder."""

from typing import Optional
import torch
import torch.nn as nn
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.history_buffer import HistoryBuffer
from src.models.latent_transformer import LatentSTTransformer


class LatentForecaster(nn.Module):
    """Integrates spatial Encoder, LatentSTTransformer, and spatial Decoder.

    Args:
        encoder: Encoder2D instance.
        transformer: LatentSTTransformer instance.
        decoder: Decoder2D instance.
        freeze_representation: If True, disables gradient computation on Encoder and Decoder.
    """

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
        re: Optional[torch.Tensor] = None,
        sc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """q_hist: (B, L, 4, Ny, Nx) -> q_pred: (B, 1, 4, Ny, Nx)."""
        if self.freeze_representation:
            with torch.no_grad():
                z_hist = self.encoder(q_hist)
        else:
            z_hist = self.encoder(q_hist)

        z_next = self.transformer(z_hist, re=re, sc=sc)
        q_next = self.decoder(z_next)
        return q_next

    def forward_rollout(
        self,
        q_hist: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        sc: Optional[torch.Tensor] = None,
        horizon: int = 30,
        pushforward_steps: int = 0,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Roll out H steps entirely in latent space, then decode.

        Args:
            q_hist: History physical fields of shape (B, L, C, Ny, Nx).
            re: Optional Reynolds number tensor (B,).
            sc: Optional Schmidt number tensor (B,).
            horizon: Prediction horizon H (default: 30).
            pushforward_steps: Number of warmup rollout steps executed without gradients (default: 0).
            noise_std: Standard deviation of Gaussian perturbation injected during rollout (default: 0.0).

        Returns:
            q_rollout: Predicted physical fields of shape (B, horizon, C, Ny, Nx).
        """
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

        z_rollout = buf.rollout(step_fn, steps=horizon, noise_std=noise_std)  # (B, H, C_z, H_z, W_z)
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
        """Unified forward interface matching baseline models.

        Args:
            q_hist: History physical fields of shape (B, L, C, Ny, Nx).
            re: Optional Reynolds number tensor (B,).
            sc: Optional Schmidt number tensor (B,).
            horizon: Prediction horizon H (default: 1).
            pushforward_steps: Number of warmup rollout steps without gradients (default: 0).
            noise_std: Standard deviation of Gaussian noise injected (default: 0.0).

        Returns:
            Predicted physical fields of shape (B, H, C, Ny, Nx).
        """
        if horizon == 1 and pushforward_steps == 0:
            return self.forward_single_step(q_hist, re=re, sc=sc)
        else:
            return self.forward_rollout(
                q_hist,
                re=re,
                sc=sc,
                horizon=horizon,
                pushforward_steps=pushforward_steps,
                noise_std=noise_std,
            )


