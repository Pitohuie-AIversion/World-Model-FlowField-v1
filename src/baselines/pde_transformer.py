"""PDE-Transformer baseline for spatio-temporal physical flow field forecasting.

Operates directly on physical fields q in R^(B, L, C, Ny, Nx) by patchifying
the 2D domain, applying factorized spatio-temporal self-attention blocks,
and projecting back to the physical mesh. Serves as the standard direct-space
Transformer baseline for fair comparison with the Latent World Model.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.models.direct_transformer import DirectSTTransformer


class PDETransformer(DirectSTTransformer):
    """PDE-Transformer baseline model operating directly on physical patches.

    Args:
        in_channels: Number of physical field channels (default: 4 for [u, v, p, s]).
        patch_size: 2D patch dimensions (default: (8, 8)).
        embed_dim: Hidden representation dimension D (default: 256).
        cond_dim: Physical condition dimension for Re/Sc (default: 128).
        depth: Number of Transformer blocks (default: 6).
        num_heads: Number of attention heads (default: 8).
        history_length: Length of history window L (default: 4).
        prediction_mode: 'direct' or 'residual' (default: 'direct').
    """

    def __init__(
        self,
        in_channels: int = 4,
        patch_size: Tuple[int, int] = (8, 8),
        embed_dim: int = 256,
        cond_dim: int = 128,
        depth: int = 6,
        num_heads: int = 8,
        history_length: int = 4,
        prediction_mode: str = "direct",
    ):
        super().__init__(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            cond_dim=cond_dim,
            depth=depth,
            num_heads=num_heads,
            history_length=history_length,
            prediction_mode=prediction_mode,
        )
