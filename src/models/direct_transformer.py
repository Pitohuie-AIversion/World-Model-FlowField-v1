"""Direct Spatio-Temporal Transformer baseline (Baseline 4).

Operates directly on physical state patches q in R^(B, L, C, Ny, Nx) without
latent compression, allowing a 1:1 scientific comparison with Latent ST Transformer.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.models.latent_transformer import FactorizedSTBlock
from src.models.conditioning import AdaLN, PhysicalConditionEmbedding


class DirectSTTransformer(nn.Module):
    """Direct Spatio-Temporal Transformer operating on physical grid patches.

    Args:
        in_channels: Number of physical channels (default: 4 for [u, v, p, s]).
        patch_size: (patch_y, patch_x) downsampling stride (default: (8, 8)).
        embed_dim: Hidden dimension D (default: 256).
        cond_dim: Condition embedding dimension (default: 128).
        depth: Transformer depth (default: 6).
        num_heads: Attention heads (default: 8).
        history_length: Historical sequence length L (default: 4).
        prediction_mode: 'direct' or 'residual'. Default 'direct'.
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
        super().__init__()
        assert prediction_mode in ("direct", "residual")
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.history_length = history_length
        self.prediction_mode = prediction_mode

        py, px = patch_size
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=(py, px), stride=(py, px))

        # Temporal positional embeddings (1, L, 1, D)
        self.temp_pos_embed = nn.Parameter(torch.zeros(1, history_length, 1, embed_dim))

        # Condition encoder
        self.cond_embed = PhysicalConditionEmbedding(embed_dim=cond_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            FactorizedSTBlock(embed_dim, cond_dim, num_heads=num_heads) for _ in range(depth)
        ])

        self.final_norm = AdaLN(embed_dim, cond_dim)
        # Unpatch projection: project D -> in_channels * py * px
        self.out_proj = nn.Linear(embed_dim, in_channels * py * px)

    def forward(
        self,
        q_hist: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        sc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            q_hist: Physical state history of shape (B, L, C, Ny, Nx).
            re: Optional Reynolds number tensor (B,).
            sc: Optional Schmidt number tensor (B,).

        Returns:
            q_next: Predicted next state of shape (B, 1, C, Ny, Nx).
        """
        b, l, c, ny, nx = q_hist.shape
        py, px = self.patch_size
        n_y, n_x = ny // py, nx // px
        n_spatial = n_y * n_x

        # 1. Patchify each time step: (B * L, C, Ny, Nx) -> (B * L, D, n_y, n_x)
        h = self.patch_embed(q_hist.view(b * l, c, ny, nx))
        h = h.flatten(2).transpose(1, 2)  # (B * L, n_spatial, D)
        x = h.view(b, l, n_spatial, self.embed_dim) + self.temp_pos_embed[:, :l]

        # Condition
        cond = None
        if re is not None and sc is not None:
            cond = self.cond_embed(re, sc)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, cond=cond, b=b, l=l, n=n_spatial)

        # Final prediction on last time step
        x_last = x[:, -1:]  # (B, 1, n_spatial, D)
        x_last = self.final_norm(x_last, cond)
        pred_patches = self.out_proj(x_last)  # (B, 1, n_spatial, C * py * px)

        # Unpatchify back to (B, 1, C, Ny, Nx)
        pred_patches = pred_patches.view(b, 1, n_y, n_x, c, py, px)
        # Permute to (B, 1, c, n_y, py, n_x, px)
        pred_fields = pred_patches.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        pred_fields = pred_fields.view(b, 1, c, ny, nx)

        if self.prediction_mode == "residual":
            return q_hist[:, -1:] + pred_fields
        else:
            return pred_fields
