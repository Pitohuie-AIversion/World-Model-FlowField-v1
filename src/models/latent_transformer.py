"""Factorized Spatio-Temporal Latent Transformer for Physical World Modeling.

Operates on latent state sequences Z_{t-L+1:t} in R^(B, L, C_z, H_z, W_z).
Alternates between Spatial Self-Attention and Temporal Self-Attention,
injecting physical parameters (Re, Sc) via AdaLN modulation.
Supports both Direct and Residual latent prediction heads.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.models.conditioning import AdaLN, PhysicalConditionEmbedding


class SpatialAttention(nn.Module):
    """Multi-head self-attention operating across spatial tokens for each time step."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B_total, N_spatial, D)
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class TemporalAttention(nn.Module):
    """Multi-head self-attention operating across historical time steps for each spatial token."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B_total, L_time, D)
        b, l, d = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, l, d)
        return self.proj(out)


class FactorizedSTBlock(nn.Module):
    """Factorized Spatio-Temporal Transformer block with AdaLN condition injection."""

    def __init__(self, dim: int, cond_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_spatial = AdaLN(dim, cond_dim)
        self.spatial_attn = SpatialAttention(dim, num_heads)

        self.norm_temporal = AdaLN(dim, cond_dim)
        self.temporal_attn = TemporalAttention(dim, num_heads)

        self.norm_mlp = AdaLN(dim, cond_dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        b: int = 1,
        l: int = 4,
        n: int = 512,
    ) -> torch.Tensor:
        """Args:

        x: Flattened tokens of shape (B, L, N, D).
        cond: Condition embedding of shape (B, cond_dim).
        """
        # 1. Spatial Attention over N tokens at each (b, t)
        h = self.norm_spatial(x, cond)  # (B, L, N, D)
        h_spatial = h.view(b * l, n, -1)
        res_spatial = self.spatial_attn(h_spatial).view(b, l, n, -1)
        x = x + res_spatial

        # 2. Temporal Attention over L tokens at each (b, s)
        h = self.norm_temporal(x, cond)  # (B, L, N, D)
        h_temporal = h.permute(0, 2, 1, 3).reshape(b * n, l, -1)  # (B*N, L, D)
        res_temporal = self.temporal_attn(h_temporal).view(b, n, l, -1).permute(0, 2, 1, 3)
        x = x + res_temporal

        # 3. Feedforward MLP
        h = self.norm_mlp(x, cond)
        x = x + self.mlp(h)
        return x


class LatentSTTransformer(nn.Module):
    """Factorized Spatio-Temporal Transformer for Latent State Dynamics.

    Args:
        latent_channels: C_z channels from encoder (default: 64).
        embed_dim: Transformer hidden dimension D (default: 256).
        cond_dim: Physical condition dimension (default: 128).
        depth: Number of factorized ST blocks (default: 6).
        num_heads: Attention heads (default: 8).
        history_length: Historical window L (default: 4).
        prediction_mode: 'direct' or 'residual'. Default 'direct'.
    """

    def __init__(
        self,
        latent_channels: int = 64,
        embed_dim: int = 256,
        cond_dim: int = 128,
        depth: int = 6,
        num_heads: int = 8,
        history_length: int = 4,
        prediction_mode: str = "direct",
    ):
        super().__init__()
        assert prediction_mode in ("direct", "residual")
        self.latent_channels = latent_channels
        self.embed_dim = embed_dim
        self.history_length = history_length
        self.prediction_mode = prediction_mode

        # Input patch/channel projection
        self.in_proj = nn.Linear(latent_channels, embed_dim)

        # Learnable temporal positional embeddings (L,)
        self.temp_pos_embed = nn.Parameter(torch.zeros(1, history_length, 1, embed_dim))

        # Condition encoder
        self.cond_embed = PhysicalConditionEmbedding(embed_dim=cond_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            FactorizedSTBlock(embed_dim, cond_dim, num_heads=num_heads) for _ in range(depth)
        ])

        self.final_norm = AdaLN(embed_dim, cond_dim)
        self.out_proj = nn.Linear(embed_dim, latent_channels)

    def forward(
        self,
        z_hist: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        sc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass predicting the next latent state Z_{t+1}.

        Args:
            z_hist: Latent sequence of shape (B, L, C_z, H_z, W_z).
            re: Optional Reynolds number tensor (B,).
            sc: Optional Schmidt number tensor (B,).

        Returns:
            z_next: Next latent state prediction of shape (B, 1, C_z, H_z, W_z).
        """
        b, l, c_z, h_z, w_z = z_hist.shape
        n_spatial = h_z * w_z

        # (B, L, C_z, H_z, W_z) -> (B, L, N, C_z)
        x = z_hist.permute(0, 1, 3, 4, 2).reshape(b, l, n_spatial, c_z)
        x = self.in_proj(x) + self.temp_pos_embed[:, :l]

        # Condition embedding
        cond = None
        if re is not None and sc is not None:
            cond = self.cond_embed(re, sc)

        # Factorized Transformer blocks
        for block in self.blocks:
            x = block(x, cond=cond, b=b, l=l, n=n_spatial)

        # Aggregate or take the latest time-slice for next step prediction
        # Use final time step representation
        x_last = x[:, -1:]  # (B, 1, N, D)
        x_last = self.final_norm(x_last, cond)
        delta_or_pred = self.out_proj(x_last)  # (B, 1, N, C_z)

        # Reshape to (B, 1, C_z, H_z, W_z)
        pred_latent = delta_or_pred.view(b, 1, h_z, w_z, c_z).permute(0, 1, 4, 2, 3)

        if self.prediction_mode == "residual":
            z_last = z_hist[:, -1:]  # (B, 1, C_z, H_z, W_z)
            return z_last + pred_latent
        else:
            return pred_latent
