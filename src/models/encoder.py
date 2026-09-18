"""Deterministic 2D spatial encoder for physical flow fields.

Downsamples physical states q = [u, v, p, s] (4 channels)
by an 8x spatial reduction factor into latent representations Z in R^(H_z x W_z x C_z).
Supports circular padding to respect periodic boundary conditions.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class ResConvBlock2D(nn.Module):
    """Residual convolutional block with circular padding for periodic boundaries."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            padding_mode="circular",
        )
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act2 = nn.SiLU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act2(h + res)


class Encoder2D(nn.Module):
    """Spatial encoder downsampling 4-channel physical fields by 8x.

    Args:
        in_channels: Number of physical input channels (default: 4 for [u, v, p, s]).
        latent_channels: Latent channel dimension C_z (default: 64).
        channel_mult: Multipliers for intermediate channel dimensions (default: [1, 2, 4]).
    """

    def __init__(
        self,
        in_channels: int = 4,
        latent_channels: int = 64,
        base_channels: int = 32,
        channel_mult: Optional[List[int]] = None,
    ):
        super().__init__()
        channel_mult = channel_mult or [1, 2, 4]  # 3 stages -> 8x downsampling

        self.in_conv = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )

        stages = []
        curr_ch = base_channels
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            # Downsampling stage (stride=2 in first block)
            stages.append(ResConvBlock2D(curr_ch, out_ch, stride=2))
            stages.append(ResConvBlock2D(out_ch, out_ch, stride=1))
            curr_ch = out_ch

        self.down_stages = nn.Sequential(*stages)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, curr_ch), curr_ch),
            nn.SiLU(),
            nn.Conv2d(curr_ch, latent_channels, kernel_size=3, padding=1, padding_mode="circular"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, in_channels, H, W) or (B, T, in_channels, H, W).

        Returns:
            z: Latent tensor of shape (B, latent_channels, H_z, W_z)
               or (B, T, latent_channels, H_z, W_z).
        """
        orig_ndim = x.ndim
        if orig_ndim == 5:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)

        h = self.in_conv(x)
        h = self.down_stages(h)
        z = self.out_conv(h)

        if orig_ndim == 5:
            _, c_z, h_z, w_z = z.shape
            z = z.view(b, t, c_z, h_z, w_z)

        return z
