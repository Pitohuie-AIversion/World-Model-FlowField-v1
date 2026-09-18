"""Deterministic 2D spatial decoder for physical flow fields.

Upsamples latent representations Z in R^(H_z x W_z x C_z) by an 8x factor
back into physical state fields q_tilde = [u, v, p, s].
Supports circular padding and optional zero-mean gauge projection on the pressure channel.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from src.models.encoder import ResConvBlock2D
from src.utils.fft_derivatives import project_zero_mean_pressure


class UpBlock2D(nn.Module):
    """Upsampling block combining bilinear interpolation with residual conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.res_block = ResConvBlock2D(out_channels, out_channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2x spatial upsampling using bilinear interpolation
        # Periodic boundaries are preserved by interpolation
        h = nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        h = self.conv(h)
        return self.res_block(h)


class Decoder2D(nn.Module):
    """Spatial decoder upsampling latent states by 8x to reconstruct physical fields.

    Args:
        latent_channels: Latent dimension C_z (default: 64).
        out_channels: Number of physical output channels (default: 4 for [u, v, p, s]).
        base_channels: Base channel count of decoder (default: 32).
        channel_mult: Multipliers in reverse order (default: [4, 2, 1]).
        project_pressure: Whether to enforce zero spatial mean on output pressure channel.
    """

    def __init__(
        self,
        latent_channels: int = 64,
        out_channels: int = 4,
        base_channels: int = 32,
        channel_mult: Optional[List[int]] = None,
        project_pressure: bool = False,
    ):
        super().__init__()
        channel_mult = channel_mult or [4, 2, 1]
        self.project_pressure = project_pressure

        curr_ch = base_channels * channel_mult[0]
        self.in_conv = nn.Conv2d(
            latent_channels,
            curr_ch,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )

        stages = []
        for mult in channel_mult[1:]:
            out_ch = base_channels * mult
            stages.append(UpBlock2D(curr_ch, out_ch))
            curr_ch = out_ch

        # Final 2x upsampling to base_channels
        stages.append(UpBlock2D(curr_ch, base_channels))
        self.up_stages = nn.Sequential(*stages)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Latent tensor of shape (B, latent_channels, H_z, W_z)
               or (B, T, latent_channels, H_z, W_z).

        Returns:
            q: Physical fields of shape (B, out_channels, H, W)
               or (B, T, out_channels, H, W).
        """
        orig_ndim = z.ndim
        if orig_ndim == 5:
            b, t, c_z, h_z, w_z = z.shape
            z = z.view(b * t, c_z, h_z, w_z)

        h = self.in_conv(z)
        h = self.up_stages(h)
        q = self.out_conv(h)

        if self.project_pressure:
            # Channel 2 is pressure p in [u, v, p, s]
            p = q[:, 2:3, :, :]
            q[:, 2:3, :, :] = project_zero_mean_pressure(p)

        if orig_ndim == 5:
            _, c, h_out, w_out = q.shape
            q = q.view(b, t, c, h_out, w_out)

        return q
