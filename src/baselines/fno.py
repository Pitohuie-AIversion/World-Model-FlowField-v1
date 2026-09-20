"""Fourier Neural Operator (FNO-2D) baseline for periodic flow field dynamics."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.fft


class SpectralConv2d(nn.Module):
    """2D Fourier layer with periodic boundary handling.

    Weights are stored as real float32 tensors (real + imag stacked on last dim)
    to be compatible with AMP GradScaler (which does not support cfloat gradients).
    They are reassembled into complex tensors via view_as_complex during forward.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Wavenumbers along Ny
        self.modes2 = modes2  # Wavenumbers along Nx

        scale = 1.0 / (in_channels * out_channels)
        # Store as real(..., 2) to avoid AMP cfloat GradScaler incompatibility
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        batchsize = x.shape[0]
        # Compute 2D Fourier coefficients
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Reassemble real(...,2) weights as complex tensors (AMP-compatible trick)
        w1 = torch.view_as_complex(self.weights1.float().contiguous())  # (in, out, m1, m2)
        w2 = torch.view_as_complex(self.weights2.float().contiguous())  # (in, out, m1, m2)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], w1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], w2
        )

        # Return to spatial domain
        x_out = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x_out.to(dtype=orig_dtype)


class FNO2D(nn.Module):
    """Fourier Neural Operator (FNO) mapping historical state frames to the next frame.

    Args:
        in_channels: Input channels (L * C_state, e.g. 4 * 4 = 16).
        out_channels: Output channels (C_state, e.g. 4 for [u, v, p, s]).
        modes1: Truncated Fourier modes in y dimension (default: 16).
        modes2: Truncated Fourier modes in x dimension (default: 16).
        width: Hidden channel dimension (default: 64).
        num_layers: Number of Fourier layers (default: 4).
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 4,
        modes1: int = 16,
        modes2: int = 16,
        width: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()
        self.width = width
        self.num_layers = num_layers

        # Lifting layer
        self.fc0 = nn.Conv2d(in_channels, width, kernel_size=1)

        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(num_layers)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1) for _ in range(num_layers)
        ])
        self.act = nn.GELU()

        # Projection layers
        self.fc1 = nn.Conv2d(width, 128, kernel_size=1)
        self.fc2 = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, L * C, Ny, Nx) or (B, L, C, Ny, Nx).

        Returns:
            Next state prediction of shape (B, 1, C, Ny, Nx) or (B, C, Ny, Nx).
        """
        is_5d = x.ndim == 5
        if is_5d:
            b, l, c, ny, nx = x.shape
            x = x.view(b, l * c, ny, nx)

        h = self.fc0(x)
        for spectral, w in zip(self.spectral_layers, self.w_layers):
            h1 = spectral(h)
            h2 = w(h)
            h = self.act(h1 + h2)

        out = self.fc2(self.act(self.fc1(h)))

        if is_5d:
            out = out.unsqueeze(1)  # (B, 1, C, Ny, Nx)

        return out
