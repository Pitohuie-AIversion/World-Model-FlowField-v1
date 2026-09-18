"""Channel-wise normalization and denormalization utilities for flow fields.

Supported representations:
    q = [u, v, p, s]
where:
    u: horizontal velocity
    v: vertical velocity
    p: pressure
    s: passive tracer scalar
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


class FieldNormalizer(nn.Module):
    """Channel-wise normalizer supporting Z-score (standardization) or Min-Max scaling.

    Args:
        mean: Optional tensor of shape (C,) or (1, C, 1, 1) containing channel means.
        std: Optional tensor of shape (C,) or (1, C, 1, 1) containing channel standard deviations.
        eps: Small epsilon to prevent division by zero.
    """

    def __init__(
        self,
        mean: Optional[Union[torch.Tensor, list]] = None,
        std: Optional[Union[torch.Tensor, list]] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        if mean is not None:
            mean = torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
            self.register_buffer("mean", mean)
        else:
            self.register_buffer("mean", None)

        if std is not None:
            std = torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
            self.register_buffer("std", std)
        else:
            self.register_buffer("std", None)

    def fit(self, data: torch.Tensor, dim: Tuple[int, ...] = (0, 2, 3)) -> "FieldNormalizer":
        """Compute mean and std across spatial dimensions and samples.

        Args:
            data: Tensor of shape (N, C, H, W) or (N, T, C, H, W).
            dim: Dimensions over which to compute statistics.
        """
        if data.ndim == 5:
            # (N, T, C, H, W) -> permute to (N, T, H, W, C) or compute along (0, 1, 3, 4)
            mean = data.mean(dim=(0, 1, 3, 4), keepdim=True).squeeze(1)  # (1, C, 1, 1)
            std = data.std(dim=(0, 1, 3, 4), keepdim=True).squeeze(1)
        elif data.ndim == 4:
            mean = data.mean(dim=(0, 2, 3), keepdim=True)
            std = data.std(dim=(0, 2, 3), keepdim=True)
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {data.shape}")

        std = torch.clamp(std, min=self.eps)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor x with shape (..., C, H, W)."""
        if self.mean is None or self.std is None:
            return x

        # Match broadcasting dimensions
        # If x is (B, T, C, H, W), mean/std need shape (1, 1, C, 1, 1)
        mean = self.mean
        std = self.std
        if x.ndim == 5 and mean.ndim == 4:
            mean = mean.unsqueeze(1)
            std = std.unsqueeze(1)

        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor x with shape (..., C, H, W)."""
        if self.mean is None or self.std is None:
            return x

        mean = self.mean
        std = self.std
        if x.ndim == 5 and mean.ndim == 4:
            mean = mean.unsqueeze(1)
            std = std.unsqueeze(1)

        return x * std + mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)

    def state_dict(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return {
            "mean": self.mean.cpu() if self.mean is not None else None,
            "std": self.std.cpu() if self.std is not None else None,
            "eps": self.eps,
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        mean = state_dict.get("mean")
        std = state_dict.get("std")
        self.eps = state_dict.get("eps", 1e-6)
        if mean is not None:
            self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        if std is not None:
            self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1))
