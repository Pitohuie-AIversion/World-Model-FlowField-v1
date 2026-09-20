"""Field reconstruction and prediction losses for flow fields [u, v, p, s]."""

from typing import List, Optional
import torch
import torch.nn as nn


class FieldLoss(nn.Module):
    """Calculates channel-weighted loss across physical state variables.

    Args:
        loss_type: 'mse', 'relative_l2', or 'smooth_l1'. Default 'mse'.
        channel_weights: Weights for [u, v, p, s]. Default [1.0, 1.0, 1.0, 1.0].
        eps: Small constant to avoid division by zero in relative loss.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        channel_weights: Optional[List[float]] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.eps = eps
        weights = channel_weights or [1.0, 1.0, 1.0, 1.0]
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute field loss.

        Args:
            pred: Predicted fields, shape (..., C, H, W).
            target: Ground truth fields, shape (..., C, H, W).

        Returns:
            loss: Scalar loss tensor.
        """
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

        weights = self.weights.to(pred.device)
        if pred.ndim == 5 and weights.ndim == 4:
            weights = weights.unsqueeze(1)  # (1, 1, C, 1, 1)

        if self.loss_type == "mse":
            sq_diff = (pred - target) ** 2
            weighted_diff = sq_diff * weights
            return weighted_diff.mean()

        elif self.loss_type == "relative_l2":
            # Relative L2 per channel: ||pred - target||_2 / (||target||_2 + eps)
            diff_norm = torch.norm(pred - target, p=2, dim=(-2, -1))
            target_norm = torch.norm(target, p=2, dim=(-2, -1))
            rel_err = diff_norm / (target_norm + self.eps)
            # Weights applied to channel dim
            w = self.weights.to(pred.device).view(1, -1)
            if rel_err.ndim == 3:  # (B, T, C)
                w = w.unsqueeze(1)
            return (rel_err * w).mean()

        elif self.loss_type == "smooth_l1":
            diff = nn.functional.smooth_l1_loss(pred, target, reduction="none")
            return (diff * weights).mean()

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
