"""Multi-step autoregressive rollout loss with configurable time-step weighting."""

from typing import List, Optional, Union
import torch
import torch.nn as nn
from src.losses.field import FieldLoss


class RolloutLoss(nn.Module):
    """Calculates weighted multi-step future prediction loss over an H-step horizon.

    Args:
        field_loss: Base FieldLoss instance (or initialized with default).
        weights: Weight schedule for horizon steps h=1..H.
                 Can be 'uniform', 'discount', or a custom list of floats.
        discount_factor: Discount gamma if weights='discount' (w_h = gamma^(h-1)).
    """

    def __init__(
        self,
        field_loss: Optional[FieldLoss] = None,
        weight_mode: str = "uniform",
        discount_factor: float = 0.95,
        custom_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.field_loss = field_loss or FieldLoss(loss_type="mse")
        self.weight_mode = weight_mode
        self.discount_factor = discount_factor
        self.custom_weights = custom_weights

    def get_step_weights(self, horizon: int, device: torch.device) -> torch.Tensor:
        """Compute normalized weights w_h for h in [1, horizon]."""
        if self.custom_weights is not None:
            w = torch.tensor(self.custom_weights[:horizon], dtype=torch.float32, device=device)
            if len(w) < horizon:
                pad = torch.ones(horizon - len(w), device=device) * w[-1]
                w = torch.cat([w, pad])
        elif self.weight_mode == "uniform":
            w = torch.ones(horizon, dtype=torch.float32, device=device) / horizon
        elif self.weight_mode == "discount":
            steps = torch.arange(horizon, dtype=torch.float32, device=device)
            w = self.discount_factor**steps
            w = w / w.sum()
        else:
            raise ValueError(f"Unknown weight_mode: {self.weight_mode}")
        return w

    def forward(self, pred_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
        """Compute multi-step rollout loss.

        Args:
            pred_seq: Predicted rollout states of shape (B, H, C, Ny, Nx).
            target_seq: Target ground truth states of shape (B, H, C, Ny, Nx).

        Returns:
            loss: Scalar weighted multi-step loss.
        """
        assert pred_seq.ndim == 5, f"Expected 5D tensor (B, H, C, Ny, Nx), got {pred_seq.shape}"
        assert pred_seq.shape == target_seq.shape, "Shape mismatch"

        horizon = pred_seq.shape[1]
        w = self.get_step_weights(horizon, pred_seq.device)

        step_losses = []
        for h in range(horizon):
            loss_h = self.field_loss(pred_seq[:, h], target_seq[:, h])
            step_losses.append(loss_h * w[h])

        return torch.stack(step_losses).sum()
