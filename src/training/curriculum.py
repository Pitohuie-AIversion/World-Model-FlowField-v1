"""Curriculum rollout scheduling and Pushforward autoregressive training algorithms.

References:
    Brandstetter et al., 'Message Passing Neural PDE Solvers', ICLR 2022.
    Takamoto et al., 'PDE-Arena: Benchmarking Neural PDE Solvers', NeurIPS 2022.
    FluidWorld: Autoregressive Latent World Models for Unsteady Fluid Flow, 2024.
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn


@dataclass
class CurriculumConfig:
    """Configuration for progressive curriculum rollout and pushforward training.

    Attributes:
        enabled: Whether curriculum rollout scheduling is active. If False, horizon is fixed to target_horizon.
        start_horizon: Starting training horizon H at epoch 1 (default: 2).
        target_horizon: Maximum/target training horizon H (default: 16).
        step_epochs: Number of epochs to train before advancing to next horizon step (default: 3).
        schedule: Progression strategy:
            - 'doubling': H doubles every step_epochs (e.g. 2 -> 4 -> 8 -> 16).
            - 'linear': H increases by start_horizon every step_epochs.
            - 'fixed': H remains constant at target_horizon.
        pushforward_steps: Number of unrolled autoregressive steps executed without gradient (stop-gradient)
            before beginning the BPTT rollout. Mitigates training-inference distribution shift (default: 0).
        pushforward_noise_std: Standard deviation of Gaussian perturbation injected into latent history buffer
            during pushforward unrolling to promote contractive dynamics (default: 0.0).
        pushforward_mode:
            - 'future': Unrolls pushforward_steps forward in time, target ground truth is sliced from q_future[K:K+H].
            - 'history': Unrolls pushforward_steps inside the history window, starting from seed q_hist[:L-K],
              target ground truth remains q_future[:H].
    """
    enabled: bool = False
    start_horizon: int = 2
    target_horizon: int = 16
    step_epochs: int = 3
    schedule: str = "doubling"
    pushforward_steps: int = 0
    pushforward_noise_std: float = 0.0
    pushforward_mode: str = "future"

    def __post_init__(self):
        if self.start_horizon < 1:
            raise ValueError(f"start_horizon must be >= 1, got {self.start_horizon}")
        if self.target_horizon < self.start_horizon:
            raise ValueError(
                f"target_horizon ({self.target_horizon}) cannot be less than start_horizon ({self.start_horizon})"
            )
        if self.step_epochs < 1:
            raise ValueError(f"step_epochs must be >= 1, got {self.step_epochs}")
        if self.schedule not in ("doubling", "linear", "fixed"):
            raise ValueError(f"Unknown schedule: {self.schedule}. Must be one of ('doubling', 'linear', 'fixed')")
        if self.pushforward_steps < 0:
            raise ValueError(f"pushforward_steps must be >= 0, got {self.pushforward_steps}")
        if self.pushforward_noise_std < 0.0:
            raise ValueError(f"pushforward_noise_std must be >= 0.0, got {self.pushforward_noise_std}")
        if self.pushforward_mode not in ("future", "history"):
            raise ValueError(f"Unknown pushforward_mode: {self.pushforward_mode}. Must be 'future' or 'history'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurriculumConfig":
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


class CurriculumRolloutScheduler:
    """Manages epoch-dependent training horizon and pushforward parameters.

    Provides deterministic, epoch-based progression without mutable internal epoch counters,
    allowing transparent integration with PyTorch training loops, checkpointing, and DDP.
    """

    def __init__(self, config: Optional[CurriculumConfig] = None):
        self.config = config or CurriculumConfig()

    def get_horizon(self, epoch: int) -> int:
        """Compute the training prediction horizon H for a given 1-indexed epoch.

        Args:
            epoch: 1-indexed epoch number (>= 1).

        Returns:
            int: Training horizon H for the current epoch, clamped to target_horizon.
        """
        if epoch < 1:
            epoch = 1

        if not self.config.enabled or self.config.schedule == "fixed":
            return self.config.target_horizon

        step_idx = (epoch - 1) // self.config.step_epochs

        if self.config.schedule == "doubling":
            current_h = self.config.start_horizon * (2 ** step_idx)
        elif self.config.schedule == "linear":
            current_h = self.config.start_horizon + step_idx * self.config.start_horizon
        else:
            current_h = self.config.target_horizon

        return min(self.config.target_horizon, current_h)

    def get_pushforward_steps(self, epoch: int) -> int:
        """Get the number of stop-gradient warmup steps for the current epoch.

        When curriculum is enabled, pushforward steps can scale proportionally
        with current horizon if pushforward_steps is configured > 0.
        """
        if self.config.pushforward_steps <= 0:
            return 0

        # If curriculum is disabled, use configured static steps
        if not self.config.enabled:
            return self.config.pushforward_steps

        # Proportional scale: scale pushforward steps with horizon ratio
        current_h = self.get_horizon(epoch)
        ratio = current_h / self.config.target_horizon
        effective_steps = max(1, int(round(self.config.pushforward_steps * ratio)))
        return min(self.config.pushforward_steps, effective_steps)

    def get_noise_std(self, epoch: int) -> float:
        """Get the perturbation noise scale for the current epoch."""
        return float(self.config.pushforward_noise_std)

    def get_epoch_plan(self, max_epochs: int) -> List[Dict[str, Any]]:
        """Generate a complete schedule plan across all training epochs for inspection and logging."""
        plan = []
        for ep in range(1, max_epochs + 1):
            plan.append({
                "epoch": ep,
                "horizon": self.get_horizon(ep),
                "pushforward_steps": self.get_pushforward_steps(ep),
                "noise_std": self.get_noise_std(ep),
            })
        return plan

    def state_dict(self) -> Dict[str, Any]:
        return {"config": self.config.to_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "config" in state:
            self.config = CurriculumConfig.from_dict(state["config"])


@torch.no_grad()
def apply_pushforward_warmup(
    history_buffer: Any,
    step_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    steps: int,
    condition: Optional[torch.Tensor] = None,
    noise_std: float = 0.0,
) -> None:
    """Executes K stop-gradient autoregressive rollout steps on a HistoryBuffer.

    Pushes the history buffer into the model's own predictive distribution without
    building a computation graph, mitigating covariate shift during subsequent BPTT.

    Args:
        history_buffer: An instance of HistoryBuffer holding state (B, L, *dims).
        step_fn: Function mapping (hist_tensor, condition) -> next_predicted_state.
        steps: Number of pushforward warmup steps to advance (>= 1).
        condition: Optional physical condition embedding tensor.
        noise_std: Gaussian noise standard deviation added to intermediate predictions.
    """
    if steps <= 0:
        return

    for _ in range(steps):
        pred_next = step_fn(history_buffer.current, condition)
        if noise_std > 0.0:
            noise = torch.randn_like(pred_next) * noise_std
            pred_next = pred_next + noise
        history_buffer.push(pred_next)
