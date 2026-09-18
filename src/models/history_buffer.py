"""History buffer mechanism for autoregressive rollout in latent or physical state spaces.

Ensures:
1. Fixed historical window length L.
2. Popping oldest state and appending newest prediction without reading ground truth.
3. Completely latent-space rollout avoiding wasteful decode-encode roundtrips.
"""

from typing import Callable, List, Optional, Tuple
import torch


class HistoryBuffer:
    """Maintains a rolling historical sequence of states for autoregressive rollout.

    Args:
        history_length: Length of history window L (default: 4).
    """

    def __init__(self, history_length: int = 4):
        self.history_length = history_length
        self._buffer: Optional[torch.Tensor] = None

    def reset(self, initial_history: torch.Tensor) -> "HistoryBuffer":
        """Initialize buffer with past L states.

        Args:
            initial_history: Tensor of shape (B, L, *dims).
        """
        assert initial_history.shape[1] == self.history_length, (
            f"Expected sequence length {self.history_length}, got {initial_history.shape[1]}"
        )
        self._buffer = initial_history.clone()
        return self

    @property
    def current(self) -> torch.Tensor:
        """Get the current history window of shape (B, L, *dims)."""
        if self._buffer is None:
            raise RuntimeError("HistoryBuffer is not initialized. Call reset() first.")
        return self._buffer

    def push(self, next_state: torch.Tensor) -> torch.Tensor:
        """Evict oldest time step and append next_state at the end.

        Args:
            next_state: Tensor of shape (B, *dims) or (B, 1, *dims).

        Returns:
            Updated buffer of shape (B, L, *dims).
        """
        if self._buffer is None:
            raise RuntimeError("HistoryBuffer is not initialized. Call reset() first.")

        if next_state.ndim == self._buffer.ndim - 1:
            next_state = next_state.unsqueeze(1)

        assert next_state.shape[1] == 1, f"Expected single step to push, got shape {next_state.shape}"

        # Drop oldest (index 0) and append next_state
        self._buffer = torch.cat([self._buffer[:, 1:], next_state], dim=1)
        return self._buffer

    def rollout(
        self,
        step_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        steps: int,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressively roll out for H steps without reading future ground truth.

        Args:
            step_fn: Function mapping (history_tensor, condition) -> next_predicted_state (B, 1, *dims)
                     or (B, *dims).
            steps: Number of future steps H to roll out.
            condition: Optional physical condition embedding tensor.

        Returns:
            trajectory: Predicted future states of shape (B, steps, *dims).
        """
        predictions: List[torch.Tensor] = []
        for _ in range(steps):
            pred_next = step_fn(self.current, condition)
            if pred_next.ndim == self.current.ndim - 1:
                pred_next = pred_next.unsqueeze(1)
            predictions.append(pred_next)
            self.push(pred_next)

        return torch.cat(predictions, dim=1)
