"""Physical state specification contract for physical world models.

Defines the semantic identity, channel layout, and spatial dimensionality
of variables comprising the physical world state.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch


@dataclass(frozen=True)
class StateSpec:
    """Immutable contract describing physical state variables and channel dimensionality.

    Args:
        variables: Sequence of state variable names (e.g. ("u", "v", "p", "s")).
        num_channels: Total number of physical feature channels.
        spatial_dim: Number of spatial dimensions (default: 2 for 2D field models).
    """

    variables: Tuple[str, ...]
    num_channels: int
    spatial_dim: int = 2

    def __init__(
        self,
        variables: Sequence[str],
        num_channels: Optional[int] = None,
        spatial_dim: int = 2,
    ):
        vars_tuple = tuple(str(v).strip() for v in variables)
        if len(vars_tuple) == 0:
            raise ValueError("StateSpec variables sequence must not be empty.")

        for v in vars_tuple:
            if not v:
                raise ValueError("StateSpec variable names must be non-empty strings.")

        if len(set(vars_tuple)) != len(vars_tuple):
            raise ValueError(f"StateSpec variables contain duplicate names: {vars_tuple}")

        resolved_channels = len(vars_tuple) if num_channels is None else int(num_channels)
        if resolved_channels != len(vars_tuple):
            raise ValueError(
                f"StateSpec channel count mismatch: num_channels={resolved_channels}, "
                f"but len(variables)={len(vars_tuple)}"
            )

        if spatial_dim <= 0:
            raise ValueError(f"StateSpec spatial_dim must be positive (>0), got {spatial_dim}")

        object.__setattr__(self, "variables", vars_tuple)
        object.__setattr__(self, "num_channels", resolved_channels)
        object.__setattr__(self, "spatial_dim", int(spatial_dim))

    def channel_index(self, var_name: str) -> int:
        """Return 0-indexed channel offset for a variable name."""
        try:
            return self.variables.index(var_name)
        except ValueError:
            raise KeyError(
                f"Variable '{var_name}' not defined in StateSpec variables {self.variables}"
            )

    def has_variable(self, var_name: str) -> bool:
        """Check if a variable name exists in this specification."""
        return var_name in self.variables

    def validate_tensor(self, tensor: torch.Tensor, channel_dim: int = 1) -> None:
        """Assert that a state tensor's channel dimension matches num_channels.

        Args:
            tensor: PyTorch tensor to validate.
            channel_dim: Dimension index of channels (e.g. 1 in (B, C, H, W) or 2 in (B, L, C, H, W)).

        Raises:
            ValueError: If tensor shape does not match num_channels.
        """
        actual_channels = tensor.shape[channel_dim]
        if actual_channels != self.num_channels:
            raise ValueError(
                f"Tensor channel mismatch: dim {channel_dim} has size {actual_channels}, "
                f"expected num_channels={self.num_channels} for variables {self.variables}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize StateSpec to a standard dictionary."""
        return {
            "variables": list(self.variables),
            "num_channels": self.num_channels,
            "spatial_dim": self.spatial_dim,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateSpec":
        """Reconstruct StateSpec from a serialized dictionary."""
        return cls(
            variables=data["variables"],
            num_channels=data.get("num_channels"),
            spatial_dim=data.get("spatial_dim", 2),
        )


# Canonical specification for 2D incompressible shear flow with passive tracer
SHEAR_FLOW_STATE_SPEC = StateSpec(
    variables=("u", "v", "p", "s"),
    num_channels=4,
    spatial_dim=2,
)
