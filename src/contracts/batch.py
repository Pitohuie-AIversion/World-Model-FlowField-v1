"""WorldModelBatch contract for physical world models.

Provides a unified, mapping-compatible batch contract encapsulating:
- State tensors (history, future)
- Multi-modal conditioning context (physical Re/Sc, geometry, boundaries)
- State specifications
- Spatiotemporal coordinates
- Provenance metadata
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import torch

from src.contracts.context import Context, PhysicalContext
from src.contracts.state_spec import StateSpec, SHEAR_FLOW_STATE_SPEC


@dataclass
class WorldModelBatch(Mapping):
    """Unified batch container for World Model training and inference.

    Implements the Mapping protocol for full backwards compatibility with
    legacy dictionary-style batch access (e.g. batch["history"], batch["re"]).

    Args:
        history: Historical physical field tensor of shape (B, L, C, Ny, Nx).
        future: Optional ground truth future field tensor of shape (B, H, C, Ny, Nx).
        context: Optional Context object containing physical (Re, Sc), boundary, geometry, etc.
        state_spec: StateSpec defining variable identities and channel count.
        coordinates: Optional spatiotemporal coordinates (e.g. dt, continuous time grid).
        boundary: Optional boundary specification (default: "periodic" for shear_flow).
        geometry: Optional spatial geometry or domain mask (reserved for obstacle extensions).
        metadata: Trajectory-level metadata (source_file, traj_idx, start_t, cluster_id).
    """

    history: torch.Tensor
    future: Optional[torch.Tensor] = None
    context: Optional[Context] = None
    state_spec: StateSpec = SHEAR_FLOW_STATE_SPEC
    coordinates: Optional[Dict[str, Any]] = None
    boundary: Optional[Any] = "periodic"
    geometry: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate history channel dimension against state_spec
        if isinstance(self.history, torch.Tensor) and self.history.ndim >= 3:
            # Assumes shape (B, L, C, ...) or (L, C, ...)
            c_dim = 2 if self.history.ndim >= 4 else 1
            if self.history.shape[c_dim] != self.state_spec.num_channels:
                raise ValueError(
                    f"history channel count mismatch: shape={tuple(self.history.shape)}, "
                    f"expected {self.state_spec.num_channels} channels from StateSpec."
                )

        if self.future is not None and isinstance(self.future, torch.Tensor) and self.future.ndim >= 3:
            c_dim = 2 if self.future.ndim >= 4 else 1
            if self.future.shape[c_dim] != self.state_spec.num_channels:
                raise ValueError(
                    f"future channel count mismatch: shape={tuple(self.future.shape)}, "
                    f"expected {self.state_spec.num_channels} channels from StateSpec."
                )

    @property
    def batch_size(self) -> int:
        """Return batch dimension size B."""
        return self.history.shape[0] if isinstance(self.history, torch.Tensor) else 0

    @classmethod
    def from_batch_dict(
        cls,
        batch_dict: Dict[str, Any],
        state_spec: StateSpec = SHEAR_FLOW_STATE_SPEC,
        boundary: Optional[Any] = "periodic",
        geometry: Optional[Any] = None,
    ) -> "WorldModelBatch":
        """Convert a standard DataLoader output dictionary into a WorldModelBatch.

        Args:
            batch_dict: Dictionary returned by PyTorch default_collate.
            state_spec: Target StateSpec contract (default: SHEAR_FLOW_STATE_SPEC).
            boundary: Optional boundary specification.
            geometry: Optional geometry specification.
        """
        history = batch_dict["history"]
        future = batch_dict.get("future")

        # Resolve Context from physical parameters
        re = batch_dict.get("re")
        sc = batch_dict.get("sc")
        context = None
        if re is not None or sc is not None:
            context = Context.from_re_sc(re=re, sc=sc)

        # Coordinate information
        coordinates = {}
        if "dt" in batch_dict:
            coordinates["dt"] = batch_dict["dt"]
        if "time" in batch_dict:
            coordinates["time"] = batch_dict["time"]

        # Trajectory metadata
        metadata = {}
        for meta_key in ("source_file", "traj_idx", "start_t", "cluster_id", "sim_idx"):
            if meta_key in batch_dict:
                metadata[meta_key] = batch_dict[meta_key]

        return cls(
            history=history,
            future=future,
            context=context,
            state_spec=state_spec,
            coordinates=coordinates,
            boundary=boundary,
            geometry=geometry,
            metadata=metadata,
        )

    def to(self, *args, **kwargs) -> "WorldModelBatch":
        """Move all contained tensors to target device and dtype."""
        new_hist = self.history.to(*args, **kwargs) if isinstance(self.history, torch.Tensor) else self.history
        new_fut = self.future.to(*args, **kwargs) if isinstance(self.future, torch.Tensor) else self.future
        new_ctx = self.context.to(*args, **kwargs) if self.context is not None else None

        new_coords = None
        if self.coordinates is not None:
            new_coords = {}
            for k, v in self.coordinates.items():
                new_coords[k] = v.to(*args, **kwargs) if isinstance(v, torch.Tensor) else v

        new_geom = self.geometry.to(*args, **kwargs) if hasattr(self.geometry, "to") else self.geometry

        return WorldModelBatch(
            history=new_hist,
            future=new_fut,
            context=new_ctx,
            state_spec=self.state_spec,
            coordinates=new_coords,
            boundary=self.boundary,
            geometry=new_geom,
            metadata=self.metadata,
        )

    # --- Mapping Protocol for 100% dictionary backwards-compatibility ---
    def __getitem__(self, key: str) -> Any:
        if key == "history":
            return self.history
        if key == "future":
            if self.future is not None:
                return self.future
            raise KeyError("future")
        if key == "context":
            return self.context
        if key == "state_spec":
            return self.state_spec
        if key == "coordinates":
            return self.coordinates
        if key == "boundary":
            return self.boundary
        if key == "geometry":
            return self.geometry
        if key == "metadata":
            return self.metadata

        # Accessors for physical context
        if self.context is not None:
            if key == "re":
                if self.context.re is not None:
                    return self.context.re
            elif key == "sc":
                if self.context.sc is not None:
                    return self.context.sc

        # Accessors for coordinates
        if self.coordinates is not None and key in self.coordinates:
            return self.coordinates[key]

        # Accessors for metadata
        if key in self.metadata:
            return self.metadata[key]

        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        keys = ["history"]
        if self.future is not None:
            keys.append("future")
        if self.context is not None:
            if self.context.re is not None:
                keys.append("re")
            if self.context.sc is not None:
                keys.append("sc")
        if self.coordinates:
            keys.extend(self.coordinates.keys())
        keys.extend(self.metadata.keys())
        return iter(keys)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            self[key]
            return True
        except KeyError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert WorldModelBatch into a flat dictionary matching legacy batch structure."""
        out: Dict[str, Any] = {"history": self.history}
        if self.future is not None:
            out["future"] = self.future
        if self.context is not None:
            if self.context.re is not None:
                out["re"] = self.context.re
            if self.context.sc is not None:
                out["sc"] = self.context.sc
        if self.coordinates:
            out.update(self.coordinates)
        out.update(self.metadata)
        return out


def collate_world_model_batch(
    batch_list: List[Dict[str, Any]],
    state_spec: StateSpec = SHEAR_FLOW_STATE_SPEC,
) -> WorldModelBatch:
    """Collate function for PyTorch DataLoader returning a WorldModelBatch directly.

    Args:
        batch_list: List of sample dictionaries from Dataset.__getitem__.
        state_spec: Target StateSpec contract.

    Returns:
        Structured WorldModelBatch instance.
    """
    from torch.utils.data.dataloader import default_collate
    collated_dict = default_collate(batch_list)
    return WorldModelBatch.from_batch_dict(collated_dict, state_spec=state_spec)
