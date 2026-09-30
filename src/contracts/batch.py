"""WorldModelBatch contract for physical world models.

Provides a unified, mapping-compatible batch contract encapsulating:
- State tensors (history, future)
- Multi-modal conditioning context (physical Re/Sc, geometry, boundaries)
- State specifications
- Spatiotemporal coordinates
- Provenance metadata

Architectural Principles (Phase 1.1 Hardening):
1. Single Source of Truth: Geometry and Boundary conditions belong exclusively
   to Context. WorldModelBatch accesses them via properties delegating to context.
2. Generic Contracts: WorldModelBatch requires explicit StateSpec and does not
   silently default to shear_flow or 'periodic' boundaries. Domain-specific
   adapters (e.g. shear_flow_batch_adapter) provide dataset bindings.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import torch

from src.contracts.context import Context, PhysicalContext, resolve_context
from src.contracts.state_spec import StateSpec, SHEAR_FLOW_STATE_SPEC


@dataclass
class WorldModelBatch(Mapping):
    """Unified batch container for World Model training and inference.

    Implements the Mapping protocol for backwards compatibility with
    legacy dictionary-style batch access (e.g. batch["history"], batch["re"]).

    Single Source of Truth:
    Boundary and geometry specifications belong strictly to Context.
    WorldModelBatch exposes them as properties delegating to self.context.

    Generic Contract:
    Requires explicit state_spec without assuming a default domain schema.

    Args:
        history: Historical physical field tensor of shape (B, L, C, Ny, Nx).
        state_spec: StateSpec defining variable identities and channel count.
        future: Optional ground truth future field tensor of shape (B, H, C, Ny, Nx).
        context: Optional Context object containing physical (Re, Sc), boundary, geometry, etc.
        coordinates: Optional spatiotemporal coordinates (e.g. dt, continuous time grid).
        metadata: Trajectory-level metadata (source_file, traj_idx, start_t, cluster_id).
    """

    history: torch.Tensor
    state_spec: StateSpec
    future: Optional[torch.Tensor] = None
    context: Optional[Context] = None
    coordinates: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.state_spec, StateSpec):
            raise TypeError(
                f"state_spec must be an instance of StateSpec, got {type(self.state_spec)}"
            )

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

    @property
    def boundary(self) -> Optional[Any]:
        """Boundary specification delegating to single source of truth in Context."""
        return self.context.boundary if self.context is not None else None

    @property
    def geometry(self) -> Optional[Any]:
        """Geometry specification delegating to single source of truth in Context."""
        return self.context.geometry if self.context is not None else None

    @classmethod
    def from_batch_dict(
        cls,
        batch_dict: Dict[str, Any],
        state_spec: StateSpec,
        boundary: Optional[Any] = None,
        geometry: Optional[Any] = None,
        context: Optional[Union[Context, Dict[str, Any]]] = None,
    ) -> "WorldModelBatch":
        """Convert a standard DataLoader output dictionary into a WorldModelBatch.

        Args:
            batch_dict: Dictionary returned by PyTorch default_collate.
            state_spec: Target StateSpec contract (explicitly required).
            boundary: Optional boundary specification to bind into Context.
            geometry: Optional geometry specification to bind into Context.
            context: Optional explicit Context override or dict.
        """
        history = batch_dict["history"]
        future = batch_dict.get("future")

        # Resolve Context: combine explicit context, batch_dict['context'], and legacy re/sc
        base_ctx = context if context is not None else batch_dict.get("context")
        resolved_ctx = resolve_context(base_ctx) if base_ctx is not None else None

        re = batch_dict.get("re")
        sc = batch_dict.get("sc")
        if re is not None or sc is not None:
            resolved_ctx = resolve_context(context=resolved_ctx, re=re, sc=sc)

        # Boundary and geometry handling: strictly embed into Context
        b_val = boundary if boundary is not None else batch_dict.get("boundary")
        g_val = geometry if geometry is not None else batch_dict.get("geometry")

        if b_val is not None or g_val is not None:
            if resolved_ctx is None:
                resolved_ctx = Context(boundary=b_val, geometry=g_val)
            else:
                if b_val is not None:
                    if resolved_ctx.boundary is not None and resolved_ctx.boundary != b_val:
                        raise ValueError(
                            f"Boundary conflict: Context has boundary='{resolved_ctx.boundary}', "
                            f"but divergent boundary='{b_val}' was provided (Fail-Closed)."
                        )
                    resolved_ctx.boundary = b_val
                if g_val is not None:
                    if resolved_ctx.geometry is not None and resolved_ctx.geometry != g_val:
                        raise ValueError(
                            f"Geometry conflict: Context has geometry='{resolved_ctx.geometry}', "
                            f"but divergent geometry='{g_val}' was provided (Fail-Closed)."
                        )
                    resolved_ctx.geometry = g_val

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
            state_spec=state_spec,
            future=future,
            context=resolved_ctx,
            coordinates=coordinates,
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

        return WorldModelBatch(
            history=new_hist,
            state_spec=self.state_spec,
            future=new_fut,
            context=new_ctx,
            coordinates=new_coords,
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
        if key == "metadata":
            return self.metadata
        if key == "boundary":
            if self.boundary is not None:
                return self.boundary
            raise KeyError("boundary")
        if key == "geometry":
            if self.geometry is not None:
                return self.geometry
            raise KeyError("geometry")

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
            if self.context.boundary is not None:
                keys.append("boundary")
            if self.context.geometry is not None:
                keys.append("geometry")
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
            if self.context.boundary is not None:
                out["boundary"] = self.context.boundary
            if self.context.geometry is not None:
                out["geometry"] = self.context.geometry
        if self.coordinates:
            out.update(self.coordinates)
        out.update(self.metadata)
        return out


def collate_world_model_batch(
    batch_list: List[Dict[str, Any]],
    state_spec: StateSpec,
    boundary: Optional[Any] = None,
    geometry: Optional[Any] = None,
) -> WorldModelBatch:
    """Collate function for PyTorch DataLoader returning a WorldModelBatch directly.

    Requires explicit StateSpec to avoid silent domain assumptions.

    Args:
        batch_list: List of sample dictionaries from Dataset.__getitem__.
        state_spec: Target StateSpec contract (explicitly required).
        boundary: Optional boundary specification.
        geometry: Optional geometry specification.

    Returns:
        Structured WorldModelBatch instance.
    """
    from torch.utils.data.dataloader import default_collate
    collated_dict = default_collate(batch_list)
    return WorldModelBatch.from_batch_dict(
        collated_dict,
        state_spec=state_spec,
        boundary=boundary,
        geometry=geometry,
    )


def shear_flow_batch_adapter(
    batch_dict: Dict[str, Any],
    boundary: Optional[Any] = "periodic",
    geometry: Optional[Any] = None,
) -> WorldModelBatch:
    """Domain adapter for 2D Kolmogorov shear flow dataset.

    Injects SHEAR_FLOW_STATE_SPEC and default 'periodic' boundary condition.

    Args:
        batch_dict: Dictionary returned by PyTorch default_collate.
        boundary: Boundary condition (default: "periodic").
        geometry: Domain geometry (default: None for open/unbounded rectangular torus).

    Returns:
        Structured WorldModelBatch bound to SHEAR_FLOW_STATE_SPEC.
    """
    return WorldModelBatch.from_batch_dict(
        batch_dict=batch_dict,
        state_spec=SHEAR_FLOW_STATE_SPEC,
        boundary=boundary,
        geometry=geometry,
    )


def collate_shear_flow_batch(
    batch_list: List[Dict[str, Any]],
    boundary: Optional[Any] = "periodic",
    geometry: Optional[Any] = None,
) -> WorldModelBatch:
    """Collate function for 2D shear flow dataset with SHEAR_FLOW_STATE_SPEC."""
    from torch.utils.data.dataloader import default_collate
    collated = default_collate(batch_list)
    return shear_flow_batch_adapter(collated, boundary=boundary, geometry=geometry)
