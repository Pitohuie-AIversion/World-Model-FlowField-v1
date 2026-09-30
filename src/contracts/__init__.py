"""World Model Core Contract Abstractions (Phase 1 & Phase 1.1).

Establishes the fundamental software contracts governing:
1. StateSpec: Physical state variables and channel dimensionality.
2. Context: Multi-modal world conditioning (physical parameters, geometry, boundary, forcing).
3. WorldModelBatch: Structured, mapping-compatible batch representation with single source of truth.
4. LatentDynamics: Unified interface spanning deterministic, Gaussian, and Flow Matching dynamics.
"""

from src.contracts.state_spec import StateSpec, SHEAR_FLOW_STATE_SPEC
from src.contracts.context import Context, PhysicalContext, resolve_context
from src.contracts.batch import (
    WorldModelBatch,
    collate_world_model_batch,
    shear_flow_batch_adapter,
    collate_shear_flow_batch,
)
from src.contracts.latent_dynamics import (
    LatentDynamics,
    DeterministicLatentDynamics,
    GaussianLatentDynamics,
    FlowMatchingLatentDynamics,
)

__all__ = [
    "StateSpec",
    "SHEAR_FLOW_STATE_SPEC",
    "Context",
    "PhysicalContext",
    "resolve_context",
    "WorldModelBatch",
    "collate_world_model_batch",
    "shear_flow_batch_adapter",
    "collate_shear_flow_batch",
    "LatentDynamics",
    "DeterministicLatentDynamics",
    "GaussianLatentDynamics",
    "FlowMatchingLatentDynamics",
]
