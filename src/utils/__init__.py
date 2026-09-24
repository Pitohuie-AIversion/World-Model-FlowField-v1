"""Utils package."""

from src.utils.fft_derivatives import (
    spectral_grad_2d,
    spectral_grad_xy,
    compute_vorticity,
    compute_divergence,
    compute_kinetic_energy,
    compute_enstrophy,
    compute_laplacian_2d,
    project_zero_mean_pressure,
    project_divergence_free_2d,
    project_incompressible_state,
)
from src.utils.physics_contract import (
    PHYSICS_PROTOCOL,
    SPATIAL_AXIS_CONTRACT,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
    ABLATION_SEMANTIC_SPECS,
    validate_ablation_checkpoint_semantics,
)

__all__ = [
    "spectral_grad_2d",
    "spectral_grad_xy",
    "compute_vorticity",
    "compute_divergence",
    "compute_kinetic_energy",
    "compute_enstrophy",
    "compute_laplacian_2d",
    "project_zero_mean_pressure",
    "project_divergence_free_2d",
    "project_incompressible_state",
    "PHYSICS_PROTOCOL",
    "SPATIAL_AXIS_CONTRACT",
    "SHEAR_FLOW_DOMAIN_SIZE_XY",
    "ABLATION_SEMANTIC_SPECS",
    "validate_ablation_checkpoint_semantics",
]

