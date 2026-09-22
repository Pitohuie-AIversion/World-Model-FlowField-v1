"""Utils package."""

from src.utils.fft_derivatives import (
    spectral_grad_xy,
    spectral_grad_2d,
    compute_vorticity,
    compute_divergence,
    compute_kinetic_energy,
    compute_enstrophy,
    compute_laplacian_2d,
    project_zero_mean_pressure,
)
from src.utils.physics_contract import (
    PHYSICS_PROTOCOL,
    SPATIAL_AXIS_CONTRACT,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
)

__all__ = [
    "spectral_grad_xy",
    "spectral_grad_2d",
    "compute_vorticity",
    "compute_divergence",
    "compute_kinetic_energy",
    "compute_enstrophy",
    "compute_laplacian_2d",
    "project_zero_mean_pressure",
    "PHYSICS_PROTOCOL",
    "SPATIAL_AXIS_CONTRACT",
    "SHEAR_FLOW_DOMAIN_SIZE_XY",
]
