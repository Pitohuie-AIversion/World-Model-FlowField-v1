"""Canonical physical-space contract for The Well shear_flow Closure-R4.

Tensor layout is intentionally kept identical to the HDF5/Dataset layout:
    (..., C, Nx, Ny)
with dim -2 -> x and dim -1 -> y.

The physical derivative extents are Lx=1 and Ly=2. HDF5 coordinate labels are
normalized, so derivative operators must use the physical extents below.
"""

from typing import Tuple

PHYSICS_PROTOCOL = "Closure-R4"
SPATIAL_AXIS_CONTRACT = "tensor(...,C,Nx,Ny):dim-2=x,dim-1=y"
SHEAR_FLOW_DOMAIN_SIZE_XY: Tuple[float, float] = (1.0, 2.0)
