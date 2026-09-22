# Closure-R4 Spatial-Axis Contract Fix

Closure-R4 supersedes the pre-R4 physics-operator protocol for The Well shear_flow.

## Canonical contract

- Dataset/HDF5 tensor layout remains (..., C, Nx, Ny).
- dim -2 = x, dim -1 = y.
- Physical derivative extents are Lx=1.0, Ly=2.0.
- FFT gradients return (df/dx, df/dy).
- No Dataset transpose is introduced.

## Invalidated assets

Any checkpoint trained with non-zero divergence or vorticity loss before
Closure-R4 used the swapped-axis physics operator and is invalid for formal
scientific comparison.

| Asset | Disposition |
| --- | --- |
| Representation | keep |
| E0 field-only | keep weights; recompute physics metrics |
| E1 rollout field-only | keep weights; recompute physics metrics |
| E2 + divergence | retrain |
| E3 + vorticity | retrain |
| E4 full physics | retrain |
| H2/H4/H8 | retrain iff lambda_div or lambda_vort was non-zero |
| Structural ablations | apply the same rule |

The pre-R4 closure_r3 physics-ablation metrics are archived only and must not be
used for final physical conclusions.
