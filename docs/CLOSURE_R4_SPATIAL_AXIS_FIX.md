# Closure-R4: Spatial Axis Contract & Physics Asset Governance

## 1. Executive Summary

This document formalizes the **Closure-R4** scientific contract for `The Well: shear_flow` in World-Model-FlowField-v1.
Closure-R4 addresses two critical areas:
1. **Mathematical Spatial-Axis Contract**: Alignment of the physical domain extents ($L_x=1.0, L_y=2.0$) and tensor spatial dimension layout `(..., C, Nx=128, Ny=256)` where `dim -2` represents $x$ and `dim -1` represents $y$.
2. **Experiment Provenance & Fail-Closed Asset Governance**: Preventing pre-R4 model checkpoints trained under incorrect swapped-axis spatial derivative operators from being evaluated or reported as valid scientific results.

---

## 2. Canonical Physical Contract Constants

Defined in `src/utils/physics_contract.py`:

```python
PHYSICS_PROTOCOL = "Closure-R4"
SPATIAL_AXIS_CONTRACT = "tensor(...,C,Nx,Ny):dim-2=x,dim-1=y"
SHEAR_FLOW_DOMAIN_SIZE_XY = (1.0, 2.0)
```

- **Tensor Spatial Layout**: `(..., C, Nx, Ny)`
- **Horizontal Axis $x$**: `dim -2`, length $N_x$, physical length $L_x = 1.0$, periodic.
- **Vertical Axis $y$**: `dim -1`, length $N_y$, physical length $L_y = 2.0$, periodic.
- **Velocity Components**: Channel 0 is $u$ (horizontal velocity), Channel 1 is $v$ (vertical velocity).
- **Physical Operators**:
  - Divergence: $\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$
  - Vorticity: $\omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$
  - Laplacian: $\nabla^2 \phi = \frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2}$

---

## 3. Ground Truth Verification Evidence

On the real downsampled verification trajectory (`shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5`):

| Metric | Pre-R4 Swapped Operator | Closure-R4 Corrected Operator | Reduction |
| :--- | :--- | :--- | :--- |
| **Ground Truth Divergence RMSE** | **5.8208** | **0.0019** | **99.78%** |
| $\text{std}(\partial u / \partial x)$ vs $\text{std}(\partial v / \partial y)$ | Unbalanced | $0.3418 \approx 0.3418$ | **Exact Balance** |

Analytical streamfunction $\psi(x, y) = \sin(2\pi x / L_x) \sin(2\pi y / L_y)$ tests confirm analytical divergence $< 10^{-7}$ and $\omega = -\nabla^2 \psi$ relative error $< 5 \times 10^{-4}$.

---

## 4. Checkpoint Provenance & Fail-Closed Guard Rules

1. **Autoencoder (Representation)**:
   - Does not use spatial derivative operators or physical losses.
   - Weights in `outputs/checkpoints/representation/best_vrmse_mean.pt` remain **100% scientifically valid**.
2. **E0 (Single Step) & E1 (Rollout Field)**:
   - Pure field-space loss ($L_1 + \text{rel-}L_2$), no physics loss ($\lambda_{div} = 0, \lambda_{vort} = 0$).
   - Can be re-evaluated under Closure-R4 using `--allow_legacy_checkpoints`.
3. **E2 (+L_div), E3 (+L_vort), E4 (Full Physics)**:
   - Checkpoints trained before Closure-R4 were optimized against invalid gradients.
   - **Permanently blocked** from evaluation in:
     - `scripts/evaluate_physics_ablation.py`
     - `scripts/evaluate_rollout.py`
     - `scripts/analyze_failure_cases.py`
   - Attempting to evaluate any checkpoint with $\lambda_{div} > 0$ or $\lambda_{vort} > 0$ that does not declare `physics_protocol: "Closure-R4"` immediately raises `ValueError` / `RuntimeError`.
4. **Output Directory Isolation**:
   - Closure-R4 ablation models are trained and saved to `outputs/checkpoints/dynamics/closure_r4/`.
   - Results are saved to `outputs/metrics/closure_r4_physics_ablation.json` and `outputs/metrics/closure_r4_rollout_benchmark.json`.
