# Closure-R4: Spatial Axis Contract, Semantic Specifications & Physics Asset Governance

## 1. Executive Summary

This document formalizes the **Closure-R4** scientific contract for `The Well: shear_flow` in World-Model-FlowField-v1.
Closure-R4 addresses three critical areas:
1. **Mathematical Spatial-Axis Contract**: Alignment of the physical domain extents ($L_x=1.0, L_y=2.0$) and tensor spatial dimension layout `(..., C, Nx=128, Ny=256)` where `dim -2` represents $x$ and `dim -1` represents $y$.
2. **Ground Truth Data Audit Evidence**: Clear, reproducible numerical verification on real simulation data comparing the exact pre-R4 swapped operator vs. the corrected Closure-R4 operator.
3. **E0-E4 Semantic Contract & Fail-Closed Asset Governance**: Preventing pre-R4 model checkpoints trained under incorrect swapped-axis spatial derivative operators from being evaluated, and strictly enforcing horizon and loss-weight semantics across ablation groups.

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

## 3. Ground Truth Verification Evidence (Real Data Audit)

All values below are measured on the ground truth verification flow field (`shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5`, $t=10$). These metrics evaluate the **incompressibility of the true physical simulation data under the two operator implementations**, completely independent of any trained model prediction.

| Resolution | Operator Implementation | GT Divergence RMSE | Error Reduction | Orthogonal Gradients |
| :--- | :--- | :--- | :--- | :--- |
| **Downsampled (128x256)** | Pre-R4 Swapped Operator | **1.785469** | — | Unbalanced |
| **Downsampled (128x256)** | **Closure-R4 Corrected Operator** | **0.001926** | **99.89%** | $\text{std}(\partial_x u)=\text{std}(\partial_y v)=0.3418$ |
| **Full Res (256x512)** | Pre-R4 Swapped Operator | **1.785469** | — | Unbalanced |
| **Full Res (256x512)** | **Closure-R4 Corrected Operator** | **0.004887** | **99.73%** | $\text{std}(\partial_x u)=\text{std}(\partial_y v)=0.6791$ |

### Mathematical Explanation of the Pre-R4 Error
The pre-R4 operator assumed input `(..., Ny, Nx)` with domain size $(2.0, 1.0)$. When applied to the real data `(..., Nx=128, Ny=256)`, it effectively computed:
$$\text{div}_{\text{old}} = 2 \frac{\partial u}{\partial y} + \frac{1}{2} \frac{\partial v}{\partial x} \neq \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$$
Because true physical shear flow satisfies $\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$, the artificial mismatch in axis assignment and scaling generated an artificial divergence of $1.7855$. Correcting the axis mapping and domain sizes ($L_x=1.0, L_y=2.0$) recovers the true divergence-free physics ($\text{RMSE} = 0.0019$).

---

## 4. Canonical Semantic Specifications for Ablation Study (E0 - E4)

To prevent mislabeled or mismatched checkpoints (e.g. an E2 directory containing an E1 checkpoint), `validate_ablation_checkpoint_semantics` strictly enforces the following semantic matrix:

| Group Key | Title | Horizon ($H$) | $\lambda_{div}$ | $\lambda_{vort}$ | Allowed Protocol |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **E0_single_step** | Single-Step Pure Field Loss | 1 | 0.0 | 0.0 | Closure-R4 (or pre-R4 field-only opt-in) |
| **E1_rollout_field** | Rollout-Aware Field Loss | 2 | 0.0 | 0.0 | Closure-R4 (or pre-R4 field-only opt-in) |
| **E2_plus_L_div** | + Divergence Loss | 2 | 0.01 | 0.0 | **Closure-R4 ONLY** |
| **E3_plus_L_vort** | + Vorticity Loss | 2 | 0.0 | 0.05 | **Closure-R4 ONLY** |
| **E4_full_physics** | Full Physics Coupling | 2 | 0.01 | 0.05 | **Closure-R4 ONLY** |

### Checkpoint Provenance & Fail-Closed Guard Rules
1. **Autoencoder (Representation)**:
   - Does not use spatial derivative operators or physical losses.
   - Weights in `outputs/checkpoints/representation/best_vrmse_mean.pt` remain **100% scientifically valid**.
2. **E0 & E1**:
   - Pure field-space loss ($L_1 + \text{rel-}L_2$), no physics loss ($\lambda_{div} = 0, \lambda_{vort} = 0$).
   - Can be re-evaluated under Closure-R4 using `--allow_legacy_checkpoints`.
3. **E2, E3, E4**:
   - Checkpoints trained before Closure-R4 were optimized against invalid gradients.
   - **Permanently blocked** from evaluation in all evaluation scripts (`evaluate_physics_ablation.py`, `evaluate_rollout.py`, `analyze_failure_cases.py`).
   - Any checkpoint claiming non-zero physics losses that lacks `physics_protocol: "Closure-R4"`, `spatial_axis_contract`, or `physics_domain_size_xy: [1.0, 2.0]` fails closed immediately.
4. **Required Field Fail-Closed Verification**:
   - For all non-legacy Closure-R4 checkpoints, `horizon`, `lambda_div`, `lambda_vort`, `physics_protocol`, `spatial_axis_contract`, and `physics_domain_size_xy` are strictly required fields. Any missing field triggers an immediate fail-closed semantic violation.
   - For legacy field-only checkpoints (`is_legacy=True`), `horizon`, `lambda_div`, and `lambda_vort` are required, and physics losses must strictly be 0.0.
5. **Output Directory Isolation**:
   - Closure-R4 ablation models are trained and saved to `outputs/checkpoints/dynamics/closure_r4/`.
   - Results are saved to `outputs/metrics/closure_r4_physics_ablation.json` and `outputs/metrics/closure_r4_rollout_benchmark.json`.
