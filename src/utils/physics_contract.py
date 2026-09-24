"""Canonical physical-space contract for The Well shear_flow Closure-R4.

Tensor layout is intentionally kept identical to the HDF5/Dataset layout:
    (..., C, Nx, Ny)
with dim -2 -> x and dim -1 -> y.

The physical derivative extents are Lx=1 and Ly=2. HDF5 coordinate labels are
normalized, so derivative operators must use the physical extents below.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import torch

PHYSICS_PROTOCOL = "Closure-R4"
SPATIAL_AXIS_CONTRACT = "tensor(...,C,Nx,Ny):dim-2=x,dim-1=y"
SHEAR_FLOW_DOMAIN_SIZE_XY: Tuple[float, float] = (1.0, 2.0)


def zero_mean_pressure_gauge(
    tensor: torch.Tensor,
    pressure_channel: int = 2,
) -> torch.Tensor:
    """Enforces zero spatial-mean gauge pressure condition on physical flow field tensors.

    Supports arbitrary leading batch and time dimensions:
      - 3D: (C, Nx, Ny)
      - 4D: (T, C, Nx, Ny) or (B, C, Nx, Ny)
      - 5D: (B, T, C, Nx, Ny)

    Leaves all other channels (u, v, tracer) strictly bitwise unmodified.
    Returns:
        Tensor with zero spatial mean on pressure_channel.
    """
    if tensor.ndim < 3:
        raise ValueError(f"zero_mean_pressure_gauge expects at least 3D tensor (C, Nx, Ny), got ndim={tensor.ndim}")
    out = tensor.clone()
    p_slice = out[..., pressure_channel : pressure_channel + 1, :, :]
    out[..., pressure_channel : pressure_channel + 1, :, :] = p_slice - p_slice.mean(dim=(-2, -1), keepdim=True)
    return out

# Canonical semantic specifications for ablation groups E0 - E4
ABLATION_SEMANTIC_SPECS: Dict[str, Dict[str, Any]] = {
    "E0_single_step": {
        "title": "E0: Single-Step Pure Field Loss",
        "horizon": 1,
        "lambda_div": 0.0,
        "lambda_vort": 0.0,
    },
    "E1_rollout_field": {
        "title": "E1: Rollout-Aware Field Loss",
        "horizon": 2,
        "lambda_div": 0.0,
        "lambda_vort": 0.0,
    },
    "E2_plus_L_div": {
        "title": "E2: + Divergence Loss",
        "horizon": 2,
        "lambda_div": 0.01,
        "lambda_vort": 0.0,
    },
    "E3_plus_L_vort": {
        "title": "E3: + Vorticity Loss",
        "horizon": 2,
        "lambda_div": 0.0,
        "lambda_vort": 0.05,
    },
    "E4_full_physics": {
        "title": "E4: Full Physics Coupling",
        "horizon": 2,
        "lambda_div": 0.01,
        "lambda_vort": 0.05,
    },
}


def validate_ablation_checkpoint_semantics(
    group_key: str,
    ckpt_cfg: Dict[str, Any],
    is_legacy: bool = False,
) -> None:
    """Strictly validate that a candidate checkpoint satisfies the semantic contract for group_key.

    Raises:
        ValueError: If horizon, lambda_div, lambda_vort, or Closure-R4 identity contracts fail.
    """
    if group_key not in ABLATION_SEMANTIC_SPECS:
        raise ValueError(f"Unknown ablation group '{group_key}'. Expected one of {list(ABLATION_SEMANTIC_SPECS.keys())}")

    spec = ABLATION_SEMANTIC_SPECS[group_key]
    expected_horizon = spec["horizon"]
    expected_div = spec["lambda_div"]
    expected_vort = spec["lambda_vort"]

    mismatches: List[str] = []

    # 0. Required fields fail-closed validation
    required = [
        "horizon",
        "lambda_div",
        "lambda_vort",
    ]
    if not is_legacy:
        required += [
            "physics_protocol",
            "spatial_axis_contract",
            "physics_domain_size_xy",
        ]

    for key in required:
        if key not in ckpt_cfg:
            mismatches.append(f"missing required field: {key}")

    # 1. Horizon semantic validation
    if "horizon" in ckpt_cfg:
        ckpt_horizon = ckpt_cfg["horizon"]
        if ckpt_horizon != expected_horizon:
            mismatches.append(
                f"horizon: checkpoint={ckpt_horizon} vs expected={expected_horizon} for group '{group_key}'"
            )

    # 2. Physics loss weights semantic validation
    ckpt_div = float(ckpt_cfg.get("lambda_div", 0.0) or 0.0)
    if "lambda_div" in ckpt_cfg and not math.isclose(ckpt_div, expected_div, abs_tol=1e-5):
        mismatches.append(
            f"lambda_div: checkpoint={ckpt_div} vs expected={expected_div} for group '{group_key}'"
        )

    ckpt_vort = float(ckpt_cfg.get("lambda_vort", 0.0) or 0.0)
    if "lambda_vort" in ckpt_cfg and not math.isclose(ckpt_vort, expected_vort, abs_tol=1e-5):
        mismatches.append(
            f"lambda_vort: checkpoint={ckpt_vort} vs expected={expected_vort} for group '{group_key}'"
        )

    # 3. Protocol identity validation
    if is_legacy:
        if group_key not in ("E0_single_step", "E1_rollout_field"):
            mismatches.append(
                f"legacy opt-in refused: group '{group_key}' requires non-zero physics loss "
                f"and cannot use legacy checkpoints"
            )
        # Field-only legacy checkpoints (E0, E1) cannot have non-zero physics losses
        if ckpt_div != 0.0 or ckpt_vort != 0.0:
            mismatches.append(
                f"legacy opt-in refused: checkpoint has non-zero physics losses "
                f"(lambda_div={ckpt_div}, lambda_vort={ckpt_vort}) but is marked legacy"
            )
    else:
        # All non-legacy Closure-R4 checkpoints must declare protocol identity
        proto = ckpt_cfg.get("physics_protocol")
        if "physics_protocol" in ckpt_cfg and proto != PHYSICS_PROTOCOL:
            mismatches.append(
                f"physics_protocol: checkpoint={proto!r} vs required={PHYSICS_PROTOCOL!r}"
            )

        axis_contract = ckpt_cfg.get("spatial_axis_contract")
        if "spatial_axis_contract" in ckpt_cfg and axis_contract != SPATIAL_AXIS_CONTRACT:
            mismatches.append(
                f"spatial_axis_contract: checkpoint={axis_contract!r} vs required={SPATIAL_AXIS_CONTRACT!r}"
            )

        domain_size = ckpt_cfg.get("physics_domain_size_xy")
        if "physics_domain_size_xy" in ckpt_cfg and (
            domain_size is None or list(domain_size) != list(SHEAR_FLOW_DOMAIN_SIZE_XY)
        ):
            mismatches.append(
                f"physics_domain_size_xy: checkpoint={domain_size!r} vs required={list(SHEAR_FLOW_DOMAIN_SIZE_XY)!r}"
            )

    if mismatches:
        raise ValueError(
            f"Semantic contract violation for ablation group '{group_key}':\n"
            + "\n".join(f"  - {m}" for m in mismatches)
        )
