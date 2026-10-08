"""Context contract for multi-modal world conditioning.

Encapsulates physical parameters (Re, Sc), boundary conditions, geometry,
external forcing, language descriptors, and agent/robot actions into a unified structure.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch

# Maximum exact integer representable without roundoff in IEEE 754 binary64 (float64)
MAX_EXACT_INT = 9007199254740992  # 2**53


@dataclass
class PhysicalContext:
    """Encapsulates continuous physical parameters governing the dynamical system.

    In the shear_flow domain, this contains Reynolds (Re) and Schmidt (Sc) numbers.
    """

    re: Optional[torch.Tensor] = None
    sc: Optional[torch.Tensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to(self, *args, **kwargs) -> "PhysicalContext":
        """Move all contained tensors to target device and dtype."""
        new_re = self.re.to(*args, **kwargs) if isinstance(self.re, torch.Tensor) else self.re
        new_sc = self.sc.to(*args, **kwargs) if isinstance(self.sc, torch.Tensor) else self.sc
        new_extra = {}
        for k, v in self.extra.items():
            new_extra[k] = v.to(*args, **kwargs) if isinstance(v, torch.Tensor) else v
        return PhysicalContext(re=new_re, sc=new_sc, extra=new_extra)


@dataclass
class Context:
    """Unified context object passed into World Models and Latent Dynamics.

    Phase 1 actively utilizes:
        physical: PhysicalContext holding Reynolds and Schmidt parameters.

    Reserved for subsequent extensions:
        geometry: GeometryContext / Signed Distance Functions (SDF) / obstacle masks.
        boundary: Boundary specification and operators (periodic, wall, no-slip, etc.).
        forcing: External dynamic spatial-temporal forcing fields.
        language: Tokenized language conditioning or goal prompts.
        action: Robot or agent control inputs.
    """

    physical: Optional[PhysicalContext] = None
    geometry: Optional[Any] = None
    boundary: Optional[Any] = None
    forcing: Optional[Any] = None
    language: Optional[Any] = None
    action: Optional[Any] = None

    @property
    def re(self) -> Optional[torch.Tensor]:
        """Convenience accessor for Reynolds number tensor."""
        return self.physical.re if self.physical is not None else None

    @property
    def sc(self) -> Optional[torch.Tensor]:
        """Convenience accessor for Schmidt number tensor."""
        return self.physical.sc if self.physical is not None else None

    @classmethod
    def from_re_sc(
        cls,
        re: Optional[Union[torch.Tensor, float, int]] = None,
        sc: Optional[Union[torch.Tensor, float, int]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **extra_physical,
    ) -> "Context":
        """Factory creating a Context from Reynolds and Schmidt numbers.

        Args:
            re: Reynolds number scalar or tensor.
            sc: Schmidt number scalar or tensor.
            device: Optional target device.
            dtype: Optional target dtype (default: float32 for floats).
            **extra_physical: Additional physical parameters.
        """
        # Input type and non-empty validation
        for param_val, param_name in ((re, "re"), (sc, "sc")):
            if param_val is not None:
                if isinstance(param_val, (bool, np.bool_)):
                    raise TypeError(f"Physical parameter '{param_name}' must be numeric, got boolean {param_val}.")
                if isinstance(param_val, (complex, np.complexfloating)):
                    raise TypeError(f"Physical parameter '{param_name}' must be real-valued, got complex {param_val}.")
                if isinstance(param_val, torch.Tensor):
                    if param_val.dtype == torch.bool:
                        raise TypeError(f"Physical parameter '{param_name}' must be real numeric, got boolean tensor.")
                    if param_val.is_complex():
                        raise TypeError(f"Physical parameter '{param_name}' must be real-valued, got complex tensor.")
                    if param_val.numel() == 0:
                        raise ValueError(f"Physical parameter '{param_name}' must not be empty (got numel=0).")
                elif isinstance(param_val, np.ndarray):
                    if param_val.dtype == np.bool_:
                        raise TypeError(f"Physical parameter '{param_name}' must be real numeric, got boolean numpy array.")
                    if np.iscomplexobj(param_val):
                        raise TypeError(f"Physical parameter '{param_name}' must be real-valued, got complex numpy array.")
                    if param_val.size == 0:
                        raise ValueError(f"Physical parameter '{param_name}' must not be empty (got numel=0).")
                elif isinstance(param_val, (list, tuple)):
                    if len(param_val) == 0:
                        raise ValueError(f"Physical parameter '{param_name}' must not be empty (got len=0).")
                    for elem in param_val:
                        if isinstance(elem, (bool, np.bool_)):
                            raise TypeError(f"Physical parameter '{param_name}' sequence contains boolean element: {elem}.")
                        if isinstance(elem, (complex, np.complexfloating)):
                            raise TypeError(f"Physical parameter '{param_name}' sequence contains complex element: {elem}.")

        re_t = None
        if re is not None:
            if isinstance(re, torch.Tensor):
                re_t = re if device is None and dtype is None else re.to(device=device, dtype=dtype)
            else:
                re_t = torch.tensor(re, dtype=dtype or torch.float32, device=device)
            if re_t.ndim == 0:
                re_t = re_t.unsqueeze(0)

        sc_t = None
        if sc is not None:
            if isinstance(sc, torch.Tensor):
                sc_t = sc if device is None and dtype is None else sc.to(device=device, dtype=dtype)
            else:
                sc_t = torch.tensor(sc, dtype=dtype or torch.float32, device=device)
            if sc_t.ndim == 0:
                sc_t = sc_t.unsqueeze(0)

        phys = PhysicalContext(re=re_t, sc=sc_t, extra=extra_physical)
        return cls(physical=phys)

    def to_re_sc(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract (re, sc) tensors for direct compatibility with legacy modules."""
        if self.physical is None:
            return None, None
        return self.physical.re, self.physical.sc

    def to(self, *args, **kwargs) -> "Context":
        """Move all internal tensors (physical, geometry, etc.) to target device/dtype."""
        new_phys = self.physical.to(*args, **kwargs) if self.physical is not None else None
        new_geom = self.geometry.to(*args, **kwargs) if hasattr(self.geometry, "to") else self.geometry
        new_bound = self.boundary.to(*args, **kwargs) if hasattr(self.boundary, "to") else self.boundary
        new_force = self.forcing.to(*args, **kwargs) if hasattr(self.forcing, "to") else self.forcing
        new_act = self.action.to(*args, **kwargs) if hasattr(self.action, "to") else self.action
        return Context(
            physical=new_phys,
            geometry=new_geom,
            boundary=new_bound,
            forcing=new_force,
            language=self.language,
            action=new_act,
        )


def _canonicalize_parameter_for_comparison(
    val: Any,
    param_name: str,
) -> torch.Tensor:
    """Canonicalize a physical parameter into a 1D float64 CPU tensor for comparison.

    Rules:
    1. Type safety:
       - Boolean inputs (Python bool, numpy bool, torch bool) are strictly rejected with TypeError.
       - Complex numbers (Python complex, numpy complex, torch complex) are strictly rejected with TypeError
         BEFORE any lossy type conversions.
    2. Exact range limits:
       - Integer inputs (Python int, numpy int, torch integer) must be within [-2^53, 2^53],
         the exact lossless integer representation limit of IEEE 754 binary64.
       - Out-of-range integers raise ValueError immediately.
    3. Python floats and valid integers are directly converted to float64 tensors explicitly on CPU.
    4. Tensor inputs preserve exact numerical values and are moved to CPU and promoted to float64.
    5. Non-empty check: val must have numel > 0; empty tensors/arrays raise ValueError.
    6. Finiteness check: NaN, Inf, and -Inf are strictly rejected (torch.isfinite).
    7. Shape contract:
       - Scalars: 0-D (), 1-D (1,), or 2-D (1, 1) are normalized to 1D shape (1,).
       - Batch vectors: 1-D (B,) with B > 1, or 2-D (B, 1) with B > 1 are normalized to 1D shape (B,).
       - Any higher dimensional (>2D) or multi-column (shape[1] > 1) tensors are rejected.

    Returns:
        1D torch.Tensor of dtype float64 on CPU with numel > 0.

    Raises:
        TypeError: If val is boolean, complex, or has an unsupported data type.
        ValueError: If val is empty, non-finite, out of exact integer range, or has invalid shape.
    """
    # 1. Reject booleans strictly (in Python, bool is a subclass of int)
    if isinstance(val, (bool, np.bool_)):
        raise TypeError(f"Physical parameter '{param_name}' must be numeric, got boolean {val}.")

    # 2. Reject complex values strictly before any casting/as_tensor calls
    if isinstance(val, (complex, np.complexfloating)):
        raise TypeError(f"Physical parameter '{param_name}' must be real-valued, got complex {val}.")

    # 3. Handle Python numbers and scalars
    if isinstance(val, (int, np.integer)):
        if abs(int(val)) > MAX_EXACT_INT:
            raise ValueError(
                f"Integer physical parameter '{param_name}' exceeds maximum lossless "
                f"float64 representation limit (+/- 2^53): got {val}."
            )
        t = torch.tensor([float(val)], dtype=torch.float64, device="cpu")
    elif isinstance(val, (float, np.floating)):
        t = torch.tensor([float(val)], dtype=torch.float64, device="cpu")
    elif isinstance(val, torch.Tensor):
        if val.dtype == torch.bool:
            raise TypeError(f"Physical parameter '{param_name}' must be real numeric, got boolean tensor.")
        if val.is_complex():
            raise TypeError(f"Physical parameter '{param_name}' must be real-valued, got complex tensor dtype {val.dtype}.")
        if val.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            if val.numel() > 0 and torch.any(torch.abs(val) > MAX_EXACT_INT):
                raise ValueError(
                    f"Integer tensor for physical parameter '{param_name}' exceeds maximum lossless "
                    f"float64 representation limit (+/- 2^53)."
                )
        elif val.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError(f"Physical parameter '{param_name}' has unsupported tensor dtype {val.dtype}.")
        t = val.detach().to(device="cpu", dtype=torch.float64)
    elif isinstance(val, np.ndarray):
        if val.dtype == np.bool_:
            raise TypeError(f"Physical parameter '{param_name}' must be real numeric, got boolean numpy array.")
        if np.iscomplexobj(val):
            raise TypeError(f"Physical parameter '{param_name}' must be real-valued, got complex numpy array of dtype {val.dtype}.")
        if np.issubdtype(val.dtype, np.integer):
            if val.size > 0 and np.any(np.abs(val) > MAX_EXACT_INT):
                raise ValueError(
                    f"Integer numpy array for physical parameter '{param_name}' exceeds maximum lossless "
                    f"float64 representation limit (+/- 2^53)."
                )
        t = torch.from_numpy(val).detach().to(device="cpu", dtype=torch.float64)
    elif isinstance(val, (list, tuple)):
        for item in val:
            if isinstance(item, (bool, np.bool_)):
                raise TypeError(f"Physical parameter '{param_name}' sequence contains boolean element: {item}.")
            if isinstance(item, (complex, np.complexfloating)):
                raise TypeError(f"Physical parameter '{param_name}' sequence contains complex element: {item}.")
        try:
            t = torch.as_tensor(val, device="cpu", dtype=torch.float64)
        except Exception as err:
            raise TypeError(f"Unsupported sequence for physical parameter '{param_name}': {err}") from err
    else:
        raise TypeError(f"Unsupported type {type(val)} for physical parameter '{param_name}'.")

    # 4. Non-empty check (P2-3)
    if t.numel() == 0:
        raise ValueError(f"Physical parameter '{param_name}' must not be empty (got numel=0).")

    # 5. Finiteness validation (Fail-Fast on NaN / Inf)
    if not torch.all(torch.isfinite(t)):
        raise ValueError(
            f"Physical parameter '{param_name}' must be finite, but contains non-finite values (NaN or Inf)."
        )

    # 6. Shape contract validation
    if t.ndim == 0:
        return t.unsqueeze(0)
    elif t.ndim == 1:
        return t
    elif t.ndim == 2:
        if t.shape[1] == 1:
            return t.squeeze(1)
        elif t.shape[0] == 1 and t.shape[1] == 1:
            return t.reshape(1)
        else:
            raise ValueError(
                f"Invalid shape {tuple(val.shape if hasattr(val, 'shape') else t.shape)} for physical parameter '{param_name}': "
                f"multi-column 2D tensors are not valid scalar or 1D batch vectors."
            )
    else:
        raise ValueError(
            f"Invalid shape {tuple(val.shape if hasattr(val, 'shape') else t.shape)} for physical parameter '{param_name}': "
            f"tensors with ndim >= 3 are not valid scalar or batch conditions."
        )


def _assert_parameter_consistency(
    ctx_val: Any,
    legacy_val: Any,
    param_name: str,
    atol: float = 1e-6,
) -> None:
    """Validate numerical and shape consistency between Context and legacy parameters.

    Enforces:
    1. Independent canonicalization into float64 CPU copies without lossy cross-casting.
    2. Strict finiteness (rejects NaN, Inf on either side).
    3. Strict shape contract (rejects implicit broadcasting between scalar and batch vector,
       or mismatched batch sizes).
    4. Strict symmetric absolute difference check (max |diff| <= atol).
       Note: atol=1e-6 is an engineering absolute consistency tolerance to accommodate
       minor float32 vs float64 machine representation differences, not mathematical identity.

    Raises:
        ValueError: On non-finite values, out-of-range integers, empty inputs,
                    shape mismatch, or numerical divergence.
        TypeError: On boolean, complex, or unsupported data types.
    """
    ctx_comp = _canonicalize_parameter_for_comparison(ctx_val, f"Context physical.{param_name}")
    leg_comp = _canonicalize_parameter_for_comparison(legacy_val, f"legacy {param_name}")

    # Shape compatibility check: disallow implicit broadcasting between scalar and batch vector
    if ctx_comp.numel() != leg_comp.numel():
        raise ValueError(
            f"Context conflict: shape mismatch for '{param_name}' between Context "
            f"(size {ctx_comp.numel()}) and legacy parameter (size {leg_comp.numel()}). "
            f"Implicit broadcasting between scalar and batch vector is disallowed."
        )

    # Numerical equivalence check: symmetric absolute difference
    diff = torch.abs(ctx_comp - leg_comp)
    max_diff = torch.max(diff).item()
    if max_diff > atol:
        raise ValueError(
            f"Context conflict: Context has {param_name}={ctx_val}, "
            f"but divergent legacy {param_name}={legacy_val} was provided (Fail-Closed, "
            f"max_diff={max_diff:.6e} > atol={atol:.1e})."
        )


def resolve_context(
    context: Optional[Union[Context, Dict[str, Any]]] = None,
    re: Optional[Union[torch.Tensor, float]] = None,
    sc: Optional[Union[torch.Tensor, float]] = None,
) -> Optional[Context]:
    """Compatibility resolver supporting both new Context and legacy (re, sc) parameters.

    Fail-Closed Policy:
    If both Context and legacy (re, sc) parameters are provided, their values must be
    strictly consistent without lossy dtype truncation, non-finite values (NaN/Inf),
    or implicit shape broadcasting. Any divergence raises ValueError immediately.

    Returns:
        Unified Context instance, or None if no conditioning information is provided.

    Raises:
        ValueError: If Context and legacy parameters conflict, are non-finite, or have invalid shapes.
        TypeError: If context or parameters have unsupported types.
    """
    if context is not None:
        if isinstance(context, Context):
            resolved_ctx = context
        elif isinstance(context, dict):
            # Parse dict-based context
            phys_data = context.get("physical")
            if isinstance(phys_data, PhysicalContext):
                re_val = phys_data.re
                sc_val = phys_data.sc
            elif isinstance(phys_data, dict):
                re_val = phys_data.get("re", context.get("re"))
                sc_val = phys_data.get("sc", context.get("sc"))
            else:
                re_val = context.get("re")
                sc_val = context.get("sc")

            phys = None
            if re_val is not None or sc_val is not None:
                phys = PhysicalContext(re=re_val, sc=sc_val)

            resolved_ctx = Context(
                physical=phys,
                geometry=context.get("geometry"),
                boundary=context.get("boundary"),
                forcing=context.get("forcing"),
                language=context.get("language"),
                action=context.get("action"),
            )
        else:
            raise TypeError(f"Expected Context or dict, got {type(context)}")

        # Fail-closed validation against legacy parameters: strict physical condition identity
        if re is not None:
            if resolved_ctx.re is None:
                raise ValueError(
                    f"Context conflict: Context physical.re is None, but legacy re={re} was provided."
                )
            _assert_parameter_consistency(resolved_ctx.re, re, param_name="re")

        if sc is not None:
            if resolved_ctx.sc is None:
                raise ValueError(
                    f"Context conflict: Context physical.sc is None, but legacy sc={sc} was provided."
                )
            _assert_parameter_consistency(resolved_ctx.sc, sc, param_name="sc")

        return resolved_ctx

    if re is not None or sc is not None:
        return Context.from_re_sc(re=re, sc=sc)

    return None
