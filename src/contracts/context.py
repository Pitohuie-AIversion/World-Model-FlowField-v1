"""Context contract for multi-modal world conditioning.

Encapsulates physical parameters (Re, Sc), boundary conditions, geometry,
external forcing, language descriptors, and agent/robot actions into a unified structure.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
import torch


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


def resolve_context(
    context: Optional[Union[Context, Dict[str, Any]]] = None,
    re: Optional[Union[torch.Tensor, float]] = None,
    sc: Optional[Union[torch.Tensor, float]] = None,
) -> Optional[Context]:
    """Compatibility resolver supporting both new Context and legacy (re, sc) parameters.

    Fail-Closed Policy:
    If both Context and legacy (re, sc) parameters are provided, their values must be
    numerically equal within tolerance. Any divergence raises ValueError immediately.

    Returns:
        Unified Context instance, or None if no conditioning information is provided.

    Raises:
        ValueError: If Context and legacy parameters conflict.
        TypeError: If context has an unsupported type.
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
            ctx_re_t = torch.as_tensor(resolved_ctx.re)
            legacy_re_t = torch.as_tensor(re, device=ctx_re_t.device, dtype=ctx_re_t.dtype)
            if not torch.allclose(ctx_re_t, legacy_re_t, rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"Context conflict: Context has re={resolved_ctx.re}, "
                    f"but divergent legacy re={re} was provided (Fail-Closed)."
                )

        if sc is not None:
            if resolved_ctx.sc is None:
                raise ValueError(
                    f"Context conflict: Context physical.sc is None, but legacy sc={sc} was provided."
                )
            ctx_sc_t = torch.as_tensor(resolved_ctx.sc)
            legacy_sc_t = torch.as_tensor(sc, device=ctx_sc_t.device, dtype=ctx_sc_t.dtype)
            if not torch.allclose(ctx_sc_t, legacy_sc_t, rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"Context conflict: Context has sc={resolved_ctx.sc}, "
                    f"but divergent legacy sc={sc} was provided (Fail-Closed)."
                )

        return resolved_ctx

    if re is not None or sc is not None:
        return Context.from_re_sc(re=re, sc=sc)

    return None
