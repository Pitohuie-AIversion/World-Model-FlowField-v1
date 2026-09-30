"""LatentDynamics core contract for world models.

Establishes a unified interface for latent transition dynamics:
- DeterministicLatentDynamics (LatentSTTransformer)
- GaussianLatentDynamics (Conditional Gaussian with VarianceHead2D)
- FlowMatchingLatentDynamics (D0 backbone + Residual Latent CFM)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.contracts.context import Context, resolve_context
from src.models.history_buffer import HistoryBuffer
from src.models.latent_transformer import LatentSTTransformer
from src.models.probabilistic_latent_dynamics import VarianceHead2D, sample_next_latent
from src.models.latent_flow_matching import LatentFlowMatcher


class LatentDynamics(nn.Module, ABC):
    """Abstract base contract for latent dynamics modules in physical world models.

    All latent dynamics implementations must provide:
    1. predict_mean: Predict the deterministic/mean next latent state Z_{t+1}.
    2. sample: Generate one or more stochastic next latent state samples Z_{t+1} ~ p(Z_{t+1} | Z_{t-L+1:t}, Context).
    """

    @abstractmethod
    def predict_mean(
        self,
        latent_history: torch.Tensor,
        context: Optional[Union[Context, Dict[str, Any]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Predict expected (mean) next latent state.

        Args:
            latent_history: Historical latent sequence (B, L, C_z, H_z, W_z).
            context: Context containing physical parameters (Re, Sc), boundary, geometry.

        Returns:
            Mean latent state of shape (B, 1, C_z, H_z, W_z).
        """
        pass

    @abstractmethod
    def sample(
        self,
        latent_history: torch.Tensor,
        context: Optional[Union[Context, Dict[str, Any]]] = None,
        num_samples: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Sample next latent state(s).

        Args:
            latent_history: Historical latent sequence (B, L, C_z, H_z, W_z).
            context: Context containing physical parameters (Re, Sc), boundary, geometry.
            num_samples: Number of stochastic samples K to draw (default: 1).

        Returns:
            If num_samples == 1: Tensor of shape (B, 1, C_z, H_z, W_z).
            If num_samples > 1: Tensor of shape (B, K, 1, C_z, H_z, W_z).
        """
        pass

    def forward(
        self,
        latent_history: torch.Tensor,
        context: Optional[Union[Context, Dict[str, Any]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Standard PyTorch forward delegates to predict_mean."""
        return self.predict_mean(latent_history, context=context, **kwargs)

    def rollout(
        self,
        latent_history: torch.Tensor,
        context: Optional[Union[Context, Dict[str, Any]]] = None,
        horizon: int = 1,
        stochastic: bool = False,
        num_samples: int = 1,
        noise_std: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressively roll out H steps in latent space using HistoryBuffer.

        Args:
            latent_history: Initial latent sequence (B, L, C_z, H_z, W_z).
            context: Conditioning context.
            horizon: Number of future steps H (default: 1).
            stochastic: If True, uses sample(); if False, uses predict_mean().
            num_samples: Sample count K when stochastic=True.
            noise_std: Optional perturbation standard deviation.

        Returns:
            If num_samples == 1: Trajectory of shape (B, H, C_z, H_z, W_z).
            If num_samples > 1: Trajectories of shape (B, K, H, C_z, H_z, W_z).
        """
        resolved_ctx = resolve_context(context)
        b, l, c_z, hz, wz = latent_history.shape

        if not stochastic or num_samples == 1:
            buf = HistoryBuffer(history_length=l)
            buf.reset(latent_history)

            def step_fn(h_in, _c=None):
                if stochastic:
                    return self.sample(h_in, context=resolved_ctx, num_samples=1, **kwargs)
                return self.predict_mean(h_in, context=resolved_ctx, **kwargs)

            return buf.rollout(step_fn, steps=horizon, noise_std=noise_std)

        # Multi-sample isolated autoregressive rollout (Phase 2 & Phase 3 pattern)
        latent_exp = latent_history.repeat_interleave(num_samples, dim=0)  # (B * K, L, C_z, Hz, Wz)
        ctx_exp = None
        if resolved_ctx is not None:
            re_exp = resolved_ctx.re.repeat_interleave(num_samples, dim=0) if resolved_ctx.re is not None else None
            sc_exp = resolved_ctx.sc.repeat_interleave(num_samples, dim=0) if resolved_ctx.sc is not None else None
            ctx_exp = Context.from_re_sc(re=re_exp, sc=sc_exp)

        buf = HistoryBuffer(history_length=l)
        buf.reset(latent_exp)

        sampled_steps = []
        for _ in range(horizon):
            curr_hist = buf.current
            z_step = self.sample(curr_hist, context=ctx_exp, num_samples=1, **kwargs)
            buf.push(z_step)
            sampled_steps.append(z_step)

        # Concatenate along horizon: (B * K, H, C_z, Hz, Wz)
        all_steps = torch.cat(sampled_steps, dim=1)
        # Reshape to (B, K, H, C_z, Hz, Wz)
        return all_steps.view(b, num_samples, horizon, c_z, hz, wz)


class DeterministicLatentDynamics(LatentDynamics):
    """Deterministic latent dynamics backed by LatentSTTransformer.

    Wraps the baseline deterministic latent transformer without any architectural
    or behavioral modifications.
    """

    def __init__(self, transformer: LatentSTTransformer):
        super().__init__()
        self.transformer = transformer

    def predict_mean(
        self,
        latent_history: torch.Tensor,
        context: Optional[Union[Context, Dict[str, Any]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        ctx = resolve_context(context, re=kwargs.get("re"), sc=kwargs.get("sc"))
        re, sc = ctx.to_re_sc() if ctx is not None else (None, None)
        return self.transformer(latent_history, re=re, sc=sc)

    def sample(
        self,
        latent_history: torch.Tensor,
        context: Optional[Union[Context, Dict[str, Any]]] = None,
        num_samples: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        mean = self.predict_mean(latent_history, context=context, **kwargs)
        if num_samples == 1:
            return mean
        # Deterministic dynamics has zero variance: replicates mean across K samples
        # shape: (B, K, 1, C_z, H_z, W_z)
        return mean.unsqueeze(1).expand(-1, num_samples, -1, -1, -1, -1)


class GaussianLatentDynamics(LatentDynamics):
    """Conditional Gaussian probabilistic latent dynamics.

    Wraps LatentSTTransformer + VarianceHead2D with reparameterized sampling
    and exact variance floor governance.
    """

    def __init__(
        self,
        transformer: LatentSTTransformer,
        variance_head: Optional[VarianceHead2D] = None,
    ):
        super().__init__()
        self.transformer = transformer
        if variance_head is not None:
            self.transformer.attach_variance_head(variance_head)

    @property
    def variance_head(self) -> Optional[nn.Module]:
        return self.transformer.variance_head

    def predict_distribution(
        self,
        latent_history: torch.Tensor,
        context: Optional[Union[Context, Dict[str, Any]]] = None,
        variance_head: Optional[nn.Module] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict conditional Gaussian parameters (mu, variance)."""
        ctx = resolve_context(context, re=kwargs.get("re"), sc=kwargs.get("sc"))
        re, sc = ctx.to_re_sc() if ctx is not None else (None, None)
        return self.transformer.predict_distribution(
            z_hist=latent_history,
            re=re,
            sc=sc,
            variance_head=variance_head,
        )

    def predict_mean(
        self,
        latent_history: torch.Tensor,
        context: Optional[Union[Context, Dict[str, Any]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        mu, _ = self.predict_distribution(latent_history, context=context, **kwargs)
        return mu

    def sample(
        self,
        latent_history: torch.Tensor,
        context: Optional[Union[Context, Dict[str, Any]]] = None,
        num_samples: int = 1,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        mu, variance = self.predict_distribution(latent_history, context=context, **kwargs)
        if num_samples == 1:
            return sample_next_latent(mu=mu, variance=variance, generator=generator, seed=seed)

        # Multi-sample draw
        b = mu.shape[0]
        mu_exp = mu.repeat_interleave(num_samples, dim=0)
        var_exp = variance.repeat_interleave(num_samples, dim=0)
        samples = sample_next_latent(mu=mu_exp, variance=var_exp, generator=generator, seed=seed)
        # Reshape to (B, K, 1, C_z, H_z, W_z)
        return samples.view(b, num_samples, *mu.shape[1:])


class FlowMatchingLatentDynamics(LatentDynamics):
    """Residual Latent Conditional Flow Matching dynamics.

    Wraps the deterministic mean backbone (D0) with LatentFlowMatcher for
    continuous-time probability path ODE sampling.
    """

    def __init__(
        self,
        backbone: LatentSTTransformer,
        flow_matcher: LatentFlowMatcher,
    ):
        super().__init__()
        self.backbone = backbone
        self.flow_matcher = flow_matcher

    def predict_mean(
        self,
        latent_history: torch.Tensor,
        context: Optional[Union[Context, Dict[str, Any]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Predict deterministic D0 mean (identical to deterministic baseline)."""
        ctx = resolve_context(context, re=kwargs.get("re"), sc=kwargs.get("sc"))
        re, sc = ctx.to_re_sc() if ctx is not None else (None, None)
        return self.backbone(latent_history, re=re, sc=sc)

    def sample(
        self,
        latent_history: torch.Tensor,
        context: Optional[Union[Context, Dict[str, Any]]] = None,
        num_samples: int = 1,
        num_steps: int = 10,
        solver: str = "midpoint",
        noise_scale: float = 1.0,
        deterministic_fallback: bool = False,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample next latent state by integrating the learned neural ODE."""
        mu = self.predict_mean(latent_history, context=context, **kwargs)
        if deterministic_fallback:
            if num_samples == 1:
                return mu
            return mu.unsqueeze(1).expand(-1, num_samples, -1, -1, -1, -1)

        ctx = resolve_context(context, re=kwargs.get("re"), sc=kwargs.get("sc"))
        re, sc = ctx.to_re_sc() if ctx is not None else (None, None)

        if num_samples == 1:
            return self.flow_matcher.sample_next_latent(
                mu=mu,
                re=re,
                sc=sc,
                num_steps=num_steps,
                solver=solver,
                noise_scale=noise_scale,
                deterministic_fallback=False,
                generator=generator,
                seed=seed,
            )

        # Multi-sample draw
        b = mu.shape[0]
        mu_exp = mu.repeat_interleave(num_samples, dim=0)
        re_exp = re.repeat_interleave(num_samples, dim=0) if re is not None else None
        sc_exp = sc.repeat_interleave(num_samples, dim=0) if sc is not None else None

        samples = self.flow_matcher.sample_next_latent(
            mu=mu_exp,
            re=re_exp,
            sc=sc_exp,
            num_steps=num_steps,
            solver=solver,
            noise_scale=noise_scale,
            deterministic_fallback=False,
            generator=generator,
            seed=seed,
        )
        return samples.view(b, num_samples, *mu.shape[1:])

    def compute_loss(
        self,
        z_next: torch.Tensor,
        latent_history: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        context: Optional[Union[Context, Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute CFM loss on normalized latent residual."""
        if mu is None:
            if latent_history is None:
                raise ValueError("Either mu or latent_history must be provided to compute_loss")
            with torch.no_grad():
                mu = self.predict_mean(latent_history, context=context, **kwargs)

        ctx = resolve_context(context, re=kwargs.get("re"), sc=kwargs.get("sc"))
        re, sc = ctx.to_re_sc() if ctx is not None else (None, None)

        return self.flow_matcher.compute_loss(
            z_next=z_next,
            mu=mu,
            re=re,
            sc=sc,
        )
