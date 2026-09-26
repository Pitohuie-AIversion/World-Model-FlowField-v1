"""End-to-end Latent Forecaster integrating Encoder, LatentSTTransformer, and Decoder."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.history_buffer import HistoryBuffer
from src.models.latent_transformer import LatentSTTransformer
from src.models.probabilistic_latent_dynamics import sample_next_latent


class LatentForecaster(nn.Module):
    """Integrates spatial Encoder, LatentSTTransformer, and spatial Decoder.

    Args:
        encoder: Encoder2D instance.
        transformer: LatentSTTransformer instance.
        decoder: Decoder2D instance.
        freeze_representation: If True, disables gradient computation on Encoder and Decoder.
    """

    def __init__(
        self,
        encoder: Encoder2D,
        transformer: LatentSTTransformer,
        decoder: Decoder2D,
        freeze_representation: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.transformer = transformer
        self.decoder = decoder
        self.freeze_representation = freeze_representation

        if freeze_representation:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False

    def forward_single_step(
        self,
        q_hist: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        sc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """q_hist: (B, L, 4, Ny, Nx) -> q_pred: (B, 1, 4, Ny, Nx)."""
        if self.freeze_representation:
            with torch.no_grad():
                z_hist = self.encoder(q_hist)
        else:
            z_hist = self.encoder(q_hist)

        z_next = self.transformer(z_hist, re=re, sc=sc)
        q_next = self.decoder(z_next)
        return q_next

    def forward_rollout(
        self,
        q_hist: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        sc: Optional[torch.Tensor] = None,
        horizon: int = 30,
        pushforward_steps: int = 0,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Roll out H steps entirely in latent space, then decode.

        Args:
            q_hist: History physical fields of shape (B, L, C, Ny, Nx).
            re: Optional Reynolds number tensor (B,).
            sc: Optional Schmidt number tensor (B,).
            horizon: Prediction horizon H (default: 30).
            pushforward_steps: Number of warmup rollout steps executed without gradients (default: 0).
            noise_std: Standard deviation of Gaussian perturbation injected during rollout (default: 0.0).

        Returns:
            q_rollout: Predicted physical fields of shape (B, horizon, C, Ny, Nx).
        """
        if self.freeze_representation:
            with torch.no_grad():
                z_hist = self.encoder(q_hist)
        else:
            z_hist = self.encoder(q_hist)

        buf = HistoryBuffer(history_length=q_hist.shape[1])
        buf.reset(z_hist)

        def step_fn(hist_z, _cond=None):
            return self.transformer(hist_z, re=re, sc=sc)

        if pushforward_steps > 0:
            buf.pushforward(step_fn, steps=pushforward_steps, noise_std=noise_std)

        z_rollout = buf.rollout(step_fn, steps=horizon, noise_std=noise_std)  # (B, H, C_z, H_z, W_z)
        q_rollout = self.decoder(z_rollout)
        return q_rollout

    def forward(
        self,
        q_hist: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        sc: Optional[torch.Tensor] = None,
        horizon: int = 1,
        pushforward_steps: int = 0,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Unified forward interface matching baseline models.

        Args:
            q_hist: History physical fields of shape (B, L, C, Ny, Nx).
            re: Optional Reynolds number tensor (B,).
            sc: Optional Schmidt number tensor (B,).
            horizon: Prediction horizon H (default: 1).
            pushforward_steps: Number of warmup rollout steps without gradients (default: 0).
            noise_std: Standard deviation of Gaussian noise injected (default: 0.0).

        Returns:
            Predicted physical fields of shape (B, H, C, Ny, Nx).
        """
        if horizon == 1 and pushforward_steps == 0:
            return self.forward_single_step(q_hist, re=re, sc=sc)
        else:
            return self.forward_rollout(
                q_hist,
                re=re,
                sc=sc,
                horizon=horizon,
                pushforward_steps=pushforward_steps,
                noise_std=noise_std,
            )

    def attach_variance_head(self, variance_head: nn.Module) -> None:
        """Attach variance projection head to the transformer module."""
        self.transformer.attach_variance_head(variance_head)

    @property
    def has_variance_head(self) -> bool:
        """Return True if a variance head is attached to the transformer."""
        return self.transformer.has_variance_head

    def predict_distribution_single_step(
        self,
        q_hist: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        sc: Optional[torch.Tensor] = None,
        variance_head: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode physical history and predict conditional (mu, variance) in latent space.

        Args:
            q_hist: History physical fields of shape (B, L, C, Ny, Nx).
            re: Optional Reynolds number tensor (B,).
            sc: Optional Schmidt number tensor (B,).
            variance_head: Optional variance head override.

        Returns:
            mu: Predicted mean latent state of shape (B, 1, C_z, H_z, W_z).
            variance: Predicted conditional variance of shape (B, 1, C_z, H_z, W_z).
        """
        if self.freeze_representation:
            with torch.no_grad():
                z_hist = self.encoder(q_hist)
        else:
            z_hist = self.encoder(q_hist)

        return self.transformer.predict_distribution(
            z_hist=z_hist,
            re=re,
            sc=sc,
            variance_head=variance_head,
        )

    def sample_rollout(
        self,
        q_hist: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        sc: Optional[torch.Tensor] = None,
        horizon: int = 30,
        num_samples: int = 8,
        seed: Optional[int] = 42,
        generator: Optional[torch.Generator] = None,
        variance_head: Optional[nn.Module] = None,
        decode_samples: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Execute single-source multi-trajectory probabilistic rollout in latent space.

        Ensures each sample trajectory k in {1..num_samples} maintains its own isolated
        autoregressive history buffer without cross-trajectory contamination.

        Args:
            q_hist: History physical fields of shape (B, L, C, Ny, Nx).
            re: Optional Reynolds number tensor (B,).
            sc: Optional Schmidt number tensor (B,).
            horizon: Prediction horizon H (default: 30).
            num_samples: Number of sample trajectories K to generate (default: 8).
            seed: Optional integer seed for reproducibility (default: 42).
            generator: Optional PyTorch Generator.
            variance_head: Optional variance head override.
            decode_samples: If True, decodes sample latent trajectories to physical space.

        Returns:
            Dictionary containing:
                - "deterministic_rollout": (B, H, C, Ny, Nx) physical fields from deterministic mean.
                - "sample_trajectories": (B, K, H, C, Ny, Nx) decoded physical fields (if decode_samples=True).
                - "ensemble_mean": (B, H, C, Ny, Nx) mean across K physical trajectories (if decode_samples=True).
                - "latent_samples": (B, K, H, C_z, H_z, W_z) sampled latent trajectories.
                - "latent_variances": (B, K, H, C_z, H_z, W_z) conditional variance at each step.
        """
        b, l, c_in, ny, nx = q_hist.shape
        k = num_samples

        if self.freeze_representation:
            with torch.no_grad():
                z_hist = self.encoder(q_hist)
        else:
            z_hist = self.encoder(q_hist)

        # 1. Deterministic baseline rollout for reference
        buf_det = HistoryBuffer(history_length=l)
        buf_det.reset(z_hist)

        def det_step(hz, _c=None):
            return self.transformer(hz, re=re, sc=sc)

        z_det_rollout = buf_det.rollout(det_step, steps=horizon, noise_std=0.0)
        q_det_rollout = self.decoder(z_det_rollout)

        # 2. Multi-sample probabilistic rollout
        # Expand batch by K so all K trajectories advance in parallel with complete isolation
        z_hist_exp = z_hist.repeat_interleave(k, dim=0)  # (B * K, L, C_z, Hz, Wz)
        re_exp = re.repeat_interleave(k, dim=0) if re is not None else None
        sc_exp = sc.repeat_interleave(k, dim=0) if sc is not None else None

        if seed is not None and generator is None:
            gen = torch.Generator(device=q_hist.device if q_hist.device.type != "mps" else "cpu")
            gen.manual_seed(seed)
        else:
            gen = generator

        buf_prob = HistoryBuffer(history_length=l)
        buf_prob.reset(z_hist_exp)

        sampled_latents = []
        sampled_variances = []

        for _ in range(horizon):
            curr_hist = buf_prob.current  # (B * K, L, C_z, Hz, Wz)
            mu_t, var_t = self.transformer.predict_distribution(
                z_hist=curr_hist,
                re=re_exp,
                sc=sc_exp,
                variance_head=variance_head,
            )  # (B * K, 1, C_z, Hz, Wz)
            z_sample_t = sample_next_latent(mu=mu_t, variance=var_t, generator=gen)
            buf_prob.push(z_sample_t)
            sampled_latents.append(z_sample_t)
            sampled_variances.append(var_t)

        # Concat along horizon dimension: (B * K, H, C_z, Hz, Wz)
        z_samples_all = torch.cat(sampled_latents, dim=1)
        var_samples_all = torch.cat(sampled_variances, dim=1)

        hz, wz = z_samples_all.shape[-2], z_samples_all.shape[-1]
        c_z = z_samples_all.shape[2]

        # Reshape to (B, K, H, C_z, Hz, Wz)
        z_samples_reshaped = z_samples_all.view(b, k, horizon, c_z, hz, wz)
        var_samples_reshaped = var_samples_all.view(b, k, horizon, c_z, hz, wz)

        results = {
            "deterministic_rollout": q_det_rollout,
            "latent_samples": z_samples_reshaped,
            "latent_variances": var_samples_reshaped,
        }

        if decode_samples:
            flat_z = z_samples_all.view(b * k * horizon, c_z, hz, wz).unsqueeze(1)
            q_samples_flat = self.decoder(flat_z).squeeze(1)  # (B*K*H, C_in, Ny, Nx)
            q_samples = q_samples_flat.view(b, k, horizon, c_in, ny, nx)
            results["sample_trajectories"] = q_samples
            results["ensemble_mean"] = q_samples.mean(dim=1)

        return results

