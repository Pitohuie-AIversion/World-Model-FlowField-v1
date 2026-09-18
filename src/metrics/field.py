"""Field error metrics: nMSE, VRMSE (The Well benchmark metric), and Max Error."""

from typing import Dict
import torch


def compute_nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalized Mean Squared Error (nMSE): ||pred - target||^2 / ||target||^2."""
    diff_norm_sq = torch.sum((pred - target) ** 2, dim=(-2, -1))
    target_norm_sq = torch.sum(target**2, dim=(-2, -1))
    return torch.mean(diff_norm_sq / (target_norm_sq + eps))


def compute_vrmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Variance-scaled Root Mean Squared Error (VRMSE) as defined in The Well benchmark:

    VRMSE = sqrt( mean((pred - target)^2) / (Var(target) + eps) )
    Predicting the constant mean value of target field yields a score of ~1.0.
    """
    mse = torch.mean((pred - target) ** 2, dim=(-2, -1))
    var = torch.var(target, dim=(-2, -1), unbiased=False)
    vrmse = torch.sqrt(mse / (var + eps))
    return torch.mean(vrmse)


def compute_max_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Maximum absolute error: max |pred - target| across spatial points."""
    return torch.max(torch.abs(pred - target))


def evaluate_field_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_names: tuple = ("u", "v", "p", "s"),
) -> Dict[str, float]:
    """Computes comprehensive field errors per physical channel.

    Args:
        pred: Tensor of shape (..., C, Ny, Nx).
        target: Tensor of shape (..., C, Ny, Nx).
        channel_names: Tuple of names for each channel.

    Returns:
        Dictionary mapping metric names (e.g. 'vrmse_u', 'nmse_p') to float values.
    """
    metrics = {}
    c_dim = -3
    num_channels = pred.shape[c_dim]

    for c in range(num_channels):
        name = channel_names[c] if c < len(channel_names) else f"c{c}"
        p_c = pred.select(c_dim, c)
        t_c = target.select(c_dim, c)

        metrics[f"vrmse_{name}"] = float(compute_vrmse(p_c, t_c).item())
        metrics[f"nmse_{name}"] = float(compute_nmse(p_c, t_c).item())
        metrics[f"max_err_{name}"] = float(compute_max_error(p_c, t_c).item())

    # Overall VRMSE average
    metrics["vrmse_mean"] = sum(metrics[f"vrmse_{name}"] for name in channel_names) / len(channel_names)
    return metrics
