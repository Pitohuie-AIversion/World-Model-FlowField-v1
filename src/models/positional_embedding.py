"""2D Sinusoidal / Fourier positional embeddings for spatio-temporal flow models."""

from typing import Dict, Tuple
import torch

_POS_EMBED_2D_CACHE: Dict[Tuple[int, int, int, torch.device, torch.dtype], torch.Tensor] = {}


def build_2d_sincos_position_embedding(
    embed_dim: int,
    height: int,
    width: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    temperature: float = 10000.0,
) -> torch.Tensor:
    """Generate 2D sinusoidal (sine-cosine) spatial positional embeddings.

    Decomposes `embed_dim` into two halves:
        - dim_h for vertical spatial coordinate y (height)
        - dim_w for horizontal spatial coordinate x (width)
    Produces parameter-free, deterministic coordinate embeddings compatible with
    arbitrary grid resolutions, breaking spatial permutation equivariance without
    adding parameters to state_dict.

    Args:
        embed_dim: Total embedding channel dimension (typically even, e.g. 128, 256, 512).
        height: Spatial grid height (e.g. H_z or Ny // py).
        width: Spatial grid width (e.g. W_z or Nx // px).
        device: Target torch device.
        dtype: Target torch data type.
        temperature: Frequency base scaling factor (default: 10000.0).

    Returns:
        pos_embed: Tensor of shape (1, 1, height * width, embed_dim) formatted
            for direct broadcasting over (B, L, N, D) sequence representations.
    """
    dim_h = embed_dim // 2
    dim_w = embed_dim - dim_h

    # 1. Height (y) positional embedding
    grid_y = torch.arange(height, device=device, dtype=dtype)
    omega_h = torch.arange(dim_h // 2, device=device, dtype=dtype)
    omega_h = 1.0 / (temperature ** (2.0 * omega_h / dim_h))
    out_y = torch.outer(grid_y, omega_h)  # (height, dim_h // 2)
    pe_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=-1)  # (height, 2 * (dim_h // 2))
    if pe_y.shape[-1] < dim_h:
        pe_y = torch.nn.functional.pad(pe_y, (0, dim_h - pe_y.shape[-1]))

    # 2. Width (x) positional embedding
    grid_x = torch.arange(width, device=device, dtype=dtype)
    omega_w = torch.arange(dim_w // 2, device=device, dtype=dtype)
    omega_w = 1.0 / (temperature ** (2.0 * omega_w / dim_w))
    out_x = torch.outer(grid_x, omega_w)  # (width, dim_w // 2)
    pe_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=-1)  # (width, 2 * (dim_w // 2))
    if pe_x.shape[-1] < dim_w:
        pe_x = torch.nn.functional.pad(pe_x, (0, dim_w - pe_x.shape[-1]))

    # 3. Outer grid composition: pe_y -> (height, 1, dim_h), pe_x -> (1, width, dim_w)
    pe_y = pe_y.unsqueeze(1).expand(height, width, dim_h)
    pe_x = pe_x.unsqueeze(0).expand(height, width, dim_w)
    pe_2d = torch.cat([pe_y, pe_x], dim=-1)  # (height, width, embed_dim)

    # Flatten spatial grid to (1, 1, height * width, embed_dim)
    return pe_2d.reshape(1, 1, height * width, embed_dim)


def get_2d_sincos_position_embedding(
    embed_dim: int,
    height: int,
    width: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    temperature: float = 10000.0,
) -> torch.Tensor:
    """Retrieve or compute cached 2D sinusoidal spatial positional embeddings."""
    key = (embed_dim, height, width, device, dtype)
    cached = _POS_EMBED_2D_CACHE.get(key)
    if cached is not None:
        return cached

    pe = build_2d_sincos_position_embedding(
        embed_dim=embed_dim,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
        temperature=temperature,
    )
    _POS_EMBED_2D_CACHE[key] = pe
    return pe


def clear_pos_embed_cache() -> None:
    """Clear cached 2D positional embeddings."""
    _POS_EMBED_2D_CACHE.clear()
