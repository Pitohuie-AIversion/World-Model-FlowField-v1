"""Computational cost metrics: parameter count, peak GPU memory, and inference throughput."""

import time
from typing import Callable, Dict
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """Count total and trainable parameters in Millions."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total_m": total_params / 1e6,
        "params_trainable_m": trainable_params / 1e6,
    }


def benchmark_inference(
    forward_fn: Callable[[], torch.Tensor],
    warmup_steps: int = 5,
    benchmark_steps: int = 20,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, float]:
    """Measure inference throughput (frames/sec) and peak GPU memory (GB)."""
    # Warmup
    for _ in range(warmup_steps):
        _ = forward_fn()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()
    for _ in range(benchmark_steps):
        out = forward_fn()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start_time
    total_time_ms = (elapsed / benchmark_steps) * 1000.0

    # Determine batch size and frames produced
    batch_size = out.shape[0] if hasattr(out, "shape") else 1
    frames = out.shape[1] if (hasattr(out, "ndim") and out.ndim == 5) else 1
    total_frames = batch_size * frames
    fps = (total_frames * benchmark_steps) / elapsed

    results = {
        "latency_per_call_ms": float(total_time_ms),
        "throughput_fps": float(fps),
    }

    if device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        results["peak_memory_gb"] = float(peak_mem_gb)

    return results
