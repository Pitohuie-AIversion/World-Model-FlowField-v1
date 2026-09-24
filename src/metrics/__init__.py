"""Metrics package for World-Model-FlowField-v1.

Exports field-level error metrics, multi-step rollout evaluation,
spectral energy analysis, tracer dynamics, and inference benchmarks.
"""

from src.metrics.compute import benchmark_inference, count_parameters
from src.metrics.field import (
    evaluate_field_metrics,
    compute_vrmse,
    compute_nmse,
    compute_max_error,
)
from src.metrics.rollout import evaluate_rollout_trajectory
from src.metrics.spectral import (
    compute_radial_energy_spectrum,
    compute_spectral_error,
)
from src.metrics.tracer import compute_tracer_metrics

__all__ = [
    "benchmark_inference",
    "count_parameters",
    "evaluate_field_metrics",
    "compute_vrmse",
    "compute_nmse",
    "compute_max_error",
    "evaluate_rollout_trajectory",
    "compute_radial_energy_spectrum",
    "compute_spectral_error",
    "compute_tracer_metrics",
]
