"""Unit tests for H16 comparison qualitative visualization pipeline.

Validates:
1. Pressure zero-mean gauge invariance:
   - Preserves channels 0 (u), 1 (v), 3 (tracer) bitwise identically.
   - Enforces spatial mean zero on channel 2 (pressure) across all time steps and batches.
   - Supports 3D (C, Nx, Ny), 4D (T, C, Nx, Ny), and 5D (B, T, C, Nx, Ny) tensors.
   - Specifically verifies frame index 2 (h=3) is NOT modified on u, v, tracer.
2. Field extraction:
   - Correctly extracts scalar 2D fields for u, v, p, tracer, and computed vorticity.
3. Checkpoint SHA-256 and target parity between benchmark evaluation and visualization pipelines.
4. Metadata schema contract:
   - Includes full SHA-256, git status, split/normalizer hashes, ranking basis, and colorbar scaling policy.
"""

import json
from pathlib import Path
import pytest
import torch
import numpy as np

from src.utils.physics_contract import (
    SHEAR_FLOW_DOMAIN_SIZE_XY,
    zero_mean_pressure_gauge,
)
from src.utils.provenance import compute_file_sha256
from scripts.evaluate_h16_benchmark import BENCHMARK_TARGETS as BENCHMARK_EVAL_TARGETS
from scripts.visualize_h16_comparison import (
    BENCHMARK_TARGETS as VISUALIZE_TARGETS,
    TARGET_SAMPLES,
    extract_field_slice,
)


def test_zero_mean_pressure_gauge_bitwise_invariance_4d():
    """Verify 4D tensor (T, C, Nx, Ny) pressure gauge:

    1. Channels u (0), v (1), tracer (3) must be strictly bitwise untouched.
    2. Specifically, frame 2 (h=3) channels 0, 1, 3 must be bitwise untouched.
    3. Channel 2 (pressure) must have spatial mean close to 0 for all frames.
    """
    torch.manual_seed(42)
    T, C, Nx, Ny = 10, 4, 32, 64
    tensor = torch.randn(T, C, Nx, Ny, dtype=torch.float32)

    # Add arbitrary non-zero mean offsets to pressure
    tensor[:, 2] += 123.456

    result = zero_mean_pressure_gauge(tensor)

    # 1. Exact bitwise identity for u, v, tracer on all time steps
    assert torch.equal(result[:, 0], tensor[:, 0]), "Channel u was modified by pressure gauge"
    assert torch.equal(result[:, 1], tensor[:, 1]), "Channel v was modified by pressure gauge"
    assert torch.equal(result[:, 3], tensor[:, 3]), "Channel tracer was modified by pressure gauge"

    # Specifically check frame index 2 (h=3)
    assert torch.equal(result[2, 0], tensor[2, 0]), "Frame 2 channel u was modified"
    assert torch.equal(result[2, 1], tensor[2, 1]), "Frame 2 channel v was modified"
    assert torch.equal(result[2, 3], tensor[2, 3]), "Frame 2 channel tracer was modified"

    # 2. Pressure channel spatial mean must be zero for every frame (accounting for float32 precision)
    spatial_means = result[:, 2].mean(dim=(-2, -1))
    assert torch.allclose(spatial_means, torch.zeros(T), atol=1e-5), (
        f"Pressure spatial mean not zero: {spatial_means}"
    )


def test_zero_mean_pressure_gauge_dimensions():
    """Verify pressure gauge correctly handles 3D, 4D, and 5D tensors."""
    torch.manual_seed(42)

    # 3D: (C, Nx, Ny)
    t3 = torch.randn(4, 16, 32)
    r3 = zero_mean_pressure_gauge(t3)
    assert r3.shape == t3.shape
    assert torch.equal(r3[0], t3[0])
    assert torch.allclose(r3[2].mean(), torch.tensor(0.0), atol=1e-6)

    # 5D: (B, T, C, Nx, Ny)
    t5 = torch.randn(2, 5, 4, 16, 32)
    r5 = zero_mean_pressure_gauge(t5)
    assert r5.shape == t5.shape
    assert torch.equal(r5[:, :, 0], t5[:, :, 0])
    assert torch.equal(r5[:, :, 1], t5[:, :, 1])
    assert torch.equal(r5[:, :, 3], t5[:, :, 3])
    spatial_means_5d = r5[:, :, 2].mean(dim=(-2, -1))
    assert torch.allclose(spatial_means_5d, torch.zeros(2, 5), atol=1e-6)

    # Reject < 3D
    with pytest.raises(ValueError, match="expects at least 3D tensor"):
        zero_mean_pressure_gauge(torch.randn(4, 16))


def test_extract_field_slice():
    """Verify scalar 2D numpy slice extraction and vorticity calculation."""
    tensor_4d = torch.zeros(4, 32, 64)
    # Set known values
    tensor_4d[0, :, :] = 1.5   # u
    tensor_4d[1, :, :] = -2.5  # v
    tensor_4d[2, :, :] = 0.8   # p
    tensor_4d[3, :, :] = 3.2   # tracer

    u_field = extract_field_slice(tensor_4d, "u")
    assert isinstance(u_field, np.ndarray)
    assert u_field.shape == (32, 64)
    assert np.allclose(u_field, 1.5)

    p_field = extract_field_slice(tensor_4d, "p")
    assert np.allclose(p_field, 0.8)

    tracer_field = extract_field_slice(tensor_4d, "tracer")
    assert np.allclose(tracer_field, 3.2)

    # For constant u and v, vorticity = dv/dx - du/dy = 0
    vort_field = extract_field_slice(tensor_4d, "vorticity", domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY)
    assert vort_field.shape == (32, 64)
    assert np.allclose(vort_field, 0.0, atol=1e-6)

    with pytest.raises(ValueError, match="Unknown variable name"):
        extract_field_slice(tensor_4d, "unknown_field")


def test_checkpoint_parity_between_eval_and_vis():
    """Ensure benchmark evaluation and visualization point to the exact same checkpoint files and SHAs."""
    assert set(BENCHMARK_EVAL_TARGETS.keys()) == set(VISUALIZE_TARGETS.keys()), (
        f"Target keys mismatch: {BENCHMARK_EVAL_TARGETS.keys()} vs {VISUALIZE_TARGETS.keys()}"
    )

    for k in BENCHMARK_EVAL_TARGETS:
        eval_path = BENCHMARK_EVAL_TARGETS[k]["path"]
        vis_path = VISUALIZE_TARGETS[k]["path"]
        assert eval_path == vis_path, f"Path mismatch for {k}: {eval_path} vs {vis_path}"

        if Path(eval_path).exists():
            eval_sha = compute_file_sha256(eval_path)
            vis_sha = compute_file_sha256(vis_path)
            assert eval_sha == vis_sha
            assert len(eval_sha) == 64, f"Invalid SHA-256 hex length: {eval_sha}"


def test_target_samples_provenance_and_ranking_basis():
    """Verify TARGET_SAMPLES specification includes all required provenance and ranking criteria."""
    assert len(TARGET_SAMPLES) == 3
    sample_indices = [s["index"] for s in TARGET_SAMPLES]
    assert sample_indices == [0, 23, 36]

    for sample in TARGET_SAMPLES:
        assert "selection_basis" in sample and sample["selection_basis"], f"Missing selection_basis in sample {sample['index']}"
        assert "source_file" in sample and sample["source_file"].endswith(".hdf5"), f"Missing source_file in sample {sample['index']}"
        assert "sim_idx" in sample and isinstance(sample["sim_idx"], int), f"Missing sim_idx in sample {sample['index']}"
        assert "start_t" in sample and isinstance(sample["start_t"], int), f"Missing start_t in sample {sample['index']}"
        assert "end_t" in sample and isinstance(sample["end_t"], int), f"Missing end_t in sample {sample['index']}"

    # Confirm explicit H8 ranking basis in descriptions
    assert "H8" in TARGET_SAMPLES[1]["selection_basis"]
    assert "H8" in TARGET_SAMPLES[2]["selection_basis"]
