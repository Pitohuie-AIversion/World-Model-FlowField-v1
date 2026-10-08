"""Comprehensive test suite for StocBench data audit and contract compatibility.

Verifies:
1. Legal single-channel training data reading and axis semantics (StocBenchTrainDataset).
2. Legal bifurcation file keys, shapes, and reference member axis K_ref.
3. Fail-fast behavior when 'init' or 'raw' keys are missing in bifurcation files.
4. Fail-fast behavior on NaN, Inf, empty arrays, or illegal dimensionality.
5. Fail-fast detection of degenerate or identical futures (zero bifurcation).
6. Statistical mean/std consistency check (verifying ddof=0 population std).
7. Complete retention of dataset identity and provenance metadata.
8. Train/test isolation: ensuring test bifurcation data is strictly isolated from training.
9. Single-channel WorldModelBatch construction, mapping protocol, and device migration.
10. Zero regression on existing shear_flow dataset and batch contract behavior.
"""

import os
from pathlib import Path
import tempfile
import pytest
import numpy as np
import torch

from src.contracts.batch import WorldModelBatch, shear_flow_batch_adapter
from src.contracts.context import Context
from src.contracts.state_spec import StateSpec, SHEAR_FLOW_STATE_SPEC
from src.data.stocbench_dataset import (
    STOCBENCH_SAMPLE_DT,
    STOCBENCH_SOLVER_DT,
    STOCBENCH_STATE_SPEC,
    StocBenchReferenceEnsemble,
    StocBenchTrainDataset,
    collate_stocbench_batch,
    stocbench_batch_adapter,
)


# Helper: create mock synthetic StocBench trajectory file (traj_seed_*.npy)
def create_mock_traj_file(file_path: Path, n_sims: int = 4, t_frames: int = 10, ny: int = 64, nx: int = 64):
    shape = (n_sims, t_frames, 1, ny, nx)
    # Generate continuous realistic vortex data
    data = np.random.randn(*shape).astype(np.float32)
    np.save(file_path, data)


# Helper: create mock synthetic StocBench bifurcation file (step_seed_*.npz)
def create_mock_step_file(
    file_path: Path,
    k_ref: int = 50,
    ny: int = 64,
    nx: int = 64,
    degenerate: bool = False,
    include_mean_std: bool = True,
    mean_noise: float = 0.0,
):
    init = np.random.randn(1, ny, nx).astype(np.float32)
    if degenerate:
        # All futures identical to first future
        single_fut = np.random.randn(1, 1, ny, nx).astype(np.float32)
        raw = np.repeat(single_fut, k_ref, axis=0)  # (K, 1, 1, ny, nx)
    else:
        # Different stochastic futures
        raw = np.random.randn(k_ref, 1, 1, ny, nx).astype(np.float32)

    kwargs = {"init": init, "raw": raw}
    if include_mean_std:
        # ddof=0 population standard deviation as per StocBench solver/data implementation
        mean = np.mean(raw, axis=0) + mean_noise
        std = np.std(raw, axis=0, ddof=0)
        kwargs["mean"] = mean
        kwargs["std"] = std

    np.savez(file_path, **kwargs)


# ==============================================================================
# 1. Legal Single-Channel Training Data Reading & Axis Semantics
# ==============================================================================

def test_stocbench_train_dataset_axes_and_sliding_windows():
    """Verify StocBenchTrainDataset reading, axes semantics, and sliding windows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_path = Path(tmpdir) / "traj_seed_42.npy"
        create_mock_traj_file(traj_path, n_sims=3, t_frames=8, ny=64, nx=64)

        ds = StocBenchTrainDataset(
            file_path=traj_path,
            history_length=1,
            horizon=1,
            stride=1,
            dataset_id="stocbench_test",
            dataset_revision="test_rev_123",
        )

        # Windows per trajectory: (8 - (1 + 1)) // 1 + 1 = 7 windows
        # 3 trajectories * 7 windows = 21 samples
        assert len(ds) == 21

        sample = ds[0]
        assert "history" in sample
        assert "future" in sample
        assert sample["history"].shape == (1, 1, 64, 64)  # (L=1, C=1, Ny, Nx)
        assert sample["future"].shape == (1, 1, 64, 64)   # (H=1, C=1, Ny, Nx)
        assert sample["history"].dtype == torch.float32
        assert sample["future"].dtype == torch.float32
        assert sample["dt"].item() == pytest.approx(STOCBENCH_SAMPLE_DT)
        assert sample["dataset_id"] == "stocbench_test"
        assert sample["dataset_revision"] == "test_rev_123"
        assert sample["state_variables"] == ("vorticity",)


# ==============================================================================
# 2. Legal Bifurcation File Keys, Shapes, and Member Axis
# ==============================================================================

def test_stocbench_reference_ensemble_keys_and_shapes():
    """Verify StocBenchReferenceEnsemble correctly parses bifurcation file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        step_path = Path(tmpdir) / "step_seed_100.npz"
        create_mock_step_file(step_path, k_ref=100, ny=64, nx=64)

        ref = StocBenchReferenceEnsemble(step_path)
        assert ref.num_members == 100
        assert ref.spatial_shape == (64, 64)

        init_t, raw_t = ref.get_canonical_tensors()
        # Condition history: (B=1, L=1, C=1, Ny, Nx)
        assert init_t.shape == (1, 1, 1, 64, 64)
        # Reference ensemble: (B=1, K_ref=100, H=1, C=1, Ny, Nx)
        assert raw_t.shape == (1, 100, 1, 1, 64, 64)


# ==============================================================================
# 3. Missing 'init' or 'raw' Keys Must Fail Fast
# ==============================================================================

def test_stocbench_reference_missing_keys_fail_fast():
    """Verify that missing 'init' or 'raw' keys immediately raises KeyError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Missing 'raw'
        p1 = Path(tmpdir) / "missing_raw.npz"
        np.savez(p1, init=np.zeros((1, 64, 64), dtype=np.float32))
        with pytest.raises(KeyError, match="Missing required key 'raw'"):
            StocBenchReferenceEnsemble(p1)

        # Missing 'init'
        p2 = Path(tmpdir) / "missing_init.npz"
        np.savez(p2, raw=np.zeros((10, 1, 1, 64, 64), dtype=np.float32))
        with pytest.raises(KeyError, match="Missing required key 'init'"):
            StocBenchReferenceEnsemble(p2)


# ==============================================================================
# 4. NaN, Inf, Empty Array, or Illegal Dimension Fail-Fast
# ==============================================================================

def test_stocbench_fail_fast_on_nan_inf_and_dimensions():
    """Verify rejection of NaN, Inf, empty arrays, or mismatched shapes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Trajectory with NaN
        p_nan = Path(tmpdir) / "traj_nan.npy"
        nan_data = np.zeros((2, 5, 1, 64, 64), dtype=np.float32)
        nan_data[0, 1, 0, 10, 10] = np.nan
        np.save(p_nan, nan_data)

        ds = StocBenchTrainDataset(p_nan, history_length=1, horizon=1)
        # Reading window with NaN should raise ValueError
        with pytest.raises(ValueError, match="Non-finite"):
            _ = ds[0]

        # Trajectory with wrong channels (e.g. 4 channels instead of 1)
        p_ch4 = Path(tmpdir) / "traj_ch4.npy"
        ch4_data = np.zeros((2, 5, 4, 64, 64), dtype=np.float32)
        np.save(p_ch4, ch4_data)
        with pytest.raises(ValueError, match="Expected 1 vorticity channel, got 4"):
            StocBenchTrainDataset(p_ch4)

        # Reference with Inf
        p_inf = Path(tmpdir) / "ref_inf.npz"
        raw_inf = np.random.randn(10, 1, 1, 64, 64).astype(np.float32)
        raw_inf[0, 0, 0, 0, 0] = np.inf
        np.savez(p_inf, init=np.zeros((1, 64, 64), dtype=np.float32), raw=raw_inf)
        with pytest.raises(ValueError, match="Non-finite"):
            StocBenchReferenceEnsemble(p_inf)


# ==============================================================================
# 5. Degenerate or Identical Futures Must Fail Fast
# ==============================================================================

def test_stocbench_fail_fast_on_degenerate_futures():
    """Verify that identical future realizations are rejected as degenerate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_deg = Path(tmpdir) / "degenerate.npz"
        create_mock_step_file(p_deg, k_ref=50, degenerate=True)

        with pytest.raises(ValueError, match="Degenerate reference ensemble.*all members are virtually identical"):
            StocBenchReferenceEnsemble(p_deg)


# ==============================================================================
# 6. Statistical Consistency Verification (Mean / Std with ddof=0)
# ==============================================================================

def test_stocbench_statistical_consistency_check():
    """Verify statistical consistency check detects intentional mean/std divergence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Valid consistent file
        p_good = Path(tmpdir) / "step_good.npz"
        create_mock_step_file(p_good, k_ref=100, include_mean_std=True, mean_noise=0.0)
        ref_good = StocBenchReferenceEnsemble(p_good)
        res = ref_good.verify_statistical_consistency(atol=1e-4)
        assert res["max_mean_discrepancy"] < 1e-4
        assert res["max_std_discrepancy"] < 1e-4

        # Inconsistent file (tampered mean)
        p_bad = Path(tmpdir) / "step_bad.npz"
        create_mock_step_file(p_bad, k_ref=100, include_mean_std=True, mean_noise=0.1)
        ref_bad = StocBenchReferenceEnsemble(p_bad)
        with pytest.raises(ValueError, match="Statistical mean mismatch"):
            ref_bad.verify_statistical_consistency(atol=1e-4)


# ==============================================================================
# 7. Retention of Dataset Identity and Provenance Metadata
# ==============================================================================

def test_stocbench_provenance_metadata_retention():
    """Verify that dataset_id, revision, and physical_dt are preserved end-to-end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_path = Path(tmpdir) / "traj_seed_42.npy"
        create_mock_traj_file(traj_path, n_sims=2, t_frames=5)

        ds = StocBenchTrainDataset(
            file_path=traj_path,
            dataset_id="stocbench_v1",
            dataset_revision="3a5f50398cf6d14f",
        )
        sample = ds[0]

        batch = stocbench_batch_adapter({
            "history": sample["history"].unsqueeze(0),
            "future": sample["future"].unsqueeze(0),
            "dt": sample["dt"],
            "dataset_id": sample["dataset_id"],
            "dataset_revision": sample["dataset_revision"],
            "source_file": sample["source_file"],
            "traj_idx": sample["traj_idx"],
            "start_t": sample["start_t"],
            "state_variables": sample["state_variables"],
        })

        assert batch.metadata["dataset_id"] == "stocbench_v1"
        assert batch.metadata["dataset_revision"] == "3a5f50398cf6d14f"
        assert batch.coordinates["dt"].item() == pytest.approx(0.5)
        assert batch.metadata["source_file"] == "traj_seed_42.npy"


# ==============================================================================
# 8. Train / Test Usage Isolation Contract
# ==============================================================================

def test_stocbench_train_test_usage_isolation():
    """Assert that step_seed_*.npz files cannot be accidentally loaded as StocBenchTrainDataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        step_path = Path(tmpdir) / "step_seed_100.npz"
        create_mock_step_file(step_path, k_ref=10)

        # Attempting to load .npz into StocBenchTrainDataset must fail
        with pytest.raises(ValueError, match="Invalid trajectory shape"):
            StocBenchTrainDataset(step_path)


# ==============================================================================
# 9. Single-Channel WorldModelBatch Construction & Device Migration
# ==============================================================================

def test_stocbench_world_model_batch_contract_and_to_device():
    """Verify WorldModelBatch handles STOCBENCH_STATE_SPEC, mapping, and .to()."""
    spec = STOCBENCH_STATE_SPEC
    assert spec.num_channels == 1
    assert spec.variables == ("vorticity",)

    hist = torch.randn(2, 1, 1, 64, 64)
    fut = torch.randn(2, 1, 1, 64, 64)

    batch = WorldModelBatch(
        history=hist,
        state_spec=spec,
        future=fut,
        context=Context(boundary="periodic"),
        coordinates={"dt": torch.tensor(0.5)},
        metadata={"dataset_id": "stocbench"},
    )

    assert batch.batch_size == 2
    assert batch.state_spec.num_channels == 1
    assert batch["history"].shape == (2, 1, 1, 64, 64)
    assert batch["future"].shape == (2, 1, 1, 64, 64)
    assert batch["dt"].item() == pytest.approx(0.5)
    assert batch["boundary"] == "periodic"

    # Device migration
    batch_cpu = batch.to(device="cpu", dtype=torch.float32)
    assert batch_cpu.history.device.type == "cpu"
    assert batch_cpu.future.device.type == "cpu"

    # Channel mismatch validation
    mismatched_hist = torch.randn(2, 1, 4, 64, 64)
    with pytest.raises(ValueError, match="history channel count mismatch"):
        WorldModelBatch(history=mismatched_hist, state_spec=spec)


# ==============================================================================
# 10. Existing shear_flow Path Behavior Invariance (Zero Regression)
# ==============================================================================

def test_existing_shear_flow_contracts_unaffected():
    """Verify that SHEAR_FLOW_STATE_SPEC and shear_flow_batch_adapter behave identically."""
    shear_spec = SHEAR_FLOW_STATE_SPEC
    assert shear_spec.num_channels == 4
    assert shear_spec.variables == ("u", "v", "p", "s")

    # Construct shear_flow batch
    hist_4ch = torch.randn(2, 4, 4, 32, 64)
    fut_4ch = torch.randn(2, 1, 4, 32, 64)

    batch = shear_flow_batch_adapter({
        "history": hist_4ch,
        "future": fut_4ch,
        "re": torch.tensor([1e4, 1e4]),
        "sc": torch.tensor([1.0, 1.0]),
    })

    assert batch.state_spec == SHEAR_FLOW_STATE_SPEC
    assert batch.batch_size == 2
    assert batch["history"].shape == (2, 4, 4, 32, 64)
    assert batch["future"].shape == (2, 1, 4, 32, 64)
    assert batch["re"].shape == (2,)
    assert batch["sc"].shape == (2,)
    assert batch["boundary"] == "periodic"


# ==============================================================================
# 11. Real Downloaded StocBench Data Contract Acceptance
# ==============================================================================

def test_real_downloaded_stocbench_files_acceptance():
    """Verify contracts against real downloaded StocBench files.

    Execution policy:
    - Default offline suite: Skipped if STOCBENCH_REQUIRE_REAL_DATA != '1' and files absent.
    - StocBench data audit acceptance: Fail-closed assertion when STOCBENCH_REQUIRE_REAL_DATA == '1'.
    """
    data_dir = Path(os.environ.get("STOCBENCH_DATA_DIR", "data/stocbench/incns_stoc/64"))
    real_traj_path = data_dir / "traj_seed_42.npy"
    real_step_path = data_dir / "step_seed_100.npz"

    require_real = os.environ.get("STOCBENCH_REQUIRE_REAL_DATA", "0") == "1"
    if require_real:
        assert real_traj_path.exists(), (
            f"FAIL-CLOSED: Real trajectory file '{real_traj_path}' not found! "
            "Live data acceptance requires real files."
        )
        assert real_step_path.exists(), (
            f"FAIL-CLOSED: Real bifurcation file '{real_step_path}' not found! "
            "Live data acceptance requires real files."
        )
    elif not real_traj_path.exists() or not real_step_path.exists():
        pytest.skip(
            "Real StocBench data files not found. Skipping live acceptance in offline environment. "
            "Set STOCBENCH_REQUIRE_REAL_DATA=1 to enforce live acceptance."
        )

    # 1. Real trajectory verification
    ds = StocBenchTrainDataset(real_traj_path, history_length=1, horizon=1)
    assert len(ds) == 99500
    sample = ds[0]
    assert sample["history"].shape == (1, 1, 64, 64)
    assert sample["future"].shape == (1, 1, 64, 64)

    # 2. Real bifurcation reference verification
    ref = StocBenchReferenceEnsemble(real_step_path)
    assert ref.num_members == 5000
    assert ref.spatial_shape == (64, 64)
    consistency = ref.verify_statistical_consistency(atol=1e-4)
    assert consistency["max_mean_discrepancy"] < 1e-4
    assert consistency["max_std_discrepancy"] < 1e-4

    # 3. Batch adapter verification with real sample
    batch = stocbench_batch_adapter({
        "history": sample["history"].unsqueeze(0),
        "future": sample["future"].unsqueeze(0),
        "dt": sample["dt"],
        "source_file": sample["source_file"],
        "traj_idx": sample["traj_idx"],
        "start_t": sample["start_t"],
        "dataset_id": sample["dataset_id"],
        "dataset_revision": sample["dataset_revision"],
        "state_variables": sample["state_variables"],
    })
    assert batch.state_spec == STOCBENCH_STATE_SPEC
    assert batch.batch_size == 1
    assert batch["history"].shape == (1, 1, 1, 64, 64)
    assert batch["future"].shape == (1, 1, 1, 64, 64)
    assert batch.metadata["dataset_id"] == "stocbench"
