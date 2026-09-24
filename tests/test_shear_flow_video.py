"""Unit and integration tests for shear flow temporal video generator."""

import json
import os
import shutil
import tempfile
import pytest
import numpy as np
import torch
import h5py

from scripts.generate_shear_flow_video import (
    FIELD_CONFIGS,
    load_trajectory_data,
    render_shear_flow_video,
)

REAL_HDF5 = "/root/autodl-tmp/datasets/shear_flow/data/valid/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5"


def _has_video_writer() -> bool:
    if shutil.which("ffmpeg") is not None:
        return True
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def mock_hdf5_file(tmp_path):
    """Create a minimal synthetic HDF5 file matching shear_flow schema."""
    file_path = str(tmp_path / "mock_shear_flow.hdf5")
    n_sims = 2
    n_times = 6
    nx, ny = 32, 64

    with h5py.File(file_path, "w") as f:
        # velocity: (n_sims, n_times, nx, ny, 2)
        f.create_dataset(
            "t1_fields/velocity",
            data=np.random.randn(n_sims, n_times, nx, ny, 2).astype(np.float32) * 0.1,
        )
        # tracer: (n_sims, n_times, nx, ny)
        f.create_dataset(
            "t0_fields/tracer",
            data=np.random.randn(n_sims, n_times, nx, ny).astype(np.float32) * 0.2,
        )
        # pressure: (n_sims, n_times, nx, ny)
        f.create_dataset(
            "t0_fields/pressure",
            data=np.random.randn(n_sims, n_times, nx, ny).astype(np.float32) * 0.01,
        )
        f.create_dataset("dimensions/time", data=np.linspace(0.0, 0.5, n_times))
        f.create_dataset("scalars/Reynolds", data=10000.0)
        f.create_dataset("scalars/Schmidt", data=0.1)

    return file_path


def test_load_trajectory_data_mock(mock_hdf5_file):
    """Verify loading from synthetic HDF5 file."""
    data = load_trajectory_data(mock_hdf5_file, sim_idx=0, max_frames=4)
    assert data["n_frames"] == 4
    assert data["u"].shape == (4, 32, 64)
    assert data["v"].shape == (4, 32, 64)
    assert data["vorticity"].shape == (4, 32, 64)
    assert data["tracer"].shape == (4, 32, 64)
    assert data["speed"].shape == (4, 32, 64)
    assert data["re"] == 10000.0
    assert data["sc"] == 0.1
    assert data["sim_idx"] == 0


def test_load_trajectory_data_invalid_file():
    """Verify FileNotFoundError for nonexistent dataset."""
    with pytest.raises(FileNotFoundError):
        load_trajectory_data("/nonexistent/path/data.hdf5", sim_idx=0)


def test_load_trajectory_data_invalid_sim_idx(mock_hdf5_file):
    """Verify ValueError for out-of-range simulation index."""
    with pytest.raises(ValueError, match="Requested sim_idx=99 out of range"):
        load_trajectory_data(mock_hdf5_file, sim_idx=99)


@pytest.mark.skipif(not _has_video_writer(), reason="Neither ffmpeg nor cv2 available for video rendering")
def test_render_shear_flow_video_mock(mock_hdf5_file, tmp_path):
    """Verify rendering pipeline produces valid video and metadata."""
    out_mp4 = str(tmp_path / "mock_output.mp4")
    meta = render_shear_flow_video(
        hdf5_path=mock_hdf5_file,
        output_video=out_mp4,
        sim_idx=0,
        fps=10,
        layout="quad",
        max_frames=3,
        dpi=72,
        export_gif=True,
    )

    assert os.path.exists(out_mp4)
    assert os.path.getsize(out_mp4) > 0

    meta_json = str(tmp_path / "mock_output_metadata.json")
    assert os.path.exists(meta_json)
    with open(meta_json, "r") as f:
        loaded = json.load(f)

    assert loaded["total_frames"] == 3
    assert loaded["simulation_index"] == 0
    assert loaded["layout"] == "quad"
    assert loaded["panels"] == ["vorticity", "u", "tracer", "speed"]
    assert "resolution" in loaded
    assert loaded["resolution"]["width"] > 0
    assert loaded["resolution"]["height"] > 0


@pytest.mark.skipif(not _has_video_writer(), reason="Neither ffmpeg nor cv2 available for video rendering")
def test_render_shear_flow_video_triple_layout(mock_hdf5_file, tmp_path):
    """Verify rendering with triple-panel layout."""
    out_mp4 = str(tmp_path / "mock_triple.mp4")
    meta = render_shear_flow_video(
        hdf5_path=mock_hdf5_file,
        output_video=out_mp4,
        sim_idx=1,
        fps=5,
        layout="triple",
        max_frames=2,
        dpi=72,
        export_gif=False,
    )
    assert os.path.exists(out_mp4)
    assert meta["layout"] == "triple"
    assert meta["panels"] == ["vorticity", "u", "tracer"]


@pytest.mark.skipif(not os.path.exists(REAL_HDF5), reason="Real shear flow dataset not found")
def test_load_real_dataset_shape():
    """Verify real dataset dimensions if present on filesystem."""
    data = load_trajectory_data(REAL_HDF5, sim_idx=0, max_frames=2)
    assert data["n_frames"] == 2
    assert data["shape"] == (256, 512)
    assert data["u"].shape == (2, 256, 512)
    assert data["re"] == 10000.0
