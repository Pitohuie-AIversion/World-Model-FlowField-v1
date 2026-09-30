"""Unit tests for the publication figures visualization pipeline."""

from pathlib import Path
import json
import pytest
import numpy as np

from scripts.visualize_paper_figures import (
    load_trajectory_paired_data,
    extract_seed_summary,
    plot_figure2_primary_multiseed,
    plot_figure3_secondary_physics,
    SEED_COLORS,
    SEED_MARKERS,
)


@pytest.fixture
def mock_paired_data(tmp_path: Path) -> Path:
    """Create a minimal valid mock paired trajectory JSON artifact."""
    mock_data = {
        "metadata": {
            "all_seeds": [42, 43, 44, 45, 46],
            "discovery_seed": 42,
            "replication_seeds": [43, 44, 45, 46],
            "total_seeds": 5,
            "trajectories_per_seed": 2,
            "total_trajectory_pairs": 10,
        },
        "by_seed_trajectory": {
            str(s): {
                f"traj_{t}": {
                    "source_file": "mock.h5",
                    "traj_idx": t,
                    "metrics": {
                        "primary1_h10_ens_vrmse": {"C2": 1.0 + s * 0.01, "R2_A": 0.95 + s * 0.01, "delta": -0.05, "relative_change_pct": -5.0},
                        "primary2_h10_ens_spec_rel_err": {"C2": 0.05, "R2_A": 0.04 if s != 45 else 0.06, "delta": -0.01 if s != 45 else 0.01, "relative_change_pct": -20.0 if s != 45 else 20.0},
                        "primary3_h10_indiv_spec_rel_err": {"C2": 0.06, "R2_A": 0.05, "delta": -0.01, "relative_change_pct": -16.6},
                        "primary4_h5_ens_vrmse": {"C2": 0.5, "R2_A": 0.501, "delta": 0.001, "relative_change_pct": 0.2},
                        "secondary_h10_samp_vrmse": {"C2": 2.0, "R2_A": 1.9, "delta": -0.1, "relative_change_pct": -5.0},
                        "secondary_h10_samp_div_rms": {"C2": 1.2, "R2_A": 1.1, "delta": -0.1, "relative_change_pct": -8.3},
                        "secondary_h10_samp_vort_rmse": {"C2": 1.8, "R2_A": 1.7, "delta": -0.1, "relative_change_pct": -5.5},
                    },
                }
                for t in range(2)
            }
            for s in [42, 43, 44, 45, 46]
        },
    }
    p = tmp_path / "mock_paired_analysis.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(mock_data, f)
    return p


def test_load_trajectory_paired_data(mock_paired_data: Path):
    """Verify loading and validating paired data."""
    data = load_trajectory_paired_data(mock_paired_data)
    assert "metadata" in data
    assert "by_seed_trajectory" in data
    assert len(data["by_seed_trajectory"]) == 5


def test_load_trajectory_paired_data_missing():
    """Verify fail-closed behavior on missing artifact."""
    with pytest.raises(FileNotFoundError):
        load_trajectory_paired_data("non_existent_file_path.json")


def test_extract_seed_summary(mock_paired_data: Path):
    """Verify seed-level aggregation logic."""
    data = load_trajectory_paired_data(mock_paired_data)
    summary = extract_seed_summary(data, "primary1_h10_ens_vrmse", [42, 43, 44])
    assert len(summary) == 3
    assert summary[0]["seed"] == 42
    expected_rel = (-0.05 / 1.42) * 100.0
    assert np.isclose(summary[0]["rel_change_pct"], expected_rel, atol=1e-3)


def test_render_figures_mock(mock_paired_data: Path, tmp_path: Path):
    """Verify Figure 2 and Figure 3 rendering pipeline produces all expected files."""
    data = load_trajectory_paired_data(mock_paired_data)
    out_dir = tmp_path / "figures"

    # Figure 2
    f2_paths = plot_figure2_primary_multiseed(data, out_dir, prefix="test_fig2")
    for ext in ["pdf", "png", "svg"]:
        assert ext in f2_paths
        assert f2_paths[ext].exists()
        assert f2_paths[ext].stat().st_size > 1000

    # Figure 3
    f3_paths = plot_figure3_secondary_physics(data, out_dir, prefix="test_fig3")
    for ext in ["pdf", "png", "svg"]:
        assert ext in f3_paths
        assert f3_paths[ext].exists()
        assert f3_paths[ext].stat().st_size > 1000


def test_render_figures_real_artifact(tmp_path: Path):
    """Verify rendering with the actual repo artifact if it exists."""
    real_artifact = Path("outputs/metrics/fm_r2_multiseed_trajectory_paired_analysis.json")
    if not real_artifact.exists():
        pytest.skip("Real artifact not present")

    data = load_trajectory_paired_data(real_artifact)
    out_dir = tmp_path / "figures_real"

    f2_paths = plot_figure2_primary_multiseed(data, out_dir, prefix="fig2_real")
    assert f2_paths["pdf"].exists()
    assert f2_paths["png"].exists()
    assert f2_paths["svg"].exists()

    f3_paths = plot_figure3_secondary_physics(data, out_dir, prefix="fig3_real")
    assert f3_paths["pdf"].exists()
    assert f3_paths["png"].exists()
    assert f3_paths["svg"].exists()
