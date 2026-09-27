"""Unit test suite for Phase 3 probabilistic evaluation plotting and figure generation."""

import json
import math
from pathlib import Path

import pytest
from scripts.plot_prob_latent_phase3 import (
    extract_calibration_data,
    extract_spread_skill_data,
    extract_vrmse_data,
    load_phase3_metrics,
    plot_figure_1_vrmse,
    plot_figure_2_calibration,
    plot_figure_3_spread_skill,
    plot_figure_summary_3panel,
)


@pytest.fixture
def canonical_metrics_path() -> Path:
    """Path to the actual canonical Phase 3 metrics report."""
    path = Path("outputs/metrics/phase3_probabilistic_evaluation.json")
    if not path.exists():
        pytest.skip(f"Canonical metrics file {path} not found.")
    return path


class TestMetricsExtractionAndUnitIntegrity:
    """Verify that plotting utilities read frozen metrics with correct units, fields, and labels."""

    def test_load_phase3_metrics_valid(self, canonical_metrics_path):
        data = load_phase3_metrics(canonical_metrics_path)
        assert "metadata" in data
        assert "step1_single_step_test_evaluation" in data
        assert "step2_autoregressive_rollout_evaluation" in data

    def test_load_phase3_metrics_missing_file_raises(self, tmp_path):
        non_existent = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            load_phase3_metrics(non_existent)

    def test_load_phase3_metrics_malformed_raises(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        with open(bad_json, "w") as f:
            json.dump({"incomplete": True}, f)
        with pytest.raises(KeyError):
            load_phase3_metrics(bad_json)

    def test_extract_vrmse_data_accuracy_and_distinction(self, canonical_metrics_path):
        data = load_phase3_metrics(canonical_metrics_path)
        horizons, std_vrmse, rms_diag_vrmse = extract_vrmse_data(data)

        # Check horizons
        assert horizons == [1, 5, 10, 20, 30]

        # Verify exact numerical match at Step 30 for standard VRMSE
        assert std_vrmse["D0"][-1] == pytest.approx(0.480226, rel=1e-4)
        assert std_vrmse["G0"][-1] == pytest.approx(0.550766, rel=1e-4)
        assert std_vrmse["G1"][-1] == pytest.approx(0.591396, rel=1e-4)

        # Verify RMS diagnostic values are distinct from standard VRMSE
        assert rms_diag_vrmse["D0"][-1] == pytest.approx(0.506010, rel=1e-4)
        assert rms_diag_vrmse["G0"][-1] == pytest.approx(0.579917, rel=1e-4)
        assert rms_diag_vrmse["G1"][-1] == pytest.approx(0.618277, rel=1e-4)

        for m in ("D0", "G0", "G1"):
            # RMS of window VRMSE must be strictly greater than or equal to standard mean VRMSE
            for s_val, r_val in zip(std_vrmse[m], rms_diag_vrmse[m]):
                assert r_val >= s_val - 1e-6

    def test_extract_calibration_data_percentage_points(self, canonical_metrics_path):
        data = load_phase3_metrics(canonical_metrics_path)
        nominal_pcts, empirical_pcts, deviation_pct_points = extract_calibration_data(data)

        assert nominal_pcts == [50.0, 80.0, 90.0, 95.0]

        # G1 at nominal 50% should have empirical coverage ~ 26.51%
        assert empirical_pcts["G1"][0] == pytest.approx(26.505, abs=0.1)
        # Deviation in percentage points: ~ -23.49 pp
        assert deviation_pct_points["G1"][0] == pytest.approx(-23.495, abs=0.1)

        # G1 at nominal 90% should have empirical coverage ~ 94.94%
        assert empirical_pcts["G1"][2] == pytest.approx(94.943, abs=0.1)
        # Deviation in percentage points: ~ +4.94 pp
        assert deviation_pct_points["G1"][2] == pytest.approx(+4.943, abs=0.1)

    def test_extract_spread_skill_data_velocity(self, canonical_metrics_path):
        data = load_phase3_metrics(canonical_metrics_path)
        horizons, ss_data = extract_spread_skill_data(data)

        assert horizons == [1, 5, 10, 20, 30]

        # At Step 30:
        # G0 Pooled Spread ~ 0.1110, Pooled RMSE ~ 0.1100, SSR ~ 1.0092
        assert ss_data["G0"]["pooled_spread"][-1] == pytest.approx(0.110998, rel=1e-4)
        assert ss_data["G0"]["pooled_rmse"][-1] == pytest.approx(0.109985, rel=1e-4)
        assert ss_data["G0"]["pooled_ssr"][-1] == pytest.approx(1.009217, rel=1e-4)

        # G1 Pooled Spread ~ 0.1309, Pooled RMSE ~ 0.1128, SSR ~ 1.1601
        assert ss_data["G1"]["pooled_spread"][-1] == pytest.approx(0.130899, rel=1e-4)
        assert ss_data["G1"]["pooled_rmse"][-1] == pytest.approx(0.112839, rel=1e-4)
        assert ss_data["G1"]["pooled_ssr"][-1] == pytest.approx(1.160053, rel=1e-4)

        # Finite-K inflation factor check
        k_factor = math.sqrt(33.0 / 32.0)
        assert ss_data["G1"]["pooled_spread_adj"][-1] == pytest.approx(
            ss_data["G1"]["pooled_spread"][-1] * k_factor, rel=1e-4
        )


class TestFigureGenerationEndToEnd:
    """Verify that all figures are rendered cleanly to disk without runtime errors."""

    def test_generate_all_figures_to_tmp_dir(self, canonical_metrics_path, tmp_path):
        data = load_phase3_metrics(canonical_metrics_path)

        horizons, std_vrmse, rms_diag_vrmse = extract_vrmse_data(data)
        nominal_pcts, empirical_pcts, deviation_pct_points = extract_calibration_data(data)
        horizons_ss, ss_data = extract_spread_skill_data(data)

        fig1_path = tmp_path / "figure_1_vrmse_evolution.png"
        plot_figure_1_vrmse(horizons, std_vrmse, rms_diag_vrmse, fig1_path)
        assert fig1_path.exists()
        assert fig1_path.stat().st_size > 10000

        fig2_path = tmp_path / "figure_2_interval_calibration.png"
        plot_figure_2_calibration(nominal_pcts, empirical_pcts, deviation_pct_points, fig2_path)
        assert fig2_path.exists()
        assert fig2_path.stat().st_size > 10000

        fig3_path = tmp_path / "figure_3_spread_skill_relationship.png"
        plot_figure_3_spread_skill(horizons_ss, ss_data, fig3_path)
        assert fig3_path.exists()
        assert fig3_path.stat().st_size > 10000

        fig_summary_path = tmp_path / "figure_summary_phase3.png"
        plot_figure_summary_3panel(horizons, std_vrmse, nominal_pcts, empirical_pcts, ss_data, fig_summary_path)
        assert fig_summary_path.exists()
        assert fig_summary_path.stat().st_size > 10000
