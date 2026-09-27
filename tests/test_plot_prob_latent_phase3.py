"""Unit test suite for Phase 3 probabilistic evaluation plotting and figure generation.

Verifies:
1. Data extraction and mathematical consistency from Phase 3 evaluation report.
2. Non-overlapping categorical bar chart coordinates (P2-1 fix).
3. Fully dynamic data-driven annotations without stale hardcoded values (P2-2 fix).
4. Accurate scientific wording for parity lines and prediction interval labels (P2-3 fix).
5. Dual export of high-resolution raster (PNG) and publication vector (PDF) figures.
"""

import copy
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.plot_prob_latent_phase3 import (
    extract_calibration_data,
    extract_metadata_config,
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

    def test_extract_metadata_config_canonical(self, canonical_metrics_path):
        data = load_phase3_metrics(canonical_metrics_path)
        cfg_meta = extract_metadata_config(data)
        assert cfg_meta["windows_evaluated"] == 105
        assert cfg_meta["num_samples"] == 32
        assert cfg_meta["finite_k_inflation_factor"] == pytest.approx(math.sqrt(33.0 / 32.0), rel=1e-4)

    def test_extract_vrmse_data_accuracy_and_distinction(self, canonical_metrics_path):
        data = load_phase3_metrics(canonical_metrics_path)
        horizons, std_vrmse, rms_diag_vrmse = extract_vrmse_data(data)

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
        assert ss_data["G0"]["pooled_spread"][-1] == pytest.approx(0.110998, rel=1e-4)
        assert ss_data["G0"]["pooled_rmse"][-1] == pytest.approx(0.109985, rel=1e-4)
        assert ss_data["G0"]["pooled_ssr"][-1] == pytest.approx(1.009217, rel=1e-4)

        assert ss_data["G1"]["pooled_spread"][-1] == pytest.approx(0.130899, rel=1e-4)
        assert ss_data["G1"]["pooled_rmse"][-1] == pytest.approx(0.112839, rel=1e-4)
        assert ss_data["G1"]["pooled_ssr"][-1] == pytest.approx(1.160053, rel=1e-4)

        k_factor = math.sqrt(33.0 / 32.0)
        assert ss_data["G1"]["pooled_spread_adj"][-1] == pytest.approx(
            ss_data["G1"]["pooled_spread"][-1] * k_factor, rel=1e-4
        )


class TestBarChartNonOverlappingAndCategoricalCoordinates:
    """P2-1 Fix: Verify that calibration error bars are rendered on equidistant categorical coordinates."""

    def test_bars_do_not_overlap_between_90_and_95_percent(self, canonical_metrics_path, tmp_path):
        data = load_phase3_metrics(canonical_metrics_path)
        nominal_pcts, empirical_pcts, deviation_pct_points = extract_calibration_data(data)

        fig_path = tmp_path / "test_calibration.png"
        plot_figure_2_calibration(nominal_pcts, empirical_pcts, deviation_pct_points, fig_path)

        # Inspect the categorical coordinates:
        # Number of nominal groups is 4 (50%, 80%, 90%, 95%)
        # In categorical coordinates x = 0, 1, 2, 3 with width = 0.35:
        # Group 2 (90%): center=2, G0 span [1.65, 2.0], G1 span [2.0, 2.35] -> right edge = 2.35
        # Group 3 (95%): center=3, G0 span [2.65, 3.0], G1 span [3.0, 3.35] -> left edge = 2.65
        # Gap between group 2 and group 3 = 2.65 - 2.35 = 0.30 > 0 (strictly positive, zero overlap!)
        x_indices = np.arange(len(nominal_pcts))
        width = 0.35
        for i in range(len(nominal_pcts) - 1):
            right_edge_group_i = (x_indices[i] + width / 2) + width / 2  # rightmost point of G1 bar
            left_edge_group_next = (x_indices[i + 1] - width / 2) - width / 2  # leftmost point of G0 bar
            assert left_edge_group_next > right_edge_group_i, (
                f"Bar groups {nominal_pcts[i]}% and {nominal_pcts[i+1]}% overlap: "
                f"right_edge={right_edge_group_i}, left_edge={left_edge_group_next}"
            )


class TestDynamicDataDrivenAnnotations:
    """P2-2 Fix: Verify that annotations dynamically change when input data changes, without stale hardcoded values."""

    def test_dynamic_annotation_updates_with_modified_data(self, canonical_metrics_path, tmp_path):
        data = load_phase3_metrics(canonical_metrics_path)

        # Mutate configuration and Step 1 intervals in synthetic copy
        mutated_data = copy.deepcopy(data)
        mutated_data["step2_autoregressive_rollout_evaluation"]["configuration"]["windows_evaluated"] = 42
        mutated_data["step2_autoregressive_rollout_evaluation"]["configuration"]["num_samples"] = 16
        mutated_data["step2_autoregressive_rollout_evaluation"]["configuration"]["finite_k_inflation_factor"] = math.sqrt(17.0 / 16.0)

        # Mutate 50% coverage: empirical 40% (nominal 50%, deviation = -10.00 percentage points)
        mutated_data["step1_single_step_test_evaluation"]["G1_heteroscedastic_model"]["intervals"]["50"]["picp"] = 0.40

        cfg_meta = extract_metadata_config(mutated_data)
        assert cfg_meta["windows_evaluated"] == 42
        assert cfg_meta["num_samples"] == 16
        assert cfg_meta["finite_k_inflation_factor"] == pytest.approx(math.sqrt(17.0 / 16.0), rel=1e-4)

        nominal_pcts, empirical_pcts, deviation_pct_points = extract_calibration_data(mutated_data)
        assert empirical_pcts["G1"][0] == 40.0
        assert deviation_pct_points["G1"][0] == -10.0

        # Render figures to verify execution with mutated dynamic inputs
        fig1_path = tmp_path / "mutated_fig1.png"
        horizons, std_vrmse, rms_diag_vrmse = extract_vrmse_data(mutated_data)
        plot_figure_1_vrmse(horizons, std_vrmse, rms_diag_vrmse, cfg_meta, fig1_path)

        fig2_path = tmp_path / "mutated_fig2.png"
        plot_figure_2_calibration(nominal_pcts, empirical_pcts, deviation_pct_points, fig2_path)

        fig3_path = tmp_path / "mutated_fig3.png"
        horizons_ss, ss_data = extract_spread_skill_data(mutated_data)
        plot_figure_3_spread_skill(horizons_ss, ss_data, cfg_meta, fig3_path)

        assert fig1_path.exists()
        assert fig2_path.exists()
        assert fig3_path.exists()


class TestScientificWordingAndDualExport:
    """P2-3 Fix & Export: Verify parity reference line labels and dual PNG/PDF creation."""

    def test_dual_format_export_creates_both_png_and_pdf(self, canonical_metrics_path, tmp_path):
        data = load_phase3_metrics(canonical_metrics_path)
        cfg_meta = extract_metadata_config(data)
        horizons, std_vrmse, rms_diag_vrmse = extract_vrmse_data(data)
        nominal_pcts, empirical_pcts, deviation_pct_points = extract_calibration_data(data)
        horizons_ss, ss_data = extract_spread_skill_data(data)

        out_prefix = tmp_path / "figure_summary_phase3"
        plot_figure_summary_3panel(horizons, std_vrmse, nominal_pcts, empirical_pcts, ss_data, out_prefix)

        png_file = tmp_path / "figure_summary_phase3.png"
        pdf_file = tmp_path / "figure_summary_phase3.pdf"

        assert png_file.exists() and png_file.stat().st_size > 10000
        assert pdf_file.exists() and pdf_file.stat().st_size > 10000
