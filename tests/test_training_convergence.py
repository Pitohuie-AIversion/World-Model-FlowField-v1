"""Unit tests for training convergence analysis script."""

import json
from pathlib import Path
import pytest
import numpy as np

from scripts.analyze_training_convergence import (
    parse_log_filename,
    parse_training_log,
    compute_last_k_slope,
    compute_run_convergence_metrics,
    analyze_all_logs,
    export_convergence_json,
    plot_convergence_curves,
)


def test_parse_log_filename():
    """Verify seed and ablation group extraction across naming patterns."""
    # Seed 42 default (no explicit seed in filename)
    p1 = "outputs/train_closure_r4_ablation_E0_single_step.log"
    meta1 = parse_log_filename(p1)
    assert meta1["seed"] == 42
    assert meta1["group"] == "E0_single_step"

    p2 = "outputs/train_closure_r4_ablation_E4_full_physics.log"
    meta2 = parse_log_filename(p2)
    assert meta2["seed"] == 42
    assert meta2["group"] == "E4_full_physics"

    # Explicit seed in filename
    p3 = "outputs/train_closure_r4_seed_43_ablation_E1_rollout_field.log"
    meta3 = parse_log_filename(p3)
    assert meta3["seed"] == 43
    assert meta3["group"] == "E1_rollout_field"

    p4 = "outputs/train_closure_r4_seed_44_ablation_E3_plus_L_vort.log"
    meta4 = parse_log_filename(p4)
    assert meta4["seed"] == 44
    assert meta4["group"] == "E3_plus_L_vort"


def test_compute_last_k_slope():
    """Verify linear regression slope calculation."""
    # Strictly increasing sequence (slope = +1.0)
    assert pytest.approx(compute_last_k_slope([1.0, 2.0, 3.0, 4.0, 5.0], k=5), 1e-6) == 1.0

    # Strictly decreasing sequence (slope = -2.0)
    assert pytest.approx(compute_last_k_slope([10.0, 8.0, 6.0, 4.0, 2.0], k=5), 1e-6) == -2.0

    # Flat sequence (slope = 0.0)
    assert pytest.approx(compute_last_k_slope([0.15, 0.15, 0.15, 0.15, 0.15], k=5), 1e-6) == 0.0

    # Shorter than k: uses all available points
    assert pytest.approx(compute_last_k_slope([1.0, 3.0, 5.0], k=5), 1e-6) == 2.0

    # Edge cases: < 2 points
    assert compute_last_k_slope([1.0], k=5) == 0.0
    assert compute_last_k_slope([], k=5) == 0.0


def test_parse_training_log_mock(tmp_path):
    """Verify log parsing on mock rollout and single-step logs."""
    # 1. Rollout log
    rollout_log = tmp_path / "train_closure_r4_seed_42_ablation_E4_full_physics.log"
    rollout_content = """Loading pretrained representation weights
Training latent_transformer on cuda | Horizon: 2 | Epochs: 3
Epoch [01/03] | Train Loss: 1.5000e-01 | Val Rollout Mean VRMSE: 0.5000 | Step 1 VRMSE: 0.5100 (u: 0.07, v: 1.05, p: 0.79, s: 0.14) | Max VRAM: 16.0 GB
Epoch [02/03] | Train Loss: 2.0000e-02 | Val Rollout Mean VRMSE: 0.3000 | Step 1 VRMSE: 0.3100 (u: 0.05, v: 0.60, p: 0.50, s: 0.10) | Max VRAM: 16.0 GB
Epoch [03/03] | Train Loss: 5.0000e-03 | Val Rollout Mean VRMSE: 0.3500 | Step 1 VRMSE: 0.3400 (u: 0.06, v: 0.65, p: 0.55, s: 0.11) | Max VRAM: 16.0 GB
Training completed. Best VRMSE: 0.3000
"""
    rollout_log.write_text(rollout_content)
    parsed = parse_training_log(rollout_log)

    assert parsed["seed"] == 42
    assert parsed["group"] == "E4_full_physics"
    assert parsed["horizon"] == 2
    assert parsed["total_epochs"] == 3
    assert parsed["epochs"] == [1, 2, 3]
    assert parsed["train_losses"] == [0.15, 0.02, 0.005]
    assert parsed["val_vrmses"] == [0.5, 0.3, 0.35]
    assert parsed["step1_vrmses"] == [0.51, 0.31, 0.34]
    assert parsed["components"]["u"] == [0.07, 0.05, 0.06]

    metrics = compute_run_convergence_metrics(parsed, k_window=2)
    assert metrics["best_epoch"] == 2
    assert metrics["best_val_vrmse"] == 0.3
    assert metrics["final_val_vrmse"] == 0.35
    assert pytest.approx(metrics["best_to_final_gap"], 1e-6) == 0.05

    # 2. Single-step log
    single_step_log = tmp_path / "train_closure_r4_ablation_E0_single_step.log"
    single_content = """Loading pretrained representation weights
Training latent_transformer on cuda | Horizon: 1 | Epochs: 2
Epoch [01/02] | Train Loss: 8.0000e-02 | Val VRMSE Mean: 0.4000 (u: 0.08, v: 0.78, p: 0.60, s: 0.11) | Max VRAM: 8.0 GB
Epoch [02/02] | Train Loss: 1.0000e-02 | Val VRMSE Mean: 0.2500 (u: 0.05, v: 0.48, p: 0.49, s: 0.09) | Max VRAM: 8.0 GB
Training completed. Best VRMSE: 0.2500
"""
    single_step_log.write_text(single_content)
    parsed_s = parse_training_log(single_step_log)

    assert parsed_s["seed"] == 42
    assert parsed_s["group"] == "E0_single_step"
    assert parsed_s["horizon"] == 1
    assert parsed_s["val_vrmses"] == [0.4, 0.25]


def test_analyze_all_logs_and_export_mock(tmp_path):
    """Verify analyze_all_logs, export_convergence_json, and plot generation on mock logs."""
    # Create two synthetic logs in tmp_path
    log1 = tmp_path / "train_closure_r4_ablation_E0_single_step.log"
    log1.write_text("""Epoch [01/02] | Train Loss: 0.10 | Val VRMSE Mean: 0.40\nEpoch [02/02] | Train Loss: 0.01 | Val VRMSE Mean: 0.20\n""")

    log2 = tmp_path / "train_closure_r4_seed_43_ablation_E1_rollout_field.log"
    log2.write_text("""Epoch [01/02] | Train Loss: 0.12 | Val Rollout Mean VRMSE: 0.45\nEpoch [02/02] | Train Loss: 0.02 | Val Rollout Mean VRMSE: 0.22\n""")

    analysis = analyze_all_logs(tmp_path)
    assert analysis["aggregate"]["total_runs"] == 2
    assert len(analysis["runs"]) == 2

    # Test export JSON
    out_json = tmp_path / "summary.json"
    export_convergence_json(analysis, out_json)
    assert out_json.is_file()
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["aggregate"]["total_runs"] == 2
    assert len(data["runs"]) == 2

    # Test plot generation
    out_png = tmp_path / "curves.png"
    plot_convergence_curves(analysis, out_png)
    assert out_png.is_file()
    assert out_png.stat().st_size > 1000


def test_analyze_all_logs_on_repo_logs():
    """Verify that analyzing actual repo logs yields valid 13-run statistics when logs are present."""
    real_logs = list(Path("outputs").glob("train_closure_r4_*.log"))
    if not real_logs:
        pytest.skip("Real training logs not present in checkout environment")

    analysis = analyze_all_logs("outputs")
    agg = analysis["aggregate"]

    assert agg["total_runs"] == 13
    assert agg["min_best_epoch"] >= 20
    assert agg["max_best_epoch"] <= 29
    assert agg["runs_peaking_at_epoch_30"] == 0
    assert agg["runs_peaking_before_epoch_30"] == 13
    assert agg["fraction_peaking_before_epoch_30"] == 1.0
    assert agg["all_gaps_positive"] is True
    assert agg["mean_best_to_final_gap"] > 0.02
