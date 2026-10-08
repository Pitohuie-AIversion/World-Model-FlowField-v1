#!/usr/bin/env python3
"""Test suite for PDE-controlled and Gaussian probabilistic qualitative sampling scripts."""

from pathlib import Path
import json
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_pde_controlled_sampling_artifacts():
    pde_dir = PROJECT_ROOT / "outputs/figures/pde_controlled"
    required_files = [
        "sample_pde_flow_fields_comparison.png",
        "sample_pde_errors_comparison.png",
        "sample_pde_residuals_comparison.png",
        "sample_pde_qualitative_metadata.json",
    ]
    for rf in required_files:
        p = pde_dir / rf
        assert p.exists(), f"Missing PDE sampling artifact: {p}"
        assert p.stat().st_size > 0, f"Empty file: {p}"
        if p.suffix == ".png":
            with Image.open(p) as img:
                w, h = img.size
                assert w > 1000 and h > 1000, f"Image {rf} resolution too small: {w}x{h}"
        elif p.suffix == ".json":
            with open(p) as f:
                d = json.load(f)
            assert "sample_index" in d
            assert "figures" in d


def test_gaussian_probabilistic_sampling_artifacts():
    prob_dir = PROJECT_ROOT / "outputs/figures/probabilistic"
    required_files = [
        "sample_gaussian_ensemble_realizations.png",
        "sample_gaussian_uncertainty_vs_error.png",
        "sample_gaussian_prediction_intervals.png",
        "sample_gaussian_probabilistic_metadata.json",
    ]
    for rf in required_files:
        p = prob_dir / rf
        assert p.exists(), f"Missing probabilistic sampling artifact: {p}"
        assert p.stat().st_size > 0, f"Empty file: {p}"
        if p.suffix == ".png":
            with Image.open(p) as img:
                w, h = img.size
                assert w > 1000 and h > 1000, f"Image {rf} resolution too small: {w}x{h}"
        elif p.suffix == ".json":
            with open(p) as f:
                d = json.load(f)
            assert "sample_index" in d
            assert "num_mc_samples" in d
