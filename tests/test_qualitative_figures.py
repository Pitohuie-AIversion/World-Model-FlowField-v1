"""Unit tests for the qualitative visualization pipeline."""

import json
import os
import pytest
import numpy as np
import torch

from scripts.generate_qualitative_figures import (
    extract_scalar_field,
    get_colormap_and_norm,
    generate_panel_figure,
    generate_multihorizon_figure,
    generate_compare_figure,
    resolve_group_checkpoint_path,
    load_and_validate_forecaster,
    generate_qualitative_suite,
)
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.utils.physics_contract import (
    PHYSICS_PROTOCOL,
    SPATIAL_AXIS_CONTRACT,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
)


def _create_mock_checkpoint(
    tmp_path: str,
    group: str = "E4_full_physics",
    seed: int = 42,
    split_hash: str = "split_valid_123",
    normalizer_hash: str = "norm_valid_456",
    training_git_dirty: bool = False,
):
    """Helper to generate a lightweight, compliant test checkpoint."""
    cfg = {
        "model": "latent_transformer",
        "horizon": 2,
        "lambda_div": 0.01,
        "lambda_vort": 0.05,
        "seed": seed,
        "split_hash": split_hash,
        "normalizer_hash": normalizer_hash,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 8,
    }

    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32)
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=256,
        cond_dim=128,
        depth=6,
        num_heads=8,
        history_length=4,
        prediction_mode="direct",
    )
    model = LatentForecaster(encoder, transformer, decoder)

    ckpt_data = {
        "config": cfg,
        "seed": seed,
        "split_hash": split_hash,
        "normalizer_hash": normalizer_hash,
        "physics_protocol": PHYSICS_PROTOCOL,
        "spatial_axis_contract": SPATIAL_AXIS_CONTRACT,
        "physics_domain_size_xy": list(SHEAR_FLOW_DOMAIN_SIZE_XY),
        "training_git_commit": "mock_commit_123",
        "training_git_dirty": training_git_dirty,
        "model_state_dict": model.state_dict(),
    }
    torch.save(ckpt_data, tmp_path)


def test_extract_scalar_field_all_variables():
    """Verify field extraction for u, v, p, tracer, and derived vorticity."""
    nx, ny = 32, 64
    tensor = torch.zeros(1, 4, nx, ny)
    # Set unique values per channel
    tensor[0, 0] = 1.0  # u
    tensor[0, 1] = 2.0  # v
    tensor[0, 2] = 3.0  # p
    tensor[0, 3] = 4.0  # tracer

    u_field = extract_scalar_field(tensor, "u")
    assert u_field.shape == (nx, ny)
    assert np.allclose(u_field, 1.0)

    v_field = extract_scalar_field(tensor, "v")
    assert v_field.shape == (nx, ny)
    assert np.allclose(v_field, 2.0)

    p_field = extract_scalar_field(tensor, "p")
    assert p_field.shape == (nx, ny)
    assert np.allclose(p_field, 3.0)

    tracer_field = extract_scalar_field(tensor, "tracer")
    assert tracer_field.shape == (nx, ny)
    assert np.allclose(tracer_field, 4.0)

    # Vorticity for constant velocity should be exactly zero
    vort_field = extract_scalar_field(tensor, "vorticity")
    assert vort_field.shape == (nx, ny)
    assert np.allclose(vort_field, 0.0, atol=1e-6)


def test_extract_vorticity_shear_wave():
    """Vorticity for u = sin(2*pi*y/Ly) should be -du/dy = -(2*pi/Ly)*cos(2*pi*y/Ly)."""
    nx, ny = 32, 64
    lx, ly = 1.0, 2.0
    x = torch.arange(nx, dtype=torch.float32)
    y = torch.arange(ny, dtype=torch.float32) * (ly / ny)
    _, grid_y = torch.meshgrid(x, y, indexing="ij")

    tensor = torch.zeros(1, 4, nx, ny)
    tensor[0, 0] = torch.sin(2.0 * torch.pi * grid_y / ly)  # u
    tensor[0, 1] = 0.0  # v

    vort = extract_scalar_field(tensor, "vorticity", domain_size=(lx, ly))
    assert vort.shape == (nx, ny)

    expected_vort = -(2.0 * torch.pi / ly) * torch.cos(2.0 * torch.pi * grid_y / ly).numpy()
    assert np.allclose(vort, expected_vort, atol=1e-4)


def test_get_colormap_and_norm():
    """Verify symmetric limits for vorticity/v and proper colormaps."""
    f1 = np.array([-2.0, 5.0])
    cmap, vmin, vmax = get_colormap_and_norm("vorticity", f1)
    assert cmap == "RdBu_r"
    assert vmin == -5.0
    assert vmax == 5.0

    cmap_u, _, _ = get_colormap_and_norm("u", f1)
    assert cmap_u == "coolwarm"

    cmap_p, _, _ = get_colormap_and_norm("p", f1)
    assert cmap_p == "viridis"

    cmap_s, _, _ = get_colormap_and_norm("tracer", f1)
    assert cmap_s == "magma"


def test_generate_panel_figure_and_metadata(tmp_path):
    """Verify generation of Figure A (panel plot) and companion metadata JSON."""
    nx, ny = 16, 32
    inp = np.zeros((nx, ny))
    pred = np.ones((nx, ny)) * 1.5
    gt = np.ones((nx, ny)) * 1.0

    out_file = str(tmp_path / "test_panel.png")
    meta_info = {
        "split_hash": "split_hash_xyz",
        "normalizer_hash": "norm_hash_xyz",
        "checkpoint_sha256": "sha_xyz",
        "source_file": "dummy.hdf5",
        "sim_idx": 0,
    }

    fig_path = generate_panel_figure(
        input_field=inp,
        pred_field=pred,
        gt_field=gt,
        var_name="u",
        group="E4_full_physics",
        seed=42,
        horizon=10,
        sample_index=0,
        out_path=out_file,
        meta_info=meta_info,
    )

    assert os.path.exists(fig_path)
    assert os.path.getsize(fig_path) > 1000

    json_path = out_file.replace(".png", "_metadata.json")
    assert os.path.exists(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    assert data["figure_type"] == "panel"
    assert data["seed"] == 42
    assert data["group"] == "E4_full_physics"
    assert data["horizon"] == 10
    assert data["variable"] == "u"
    assert data["error_type"] == "absolute_error"
    assert data["mae"] == pytest.approx(0.5)
    assert data["max_err"] == pytest.approx(0.5)
    assert data["checkpoint_sha256"] == "sha_xyz"


def test_generate_multihorizon_figure_and_metadata(tmp_path):
    """Verify generation of Figure B (multihorizon plot) and companion metadata JSON."""
    nx, ny = 16, 32
    horizons = [1, 10, 30]
    preds = {h: np.ones((nx, ny)) * (h * 0.1) for h in horizons}
    gts = {h: np.zeros((nx, ny)) for h in horizons}

    out_file = str(tmp_path / "test_multihorizon.png")
    meta_info = {
        "split_hash": "split_hash_xyz",
        "normalizer_hash": "norm_hash_xyz",
        "checkpoint_sha256": "sha_xyz",
        "source_file": "dummy.hdf5",
    }

    fig_path = generate_multihorizon_figure(
        pred_fields_by_h=preds,
        gt_fields_by_h=gts,
        var_name="vorticity",
        group="E4_full_physics",
        seed=42,
        horizons=horizons,
        sample_index=2,
        out_path=out_file,
        meta_info=meta_info,
    )

    assert os.path.exists(fig_path)
    json_path = out_file.replace(".png", "_metadata.json")
    assert os.path.exists(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    assert data["figure_type"] == "multihorizon"
    assert data["horizons"] == [1, 10, 30]
    assert data["maes_by_horizon"]["1"] == pytest.approx(0.1)
    assert data["maes_by_horizon"]["10"] == pytest.approx(1.0)
    assert data["maes_by_horizon"]["30"] == pytest.approx(3.0)


def test_generate_compare_figure_and_metadata(tmp_path):
    """Verify generation of Figure C (multi-model comparison) and companion metadata JSON."""
    nx, ny = 16, 32
    groups = ["E1_rollout_field", "E4_full_physics"]
    preds = {
        "E1_rollout_field": np.ones((nx, ny)) * 2.0,
        "E4_full_physics": np.ones((nx, ny)) * 1.2,
    }
    gt = np.ones((nx, ny)) * 1.0

    out_file = str(tmp_path / "test_compare.png")
    meta_info = {
        "split_hash": "split_hash_xyz",
        "normalizer_hash": "norm_hash_xyz",
        "models_provenance": {
            "E1_rollout_field": {"checkpoint_sha256": "sha_e1"},
            "E4_full_physics": {"checkpoint_sha256": "sha_e4"},
        },
    }

    fig_path = generate_compare_figure(
        pred_fields_by_grp=preds,
        gt_field=gt,
        var_name="u",
        groups=groups,
        seed=42,
        horizon=30,
        sample_index=1,
        out_path=out_file,
        meta_info=meta_info,
    )

    assert os.path.exists(fig_path)
    json_path = out_file.replace(".png", "_metadata.json")
    assert os.path.exists(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    assert data["figure_type"] == "compare"
    assert data["compare_groups"] == groups
    assert data["maes_by_group"]["E1_rollout_field"] == pytest.approx(1.0)
    assert data["maes_by_group"]["E4_full_physics"] == pytest.approx(0.2)


def test_qualitative_checkpoint_not_found_fails_closed():
    """Missing checkpoint for group/seed raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        resolve_group_checkpoint_path("E4_full_physics", seed=9999)


def test_qualitative_wrong_seed_rejection(tmp_path):
    """Evaluating a checkpoint against an unexpected seed must fail closed."""
    ckpt_file = str(tmp_path / "mock_ckpt.pt")
    _create_mock_checkpoint(ckpt_file, seed=42)

    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        load_and_validate_forecaster(
            grp="E4_full_physics",
            ckpt_path=ckpt_file,
            seed=43,  # Mismatched expected seed!
            eval_split_hash="split_valid_123",
            eval_normalizer_hash="norm_valid_456",
            manifest_path=None,
            device=torch.device("cpu"),
        )
    assert "Seed mismatch" in str(exc_info.value)


def test_qualitative_split_hash_mismatch_rejection(tmp_path):
    """Mismatched split hash between evaluation and checkpoint must fail closed."""
    ckpt_file = str(tmp_path / "mock_ckpt.pt")
    _create_mock_checkpoint(ckpt_file, split_hash="split_A")

    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        load_and_validate_forecaster(
            grp="E4_full_physics",
            ckpt_path=ckpt_file,
            seed=42,
            eval_split_hash="split_B",  # Mismatched split hash!
            eval_normalizer_hash="norm_valid_456",
            manifest_path=None,
            device=torch.device("cpu"),
        )
    assert "Split hash mismatch" in str(exc_info.value)


def test_qualitative_normalizer_hash_mismatch_rejection(tmp_path):
    """Mismatched normalizer hash between evaluation and checkpoint must fail closed."""
    ckpt_file = str(tmp_path / "mock_ckpt.pt")
    _create_mock_checkpoint(ckpt_file, normalizer_hash="norm_A")

    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        load_and_validate_forecaster(
            grp="E4_full_physics",
            ckpt_path=ckpt_file,
            seed=42,
            eval_split_hash="split_valid_123",
            eval_normalizer_hash="norm_B",  # Mismatched normalizer hash!
            manifest_path=None,
            device=torch.device("cpu"),
        )
    assert "Normalizer hash mismatch" in str(exc_info.value)


def test_qualitative_formal_dirty_git_rejection(monkeypatch):
    """Formal qualitative suite rejects execution in dirty git working tree."""
    import scripts.generate_qualitative_figures as qual_mod

    monkeypatch.setattr(qual_mod, "is_git_dirty", lambda root: True)

    with pytest.raises(RuntimeError) as exc_info:
        qual_mod.generate_qualitative_suite(
            formal=True,
            allow_dirty=False,
        )
    assert "Working tree is dirty" in str(exc_info.value)
