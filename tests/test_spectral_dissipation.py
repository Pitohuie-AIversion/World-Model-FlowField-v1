"""Unit tests for directional spectral dissipation analysis."""

import numpy as np
import pytest
import torch

from scripts.analyze_spectral_dissipation import (
    compute_directional_energy_spectra,
    diagnose_dissipation_mode,
)


def test_directional_spectra_shapes_and_parseval():
    """Verify directional Fourier spectra obey Parseval energy conservation."""
    nx, ny = 32, 64
    lx, ly = 1.0, 2.0
    u = torch.randn(2, nx, ny)
    v = torch.randn(2, nx, ny)

    kx, e_kx, ky, e_ky = compute_directional_energy_spectra(u, v, domain_size=(lx, ly))

    assert len(kx) == nx // 2 + 1
    assert len(e_kx) == nx // 2 + 1
    assert len(ky) == ny // 2 + 1
    assert len(e_ky) == ny // 2 + 1

    assert (e_kx >= 0.0).all()
    assert (e_ky >= 0.0).all()

    # Parseval energy equality
    total_physical = 0.5 * (u**2 + v**2).mean(dim=(-2, -1)).mean().item()
    sum_kx = e_kx.sum().item()
    sum_ky = e_ky.sum().item()

    assert sum_kx == pytest.approx(total_physical, rel=1e-4)
    assert sum_ky == pytest.approx(total_physical, rel=1e-4)
    assert sum_kx == pytest.approx(sum_ky, rel=1e-5)


def test_directional_spectra_anisotropic_sinusoid():
    """Pure x-mode sinusoid should concentrate energy entirely at the fundamental kx."""
    nx, ny = 32, 64
    lx, ly = 1.0, 2.0
    x = torch.arange(nx, dtype=torch.float32) * (lx / nx)
    y = torch.arange(ny, dtype=torch.float32) * (ly / ny)
    grid_x, _ = torch.meshgrid(x, y, indexing="ij")

    # u = sin(2*pi*x / lx), v = 0
    u = torch.sin(2.0 * torch.pi * grid_x / lx)
    v = torch.zeros_like(u)

    kx, e_kx, ky, e_ky = compute_directional_energy_spectra(u, v, domain_size=(lx, ly))

    # Fundamental mode is index 1
    total_e = e_kx.sum().item()
    assert e_kx[1].item() / total_e > 0.999
    # High kx should be zero
    assert e_kx[2:].sum().item() / total_e < 1e-4


def test_diagnose_dissipation_mode():
    """Verify dissipation diagnostic correctly classifies noise vs damping."""
    k_bins = np.linspace(0, 100, 50)

    # 1. Spurious noise: ratio is 2.5 at high k
    ratio_noisy = np.ones(50)
    ratio_noisy[25:] = 2.5
    d_noisy = diagnose_dissipation_mode(k_bins, ratio_noisy, cutoff_ratio=0.5)
    assert d_noisy["diagnosis"] == "spurious_high_frequency_accumulation"
    assert d_noisy["mean_high_k_ratio"] == pytest.approx(2.5)

    # 2. Over-dissipation: ratio is 0.3 at high k
    ratio_damped = np.ones(50)
    ratio_damped[25:] = 0.3
    d_damped = diagnose_dissipation_mode(k_bins, ratio_damped, cutoff_ratio=0.5)
    assert d_damped["diagnosis"] == "numerical_over_dissipation"
    assert d_damped["mean_high_k_ratio"] == pytest.approx(0.3)

    # 3. Balanced preservation: ratio is ~1.0
    ratio_balanced = np.ones(50)
    d_balanced = diagnose_dissipation_mode(k_bins, ratio_balanced, cutoff_ratio=0.5)
    assert d_balanced["diagnosis"] == "balanced_scale_preservation"
    assert d_balanced["mean_high_k_ratio"] == pytest.approx(1.0)


def _create_mock_checkpoint(
    tmp_path: str,
    grp: str = "E2_plus_L_div",
    seed: int = 42,
    split_hash: str = "split_valid_123",
    normalizer_hash: str = "norm_valid_456",
    protocol: str = "Closure-R4",
    training_git_dirty: bool = False,
):
    from src.models.encoder import Encoder2D
    from src.models.decoder import Decoder2D
    from src.models.latent_transformer import LatentSTTransformer
    from src.models.latent_forecaster import LatentForecaster

    cfg = {
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 2,
        "prediction_mode": "direct",
        "spatial_axis_contract": "tensor(...,C,Nx,Ny):dim-2=x,dim-1=y",
        "physics_domain_size_xy": [1.0, 2.0],
        "physics_protocol": protocol,
        "horizon": 2,
        "lambda_div": 0.01,
        "lambda_vort": 0.0,
    }
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=64,
        cond_dim=128,
        depth=2,
        num_heads=2,
        history_length=4,
        prediction_mode="direct",
    )
    model = LatentForecaster(encoder, transformer, decoder)

    ckpt_data = {
        "config": cfg,
        "seed": seed,
        "split_hash": split_hash,
        "normalizer_hash": normalizer_hash,
        "physics_protocol": protocol,
        "training_git_commit": "abc12345",
        "training_git_dirty": training_git_dirty,
        "model_state_dict": model.state_dict(),
    }
    torch.save(ckpt_data, tmp_path)


def test_spectral_checkpoint_not_found_fails_closed():
    """Missing checkpoint for requested group/seed raises FileNotFoundError."""
    from scripts.analyze_spectral_dissipation import resolve_group_checkpoint_path

    with pytest.raises(FileNotFoundError):
        resolve_group_checkpoint_path("E4_full_physics", seed=9999)


def test_spectral_wrong_seed_rejection(tmp_path):
    """Evaluating a checkpoint against an unexpected seed must fail closed."""
    from scripts.analyze_spectral_dissipation import load_and_validate_forecaster

    ckpt_file = str(tmp_path / "mock_ckpt.pt")
    _create_mock_checkpoint(ckpt_file, seed=42)

    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        load_and_validate_forecaster(
            grp="E2_plus_L_div",
            ckpt_path=ckpt_file,
            seed=43,  # Mismatched expected seed!
            eval_split_hash="split_valid_123",
            eval_normalizer_hash="norm_valid_456",
            manifest_path=None,
            device=torch.device("cpu"),
        )
    assert "Seed mismatch" in str(exc_info.value)


def test_spectral_split_hash_mismatch_rejection(tmp_path):
    """Mismatched split hash between evaluation and checkpoint must fail closed."""
    from scripts.analyze_spectral_dissipation import load_and_validate_forecaster

    ckpt_file = str(tmp_path / "mock_ckpt.pt")
    _create_mock_checkpoint(ckpt_file, split_hash="split_A")

    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        load_and_validate_forecaster(
            grp="E2_plus_L_div",
            ckpt_path=ckpt_file,
            seed=42,
            eval_split_hash="split_B",  # Mismatched split hash!
            eval_normalizer_hash="norm_valid_456",
            manifest_path=None,
            device=torch.device("cpu"),
        )
    assert "Split hash mismatch" in str(exc_info.value)


def test_spectral_normalizer_hash_mismatch_rejection(tmp_path):
    """Mismatched normalizer hash between evaluation and checkpoint must fail closed."""
    from scripts.analyze_spectral_dissipation import load_and_validate_forecaster

    ckpt_file = str(tmp_path / "mock_ckpt.pt")
    _create_mock_checkpoint(ckpt_file, normalizer_hash="norm_A")

    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        load_and_validate_forecaster(
            grp="E2_plus_L_div",
            ckpt_path=ckpt_file,
            seed=42,
            eval_split_hash="split_valid_123",
            eval_normalizer_hash="norm_B",  # Mismatched normalizer hash!
            manifest_path=None,
            device=torch.device("cpu"),
        )
    assert "Normalizer hash mismatch" in str(exc_info.value)


def test_spectral_dirty_git_formal_rejection(monkeypatch):
    """Formal spectral analysis rejects execution in dirty git working tree."""
    import scripts.analyze_spectral_dissipation as spectral_mod

    monkeypatch.setattr(spectral_mod, "is_git_dirty", lambda root: True)

    with pytest.raises(RuntimeError) as exc_info:
        spectral_mod.analyze_spectral_dissipation_for_groups(
            allow_dirty=False,
        )
    assert "Working tree is dirty" in str(exc_info.value)

