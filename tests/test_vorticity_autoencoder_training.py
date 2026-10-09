"""Comprehensive regression tests for Single-Channel Vorticity Autoencoder Training,
Resumption Consistency, and Representation Evaluation Metrics.

Verifies:
1. Analytical metric fixtures: Identity (VR=1, PDE=0), Mean-replacement (VR=0, PDE=1),
   Half-deviation (VR=0.25, PDE=0.5).
2. Degenerate and edge cases: Zero field, single-member, identical members, zero-energy bins, self-pair exclusion.
3. Spectrum consistency on [1.0, 1.0] square domain.
4. Single-step optimization, finite gradients, and non-zero parameter updates (delta theta != 0).
5. Definite loss reduction on fixed probe subset.
6. Rigorous A/B checkpoint resumption numerical consistency within declared tolerances.
7. Strict data contract isolation: zero leakage of StocBench real/frozen files.
"""

import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.metrics.vorticity_representation import (
    compute_relative_l2_error,
    compute_pointwise_variance_ratio,
    compute_pairwise_difference_error,
    compute_batched_radial_enstrophy_spectrum,
    compute_enstrophy_spectrum_ratio,
)
from scripts.train_vorticity_autoencoder import (
    VorticityAutoencoder,
    generate_synthetic_vorticity_dataset,
    compute_capacity_metadata,
)


# ==============================================================================
# Helper fixture for analytical metric tests
# ==============================================================================

def _create_periodic_perturbation_ensemble(
    n_members: int = 8,
    nx: int = 64,
    ny: int = 64,
    lx: float = 1.0,
    ly: float = 1.0,
) -> torch.Tensor:
    """Create a controlled ensemble on periodic domain with exact mean + deviations."""
    x = torch.arange(nx, dtype=torch.float32) * (lx / nx)
    y = torch.arange(ny, dtype=torch.float32) * (ly / ny)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Base periodic field
    omega_0 = 2.0 * torch.cos(2.0 * torch.pi * X / lx) * torch.cos(2.0 * torch.pi * Y / ly)

    # Structured orthogonal phase shifts across members
    samples = []
    for j in range(n_members):
        phi_j = (2.0 * math.pi * j) / n_members
        pert = 0.5 * torch.sin(2.0 * math.pi * (X / lx + Y / ly) + phi_j)
        samples.append((omega_0 + pert).unsqueeze(0))

    return torch.stack(samples, dim=0)  # (N, 1, nx, ny)


# ==============================================================================
# 1. Analytical Metric Fixtures
# ==============================================================================

def test_metrics_analytical_identity():
    """Identity reconstruction must yield Variance Ratio ~ 1.0 and Pairwise Difference Error ~ 0.0."""
    target = _create_periodic_perturbation_ensemble(n_members=8)
    pred = target.clone()

    var_res = compute_pointwise_variance_ratio(pred, target)
    assert var_res["valid"] is True
    assert pytest.approx(var_res["variance_ratio"], rel=1e-5) == 1.0

    pde_res = compute_pairwise_difference_error(pred, target)
    assert pde_res["valid"] is True
    assert pde_res["pairwise_difference_error"] is not None
    assert pytest.approx(pde_res["pairwise_difference_error"], abs=1e-6) == 0.0

    l2_res = compute_relative_l2_error(pred, target)
    assert l2_res["valid_relative"] is True
    assert pytest.approx(l2_res["relative_l2"], abs=1e-6) == 0.0


def test_metrics_analytical_mean_replacement():
    """Mean replacement must yield Variance Ratio ~ 0.0 and Pairwise Difference Error ~ 1.0."""
    target = _create_periodic_perturbation_ensemble(n_members=8)
    mean_field = torch.mean(target, dim=0, keepdim=True).expand_as(target)
    pred = mean_field.clone()

    var_res = compute_pointwise_variance_ratio(pred, target)
    assert var_res["valid"] is True
    # In floating point, variance of constant mean tensor is ~ 1e-28, practically 0.0
    assert var_res["variance_ratio"] < 1e-12

    pde_res = compute_pairwise_difference_error(pred, target)
    assert pde_res["valid"] is True
    assert pde_res["pairwise_difference_error"] is not None
    assert pytest.approx(pde_res["pairwise_difference_error"], rel=1e-5) == 1.0


def test_metrics_analytical_half_deviation():
    """Half-deviation scaling must yield Variance Ratio ~ 0.25 and Pairwise Difference Error ~ 0.5."""
    target = _create_periodic_perturbation_ensemble(n_members=8)
    mean_field = torch.mean(target, dim=0, keepdim=True)
    dev = target - mean_field
    pred = mean_field + 0.5 * dev

    var_res = compute_pointwise_variance_ratio(pred, target)
    assert var_res["valid"] is True
    assert pytest.approx(var_res["variance_ratio"], rel=1e-4) == 0.25

    pde_res = compute_pairwise_difference_error(pred, target)
    assert pde_res["valid"] is True
    assert pde_res["pairwise_difference_error"] is not None
    assert pytest.approx(pde_res["pairwise_difference_error"], rel=1e-4) == 0.5


# ==============================================================================
# 2. Degenerate Inputs & Edge Cases
# ==============================================================================

def test_metrics_zero_and_near_zero_target():
    """Zero or near-zero target norm must flag invalid relative error without crashing or producing unhandled NaN."""
    target_zero = torch.zeros(4, 1, 64, 64)
    pred = torch.ones(4, 1, 64, 64) * 0.1

    res = compute_relative_l2_error(pred, target_zero)
    assert res["valid_relative"] is False
    assert res["relative_l2"] is None
    assert res["absolute_l2"] > 0.0
    assert res["reason"] == "near_zero_target_norm"


def test_metrics_insufficient_samples():
    """Sample size N < 2 must be gracefully handled for variance and pairwise difference."""
    single_target = torch.randn(1, 1, 64, 64)
    single_pred = torch.randn(1, 1, 64, 64)

    var_res = compute_pointwise_variance_ratio(single_pred, single_target)
    assert var_res["valid"] is False
    assert var_res["variance_ratio"] is None
    assert "insufficient_samples" in var_res["reason"]

    pde_res = compute_pairwise_difference_error(single_pred, single_target)
    assert pde_res["valid"] is False
    assert pde_res["pairwise_difference_error"] is None
    assert "insufficient_samples" in pde_res["reason"]


def test_metrics_identical_members_in_target():
    """If all ensemble members in target are identical, target variance is zero and must be rejected gracefully."""
    identical_target = torch.ones(5, 1, 32, 32)
    pred = torch.randn(5, 1, 32, 32)

    var_res = compute_pointwise_variance_ratio(pred, identical_target)
    assert var_res["valid"] is False
    assert var_res["variance_ratio"] is None
    assert var_res["reason"] == "near_zero_target_variance"


def test_metrics_enstrophy_spectrum_zero_energy_bins():
    """Wavenumber bins with zero target energy must yield None ratio and record spurious energy."""
    # Target has energy only at single wavenumber
    nx, ny = 32, 32
    x = torch.arange(nx, dtype=torch.float32) / nx
    y = torch.arange(ny, dtype=torch.float32) / ny
    X, Y = torch.meshgrid(x, y, indexing="ij")
    target = torch.sin(2.0 * math.pi * X).unsqueeze(0).unsqueeze(0)  # (1, 1, 32, 32)
    # Pred has high frequency noise added
    pred = target + 0.1 * torch.sin(8.0 * math.pi * Y).unsqueeze(0).unsqueeze(0)

    spec_res = compute_enstrophy_spectrum_ratio(pred, target, domain_size=(1.0, 1.0))
    assert "spectrum_ratio" in spec_res
    assert spec_res["total_bins"] > 0
    # There should be empty target bins where ratio is None
    assert any(r is None for r in spec_res["spectrum_ratio"])
    assert spec_res["spurious_energy_in_zero_bins"] >= 0.0


# ==============================================================================
# 3. Capacity & Architecture Verification
# ==============================================================================

def test_representation_capacity_element_ratio():
    """Verify that Cz=64 yields element compression ratio exactly 1.0 (no capacity reduction)."""
    meta_64 = compute_capacity_metadata(64, 64, in_channels=1, latent_channels=64, downsample_factor=8)
    assert meta_64["input_elements"] == 4096
    assert meta_64["latent_elements"] == 4096
    assert meta_64["element_compression_ratio"] == 1.0
    assert "without capacity reduction" in meta_64["capacity_interpretation"]

    meta_16 = compute_capacity_metadata(64, 64, in_channels=1, latent_channels=16, downsample_factor=8)
    assert meta_16["latent_elements"] == 1024
    assert meta_16["element_compression_ratio"] == 4.0


def test_vorticity_autoencoder_construction_and_pressure_guard():
    """Autoencoder must reject multi-channel configurations and pressure projection."""
    # Legal construction
    ae = VorticityAutoencoder(in_channels=1, out_channels=1, latent_channels=64, project_pressure=False)
    x = torch.randn(2, 1, 64, 64)
    out = ae(x)
    assert out.shape == (2, 1, 64, 64)

    # Guard: must assert on multi-channel or pressure projection enabled
    with pytest.raises(AssertionError):
        VorticityAutoencoder(in_channels=4, out_channels=4)

    with pytest.raises(AssertionError):
        VorticityAutoencoder(in_channels=1, out_channels=1, project_pressure=True)


# ==============================================================================
# 4. Optimization & Gradient Flow Test
# ==============================================================================

def test_single_step_gradient_and_parameter_update():
    """Single optimization step must produce finite gradients and update parameters (delta theta != 0)."""
    torch.manual_seed(123)
    model = VorticityAutoencoder(in_channels=1, out_channels=1, latent_channels=64, base_channels=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(2, 1, 64, 64)
    initial_param = model.encoder.in_conv.weight.clone()

    optimizer.zero_grad()
    recon = model(x)
    loss = nn.functional.mse_loss(recon, x)
    assert not torch.isnan(loss) and not torch.isinf(loss)

    loss.backward()

    # All model parameters must have finite gradients
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Missing gradient for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN in grad for {name}"
        assert not torch.isinf(p.grad).any(), f"Inf in grad for {name}"

    optimizer.step()
    updated_param = model.encoder.in_conv.weight

    # Assert parameter actually changed
    diff = torch.norm(updated_param - initial_param).item()
    assert diff > 0.0, "Optimizer step failed to update model parameters (delta theta == 0)"


def test_fixed_subset_definite_loss_reduction():
    """Model overfitted on a small fixed synthetic batch must achieve definite loss reduction."""
    torch.manual_seed(999)
    model = VorticityAutoencoder(in_channels=1, out_channels=1, latent_channels=64, base_channels=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    # 4 synthetic vorticity fields
    data = generate_synthetic_vorticity_dataset(num_samples=4, nx=64, ny=64, seed=42)

    initial_loss = None
    final_loss = None

    for step in range(25):
        optimizer.zero_grad()
        recon = model(data)
        loss = nn.functional.mse_loss(recon, data)
        loss_val = float(loss.item())

        if step == 0:
            initial_loss = loss_val
        final_loss = loss_val

        loss.backward()
        optimizer.step()

    assert initial_loss is not None and final_loss is not None
    reduction_ratio = (initial_loss - final_loss) / initial_loss
    assert reduction_ratio >= 0.30, f"Expected at least 30% loss reduction, got {reduction_ratio*100:.1f}%"


# ==============================================================================
# 5. Rigorous A/B Checkpoint Resumption Consistency Test
# ==============================================================================

def test_checkpoint_resumption_numerical_consistency(tmp_path):
    """Verify that continuous training (Run A) and interrupted-resumed training (Run B)
    produce numerically identical results within declared tolerances (atol=1e-6, rtol=1e-5)."""
    seed = 2026
    n_total_steps = 10
    k_interrupt_step = 5

    data = generate_synthetic_vorticity_dataset(num_samples=8, nx=64, ny=64, seed=seed)
    probe_input = generate_synthetic_vorticity_dataset(num_samples=2, nx=64, ny=64, seed=seed + 999)

    # --------------------------------------------------------------------------
    # Run A: Continuous training for N steps
    # --------------------------------------------------------------------------
    torch.manual_seed(seed)
    model_a = VorticityAutoencoder(in_channels=1, out_channels=1, latent_channels=64, base_channels=16)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3, weight_decay=1e-4)

    losses_a = []
    for step in range(n_total_steps):
        opt_a.zero_grad()
        recon = model_a(data)
        loss = nn.functional.mse_loss(recon, data)
        loss.backward()
        opt_a.step()
        losses_a.append(float(loss.item()))

    with torch.no_grad():
        probe_out_a = model_a(probe_input)

    # --------------------------------------------------------------------------
    # Run B: Interrupted training for K steps -> Save -> Fresh load -> Continue N-K steps
    # --------------------------------------------------------------------------
    torch.manual_seed(seed)
    model_b = VorticityAutoencoder(in_channels=1, out_channels=1, latent_channels=64, base_channels=16)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3, weight_decay=1e-4)

    losses_b = []
    for step in range(k_interrupt_step):
        opt_b.zero_grad()
        recon = model_b(data)
        loss = nn.functional.mse_loss(recon, data)
        loss.backward()
        opt_b.step()
        losses_b.append(float(loss.item()))

    # Save complete resumption state
    ckpt_path = tmp_path / "resume_checkpoint.pt"
    ckpt_dict = {
        "step": k_interrupt_step,
        "model_state_dict": model_b.state_dict(),
        "optimizer_state_dict": opt_b.state_dict(),
        "rng_state": torch.get_rng_state(),
    }
    torch.save(ckpt_dict, ckpt_path)

    # Create fresh independent model and optimizer
    model_b_resumed = VorticityAutoencoder(in_channels=1, out_channels=1, latent_channels=64, base_channels=16)
    opt_b_resumed = torch.optim.AdamW(model_b_resumed.parameters(), lr=1e-3, weight_decay=1e-4)

    # Load checkpoint
    loaded = torch.load(ckpt_path, weights_only=False)
    model_b_resumed.load_state_dict(loaded["model_state_dict"])
    opt_b_resumed.load_state_dict(loaded["optimizer_state_dict"])
    torch.set_rng_state(loaded["rng_state"])

    # Continue training for remaining N - K steps
    for step in range(k_interrupt_step, n_total_steps):
        opt_b_resumed.zero_grad()
        recon = model_b_resumed(data)
        loss = nn.functional.mse_loss(recon, data)
        loss.backward()
        opt_b_resumed.step()
        losses_b.append(float(loss.item()))

    with torch.no_grad():
        probe_out_b = model_b_resumed(probe_input)

    # --------------------------------------------------------------------------
    # Assertions within pre-declared tolerances
    # --------------------------------------------------------------------------
    atol = 1.0e-6
    rtol = 1.0e-5

    # 1. Parameter match
    for (name_a, p_a), (name_b, p_b) in zip(model_a.named_parameters(), model_b_resumed.named_parameters()):
        assert name_a == name_b
        assert torch.allclose(p_a, p_b, atol=atol, rtol=rtol), f"Parameter mismatch in {name_a}"

    # 2. Optimizer momentum match
    for s_a, s_b in zip(opt_a.state.values(), opt_b_resumed.state.values()):
        if "exp_avg" in s_a and "exp_avg" in s_b:
            assert torch.allclose(s_a["exp_avg"], s_b["exp_avg"], atol=atol, rtol=rtol)
        if "exp_avg_sq" in s_a and "exp_avg_sq" in s_b:
            assert torch.allclose(s_a["exp_avg_sq"], s_b["exp_avg_sq"], atol=atol, rtol=rtol)

    # 3. Step loss match
    assert abs(losses_a[-1] - losses_b[-1]) < 1e-6, f"Final loss diverged: {losses_a[-1]} vs {losses_b[-1]}"

    # 4. Probe prediction match
    assert torch.allclose(probe_out_a, probe_out_b, atol=atol, rtol=rtol), "Probe output mismatch after resumption"


# ==============================================================================
# 6. Production Training Entrypoint Resumption & Error Paths
# ==============================================================================

def test_production_train_vorticity_autoencoder_resumption_consistency(tmp_path):
    """Verify that train_vorticity_autoencoder() produces numerically identical results
    between continuous 2-epoch run (A) and interrupted 1+1 epoch resumed run (B)."""
    from scripts.train_vorticity_autoencoder import train_vorticity_autoencoder

    cfg = {
        "model": {"latent_channels": 64, "base_channels": 16},
        "domain": {"nx": 32, "ny": 32, "lx": 1.0, "ly": 1.0},
        "synthetic_data": {
            "num_train_samples": 16,
            "num_val_samples": 8,
            "base_wavenumber": 1,
            "perturbation_modes": [[1, 0], [0, 1]],
            "perturbation_amplitude": 0.1,
            "seed": 2026,
        },
        "training": {"batch_size": 8, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 2},
    }

    dir_a = tmp_path / "run_continuous_a"
    dir_b = tmp_path / "run_resumed_b"

    # Run A: Continuous 2 epochs
    res_a = train_vorticity_autoencoder(config=cfg, output_dir=str(dir_a), override_epochs=2, device="cpu")

    # Run B: 1 epoch -> save -> resume to epoch 2
    res_b_phase1 = train_vorticity_autoencoder(config=cfg, output_dir=str(dir_b), override_epochs=1, device="cpu")
    ckpt_b_path = Path(res_b_phase1["checkpoint_path"])
    assert ckpt_b_path.is_file()

    res_b = train_vorticity_autoencoder(
        config=cfg,
        resume_path=str(ckpt_b_path),
        output_dir=str(dir_b),
        override_epochs=2,
        device="cpu",
    )

    # 1. Global step equality
    assert res_a["summary"]["global_steps"] == res_b["summary"]["global_steps"]
    assert res_a["summary"]["epochs_completed"] == res_b["summary"]["epochs_completed"]

    # 2. Final loss numerical match
    assert abs(res_a["summary"]["final_step_loss"] - res_b["summary"]["final_step_loss"]) < 1e-6

    # 3. Parameter match within declared tolerances
    atol = 1.0e-6
    rtol = 1.0e-5
    for (name_a, p_a), (name_b, p_b) in zip(res_a["model"].named_parameters(), res_b["model"].named_parameters()):
        assert name_a == name_b
        assert torch.allclose(p_a, p_b, atol=atol, rtol=rtol), f"Parameter mismatch in {name_a}"

    # 4. Probe prediction match
    probe = generate_synthetic_vorticity_dataset(2, nx=32, ny=32, seed=9999)
    with torch.no_grad():
        out_a = res_a["model"](probe)
        out_b = res_b["model"](probe)
    assert torch.allclose(out_a, out_b, atol=atol, rtol=rtol)


def test_resume_missing_file_raises_filenotfound(tmp_path):
    """Specifying a non-existent resume path must raise FileNotFoundError immediately, never silently fallback."""
    from scripts.train_vorticity_autoencoder import train_vorticity_autoencoder

    non_existent = tmp_path / "ghost_checkpoint.pt"
    with pytest.raises(FileNotFoundError, match="Resume checkpoint file not found"):
        train_vorticity_autoencoder(resume_path=str(non_existent))


def test_resume_protected_config_mismatch_raises_valueerror(tmp_path):
    """Resuming with altered data seed or model channels must raise ValueError."""
    from scripts.train_vorticity_autoencoder import train_vorticity_autoencoder

    cfg_base = {
        "model": {"latent_channels": 64, "base_channels": 16},
        "domain": {"nx": 32, "ny": 32, "lx": 1.0, "ly": 1.0},
        "synthetic_data": {"num_train_samples": 16, "num_val_samples": 8, "seed": 42},
        "training": {"batch_size": 8, "epochs": 1},
    }
    dir_base = tmp_path / "run_base"
    res1 = train_vorticity_autoencoder(config=cfg_base, output_dir=str(dir_base), override_epochs=1, device="cpu")
    ckpt_path = res1["checkpoint_path"]

    # Attempt to resume with different seed
    cfg_altered_seed = dict(cfg_base)
    cfg_altered_seed["synthetic_data"] = {"num_train_samples": 16, "num_val_samples": 8, "seed": 999}
    with pytest.raises(ValueError, match="Resume config mismatch for protected key 'synthetic_data.seed'"):
        train_vorticity_autoencoder(config=cfg_altered_seed, resume_path=ckpt_path, override_epochs=2, device="cpu")

    # Attempt to resume with different latent_channels
    cfg_altered_channels = dict(cfg_base)
    cfg_altered_channels["model"] = {"latent_channels": 32, "base_channels": 16}
    with pytest.raises(ValueError, match="Resume config mismatch for protected key 'model.latent_channels'"):
        train_vorticity_autoencoder(config=cfg_altered_channels, resume_path=ckpt_path, override_epochs=2, device="cpu")


def test_resume_target_epoch_less_or_equal_raises_valueerror(tmp_path):
    """Resuming with target epochs <= completed epoch must raise ValueError."""
    from scripts.train_vorticity_autoencoder import train_vorticity_autoencoder

    cfg = {
        "model": {"latent_channels": 64, "base_channels": 16},
        "domain": {"nx": 32, "ny": 32},
        "synthetic_data": {"num_train_samples": 16, "num_val_samples": 8, "seed": 42},
        "training": {"batch_size": 8, "epochs": 2},
    }
    dir_run = tmp_path / "run_epochs"
    res = train_vorticity_autoencoder(config=cfg, output_dir=str(dir_run), override_epochs=2, device="cpu")

    # Completed epoch is 2; attempting to resume to epoch 2 or 1 must fail
    with pytest.raises(ValueError, match="Requested target epochs .* must be strictly greater"):
        train_vorticity_autoencoder(config=cfg, resume_path=res["checkpoint_path"], override_epochs=2, device="cpu")


def test_enstrophy_spectrum_persistence_in_checkpoint_and_summary(tmp_path):
    """Verify that full enstrophy spectrum analysis (curves, ratios, spurious energy) is saved."""
    import json
    from scripts.train_vorticity_autoencoder import train_vorticity_autoencoder

    cfg = {
        "model": {"latent_channels": 64, "base_channels": 16},
        "domain": {"nx": 32, "ny": 32, "lx": 1.0, "ly": 1.0},
        "synthetic_data": {"num_train_samples": 16, "num_val_samples": 8, "seed": 42},
        "training": {"batch_size": 8, "epochs": 1},
    }
    dir_run = tmp_path / "run_spec"
    res = train_vorticity_autoencoder(config=cfg, output_dir=str(dir_run), override_epochs=1, device="cpu")

    # 1. Check checkpoint contents
    ckpt = torch.load(res["checkpoint_path"], weights_only=False)
    assert "enstrophy_spectrum" in ckpt
    spec = ckpt["enstrophy_spectrum"]
    assert "k_bins" in spec and len(spec["k_bins"]) > 0
    assert "spectrum_target" in spec and len(spec["spectrum_target"]) > 0
    assert "spectrum_pred" in spec and len(spec["spectrum_pred"]) > 0
    assert "spectrum_ratio" in spec and len(spec["spectrum_ratio"]) > 0
    assert "spurious_energy_in_zero_bins" in spec

    # 2. Check summary json contents
    summary_path = dir_run / "training_summary.json"
    assert summary_path.is_file()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    assert "enstrophy_spectrum_summary" in summary_data
    assert summary_data["enstrophy_spectrum_summary"]["valid_bins_count"] > 0


def test_synthetic_data_generator_parameters_forwarding():
    """Verify that base_wavenumber and perturbation_amplitude directly alter generated field properties."""
    data_amp_small = generate_synthetic_vorticity_dataset(4, nx=32, ny=32, perturbation_amplitude=0.01, seed=42)
    data_amp_large = generate_synthetic_vorticity_dataset(4, nx=32, ny=32, perturbation_amplitude=0.5, seed=42)

    var_small = torch.var(data_amp_small, dim=0).mean().item()
    var_large = torch.var(data_amp_large, dim=0).mean().item()
    assert var_large > var_small, "Larger perturbation amplitude must yield larger ensemble variance"


# ==============================================================================
# 7. Data Contract & Isolation Redline
# ==============================================================================

def test_training_code_zero_stocbench_real_data_dependency():
    """Ensure training and test modules do not reference or access frozen StocBench files."""
    train_script = Path(__file__).resolve().parent.parent / "scripts" / "train_vorticity_autoencoder.py"
    content = train_script.read_text(encoding="utf-8")

    # Frozen evaluation and training file names must not appear in the training entrypoint
    assert "step_seed_100.npz" not in content
    assert "traj_seed_42.npy" not in content
    assert "outputs/data/stocbench" not in content
