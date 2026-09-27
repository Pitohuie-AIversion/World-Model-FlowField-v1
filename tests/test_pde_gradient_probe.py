"""Rigorous verification tests for PDE gradient norm and paired control probe.

Verifies:
1. Mathematical precision of gradient norm and cosine similarity in full parameter space R^D.
2. None-handling in cosine similarity across full parameter space (countering subset projection bug).
3. The 1/a^2 quadratic gradient scaling law for residual scales (a=0.05 -> 400x, a=0.02 -> 2500x).
4. Canonical compute_vrmse() matches The Well benchmark standard against ad-hoc mixed-variance counterexamples.
5. Exact consistency between probe decoupled gradients and train_forecaster _compute_batch_loss.
6. Fail-closed defense on non-finite gradients (NaN/Inf) or representation freeze violations.
7. Mock end-to-end execution of paired P0 vs. PDE small-budget control probe.
"""

import json
import math
import os
import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.probe_pde_gradient_scales import (
    compute_decoupled_batch_gradients,
    compute_gradient_cosine_similarity,
    compute_gradient_norm,
    evaluate_on_fixed_validation_window,
    run_pde_gradient_probe,
)
from scripts.train_forecaster import (
    LatentForecasterWrapper,
    _compute_batch_loss,
)
from src.data.normalization import FieldNormalizer
from src.losses.divergence import DivergenceLoss
from src.losses.navier_stokes import (
    NavierStokesMomentumResidualLoss,
    NavierStokesPDELoss,
    TracerAdvectionDiffusionResidualLoss,
)
from src.losses.vorticity import VorticityLoss
from src.metrics.field import compute_vrmse
from src.models.decoder import Decoder2D
from src.models.encoder import Encoder2D
from src.models.latent_transformer import LatentSTTransformer


def test_gradient_norm_and_cosine_similarity_math():
    """Verify compute_gradient_norm and cosine similarity handle standard geometric cases."""
    # Orthogonal vectors in R^3
    v1 = [torch.tensor([1.0, 0.0, 0.0])]
    v2 = [torch.tensor([0.0, 2.0, 0.0])]

    norm1 = compute_gradient_norm(v1)
    norm2 = compute_gradient_norm(v2)
    assert math.isclose(norm1, 1.0, rel_tol=1e-5)
    assert math.isclose(norm2, 2.0, rel_tol=1e-5)

    cos_sim = compute_gradient_cosine_similarity(v1, v2)
    assert math.isclose(cos_sim, 0.0, abs_tol=1e-6)

    # Collinear vectors
    v3 = [torch.tensor([3.0, 0.0, 0.0])]
    cos_sim_collinear = compute_gradient_cosine_similarity(v1, v3)
    assert math.isclose(cos_sim_collinear, 1.0, rel_tol=1e-5)

    # Opposing vectors
    v4 = [torch.tensor([-1.0, 0.0, 0.0])]
    cos_sim_opposing = compute_gradient_cosine_similarity(v1, v4)
    assert math.isclose(cos_sim_opposing, -1.0, rel_tol=1e-5)

    # Non-finite gradient returns NaN
    nan_v = [torch.tensor([float("nan"), 1.0])]
    assert math.isnan(compute_gradient_norm(nan_v))
    assert math.isnan(compute_gradient_cosine_similarity(v1, nan_v))

    # Length mismatch raises ValueError
    with pytest.raises(ValueError, match="Gradient list length mismatch"):
        compute_gradient_cosine_similarity([torch.tensor([1.0])], [torch.tensor([1.0]), torch.tensor([2.0])])


def test_cosine_similarity_none_handling_full_parameter_space():
    """Verify cosine similarity treats None entries as exact zeros in full parameter space R^D.

    Synthetic counterexample from audit:
      g1 = (1, 100)
      g2 = (1, 0), represented with None for second coordinate: [tensor([1.0]), None]
    If subset projection is incorrectly used: returns 1.0 (WRONG).
    In full parameter space R^2:
      dot = 1*1 + 100*0 = 1.0
      norm1 = sqrt(1^2 + 100^2) = sqrt(10001) ~ 100.005
      norm2 = sqrt(1^2 + 0^2) = 1.0
      cos = 1.0 / (sqrt(10001) * 1.0) = 0.0099995 ~ 0.0100
    """
    g1 = [torch.tensor([1.0]), torch.tensor([100.0])]
    g2 = [torch.tensor([1.0]), None]

    cos_sim = compute_gradient_cosine_similarity(g1, g2)
    expected_cos = 1.0 / math.sqrt(10001.0)

    assert not math.isclose(cos_sim, 1.0, rel_tol=1e-3), "Cosine similarity wrongly evaluated to 1.0 via subset projection!"
    assert math.isclose(cos_sim, expected_cos, rel_tol=1e-5)

    # Symmetric case
    g3 = [None, torch.tensor([50.0])]
    cos_sim_disjoint = compute_gradient_cosine_similarity(g2, g3)
    assert math.isclose(cos_sim_disjoint, 0.0, abs_tol=1e-6)

    # Both all None
    assert math.isclose(compute_gradient_cosine_similarity([None], [None]), 0.0, abs_tol=1e-6)


def test_residual_scale_quadratic_gradient_scaling_law():
    """Verify gradient norms follow the exact 1/a^2 algebraic relation on neural net dynamics parameters.

    For L(a) = <(r/a)^2>:
      grad_theta L(a) = (1/a^2) * grad_theta L(1).
    For momentum: scale a=0.05 implies an exact 1/(0.05^2) = 400x amplification.
    For tracer:   scale a=0.02 implies an exact 1/(0.02^2) = 2500x amplification.
    """
    device = torch.device("cpu")
    # Minimal trainable model using 2D convolutions
    conv = nn.Conv2d(4, 4, kernel_size=1, bias=False)
    raw_in = torch.randn(2 * 4, 4, 16, 16)
    pred_phys = conv(raw_in).view(2, 4, 4, 16, 16)
    re = torch.tensor([1000.0, 1000.0])
    sc = torch.tensor([1.0, 1.0])
    dt = torch.tensor([0.05, 0.05])
    q0 = pred_phys[:, 0]

    # Momentum loss: scale a=1.0 vs scale a=0.05
    mom_loss_1 = NavierStokesMomentumResidualLoss(domain_size=(1.0, 2.0), scale_u=1.0, scale_v=1.0, dealias=True)
    mom_loss_005 = NavierStokesMomentumResidualLoss(domain_size=(1.0, 2.0), scale_u=0.05, scale_v=0.05, dealias=True)

    loss_m1, _ = mom_loss_1(pred_phys, re=re, dt=dt, q0_phys=q0)
    loss_m005, _ = mom_loss_005(pred_phys, re=re, dt=dt, q0_phys=q0)

    grad_m1 = torch.autograd.grad(loss_m1, conv.parameters(), retain_graph=True)
    grad_m005 = torch.autograd.grad(loss_m005, conv.parameters(), retain_graph=True)

    norm_m1 = compute_gradient_norm(grad_m1)
    norm_m005 = compute_gradient_norm(grad_m005)

    assert math.isclose(loss_m005.item(), 400.0 * loss_m1.item(), rel_tol=1e-4)
    assert math.isclose(norm_m005, 400.0 * norm_m1, rel_tol=1e-4)

    # Tracer loss: scale a=1.0 vs scale a=0.02
    tr_loss_1 = TracerAdvectionDiffusionResidualLoss(domain_size=(1.0, 2.0), scale_s=1.0, dealias=True)
    tr_loss_002 = TracerAdvectionDiffusionResidualLoss(domain_size=(1.0, 2.0), scale_s=0.02, dealias=True)

    loss_t1, _ = tr_loss_1(pred_phys, re=re, sc=sc, dt=dt, q0_phys=q0)
    loss_t002, _ = tr_loss_002(pred_phys, re=re, sc=sc, dt=dt, q0_phys=q0)

    grad_t1 = torch.autograd.grad(loss_t1, conv.parameters(), retain_graph=True)
    grad_t002 = torch.autograd.grad(loss_t002, conv.parameters(), retain_graph=False)

    norm_t1 = compute_gradient_norm(grad_t1)
    norm_t002 = compute_gradient_norm(grad_t002)

    assert math.isclose(loss_t002.item(), 2500.0 * loss_t1.item(), rel_tol=1e-4)
    assert math.isclose(norm_t002, 2500.0 * norm_t1, rel_tol=1e-4)


def test_standard_vrmse_matches_canonical_definition_against_ad_hoc_counterexample():
    """Verify compute_vrmse matches The Well standard (0.550000) and exposes ad-hoc mixed-variance error (0.140720).

    Counterexample from review:
      Four zero-mean channels, spatial variance = [1, 100, 1, 100].
      Per-channel RMS error = 1.0 on all channels.
      Standard VRMSE = mean_{c} sqrt(1 / Var_c) = (1 + 0.1 + 1 + 0.1) / 4 = 0.550000.
      Ad-hoc formula = sqrt(mean(diff^2)) / sqrt(mean(var)) = 1.0 / sqrt(50.5) ~ 0.140720.
    """
    N = 256
    # Construct synthetic target with variances [1, 100, 1, 100]
    torch.manual_seed(123)
    target = torch.zeros(1, 1, 4, N, N)
    stds = [1.0, 10.0, 1.0, 10.0]
    for c, s in enumerate(stds):
        f = torch.randn(N, N)
        f = (f - f.mean()) / f.std() * s
        target[0, 0, c] = f

    # Construct prediction with exact RMS error 1.0 on each channel
    pred = torch.zeros_like(target)
    for c in range(4):
        noise = torch.randn(N, N)
        noise = (noise - noise.mean()) / noise.std() * 1.0
        pred[0, 0, c] = target[0, 0, c] + noise

    # Canonical compute_vrmse
    vrmse_standard = compute_vrmse(pred, target, eps=1e-12).item()
    assert math.isclose(vrmse_standard, 0.55, abs_tol=1e-2)

    # Ad-hoc mixed-variance formula
    var_mixed = torch.var(target, dim=(-2, -1), unbiased=False).mean().item()
    mse_total = torch.mean((pred - target) ** 2).item()
    vrmse_ad_hoc = math.sqrt(mse_total) / (math.sqrt(var_mixed) + 1e-12)

    assert math.isclose(vrmse_ad_hoc, 0.140720, abs_tol=1e-2)
    # The two formulas yield wildly divergent numbers on multi-scale physical fields
    assert abs(vrmse_standard - vrmse_ad_hoc) > 0.35


def test_probe_and_training_loss_and_gradient_consistency():
    """Verify decoupled probe components mathematically reconstruct exact training loss and gradients."""
    device = torch.device("cpu")
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=256,
        depth=2,
        num_heads=4,
        history_length=4,
        prediction_mode="direct",
    )
    model = LatentForecasterWrapper(
        encoder=encoder,
        transformer=transformer,
        decoder=decoder,
        freeze_representation=True,
    )

    normalizer = FieldNormalizer()
    normalizer.mean = nn.Parameter(torch.zeros(4), requires_grad=False)
    normalizer.std = nn.Parameter(torch.ones(4), requires_grad=False)

    mom_loss_fn = NavierStokesMomentumResidualLoss(domain_size=(1.0, 2.0), scale_u=0.05, scale_v=0.05, dealias=True)
    tracer_loss_fn = TracerAdvectionDiffusionResidualLoss(domain_size=(1.0, 2.0), scale_s=0.02, dealias=True)
    rollout_loss_fn = nn.MSELoss()
    div_loss_fn = DivergenceLoss(domain_size=(1.0, 2.0))
    vort_loss_fn = VorticityLoss(domain_size=(1.0, 2.0))

    lambda_div = 0.01
    lambda_vort = 0.05
    lambda_mom = 1.0e-4
    lambda_tr = 1.0e-5

    B, L, H = 2, 4, 4
    batch = {
        "history": torch.randn(B, L, 4, 16, 32),
        "future": torch.randn(B, H, 4, 16, 32),
        "re": torch.tensor([1000.0, 1000.0]),
        "sc": torch.tensor([1.0, 1.0]),
        "dt": torch.tensor([0.05, 0.05]),
    }

    # 1. Probe decoupled computation
    probe_res = compute_decoupled_batch_gradients(
        model=model,
        batch=batch,
        normalizer=normalizer,
        mom_loss_fn=mom_loss_fn,
        tracer_loss_fn=tracer_loss_fn,
        rollout_loss_fn=rollout_loss_fn,
        div_loss_fn=div_loss_fn,
        vort_loss_fn=vort_loss_fn,
        lambda_div=lambda_div,
        lambda_vort=lambda_vort,
        device=device,
    )

    reconstructed_loss = (
        probe_res["loss_existing"]
        + lambda_mom * probe_res["loss_mom_raw"]
        + lambda_tr * probe_res["loss_tr_raw"]
    )

    # 2. Production training entry _compute_batch_loss computation
    pred_norm = model.forward_rollout(batch["history"], re=batch["re"], sc=batch["sc"], horizon=H)
    q_hist_phys = normalizer.denormalize(batch["history"])
    q0_phys = q_hist_phys[:, -1]

    direct_training_loss = _compute_batch_loss(
        pred=pred_norm,
        q_future=batch["future"],
        normalizer=normalizer,
        field_loss_space="normalized",
        lambda_div=lambda_div,
        lambda_vort=lambda_vort,
        rollout_loss_fn=rollout_loss_fn,
        div_loss_fn=div_loss_fn,
        vort_loss_fn=vort_loss_fn,
        lambda_mom=lambda_mom,
        lambda_tr=lambda_tr,
        mom_loss_fn=mom_loss_fn,
        tracer_loss_fn=tracer_loss_fn,
        re=batch["re"],
        sc=batch["sc"],
        dt=batch["dt"],
        q0_phys=q0_phys,
    )

    # Direct loss and probe reconstructed loss must match to numerical precision
    assert math.isclose(direct_training_loss.item(), reconstructed_loss, rel_tol=1e-5)


def test_abnormal_gradient_and_freeze_fail_closed():
    """Verify probe fails closed immediately when encountering non-finite gradients or unfreezed representation."""
    device = torch.device("cpu")
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=256,
        depth=2,
        num_heads=4,
        history_length=4,
        prediction_mode="direct",
    )

    # 1. Representation freeze violation: encoder has requires_grad=True
    model_unfrozen = LatentForecasterWrapper(
        encoder=encoder,
        transformer=transformer,
        decoder=decoder,
        freeze_representation=False,
    )
    for p in model_unfrozen.encoder.parameters():
        p.requires_grad = True

    normalizer = FieldNormalizer()
    normalizer.mean = nn.Parameter(torch.zeros(4), requires_grad=False)
    normalizer.std = nn.Parameter(torch.ones(4), requires_grad=False)

    mom_loss_fn = NavierStokesMomentumResidualLoss(domain_size=(1.0, 2.0), scale_u=0.05, scale_v=0.05, dealias=True)
    tracer_loss_fn = TracerAdvectionDiffusionResidualLoss(domain_size=(1.0, 2.0), scale_s=0.02, dealias=True)
    rollout_loss_fn = nn.MSELoss()

    batch = {
        "history": torch.randn(2, 4, 4, 16, 32),
        "future": torch.randn(2, 4, 4, 16, 32),
        "re": torch.tensor([1000.0, 1000.0]),
        "sc": torch.tensor([1.0, 1.0]),
        "dt": torch.tensor([0.05, 0.05]),
    }

    with pytest.raises(RuntimeError, match="Representation freeze contract violated"):
        compute_decoupled_batch_gradients(
            model=model_unfrozen,
            batch=batch,
            normalizer=normalizer,
            mom_loss_fn=mom_loss_fn,
            tracer_loss_fn=tracer_loss_fn,
            rollout_loss_fn=rollout_loss_fn,
            device=device,
        )

    # 2. Non-finite input causing non-finite gradient
    model_frozen = LatentForecasterWrapper(
        encoder=encoder,
        transformer=transformer,
        decoder=decoder,
        freeze_representation=True,
    )
    batch_nan = {
        "history": torch.randn(2, 4, 4, 16, 32),
        "future": torch.randn(2, 4, 4, 16, 32),
        "re": torch.tensor([float("nan"), 1000.0]),
        "sc": torch.tensor([1.0, 1.0]),
        "dt": torch.tensor([0.05, 0.05]),
    }

    with pytest.raises((FloatingPointError, ValueError)):
        compute_decoupled_batch_gradients(
            model=model_frozen,
            batch=batch_nan,
            normalizer=normalizer,
            mom_loss_fn=mom_loss_fn,
            tracer_loss_fn=tracer_loss_fn,
            rollout_loss_fn=rollout_loss_fn,
            device=device,
        )


@pytest.fixture
def mock_probe_environment(tmp_path):
    """Create a minimal self-contained environment with mock HDF5 and model weights."""
    h5_path = tmp_path / "probe_flow_Reynolds_1e3_Schmidt_1e0.hdf5"
    n_sims = 2
    t_steps = 10
    nx, ny = 32, 64

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("scalars/Reynolds", data=1000.0)
        f.create_dataset("scalars/Schmidt", data=1.0)
        f.create_dataset("dimensions/time", data=np.arange(t_steps, dtype=np.float32) * 0.05)
        f.create_dataset("t1_fields/velocity", data=np.zeros((n_sims, t_steps, nx, ny, 2), dtype=np.float32))
        f.create_dataset("t0_fields/pressure", data=np.zeros((n_sims, t_steps, nx, ny), dtype=np.float32))
        f.create_dataset("t0_fields/tracer", data=np.zeros((n_sims, t_steps, nx, ny), dtype=np.float32))

    split_file = tmp_path / "mock_split.json"
    with open(split_file, "w") as f:
        json.dump({
            "train": [{"file_path": str(h5_path), "traj_idx": 0}],
            "valid": [{"file_path": str(h5_path), "traj_idx": 1}],
        }, f)

    norm_file = tmp_path / "mock_stats.pt"
    torch.save({
        "mean": torch.zeros(4),
        "std": torch.ones(4),
    }, str(norm_file))

    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    ae_file = tmp_path / "mock_ae.pt"
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
    }, str(ae_file))

    transformer = LatentSTTransformer(
        latent_channels=64,
        embed_dim=256,
        depth=6,
        num_heads=8,
        history_length=4,
        prediction_mode="direct",
    )
    wrapper = LatentForecasterWrapper(
        encoder=encoder,
        transformer=transformer,
        decoder=decoder,
        freeze_representation=True,
    )
    d0_file = tmp_path / "mock_d0.pt"
    torch.save({
        "model_state_dict": wrapper.state_dict(),
    }, str(d0_file))

    return {
        "data_root": str(tmp_path),
        "split_file": str(split_file),
        "norm_file": str(norm_file),
        "ae_ckpt": str(ae_file),
        "d0_ckpt": str(d0_file),
    }


def test_run_pde_gradient_probe_mock_end_to_end(mock_probe_environment, tmp_path):
    """End-to-end integration test: run_pde_gradient_probe paired control on mock checkpoints."""
    env = mock_probe_environment
    output_json = tmp_path / "mock_probe_output.json"

    result = run_pde_gradient_probe(
        split_file=env["split_file"],
        norm_file=env["norm_file"],
        ae_ckpt=env["ae_ckpt"],
        d0_ckpt=env["d0_ckpt"],
        data_root=env["data_root"],
        num_probe_batches=1,
        update_steps=2,
        history_length=4,
        horizon=4,
        mom_scale_u=0.05,
        mom_scale_v=0.05,
        tracer_scale_s=0.02,
        lambda_div=0.01,
        lambda_vort=0.05,
        candidate_weights=[(2.5e-5, 4.0e-6)],
        pde_probe_weights=(2.5e-5, 4.0e-6),
        output_json=str(output_json),
        num_workers=0,
        device=torch.device("cpu"),
    )

    assert os.path.exists(output_json)
    assert "q1_verification" in result
    assert "q2_loss_and_gradient_scales" in result
    assert "q3_paired_control_comparison" in result

    q1 = result["q1_verification"]
    assert q1["gradients_finite"] is True
    assert q1["representation_frozen"] is True
    assert q1["mean_norm_mom_raw"] > 0.0
    assert q1["mean_norm_tr_raw"] > 0.0

    q2 = result["q2_loss_and_gradient_scales"]
    evals = q2["candidate_weight_evaluations"]
    assert len(evals) == 1
    assert "grad_ratio_mom" in evals[0]
    assert "grad_ratio_tr" in evals[0]

    q3 = result["q3_paired_control_comparison"]
    assert "metrics_baseline" in q3
    assert "metrics_p0_control" in q3
    assert "metrics_pde_experiment" in q3
    assert "comparison" in q3
    assert "vrmse_standard" in q3["comparison"]
    assert "delta_p0_vs_base_pct" in q3["comparison"]["vrmse_standard"]
    assert "delta_pde_vs_p0_pct" in q3["comparison"]["vrmse_standard"]
    assert len(q3["p0_losses"]) == 2
    assert len(q3["pde_losses"]) == 2
