"""Comprehensive unit and integration tests for Curriculum Rollout & Pushforward Training."""

import pytest
import torch
import torch.nn as nn

from src.models.history_buffer import HistoryBuffer
from src.models.latent_forecaster import LatentForecaster
from src.models.latent_transformer import LatentSTTransformer
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D
from src.training.curriculum import (
    CurriculumConfig,
    CurriculumRolloutScheduler,
    apply_pushforward_warmup,
)


class TestCurriculumRolloutScheduler:
    """Tests for CurriculumRolloutScheduler horizon progression and pushforward scheduling."""

    def test_doubling_schedule(self):
        """Schedule 'doubling': H doubles every step_epochs until reaching target_horizon."""
        config = CurriculumConfig(
            enabled=True,
            start_horizon=2,
            target_horizon=16,
            step_epochs=3,
            schedule="doubling",
            pushforward_steps=4,
        )
        scheduler = CurriculumRolloutScheduler(config)

        # Epochs 1..3 -> H=2
        assert scheduler.get_horizon(1) == 2
        assert scheduler.get_horizon(2) == 2
        assert scheduler.get_horizon(3) == 2

        # Epochs 4..6 -> H=4
        assert scheduler.get_horizon(4) == 4
        assert scheduler.get_horizon(5) == 4
        assert scheduler.get_horizon(6) == 4

        # Epochs 7..9 -> H=8
        assert scheduler.get_horizon(7) == 8
        assert scheduler.get_horizon(8) == 8
        assert scheduler.get_horizon(9) == 8

        # Epochs 10..15 -> H=16 (clamped to target)
        assert scheduler.get_horizon(10) == 16
        assert scheduler.get_horizon(12) == 16
        assert scheduler.get_horizon(20) == 16

    def test_linear_schedule(self):
        """Schedule 'linear': H increments by start_horizon every step_epochs."""
        config = CurriculumConfig(
            enabled=True,
            start_horizon=2,
            target_horizon=10,
            step_epochs=2,
            schedule="linear",
        )
        scheduler = CurriculumRolloutScheduler(config)

        # Epochs 1..2 -> H=2
        assert scheduler.get_horizon(1) == 2
        assert scheduler.get_horizon(2) == 2

        # Epochs 3..4 -> H=4
        assert scheduler.get_horizon(3) == 4
        assert scheduler.get_horizon(4) == 4

        # Epochs 5..6 -> H=6
        assert scheduler.get_horizon(5) == 6

        # Epochs 7..8 -> H=8
        assert scheduler.get_horizon(7) == 8

        # Epochs 9+ -> H=10 (clamped)
        assert scheduler.get_horizon(9) == 10
        assert scheduler.get_horizon(15) == 10

    def test_disabled_or_fixed_schedule(self):
        """When curriculum is disabled or schedule is 'fixed', horizon is constant at target_horizon."""
        cfg_disabled = CurriculumConfig(
            enabled=False,
            start_horizon=2,
            target_horizon=16,
        )
        s_disabled = CurriculumRolloutScheduler(cfg_disabled)
        assert s_disabled.get_horizon(1) == 16
        assert s_disabled.get_horizon(10) == 16

        cfg_fixed = CurriculumConfig(
            enabled=True,
            start_horizon=2,
            target_horizon=16,
            schedule="fixed",
        )
        s_fixed = CurriculumRolloutScheduler(cfg_fixed)
        assert s_fixed.get_horizon(1) == 16
        assert s_fixed.get_horizon(10) == 16

    def test_pushforward_proportional_scaling(self):
        """Pushforward warmup steps scale proportionally with horizon when curriculum is enabled."""
        config = CurriculumConfig(
            enabled=True,
            start_horizon=2,
            target_horizon=16,
            step_epochs=2,
            schedule="doubling",
            pushforward_steps=8,
            pushforward_noise_std=0.01,
        )
        scheduler = CurriculumRolloutScheduler(config)

        # At epoch 1 (H=2, ratio=2/16=0.125): 8 * 0.125 = 1 step
        assert scheduler.get_pushforward_steps(1) == 1

        # At epoch 3 (H=4, ratio=4/16=0.25): 8 * 0.25 = 2 steps
        assert scheduler.get_pushforward_steps(3) == 2

        # At epoch 5 (H=8, ratio=8/16=0.5): 8 * 0.5 = 4 steps
        assert scheduler.get_pushforward_steps(5) == 4

        # At epoch 7 (H=16, ratio=1.0): 8 steps
        assert scheduler.get_pushforward_steps(7) == 8

        # Noise std matches config
        assert scheduler.get_noise_std(1) == 0.01

    def test_invalid_config_validation(self):
        """Fail-closed validation rejects illegal curriculum parameters."""
        with pytest.raises(ValueError, match="start_horizon must be >= 1"):
            CurriculumConfig(start_horizon=0)

        with pytest.raises(ValueError, match="cannot be less than start_horizon"):
            CurriculumConfig(start_horizon=8, target_horizon=4)

        with pytest.raises(ValueError, match="step_epochs must be >= 1"):
            CurriculumConfig(step_epochs=0)

        with pytest.raises(ValueError, match="Unknown schedule"):
            CurriculumConfig(schedule="polynomial")

        with pytest.raises(ValueError, match="pushforward_steps must be >= 0"):
            CurriculumConfig(pushforward_steps=-1)

        with pytest.raises(ValueError, match="pushforward_noise_std must be >= 0"):
            CurriculumConfig(pushforward_noise_std=-0.5)

        with pytest.raises(ValueError, match="Unknown pushforward_mode"):
            CurriculumConfig(pushforward_mode="random_mode")

    def test_state_dict_serialization_roundtrip(self):
        """Scheduler configuration correctly serializes and restores from state_dict."""
        config = CurriculumConfig(
            enabled=True,
            start_horizon=4,
            target_horizon=32,
            step_epochs=5,
            schedule="doubling",
            pushforward_steps=6,
            pushforward_noise_std=0.05,
            pushforward_mode="history",
        )
        scheduler = CurriculumRolloutScheduler(config)
        sd = scheduler.state_dict()

        restored_scheduler = CurriculumRolloutScheduler()
        restored_scheduler.load_state_dict(sd)

        assert restored_scheduler.config.enabled is True
        assert restored_scheduler.config.start_horizon == 4
        assert restored_scheduler.config.target_horizon == 32
        assert restored_scheduler.config.step_epochs == 5
        assert restored_scheduler.config.schedule == "doubling"
        assert restored_scheduler.config.pushforward_steps == 6
        assert restored_scheduler.config.pushforward_noise_std == 0.05
        assert restored_scheduler.config.pushforward_mode == "history"


class TestPushforwardHistoryBuffer:
    """Tests for HistoryBuffer pushforward and noise injection mechanics."""

    def test_pushforward_stop_gradient(self):
        """Warmup pushforward steps must advance state without creating computation graph."""
        b, l, c = 2, 4, 8
        init_history = torch.randn(b, l, c, requires_grad=True)

        buf = HistoryBuffer(history_length=l)
        buf.reset(init_history)

        linear_step = nn.Linear(c, c)

        def step_fn(hist, _cond=None):
            return linear_step(hist[:, -1])

        # Pushforward 3 steps without gradient
        buf.pushforward(step_fn, steps=3, noise_std=0.0)

        assert buf.current.shape == (b, l, c)
        # Pushforward itself must not build grad graph on linear_step weights
        assert linear_step.weight.grad is None

        # Now do 2 rollout steps with gradients
        pred = buf.rollout(step_fn, steps=2)
        assert pred.shape == (b, 2, c)

        # Loss and backward
        loss = pred.sum()
        loss.backward()

        # Gradients must exist on model weights from the rollout steps
        assert linear_step.weight.grad is not None
        assert torch.isfinite(linear_step.weight.grad).all()

    def test_pushforward_noise_injection(self):
        """Noise injection perturbs predictions while maintaining deterministic shape."""
        torch.manual_seed(42)
        b, l, c = 2, 4, 8
        init_history = torch.zeros(b, l, c)

        buf1 = HistoryBuffer(history_length=l).reset(init_history)
        buf2 = HistoryBuffer(history_length=l).reset(init_history)

        def zero_step(hist, _c=None):
            return torch.zeros(b, 1, c)

        # Zero step with 0 noise -> all zeros
        buf1.pushforward(zero_step, steps=2, noise_std=0.0)
        assert torch.all(buf1.current == 0.0)

        # Zero step with noise > 0 -> non-zero perturbed buffer
        buf2.pushforward(zero_step, steps=2, noise_std=0.5)
        assert not torch.all(buf2.current == 0.0)
        assert buf2.current.shape == (b, l, c)


class TestLatentForecasterPushforward:
    """Tests for LatentForecaster end-to-end pushforward execution and gradient flow."""

    @pytest.fixture
    def small_forecaster(self):
        encoder = Encoder2D(in_channels=4, latent_channels=16, base_channels=16)
        decoder = Decoder2D(latent_channels=16, out_channels=4, base_channels=16)
        transformer = LatentSTTransformer(
            latent_channels=16,
            embed_dim=32,
            cond_dim=16,
            depth=2,
            num_heads=2,
            history_length=4,
            prediction_mode="direct",
        )
        return LatentForecaster(
            encoder=encoder,
            transformer=transformer,
            decoder=decoder,
            freeze_representation=True,
        )

    def test_latent_forecaster_pushforward_rollout(self, small_forecaster):
        """Verify LatentForecaster.forward_rollout with pushforward_steps and noise."""
        b, l, c, ny, nx = 2, 4, 4, 16, 32
        q_hist = torch.randn(b, l, c, ny, nx)
        re = torch.tensor([1e4, 2e4])
        sc = torch.tensor([0.1, 0.5])

        # Pushforward 2 steps, then roll out 3 steps
        out = small_forecaster.forward_rollout(
            q_hist=q_hist,
            re=re,
            sc=sc,
            horizon=3,
            pushforward_steps=2,
            noise_std=0.01,
        )
        assert out.shape == (b, 3, c, ny, nx)

        # Backward pass check
        loss = out.sum()
        loss.backward()

        # Transformer weights must receive gradients
        param_with_grad = False
        for p in small_forecaster.transformer.parameters():
            if p.grad is not None:
                param_with_grad = True
                assert torch.isfinite(p.grad).all()
        assert param_with_grad, "Transformer parameters did not receive gradients"

        # Encoder weights are frozen, must have no gradients
        for p in small_forecaster.encoder.parameters():
            assert p.grad is None


class TestTrainForecasterCurriculumIntegration:
    """Integration tests for train_forecaster script curriculum/pushforward wiring."""

    def test_train_epoch_pushforward_future_mode(self, tmp_path):
        """Verify _train_epoch runs cleanly with pushforward future mode."""
        from scripts.train_forecaster import _train_epoch
        from src.losses.field import FieldLoss
        from src.losses.rollout import RolloutLoss
        from src.losses.divergence import DivergenceLoss
        from src.losses.vorticity import VorticityLoss

        b, l, c, ny, nx = 2, 4, 4, 16, 32
        device = torch.device("cpu")

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Conv2d(c, c, 3, padding=1)

            def forward(self, q_hist, re=None, sc=None, horizon=1, pushforward_steps=0, noise_std=0.0):
                # Return dummy future trajectory
                return torch.zeros(b, horizon, c, ny, nx, requires_grad=True)

        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(enabled=False)

        # Batch with future horizon = 6 (pushforward=2 + horizon=4)
        dummy_batch = {
            "history": torch.randn(b, l, c, ny, nx),
            "future": torch.randn(b, 6, c, ny, nx),
            "re": torch.tensor([1e4, 2e4]),
            "sc": torch.tensor([0.1, 0.5]),
        }
        train_loader = [dummy_batch]

        field_loss = FieldLoss(loss_type="mse")
        rollout_loss = RolloutLoss(field_loss=field_loss)
        div_loss = DivergenceLoss()
        vort_loss = VorticityLoss()

        loss_val = _train_epoch(
            model=model,
            model_type="latent_transformer",
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            normalizer=None,
            rollout_loss_fn=rollout_loss,
            div_loss_fn=div_loss,
            vort_loss_fn=vort_loss,
            horizon=4,
            grad_accum_steps=1,
            field_loss_space="normalized",
            lambda_div=0.0,
            lambda_vort=0.0,
            use_condition=True,
            use_amp=False,
            device=device,
            is_distributed=False,
            train_sampler=None,
            epoch=1,
            pushforward_steps=2,
            pushforward_noise_std=0.01,
            pushforward_mode="future",
        )
        assert isinstance(loss_val, float)
        assert loss_val >= 0.0

    def test_train_epoch_pushforward_history_mode(self):
        """Verify _train_epoch runs cleanly with pushforward history mode."""
        from scripts.train_forecaster import _train_epoch
        from src.losses.field import FieldLoss
        from src.losses.rollout import RolloutLoss
        from src.losses.divergence import DivergenceLoss
        from src.losses.vorticity import VorticityLoss

        b, l, c, ny, nx = 2, 4, 4, 16, 32
        device = torch.device("cpu")

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Conv2d(c, c, 3, padding=1)

            def forward(self, q_hist, re=None, sc=None, horizon=1, pushforward_steps=0, noise_std=0.0):
                return torch.zeros(b, horizon, c, ny, nx, requires_grad=True)

        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(enabled=False)

        dummy_batch = {
            "history": torch.randn(b, l, c, ny, nx),
            "future": torch.randn(b, 2, c, ny, nx),
            "re": torch.tensor([1e4, 2e4]),
            "sc": torch.tensor([0.1, 0.5]),
        }
        train_loader = [dummy_batch]

        loss_val = _train_epoch(
            model=model,
            model_type="latent_transformer",
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            normalizer=None,
            rollout_loss_fn=RolloutLoss(field_loss=FieldLoss()),
            div_loss_fn=DivergenceLoss(),
            vort_loss_fn=VorticityLoss(),
            horizon=2,
            grad_accum_steps=1,
            field_loss_space="normalized",
            lambda_div=0.0,
            lambda_vort=0.0,
            use_condition=True,
            use_amp=False,
            device=device,
            is_distributed=False,
            train_sampler=None,
            epoch=1,
            pushforward_steps=1,
            pushforward_noise_std=0.0,
            pushforward_mode="history",
        )
        assert isinstance(loss_val, float)
        assert loss_val >= 0.0

