"""Curriculum and pushforward rollout training mechanisms for physical world models.

Provides:
1. CurriculumRolloutScheduler: Dynamically adapts rollout horizon and pushforward steps across epochs.
2. PushforwardConfig: Configuration dataclass for curriculum and pushforward parameters.
3. apply_pushforward_warmup: Executes stop-gradient autoregressive unrolls and latent noise injection.
"""

from src.training.curriculum import (
    CurriculumConfig,
    CurriculumRolloutScheduler,
    apply_pushforward_warmup,
)

__all__ = [
    "CurriculumConfig",
    "CurriculumRolloutScheduler",
    "apply_pushforward_warmup",
]
