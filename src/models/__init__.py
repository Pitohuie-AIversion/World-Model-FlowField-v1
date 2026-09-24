"""Models package for World-Model-FlowField-v1.

Exports spatial autoencoder components (Encoder2D, Decoder2D),
history buffer for pure-latent autoregression, spatio-temporal transformers,
conditioning embedding layers, and end-to-end forecasting world models.
"""

from src.models.encoder import Encoder2D, ResConvBlock2D
from src.models.decoder import Decoder2D, UpBlock2D
from src.models.conditioning import (
    PhysicalConditionEmbedding,
    AdaLN,
    AdaLNZeroBlock,
)
from src.models.history_buffer import HistoryBuffer
from src.models.latent_transformer import (
    LatentSTTransformer,
    FactorizedSTBlock,
    SpatialAttention,
    TemporalAttention,
)
from src.models.latent_forecaster import LatentForecaster
from src.models.direct_transformer import DirectSTTransformer

__all__ = [
    "Encoder2D",
    "ResConvBlock2D",
    "Decoder2D",
    "UpBlock2D",
    "PhysicalConditionEmbedding",
    "AdaLN",
    "AdaLNZeroBlock",
    "HistoryBuffer",
    "LatentSTTransformer",
    "FactorizedSTBlock",
    "SpatialAttention",
    "TemporalAttention",
    "LatentForecaster",
    "DirectSTTransformer",
]
