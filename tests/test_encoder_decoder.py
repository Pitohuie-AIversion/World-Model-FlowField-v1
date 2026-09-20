"""Unit tests for 2D spatial Encoder and Decoder."""

import torch
import pytest
from src.models.encoder import Encoder2D
from src.models.decoder import Decoder2D


def test_encoder_decoder_shapes():
    """Verify 8x downsampling and exact shape reconstruction."""
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=True)

    # Test 128x256 resolution
    x_128 = torch.randn(2, 4, 128, 256)
    z_128 = encoder(x_128)
    assert z_128.shape == (2, 64, 16, 32), f"Unexpected latent shape: {z_128.shape}"

    x_rec_128 = decoder(z_128)
    assert x_rec_128.shape == (2, 4, 128, 256), f"Unexpected recon shape: {x_rec_128.shape}"

    # Test 256x512 resolution
    x_256 = torch.randn(1, 4, 256, 512)
    z_256 = encoder(x_256)
    assert z_256.shape == (1, 64, 32, 64), f"Unexpected latent shape: {z_256.shape}"

    x_rec_256 = decoder(z_256)
    assert x_rec_256.shape == (1, 4, 256, 512), f"Unexpected recon shape: {x_rec_256.shape}"


def test_5d_sequence_tensor():
    """Verify handling of temporal sequence (B, T, C, H, W)."""
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32)

    seq_x = torch.randn(2, 4, 4, 128, 256)  # B=2, T=4
    seq_z = encoder(seq_x)
    assert seq_z.shape == (2, 4, 64, 16, 32)

    seq_rec = decoder(seq_z)
    assert seq_rec.shape == (2, 4, 4, 128, 256)


def test_pressure_zero_mean_projection():
    """Verify decoder pressure channel has zero mean when enabled and can be toggled dynamically."""
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=True)
    z = torch.randn(2, 64, 16, 32)
    q = decoder(z)
    p = q[:, 2:3, :, :]
    mean_p = p.mean(dim=(-2, -1))
    assert torch.allclose(mean_p, torch.zeros_like(mean_p), atol=1e-5)

    # Test dynamic override project_pressure=False
    q_no_proj = decoder(z, project_pressure=False)
    assert q_no_proj.shape == q.shape

    # Test decoder initialized with False but dynamically enabled
    decoder_off = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=False)
    q_dyn_proj = decoder_off(z, project_pressure=True)
    p_dyn = q_dyn_proj[:, 2:3, :, :]
    assert torch.allclose(p_dyn.mean(dim=(-2, -1)), torch.zeros(2, 1), atol=1e-5)


def test_reconstruction_closed_loop_gradient():
    """Verify q -> Z -> q_tilde end-to-end backpropagation and zero-mean pressure."""
    encoder = Encoder2D(in_channels=4, latent_channels=64, base_channels=32)
    decoder = Decoder2D(latent_channels=64, out_channels=4, base_channels=32, project_pressure=True)

    x = torch.randn(2, 4, 128, 256, requires_grad=True)
    z = encoder(x)
    q_tilde = decoder(z)

    loss = torch.nn.functional.mse_loss(q_tilde, x)
    loss.backward()

    # Verify both encoder and decoder received gradients
    assert encoder.in_conv.weight.grad is not None
    assert decoder.out_conv[-1].weight.grad is not None
    assert x.grad is not None

    # Verify pressure zero-mean projection on output
    p_recon = q_tilde[:, 2]
    mean_p = p_recon.mean(dim=(-2, -1))
    assert torch.allclose(mean_p, torch.zeros_like(mean_p), atol=1e-5)
