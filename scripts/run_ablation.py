"""Orchestrator script for the 5 prioritized ablation experiments defined in V1 spec."""

import argparse
import subprocess
import sys


ABLATIONS = {
    1: {
        "title": "Ablation 1: Direct ST Transformer vs Latent ST Transformer",
        "commands": [
            "python scripts/train_forecaster.py --model direct_transformer --epochs 30",
            "python scripts/train_forecaster.py --model latent_transformer --epochs 30",
        ],
    },
    2: {
        "title": "Ablation 2: E0 (Single-Step) vs E1 (Rollout-Aware Training)",
        "commands": [
            "python scripts/train_forecaster.py --model latent_transformer --horizon 1 --epochs 30",
            "python scripts/train_forecaster.py --model latent_transformer --horizon 4 --epochs 30",
        ],
    },
    3: {
        "title": "Ablation 3: Physical Loss Ablation (E1 vs E2 Divergence vs E3 Vorticity)",
        "commands": [
            "python scripts/train_forecaster.py --model latent_transformer --horizon 4 --lambda_div 0.0 --lambda_vort 0.0",
            "python scripts/train_forecaster.py --model latent_transformer --horizon 4 --lambda_div 0.01 --lambda_vort 0.0",
            "python scripts/train_forecaster.py --model latent_transformer --horizon 4 --lambda_div 0.01 --lambda_vort 0.05",
        ],
    },
    4: {
        "title": "Ablation 4: Frozen vs Joint Representation",
        "commands": [
            "python scripts/train_forecaster.py --model latent_transformer --epochs 30",
        ],
    },
    5: {
        "title": "Ablation 5: Direct Latent vs Residual Latent Prediction",
        "commands": [
            "python scripts/train_forecaster.py --model latent_transformer --epochs 30",
        ],
    },
}


def run_ablation(ablation_id: int):
    if ablation_id not in ABLATIONS:
        print(f"Unknown ablation ID: {ablation_id}. Available: 1, 2, 3, 4, 5")
        sys.exit(1)

    info = ABLATIONS[ablation_id]
    print("=" * 80)
    print(f"STARTING: {info['title']}")
    print("=" * 80)

    for cmd in info["commands"]:
        print(f"\nExecuting: {cmd}")
        ret = subprocess.run(cmd, shell=True)
        if ret.returncode != 0:
            print(f"Command failed with return code {ret.returncode}: {cmd}")
            sys.exit(ret.returncode)

    print("\n" + "=" * 80)
    print(f"COMPLETED: {info['title']}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V1 ablation studies.")
    parser.add_argument("--ablation", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Ablation experiment index (1-5).")
    args = parser.parse_args()

    run_ablation(args.ablation)
