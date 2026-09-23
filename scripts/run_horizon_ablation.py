"""Orchestrator for Horizon-R1 controlled horizon ablation study (Closure-R5 / Horizon-R1).

Branches 3 strictly controlled experiments from the same E4 Seed 42 H=2 best checkpoint:
    1. E4_H2_control: Continued H=2 training (control for additional training steps)
    2. E4_H4: Horizon H=4 training (test of 2x horizon expansion)
    3. E4_H8: Horizon H=8 training (test of 4x horizon expansion)

Controls:
    - Initial weights: identical parent E4-H2 best checkpoint (SHA256 verified)
    - Effective batch size: 8 across all groups (via grad_accum_steps)
        * H2: microbatch=8, accum=1 -> eff=8
        * H4: microbatch=4, accum=2 -> eff=8
        * H8: microbatch=2, accum=4 -> eff=8
    - Epochs: 12 updates for all groups
    - Learning rate: 5e-5 for all groups (fresh AdamW optimizer, lr reset)
    - Loss terms: lambda_div=0.01, lambda_vort=0.05, physical field loss space
    - Diagnostic rollout validation: h=10, 20, 30 (tracked on validation split)
    - Checkpoint criterion: strictly 1..H mean VRMSE (preserves semantic selection)
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PARENT_CHECKPOINTS = [
    "outputs/checkpoints/dynamics/closure_r4/ablation_E4_full_physics/latent_transformer/best_vrmse_mean.pt",
    "outputs/checkpoints/dynamics/closure_r4/seed_42/ablation_E4_full_physics/latent_transformer/best_vrmse_mean.pt",
    "/root/autodl-tmp/mzy_data/World-Model-FlowField-v1/outputs/checkpoints/dynamics/closure_r4/ablation_E4_full_physics/latent_transformer/best_vrmse_mean.pt",
]

HORIZON_CONFIGS: Dict[str, Dict[str, object]] = {
    "E4_H2_control": {
        "title": "E4-H2-Control: Continued H=2 training (12 epochs, lr=5e-5, Beff=8)",
        "horizon": 2,
        "batch_size": 8,
        "grad_accum_steps": 1,
        "output_dir_suffix": "E4_H2_control",
        "log_suffix": "E4_H2_control",
    },
    "E4_H4": {
        "title": "E4-H4: Horizon H=4 training (12 epochs, lr=5e-5, Beff=8)",
        "horizon": 4,
        "batch_size": 4,
        "grad_accum_steps": 2,
        "output_dir_suffix": "E4_H4",
        "log_suffix": "E4_H4",
    },
    "E4_H8": {
        "title": "E4-H8: Horizon H=8 training (12 epochs, lr=5e-5, Beff=8)",
        "horizon": 8,
        "batch_size": 2,
        "grad_accum_steps": 4,
        "output_dir_suffix": "E4_H8",
        "log_suffix": "E4_H8",
    },
}


def resolve_parent_checkpoint(custom_path: Optional[str] = None) -> Path:
    """Resolve the canonical parent E4 Seed 42 H=2 checkpoint path."""
    if custom_path:
        p = Path(custom_path)
        try:
            if p.is_file():
                return p
        except (PermissionError, OSError):
            pass
        raise FileNotFoundError(f"Specified parent checkpoint not found: {custom_path}")

    for candidate in DEFAULT_PARENT_CHECKPOINTS:
        p = Path(candidate)
        try:
            if p.is_file():
                return p
        except (PermissionError, OSError):
            continue

    raise FileNotFoundError(
        "Could not resolve parent E4 Seed 42 H=2 checkpoint. Checked candidates:\n"
        + "\n".join(f"  - {c}" for c in DEFAULT_PARENT_CHECKPOINTS)
    )


def compute_file_sha256(filepath: Path) -> str:
    """Compute deterministic SHA256 fingerprint of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_training_command(
    group_name: str,
    gpu_id: int,
    parent_checkpoint: Path,
    seed: int = 42,
    epochs: int = 12,
    lr: float = 5e-5,
    lambda_div: float = 0.01,
    lambda_vort: float = 0.05,
    val_diagnostic_horizons: Optional[List[int]] = None,
) -> Tuple[str, str, str]:
    """Construct command string, output dir, and log file for a horizon ablation run."""
    if group_name not in HORIZON_CONFIGS:
        raise ValueError(f"Unknown group '{group_name}'. Available: {list(HORIZON_CONFIGS.keys())}")

    if val_diagnostic_horizons is None:
        val_diagnostic_horizons = [10, 20, 30]

    cfg = HORIZON_CONFIGS[group_name]
    output_dir = f"outputs/checkpoints/dynamics/horizon_r1/seed_{seed}/{cfg['output_dir_suffix']}"
    log_file = f"outputs/train_horizon_r1_seed_{seed}_{cfg['log_suffix']}.log"

    diag_str = " ".join(str(h) for h in val_diagnostic_horizons)

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python -u scripts/train_forecaster.py "
        f"--model latent_transformer "
        f"--output_dir {output_dir} "
        f"--init_checkpoint {parent_checkpoint} "
        f"--horizon {cfg['horizon']} "
        f"--epochs {epochs} "
        f"--batch_size {cfg['batch_size']} "
        f"--grad_accum_steps {cfg['grad_accum_steps']} "
        f"--lr {lr} "
        f"--lambda_div {lambda_div} "
        f"--lambda_vort {lambda_vort} "
        f"--seed {seed} "
        f"--val_diagnostic_horizons {diag_str} "
        f"--use_amp "
        f"> {log_file} 2>&1"
    )
    return cmd, output_dir, log_file


def run_horizon_group(
    group_name: str,
    gpu_id: int,
    parent_checkpoint: Path,
    seed: int = 42,
    epochs: int = 12,
    lr: float = 5e-5,
    lambda_div: float = 0.01,
    lambda_vort: float = 0.05,
    val_diagnostic_horizons: Optional[List[int]] = None,
    dry_run: bool = False,
) -> int:
    """Run or preview a single horizon ablation group."""
    cfg = HORIZON_CONFIGS[group_name]
    cmd, output_dir, log_file = build_training_command(
        group_name=group_name,
        gpu_id=gpu_id,
        parent_checkpoint=parent_checkpoint,
        seed=seed,
        epochs=epochs,
        lr=lr,
        lambda_div=lambda_div,
        lambda_vort=lambda_vort,
        val_diagnostic_horizons=val_diagnostic_horizons,
    )

    effective_batch = int(cfg["batch_size"]) * int(cfg["grad_accum_steps"])
    print(f"\n[{group_name}] {cfg['title']}")
    print(f"  - Horizon: {cfg['horizon']}")
    print(f"  - Microbatch: {cfg['batch_size']}, Accum Steps: {cfg['grad_accum_steps']} -> Effective Batch: {effective_batch}")
    print(f"  - Epochs: {epochs}, LR: {lr}")
    print(f"  - Output Dir: {output_dir}")
    print(f"  - Log File: {log_file}")
    print(f"  - Parent Checkpoint: {parent_checkpoint}")
    print(f"  - Command: {cmd}")

    if dry_run:
        print(f"  [DRY RUN] Skipped execution of {group_name}.")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print(f"[GPU {gpu_id} | Seed {seed}] Starting {cfg['title']}...", flush=True)
    start_time = time.time()
    ret = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time

    if ret.returncode == 0:
        print(f"[GPU {gpu_id} | Seed {seed}] COMPLETED: {cfg['title']} in {elapsed:.1f}s", flush=True)
        return 0
    else:
        print(f"[GPU {gpu_id} | Seed {seed}] FAILED: {cfg['title']} with code {ret.returncode}", flush=True)
        return ret.returncode


def main():
    parser = argparse.ArgumentParser(description="Horizon-R1 Controlled Horizon Ablation Orchestrator.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=12, help="Number of fine-tuning epochs (default: 12)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for warm start (default: 5e-5)")
    parser.add_argument("--gpu_id", type=int, default=1, help="GPU device ID (default: 1)")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        choices=["E4_H2_control", "E4_H4", "E4_H8"],
        help="Ablation groups to run (default: all three sequentially)",
    )
    parser.add_argument(
        "--parent_checkpoint",
        type=str,
        default=None,
        help="Custom path to parent E4 Seed 42 H=2 checkpoint",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned commands and invariant configurations without training",
    )
    args = parser.parse_args()

    parent_ckpt = resolve_parent_checkpoint(args.parent_checkpoint)
    parent_sha = compute_file_sha256(parent_ckpt)

    groups_to_run = args.groups or list(HORIZON_CONFIGS.keys())

    print("=" * 80)
    print("HORIZON-R1 CONTROLLED HORIZON ABLATION EXPERIMENT")
    print(f"Parent Checkpoint: {parent_ckpt}")
    print(f"Parent SHA256:     {parent_sha}")
    print(f"Seed:              {args.seed}")
    print(f"Epochs per group:  {args.epochs}")
    print(f"Learning Rate:     {args.lr}")
    print(f"GPU ID:            {args.gpu_id}")
    print(f"Target Groups:     {groups_to_run}")
    print(f"Dry Run:           {args.dry_run}")
    print("=" * 80)

    for grp in groups_to_run:
        code = run_horizon_group(
            group_name=grp,
            gpu_id=args.gpu_id,
            parent_checkpoint=parent_ckpt,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            dry_run=args.dry_run,
        )
        if code != 0:
            print(f"\nExecution aborted due to failure in group {grp}.")
            sys.exit(code)

    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN COMPLETED: All invariant controls verified successfully.")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print(f"ALL HORIZON-R1 ABLATION GROUPS COMPLETED SUCCESSFULLY FOR SEED {args.seed}!")
        print("=" * 80)


if __name__ == "__main__":
    main()
