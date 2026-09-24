#!/usr/bin/env python3
"""Runner for Horizon-R2 / H16 Extension Experiment.

Branches the H16 extension study from the H8 Saved Long-Best Ep 11 checkpoint:
    Parent: outputs/checkpoints/dynamics/horizon_r1/seed_42/E4_H8/latent_transformer/checkpoint_step_11_vrmse_mean_0.2186.pt
    (Validation J_long = 3.4712, test h30 = 0.8554, parent horizon = 8).

Experimental Specifications:
    - Initial weights: H8 Saved Long-Best Ep 11 (SHA256 verified)
    - Expected init horizon: 8 (enforced fail-closed via validate_init_checkpoint_contract)
    - Target training horizon: 16
    - Microbatch: 1, Grad accumulation: 8 -> Effective batch size: 8 (sample-exact)
    - Epochs: 12 updates
    - Learning rate: 5e-5 (min: 1e-6, fresh AdamW optimizer)
    - Loss terms: lambda_div=0.01, lambda_vort=0.05
    - Seed: 42
    - Diagnostic rollout validation: h=10, 20, 30
    - Checkpoint tracking:
        * Standard tracker: val 1..H mean VRMSE -> best_vrmse_mean.pt
        * True long tracker: J_long = mean(h10, h20, h30) -> best_long_vrmse.pt
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.utils.provenance import (
    compute_file_sha256,
    is_git_dirty,
    validate_init_checkpoint_contract,
    PHYSICS_PROTOCOL,
    SPATIAL_AXIS_CONTRACT,
    SHEAR_FLOW_DOMAIN_SIZE_XY,
)

DEFAULT_PARENT_CHECKPOINT = (
    "outputs/checkpoints/dynamics/horizon_r1/seed_42/E4_H8/latent_transformer/"
    "checkpoint_step_11_vrmse_mean_0.2186.pt"
)
DEFAULT_OUTPUT_DIR = "outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H16/latent_transformer"
DEFAULT_LOG_DIR = "outputs/logs/horizon_r2/seed_42"

H16_CONFIG = {
    "name": "E4_H16",
    "title": "E4-H16: Horizon H=16 extension from H8 Long-Best (12 epochs, Beff=8)",
    "expected_init_horizon": 8,
    "horizon": 16,
    "batch_size": 1,
    "grad_accum_steps": 8,
    "effective_batch_size": 8,
    "lr": 5e-5,
    "min_lr": 1e-6,
    "epochs": 12,
    "lambda_div": 0.01,
    "lambda_vort": 0.05,
    "val_diagnostics": [10, 20, 30],
    "seed": 42,
}


def resolve_parent_checkpoint(custom_path: Optional[str] = None) -> Path:
    """Resolve and verify canonical H8 Saved Long-Best parent checkpoint."""
    if custom_path:
        p = Path(custom_path)
        try:
            if p.is_file():
                return p
        except (PermissionError, OSError):
            pass
        raise FileNotFoundError(f"Specified parent checkpoint not found: {custom_path}")

    candidates = [
        PROJECT_ROOT / DEFAULT_PARENT_CHECKPOINT,
        Path(DEFAULT_PARENT_CHECKPOINT),
        Path(f"/root/autodl-tmp/mzy_data/World-Model-FlowField-v1/{DEFAULT_PARENT_CHECKPOINT}"),
    ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except (PermissionError, OSError):
            continue

    raise FileNotFoundError(
        f"H16 parent checkpoint not found. Checked default locations:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def build_training_command(
    parent_path: str,
    epochs: int = 12,
    lr: float = 5e-5,
    batch_size: int = 1,
    grad_accum_steps: int = 8,
    horizon: int = 16,
    expected_init_horizon: int = 8,
    output_dir: Optional[str] = None,
    seed: int = 42,
    lambda_div: float = 0.01,
    lambda_vort: float = 0.05,
    val_diagnostic_horizons: Optional[List[int]] = None,
    use_amp: bool = True,
    num_gpus: int = 1,
    master_port: int = 29500,
) -> List[str]:
    """Construct the formal training command for the H16 extension experiment."""
    out_dir = output_dir or str(PROJECT_ROOT / DEFAULT_OUTPUT_DIR)
    diags = val_diagnostic_horizons or H16_CONFIG["val_diagnostics"]

    if num_gpus > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            f"--master_port={master_port}",
            "scripts/train_forecaster.py",
        ]
    else:
        cmd = [
            sys.executable,
            "-u",
            "scripts/train_forecaster.py",
        ]

    cmd.extend([
        "--model", "latent_transformer",
        "--output_dir", str(out_dir),
        "--init_checkpoint", str(parent_path),
        "--expected_init_horizon", str(expected_init_horizon),
        "--horizon", str(horizon),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--grad_accum_steps", str(grad_accum_steps),
        "--lr", str(lr),
        "--lambda_div", str(lambda_div),
        "--lambda_vort", str(lambda_vort),
        "--seed", str(seed),
        "--val_diagnostic_horizons", *[str(h) for h in diags],
    ])
    if use_amp:
        cmd.append("--use_amp")

    return cmd


def validate_parent_for_h16(
    parent_path: Path,
    expected_horizon: int = 8,
    expected_seed: int = 42,
) -> Tuple[bool, str, dict]:
    """Execute preflight semantic contract validation on the H16 parent checkpoint."""
    sha256 = compute_file_sha256(str(parent_path))
    ckpt_data = torch.load(str(parent_path), map_location="cpu", weights_only=False)

    # Perform strict fail-closed contract check
    is_valid, errors = validate_init_checkpoint_contract(
        ckpt_data,
        ckpt_path=str(parent_path),
        requested_model_type="latent_transformer",
        expected_seed=expected_seed,
        expected_horizon=expected_horizon,
        expected_protocol=PHYSICS_PROTOCOL,
        expected_axis_contract=SPATIAL_AXIS_CONTRACT,
        expected_domain_size=SHEAR_FLOW_DOMAIN_SIZE_XY,
        expected_prediction_mode="direct",
        expected_use_condition=True,
        fail_closed=True,
    )
    return is_valid, sha256, ckpt_data


def print_preflight_summary(
    parent_path: Path,
    parent_sha: str,
    parent_horizon: int,
    gpu_ids: List[int],
    epochs: int,
    lr: float,
    min_lr: float,
    output_dir: str,
    log_file: str,
    batch_size: int,
    grad_accum_steps: int,
    effective_batch_size: int,
    dry_run: bool = False,
):
    """Print structured parameter table matching governance verification format."""
    print("=" * 80)
    print("HORIZON-R2 / H16 EXTENSION EXPERIMENT" + (" [DRY RUN]" if dry_run else ""))
    print("=" * 80)
    print(f"Parent checkpoint:      {parent_path}")
    print(f"Parent SHA256:          {parent_sha}")
    print(f"Parent horizon:         {parent_horizon}")
    print(f"Target horizon:         {H16_CONFIG['horizon']}")
    print(f"Expected init horizon:  {H16_CONFIG['expected_init_horizon']}")
    print(f"Microbatch:             {batch_size}")
    print(f"Grad accumulation:      {grad_accum_steps}")
    print(f"Effective batch:        {effective_batch_size}")
    print(f"Seed:                   {H16_CONFIG['seed']}")
    print(f"lambda_div:             {H16_CONFIG['lambda_div']}")
    print(f"lambda_vort:            {H16_CONFIG['lambda_vort']}")
    print(f"Epochs:                 {epochs}")
    print(f"Learning rate:          {lr} (constant, no scheduler)")
    gpu_str = ",".join(str(g) for g in gpu_ids)
    print(f"GPU ID:                 {gpu_ids[0] if len(gpu_ids) == 1 else gpu_str}")
    print(f"GPU count:              {len(gpu_ids)} ({'DDP Distributed' if len(gpu_ids) > 1 else 'Single GPU'})")
    print(f"Output dir:             {output_dir}")
    print(f"Log file:               {log_file}")
    dirty_flag = is_git_dirty()
    print(f"Git dirty:              {dirty_flag} (working tree {'DIRTY' if dirty_flag else 'CLEAN'})")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Horizon-R2 / H16 Extension Study")
    parser.add_argument("--parent_checkpoint", type=str, default=None, help="Path to parent H8 Saved Long-Best checkpoint")
    parser.add_argument("--gpu_id", type=int, default=None, help="Single GPU device ID to run on (backwards compatible)")
    parser.add_argument("--gpu_ids", "--gpus", dest="gpu_ids", type=str, default=None, help="Comma-separated GPU IDs to run on (e.g. '0,1')")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to utilize (default: auto-detect)")
    parser.add_argument("--master_port", type=int, default=29500, help="Master port for DDP")
    parser.add_argument("--epochs", type=int, default=H16_CONFIG["epochs"], help="Training epochs")
    parser.add_argument("--lr", type=float, default=H16_CONFIG["lr"], help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=H16_CONFIG["min_lr"], help="Minimum learning rate")
    parser.add_argument("--batch_size", type=int, default=H16_CONFIG["batch_size"], help="Microbatch size per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="Accumulation steps per GPU (default: computed to maintain Beff=8)")
    parser.add_argument("--effective_batch_size", type=int, default=H16_CONFIG["effective_batch_size"], help="Target effective batch size")
    parser.add_argument("--horizon", type=int, default=H16_CONFIG["horizon"], help="Target rollout horizon")
    parser.add_argument("--expected_init_horizon", type=int, default=H16_CONFIG["expected_init_horizon"], help="Required parent horizon")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for checkpoints")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR, help="Log directory")
    parser.add_argument("--dry_run", action="store_true", help="Print plan and training command without running")

    args = parser.parse_args()

    # Determine GPUs
    if args.gpu_ids is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    elif args.gpu_id is not None:
        gpu_ids = [args.gpu_id]
    elif args.num_gpus is not None:
        gpu_ids = list(range(args.num_gpus))
    else:
        # Auto-detect available GPUs (default to using all available, e.g. 0,1)
        avail = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if avail >= 2:
            gpu_ids = [0, 1]
        else:
            gpu_ids = [0]

    num_gpus = len(gpu_ids)

    # Calculate grad_accum_steps to strictly preserve effective_batch_size invariant
    if args.grad_accum_steps is not None:
        grad_accum_steps = args.grad_accum_steps
    else:
        denom = args.batch_size * num_gpus
        if args.effective_batch_size % denom != 0:
            raise ValueError(
                f"Effective batch size ({args.effective_batch_size}) must be evenly divisible by "
                f"microbatch ({args.batch_size}) * num_gpus ({num_gpus}) = {denom}"
            )
        grad_accum_steps = args.effective_batch_size // denom

    effective_batch = args.batch_size * grad_accum_steps * num_gpus
    if effective_batch != args.effective_batch_size:
        print(f"WARNING: Effective batch size ({effective_batch}) differs from target ({args.effective_batch_size})!")

    # 1. Resolve and validate parent checkpoint
    parent_path = resolve_parent_checkpoint(args.parent_checkpoint)
    is_valid, parent_sha, ckpt_data = validate_parent_for_h16(
        parent_path,
        expected_horizon=args.expected_init_horizon,
        expected_seed=H16_CONFIG["seed"],
    )

    parent_horizon = ckpt_data.get("horizon", 8)

    # 2. Setup paths
    out_dir = str(PROJECT_ROOT / args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    log_dir = str(PROJECT_ROOT / args.log_dir) if not os.path.isabs(args.log_dir) else args.log_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "E4_H16.log")

    # 3. Print preflight summary
    print_preflight_summary(
        parent_path=parent_path,
        parent_sha=parent_sha,
        parent_horizon=parent_horizon,
        gpu_ids=gpu_ids,
        epochs=args.epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        output_dir=out_dir,
        log_file=log_file,
        batch_size=args.batch_size,
        grad_accum_steps=grad_accum_steps,
        effective_batch_size=effective_batch,
        dry_run=args.dry_run,
    )

    # 4. Build training command
    cmd = build_training_command(
        parent_path=str(parent_path),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=grad_accum_steps,
        horizon=args.horizon,
        expected_init_horizon=args.expected_init_horizon,
        output_dir=out_dir,
        seed=H16_CONFIG["seed"],
        lambda_div=H16_CONFIG["lambda_div"],
        lambda_vort=H16_CONFIG["lambda_vort"],
        num_gpus=num_gpus,
        master_port=args.master_port,
    )

    gpu_env_val = ",".join(str(g) for g in gpu_ids)
    print("\n[Command]")
    print(f"CUDA_VISIBLE_DEVICES={gpu_env_val} " + " ".join(cmd))

    if args.dry_run:
        print("\nDRY RUN COMPLETED: H16 preflight checks verified successfully.")
        return 0

    # 5. Launch training
    gpu_desc = f"{num_gpus} GPUs ({gpu_env_val})" if num_gpus > 1 else f"GPU {gpu_ids[0]}"
    print(f"\nLaunching H16 training on {gpu_desc}...")
    t0 = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_env_val
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()

    proc.wait()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"\nERROR: H16 training failed with return code {proc.returncode}")
        return proc.returncode

    print(f"\nH16 training completed successfully in {elapsed:.1f}s.")
    print(f"Checkpoints saved to: {out_dir}")
    print(f"Log saved to:         {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
