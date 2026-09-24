#!/usr/bin/env python3
"""Runner for Horizon-R2 / H12 Extension Study.

Branches the H12 extension candidate comparison from the H8 Saved Long-Best Ep 11 checkpoint:
    Parent: outputs/checkpoints/dynamics/horizon_r1/seed_42/E4_H8/latent_transformer/checkpoint_step_11_vrmse_mean_0.2186.pt
    (Validation J_long = 3.4712, test h30 = 0.8554, parent horizon = 8).

Experimental Specifications (Approved Baseline):
    - Initial weights: H8 Saved Long-Best Ep 11 (Full SHA256 match enforced)
    - Expected init horizon: 8 (enforced fail-closed via validate_init_checkpoint_contract)
    - Target training horizon: 12 (fixed throughout, curriculum_rollout=False)
    - Learning rate: constant 5e-5 (fresh AdamW, no lr scheduler, no min_lr)
    - Pushforward & Spectral loss: disabled (pushforward_steps=0, lambda_spec=0.0)
    - Compile model: disabled (compile_model=False, minimize extraneous variables)
    - Loss terms: lambda_div=0.01, lambda_vort=0.05 (E4 full physics)
    - Seed: 42
    - Microbatch: 1, Grad accumulation: 4 per GPU on 2 GPUs -> Beff = 8 (sample-exact)
    - Epochs: 12
    - Diagnostic rollout validation: h=10, 20, 30
    - Checkpoint tracking:
        * Primary tracker: J_long = mean(h10, h20, h30) -> best_long_vrmse.pt
        * Auxiliary tracker: val 1..12 mean VRMSE -> best_vrmse_mean.pt
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
EXPECTED_PARENT_SHA256 = "ea038c7af0f3a0fa5d851f25dd1ea224371b8c44ab2a80b8314f5c286c224147"

DEFAULT_OUTPUT_DIR = "outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer"
DEFAULT_LOG_DIR = "outputs/logs/horizon_r2/seed_42"

H12_CONFIG = {
    "name": "E4_H12",
    "title": "E4-H12: Horizon H=12 extension from H8 Long-Best (12 epochs, Beff=8, constant lr=5e-5)",
    "expected_init_horizon": 8,
    "horizon": 12,
    "batch_size": 1,
    "grad_accum_steps": 4,  # default for 2 GPUs: 1 * 4 * 2 = 8
    "effective_batch_size": 8,
    "lr": 5e-5,
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
        f"H12 parent checkpoint not found. Checked default locations:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def validate_parent_for_h12(
    parent_path: Path,
    expected_horizon: int = 8,
    expected_seed: int = 42,
    expected_sha256: Optional[str] = EXPECTED_PARENT_SHA256,
) -> Tuple[bool, str, dict]:
    """Execute preflight semantic contract validation and full SHA-256 check on parent checkpoint."""
    sha256 = compute_file_sha256(str(parent_path))
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError(
            f"Parent checkpoint SHA-256 mismatch!\n"
            f"Expected: {expected_sha256}\n"
            f"Actual:   {sha256}\n"
            f"Refusing to train from unverified parent checkpoint."
        )

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


def build_training_command(
    parent_path: str,
    epochs: int = 12,
    lr: float = 5e-5,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    horizon: int = 12,
    expected_init_horizon: int = 8,
    output_dir: Optional[str] = None,
    seed: int = 42,
    lambda_div: float = 0.01,
    lambda_vort: float = 0.05,
    val_diagnostic_horizons: Optional[List[int]] = None,
    use_amp: bool = True,
    num_gpus: int = 2,
    master_port: int = 29501,
) -> List[str]:
    """Construct the formal training command for the H12 extension study.

    Note:
        - Constant learning rate lr=5e-5 (no lr scheduler, no min_lr).
        - Pushforward disabled (pushforward_steps=0).
        - Spectral loss disabled (lambda_spec=0.0).
        - Curriculum rollout disabled (fixed horizon=12 throughout).
        - Compile model disabled (compile_model=False).
    """
    out_dir = output_dir or str(PROJECT_ROOT / DEFAULT_OUTPUT_DIR)
    diags = val_diagnostic_horizons or H12_CONFIG["val_diagnostics"]

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
        "--lambda_spec", "0.0",
        "--pushforward_steps", "0",
        "--seed", str(seed),
        "--val_diagnostic_horizons", *[str(h) for h in diags],
    ])
    if use_amp:
        cmd.append("--use_amp")

    return cmd


def print_preflight_summary(
    parent_path: Path,
    parent_sha: str,
    parent_horizon: int,
    gpu_ids: List[int],
    epochs: int,
    lr: float,
    output_dir: str,
    log_file: str,
    batch_size: int,
    grad_accum_steps: int,
    effective_batch_size: int,
    dry_run: bool = False,
):
    """Print structured parameter table matching governance verification format."""
    print("=" * 80)
    print("HORIZON-R2 / H12 EXTENSION EXPERIMENT" + (" [DRY RUN]" if dry_run else ""))
    print("=" * 80)
    print(f"Parent checkpoint:      {parent_path}")
    print(f"Parent SHA256:          {parent_sha}")
    print(f"Parent horizon:         {parent_horizon}")
    print(f"Target horizon:         {H12_CONFIG['horizon']} (constant throughout)")
    print(f"Expected init horizon:  {H12_CONFIG['expected_init_horizon']}")
    print(f"Microbatch per GPU:     {batch_size}")
    print(f"Grad accumulation:      {grad_accum_steps}")
    print(f"Effective batch size:   {effective_batch_size}")
    print(f"Seed:                   {H12_CONFIG['seed']}")
    print(f"lambda_div:             {H12_CONFIG['lambda_div']}")
    print(f"lambda_vort:            {H12_CONFIG['lambda_vort']}")
    print(f"lambda_spec:            0.0 (disabled)")
    print(f"pushforward_steps:      0 (disabled)")
    print(f"curriculum_rollout:     False (disabled)")
    print(f"compile_model:          False (disabled)")
    print(f"Epochs:                 {epochs}")
    print(f"Learning rate:          {lr} (constant, fresh AdamW, no scheduler)")
    gpu_str = ",".join(str(g) for g in gpu_ids)
    print(f"GPU ID(s):              {gpu_ids[0] if len(gpu_ids) == 1 else gpu_str}")
    print(f"GPU count:              {len(gpu_ids)} ({'DDP Distributed' if len(gpu_ids) > 1 else 'Single GPU'})")
    print(f"Output directory:       {output_dir}")
    print(f"Log file:               {log_file}")
    dirty_flag = is_git_dirty()
    print(f"Git dirty:              {dirty_flag} (working tree {'DIRTY' if dirty_flag else 'CLEAN'})")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Horizon-R2 / H12 Extension Study")
    parser.add_argument("--parent_checkpoint", type=str, default=None, help="Path to parent H8 Saved Long-Best checkpoint")
    parser.add_argument("--gpu_id", type=int, default=None, help="Single GPU device ID to run on (backwards compatible)")
    parser.add_argument("--gpu_ids", "--gpus", dest="gpu_ids", type=str, default=None, help="Comma-separated GPU IDs to run on (e.g. '0,1')")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to utilize")
    parser.add_argument("--master_port", type=int, default=29501, help="Master port for DDP")
    parser.add_argument("--epochs", type=int, default=H12_CONFIG["epochs"], help="Training epochs")
    parser.add_argument("--lr", type=float, default=H12_CONFIG["lr"], help="Constant learning rate (default: 5e-5)")
    parser.add_argument("--batch_size", type=int, default=H12_CONFIG["batch_size"], help="Microbatch size per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="Accumulation steps per GPU (default: computed to maintain Beff=8)")
    parser.add_argument("--effective_batch_size", type=int, default=H12_CONFIG["effective_batch_size"], help="Target effective batch size")
    parser.add_argument("--horizon", type=int, default=H12_CONFIG["horizon"], help="Target rollout horizon")
    parser.add_argument("--expected_init_horizon", type=int, default=H12_CONFIG["expected_init_horizon"], help="Required parent horizon")
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
        # Default priority: use dual-GPU if available
        available_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available_count >= 2:
            gpu_ids = [0, 1]
        elif available_count == 1:
            gpu_ids = [0]
        else:
            gpu_ids = []

    num_gpus = len(gpu_ids)

    # Compute grad_accum_steps to strictly preserve Beff = 8
    target_beff = args.effective_batch_size
    microbatch = args.batch_size
    if args.grad_accum_steps is not None:
        grad_accum_steps = args.grad_accum_steps
        effective_beff = microbatch * grad_accum_steps * max(num_gpus, 1)
    else:
        if num_gpus > 1:
            grad_accum_steps = max(1, target_beff // (microbatch * num_gpus))
            effective_beff = microbatch * grad_accum_steps * num_gpus
        else:
            grad_accum_steps = max(1, target_beff // microbatch)
            effective_beff = microbatch * grad_accum_steps

    # Resolve and validate parent checkpoint
    parent_path = resolve_parent_checkpoint(args.parent_checkpoint)
    is_valid, parent_sha, ckpt_data = validate_parent_for_h12(
        parent_path=parent_path,
        expected_horizon=args.expected_init_horizon,
        expected_seed=H12_CONFIG["seed"],
        expected_sha256=EXPECTED_PARENT_SHA256,
    )
    parent_horizon = ckpt_data.get("horizon", ckpt_data.get("config", {}).get("horizon", "unknown"))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, "h12_extension_train.log")

    print_preflight_summary(
        parent_path=parent_path,
        parent_sha=parent_sha,
        parent_horizon=parent_horizon,
        gpu_ids=gpu_ids,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        log_file=log_file,
        batch_size=microbatch,
        grad_accum_steps=grad_accum_steps,
        effective_batch_size=effective_beff,
        dry_run=args.dry_run,
    )

    cmd = build_training_command(
        parent_path=str(parent_path),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=microbatch,
        grad_accum_steps=grad_accum_steps,
        horizon=args.horizon,
        expected_init_horizon=args.expected_init_horizon,
        output_dir=args.output_dir,
        seed=H12_CONFIG["seed"],
        lambda_div=H12_CONFIG["lambda_div"],
        lambda_vort=H12_CONFIG["lambda_vort"],
        val_diagnostic_horizons=H12_CONFIG["val_diagnostics"],
        use_amp=True,
        num_gpus=num_gpus,
        master_port=args.master_port,
    )

    print("\n[CMD] Executing Command:")
    print(" ".join(cmd))
    print()

    if args.dry_run:
        print("[DRY RUN] Preflight passed successfully. Command generated without execution.")
        return 0

    # Set CUDA_VISIBLE_DEVICES if specific GPUs requested
    env = os.environ.copy()
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    print(f"Launching H12 extension training (Logging to {log_file})...")
    start_time = time.time()
    with open(log_file, "w") as f_log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f_log.write(line)
            f_log.flush()
        proc.wait()

    elapsed = time.time() - start_time
    print(f"\n[DONE] Finished with returncode {proc.returncode} in {elapsed:.1f}s ({elapsed/60:.2f} min).")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
