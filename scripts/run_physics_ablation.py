"""Orchestrator for 5-group physical loss ablation study (E0 - E4) using dual GPUs.

Groups (per README spec Section 9.5 & Section 18):
    E0. ablation_E0_single_step: Single-step baseline (H=1, lambda_div=0, lambda_vort=0)
    E1. ablation_E1_rollout_field: Multi-step rollout field loss (H=2, lambda_div=0, lambda_vort=0)
    E2. ablation_E2_plus_L_div: + Incompressibility divergence penalty (H=2, lambda_div=0.01, lambda_vort=0)
    E3. ablation_E3_plus_L_vort: + Vorticity consistency penalty (H=2, lambda_div=0, lambda_vort=0.05)
    E4. ablation_E4_full_physics: Full physics coupling (H=2, lambda_div=0.01, lambda_vort=0.05)

GPU Allocation:
    GPU 0: E0_single_step -> E1_rollout_field -> E2_plus_L_div
    GPU 1: E3_plus_L_vort -> E4_full_physics
"""

import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ABLATION_CONFIGS = {
    "ablation_E0_single_step": {
        "title": "E0: Single-Step Pure Field Loss (H=1, L_field)",
        "horizon": 1,
        "lambda_div": 0.0,
        "lambda_vort": 0.0,
        "output_dir": "outputs/checkpoints/dynamics/ablation_E0_single_step",
        "log_file": "outputs/train_ablation_E0_single_step.log",
    },
    "ablation_E1_rollout_field": {
        "title": "E1: Rollout-Aware Field Loss (H=2, L_field)",
        "horizon": 2,
        "lambda_div": 0.0,
        "lambda_vort": 0.0,
        "output_dir": "outputs/checkpoints/dynamics/ablation_E1_rollout_field",
        "log_file": "outputs/train_ablation_E1_rollout_field.log",
    },
    "ablation_E2_plus_L_div": {
        "title": "E2: + Divergence Loss (H=2, +L_div)",
        "horizon": 2,
        "lambda_div": 0.01,
        "lambda_vort": 0.0,
        "output_dir": "outputs/checkpoints/dynamics/ablation_E2_plus_L_div",
        "log_file": "outputs/train_ablation_E2_plus_L_div.log",
    },
    "ablation_E3_plus_L_vort": {
        "title": "E3: + Vorticity Loss (H=2, +L_vort)",
        "horizon": 2,
        "lambda_div": 0.0,
        "lambda_vort": 0.05,
        "output_dir": "outputs/checkpoints/dynamics/ablation_E3_plus_L_vort",
        "log_file": "outputs/train_ablation_E3_plus_L_vort.log",
    },
    "ablation_E4_full_physics": {
        "title": "E4: Full Physics Coupling (H=2, +L_div + L_vort)",
        "horizon": 2,
        "lambda_div": 0.01,
        "lambda_vort": 0.05,
        "output_dir": "outputs/checkpoints/dynamics/ablation_E4_full_physics",
        "log_file": "outputs/train_ablation_E4_full_physics.log",
    },
}


def run_single_ablation(group_name: str, gpu_id: int, epochs: int = 10, batch_size: int = 8):
    cfg = ABLATION_CONFIGS[group_name]
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python -u scripts/train_forecaster.py "
        f"--model latent_transformer "
        f"--output_dir {cfg['output_dir']} "
        f"--horizon {cfg['horizon']} "
        f"--epochs {epochs} "
        f"--batch_size {batch_size} "
        f"--lambda_div {cfg['lambda_div']} "
        f"--lambda_vort {cfg['lambda_vort']} "
        f"--use_amp "
        f"> {cfg['log_file']} 2>&1"
    )

    print(f"[GPU {gpu_id}] Starting {cfg['title']}...", flush=True)
    start_time = time.time()
    ret = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time

    if ret.returncode == 0:
        print(f"[GPU {gpu_id}] COMPLETED: {cfg['title']} in {elapsed:.1f}s", flush=True)
    else:
        print(f"[GPU {gpu_id}] FAILED: {cfg['title']} with code {ret.returncode}", flush=True)
        sys.exit(ret.returncode)


def run_worker(groups, gpu_id, epochs, batch_size):
    for grp in groups:
        run_single_ablation(grp, gpu_id, epochs=epochs, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description="Run complete E0-E4 physical loss ablation study.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sequential", action="store_true", help="Run sequentially on single GPU")
    args = parser.parse_args()

    print("=" * 80)
    print("STARTING 5-GROUP PHYSICAL LOSS ABLATION STUDY (E0 - E4)")
    print("=" * 80)

    groups_p0 = ["ablation_E0_single_step", "ablation_E1_rollout_field", "ablation_E2_plus_L_div"]
    groups_p1 = ["ablation_E3_plus_L_vort", "ablation_E4_full_physics"]

    if args.sequential:
        for grp in groups_p0 + groups_p1:
            run_single_ablation(grp, 0, epochs=args.epochs, batch_size=args.batch_size)
    else:
        import multiprocessing as mp

        p0 = mp.Process(
            target=run_worker,
            args=(groups_p0, 0, args.epochs, args.batch_size),
        )
        p1 = mp.Process(
            target=run_worker,
            args=(groups_p1, 1, args.epochs, args.batch_size),
        )

        p0.start()
        p1.start()

        p0.join()
        p1.join()

        if p0.exitcode != 0 or p1.exitcode != 0:
            print(f"Error in ablation workers: p0={p0.exitcode}, p1={p1.exitcode}")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("ALL 5 ABLATION GROUPS (E0 - E4) COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
