"""Orchestrator for 4-group physical loss ablation study using dual GPUs.

Groups:
    1. L_field: Pure data-driven reconstruction loss (lambda_div=0, lambda_vort=0)
    2. plus_L_div: + Incompressibility divergence penalty (lambda_div=0.01, lambda_vort=0)
    3. plus_L_vort: + Vorticity consistency penalty (lambda_div=0, lambda_vort=0.05)
    4. plus_L_div_vort: Full physics coupling (lambda_div=0.01, lambda_vort=0.05)

GPU Allocation:
    GPU 0: ablation_L_field -> ablation_plus_L_div
    GPU 1: ablation_plus_L_vort -> ablation_plus_L_div_vort
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
    "ablation_L_field": {
        "title": "Group 1: Pure Field Loss (L_field)",
        "lambda_div": 0.0,
        "lambda_vort": 0.0,
        "output_dir": "outputs/checkpoints/dynamics/ablation_L_field",
        "log_file": "outputs/train_ablation_L_field.log",
    },
    "ablation_plus_L_div": {
        "title": "Group 2: + Divergence Loss (+L_div)",
        "lambda_div": 0.01,
        "lambda_vort": 0.0,
        "output_dir": "outputs/checkpoints/dynamics/ablation_plus_L_div",
        "log_file": "outputs/train_ablation_plus_L_div.log",
    },
    "ablation_plus_L_vort": {
        "title": "Group 3: + Vorticity Loss (+L_vort)",
        "lambda_div": 0.0,
        "lambda_vort": 0.05,
        "output_dir": "outputs/checkpoints/dynamics/ablation_plus_L_vort",
        "log_file": "outputs/train_ablation_plus_L_vort.log",
    },
    "ablation_plus_L_div_vort": {
        "title": "Group 4: Full Physics (+L_div + L_vort)",
        "lambda_div": 0.01,
        "lambda_vort": 0.05,
        "output_dir": "outputs/checkpoints/dynamics/ablation_plus_L_div_vort",
        "log_file": "outputs/train_ablation_plus_L_div_vort.log",
    },
}


def run_single_ablation(group_name: str, gpu_id: int, epochs: int = 10, horizon: int = 2, batch_size: int = 8):
    cfg = ABLATION_CONFIGS[group_name]
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python -u scripts/train_forecaster.py "
        f"--model latent_transformer "
        f"--output_dir {cfg['output_dir']} "
        f"--horizon {horizon} "
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


def run_worker(groups, gpu_id, epochs, horizon, batch_size):
    for grp in groups:
        run_single_ablation(grp, gpu_id, epochs=epochs, horizon=horizon, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sequential", action="store_true", help="Run sequentially on single GPU")
    args = parser.parse_args()

    print("=" * 80)
    print("STARTING 4-GROUP PHYSICAL LOSS ABLATION STUDY")
    print("=" * 80)

    if args.sequential:
        for grp in ["ablation_L_field", "ablation_plus_L_div", "ablation_plus_L_vort", "ablation_plus_L_div_vort"]:
            run_single_ablation(grp, 0, epochs=args.epochs, horizon=args.horizon, batch_size=args.batch_size)
    else:
        import multiprocessing as mp

        p0 = mp.Process(
            target=run_worker,
            args=(["ablation_L_field", "ablation_plus_L_div"], 0, args.epochs, args.horizon, args.batch_size),
        )
        p1 = mp.Process(
            target=run_worker,
            args=(["ablation_plus_L_vort", "ablation_plus_L_div_vort"], 1, args.epochs, args.horizon, args.batch_size),
        )

        p0.start()
        p1.start()

        p0.join()
        p1.join()

        if p0.exitcode != 0 or p1.exitcode != 0:
            print(f"Error in ablation workers: p0={p0.exitcode}, p1={p1.exitcode}")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("ALL 4 ABLATION GROUPS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
