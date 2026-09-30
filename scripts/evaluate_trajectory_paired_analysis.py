import os
import sys
import json
import math
import numpy as np
import scipy.stats as stats
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import create_flow_dataloaders
from src.data.normalization import FieldNormalizer
from src.metrics.field import compute_vrmse
from src.metrics.spectral import compute_radial_energy_spectrum
from src.utils.fft_derivatives import compute_divergence, compute_vorticity

from scripts.evaluate_latent_flow_matching_pilot import (
    build_rollout_manifest,
    load_all_models_for_evaluation,
    apply_pressure_gauge,
)

def evaluate_single_checkpoint(args_tuple):
    seed, branch, ckpt_path, device = args_tuple
    print(f"[{device}] Starting evaluation for Seed {seed} - {branch}...")
    
    dev = torch.device(device)
    normalizer = FieldNormalizer()
    normalizer.load_state_dict(torch.load("outputs/normalization/stats_grouped.pt", weights_only=True, map_location="cpu"))

    _, valid_loader, _, _ = create_flow_dataloaders(
        split_type="grouped",
        split_file="outputs/splits/grouped_split.json",
        data_root="/root/autodl-tmp/datasets/shear_flow",
        history_length=4,
        horizon=1,
        valid_horizon=10,
        train_stride=8,
        valid_stride=8,
        downsample_factor=2,
        batch_size=8,
        num_workers=2,
        normalize=True,
        normalizer=normalizer,
        seed=seed,
    )
    dataset = valid_loader.dataset

    forecaster, _, _, fm = load_all_models_for_evaluation(
        d0_checkpoint_path="outputs/checkpoints/dynamics/horizon_r2/seed_42/E4_H12/latent_transformer/best_long_vrmse.pt",
        g0_checkpoint_path="outputs/checkpoints/probabilistic/variance_head/formal_r1_seed42/g0_baseline_initialization.pt",
        g1_checkpoint_path="outputs/checkpoints/probabilistic/variance_head/formal_r1_seed42/best_g1_variance_head.pt",
        fm_checkpoint_path=ckpt_path,
        device=dev,
    )

    manifest_records, selected_indices = build_rollout_manifest(
        dataset=dataset,
        windows_per_traj=-1,
        max_trajectories=6,
    )
    total_windows = len(selected_indices)

    traj_windows = {}
    for i, r in enumerate(manifest_records):
        t_key = (r["source_file"], r["traj_idx"])
        traj_windows.setdefault(t_key, []).append(i)

    traj_metrics = {t_key: {
        "ens_vrmse_h10": [],
        "ens_vrmse_h5": [],
        "ens_spec_rel_err_h10": [],
        "indiv_spec_rel_err_h10": [],
        "samp_vrmse_h10": [],
        "samp_div_sq_h10": [],
        "samp_vort_sq_h10": [],
        "metadata": manifest_records[indices[0]],
    } for t_key, indices in traj_windows.items()}

    max_h = 10
    num_samples_K = 8
    noise_scale = 0.5
    batch_size = 8

    with torch.no_grad():
        for b_start in range(0, total_windows, batch_size):
            b_indices = selected_indices[b_start: b_start + batch_size]
            b_len = len(b_indices)
            batch_items = [dataset[i] for i in b_indices]

            first_rec = manifest_records[b_start]
            t_key = (first_rec["source_file"], first_rec["traj_idx"])

            q_hist = torch.stack([item["history"] for item in batch_items]).to(dev)
            q_gt_seq = torch.stack([item["future"][:max_h] for item in batch_items]).to(dev)
            re = torch.tensor([float(item["re"]) for item in batch_items], device=dev)
            sc = torch.tensor([float(item["sc"]) for item in batch_items], device=dev)

            q_gt_phys = apply_pressure_gauge(normalizer.denormalize(q_gt_seq))

            w_seed = seed + b_start
            r_fm = forecaster.sample_rollout_flow_matching(
                q_hist=q_hist, re=re, sc=sc, horizon=max_h,
                num_samples=num_samples_K, seed=w_seed,
                flow_matcher=fm, noise_scale=noise_scale, decode_samples=True,
            )
            fm_samps = apply_pressure_gauge(normalizer.denormalize(r_fm["sample_trajectories"].permute(1, 0, 2, 3, 4, 5)).permute(1, 0, 2, 3, 4, 5))
            pred_mean = fm_samps.mean(dim=1)

            for h in [5, 10]:
                gt_h = q_gt_phys[:, h - 1]
                pred_h = pred_mean[:, h - 1]
                vrmse_batch = compute_vrmse(pred_h, gt_h).item()
                if h == 5:
                    traj_metrics[t_key]["ens_vrmse_h5"].append(vrmse_batch)
                elif h == 10:
                    traj_metrics[t_key]["ens_vrmse_h10"].append(vrmse_batch)
                    
                    b_samp_vrmses = [compute_vrmse(fm_samps[:, k, h - 1], gt_h).item() for k in range(num_samples_K)]
                    traj_metrics[t_key]["samp_vrmse_h10"].append(float(np.mean(b_samp_vrmses)))
                    
                    vort_gt = compute_vorticity(gt_h[:, 0], gt_h[:, 1])
                    b_div_sq = []
                    b_vort_sq = []
                    for k in range(num_samples_K):
                        u_k = fm_samps[:, k, h - 1, 0]
                        v_k = fm_samps[:, k, h - 1, 1]
                        div_k = compute_divergence(u_k, v_k)
                        vort_k = compute_vorticity(u_k, v_k)
                        b_div_sq.append((div_k ** 2).mean().item())
                        b_vort_sq.append(((vort_k - vort_gt) ** 2).mean().item())
                    traj_metrics[t_key]["samp_div_sq_h10"].append(float(np.mean(b_div_sq)))
                    traj_metrics[t_key]["samp_vort_sq_h10"].append(float(np.mean(b_vort_sq)))

            for b_idx in range(b_len):
                u_gt_w = q_gt_phys[b_idx, max_h - 1, 0]
                v_gt_w = q_gt_phys[b_idx, max_h - 1, 1]
                _, e_gt = compute_radial_energy_spectrum(u_gt_w, v_gt_w)
                e_gt_arr = e_gt[:25].cpu().numpy()
                gt_norm = np.linalg.norm(e_gt_arr) + 1e-8

                u_ens_w = pred_mean[b_idx, max_h - 1, 0]
                v_ens_w = pred_mean[b_idx, max_h - 1, 1]
                _, e_ens = compute_radial_energy_spectrum(u_ens_w, v_ens_w)
                e_ens_arr = e_ens[:25].cpu().numpy()
                rel_err_ens = float(np.linalg.norm(e_ens_arr - e_gt_arr) / gt_norm)
                traj_metrics[t_key]["ens_spec_rel_err_h10"].append(rel_err_ens)

                k_indiv_rel_errs = []
                for k_idx in range(num_samples_K):
                    u_k_w = fm_samps[b_idx, k_idx, max_h - 1, 0]
                    v_k_w = fm_samps[b_idx, k_idx, max_h - 1, 1]
                    _, e_k = compute_radial_energy_spectrum(u_k_w, v_k_w)
                    e_k_arr = e_k[:25].cpu().numpy()
                    k_indiv_rel_errs.append(float(np.linalg.norm(e_k_arr - e_gt_arr) / gt_norm))
                traj_metrics[t_key]["indiv_spec_rel_err_h10"].append(float(np.mean(k_indiv_rel_errs)))

    aggregated = {}
    for idx, (t_key, m) in enumerate(traj_metrics.items()):
        meta = m["metadata"]
        aggregated[f"traj_{idx}"] = {
            "source_file": meta["source_file"].split("/")[-1],
            "traj_idx": meta["traj_idx"],
            "cluster_id": meta["cluster_id"],
            "re": meta["re"],
            "sc": meta["sc"],
            "primary1_h10_ens_vrmse": float(np.mean(m["ens_vrmse_h10"])),
            "primary2_h10_ens_spec_rel_err": float(np.mean(m["ens_spec_rel_err_h10"])),
            "primary3_h10_indiv_spec_rel_err": float(np.mean(m["indiv_spec_rel_err_h10"])),
            "primary4_h5_ens_vrmse": float(np.mean(m["ens_vrmse_h5"])),
            "secondary_h10_samp_vrmse": float(np.mean(m["samp_vrmse_h10"])),
            "secondary_h10_samp_div_rms": math.sqrt(float(np.mean(m["samp_div_sq_h10"]))),
            "secondary_h10_samp_vort_rmse": math.sqrt(float(np.mean(m["samp_vort_sq_h10"]))),
        }

    print(f"[{device}] Finished evaluation for Seed {seed} - {branch}.")
    return seed, branch, aggregated

def compute_paired_statistics(deltas_by_seed):
    # deltas_by_seed is dict: seed -> list of 6 trajectory deltas
    all_deltas = []
    cluster_means = []
    for s, deltas in deltas_by_seed.items():
        all_deltas.extend(deltas)
        cluster_means.append(float(np.mean(deltas)))

    N = len(all_deltas)
    mean_delta = float(np.mean(all_deltas))
    std_delta = float(np.std(all_deltas, ddof=1))
    se_standard = std_delta / math.sqrt(N)
    
    # Standard paired t-test (N=18, df=17)
    t_stat = mean_delta / (se_standard + 1e-12)
    p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=N - 1)))
    ci_95_standard = [
        mean_delta - stats.t.ppf(0.975, df=N - 1) * se_standard,
        mean_delta + stats.t.ppf(0.975, df=N - 1) * se_standard
    ]

    # Cluster-robust standard error (G=3 clusters)
    G = len(cluster_means)
    cluster_var = float(np.var(cluster_means, ddof=1))
    se_cluster = math.sqrt(cluster_var / G)
    ci_95_cluster = [
        mean_delta - stats.t.ppf(0.975, df=G - 1) * se_cluster,
        mean_delta + stats.t.ppf(0.975, df=G - 1) * se_cluster
    ]
    p_val_cluster = float(2 * (1 - stats.t.cdf(abs(mean_delta / (se_cluster + 1e-12)), df=G - 1)))

    return {
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "standard_se": se_standard,
        "standard_t_stat": t_stat,
        "standard_p_value": p_val,
        "standard_ci_95": ci_95_standard,
        "cluster_means": cluster_means,
        "cluster_se": se_cluster,
        "cluster_p_value": p_val_cluster,
        "cluster_ci_95": ci_95_cluster,
    }

def main():
    checkpoints = {
        (42, "C2"): ("outputs/checkpoints/probabilistic/flow_matching_r2_alpha05/C2/best_latent_flow_matcher.pt", "cuda:0"),
        (42, "R2_A"): ("outputs/checkpoints/probabilistic/flow_matching_r2_alpha05/R2_A/best_latent_flow_matcher.pt", "cuda:0"),
        (43, "C2"): ("outputs/checkpoints/probabilistic/flow_matching_r2_alpha05_seeds/43/C2/best_latent_flow_matcher.pt", "cuda:0"),
        (43, "R2_A"): ("outputs/checkpoints/probabilistic/flow_matching_r2_alpha05_seeds/43/R2_A/best_latent_flow_matcher.pt", "cuda:1"),
        (44, "C2"): ("outputs/checkpoints/probabilistic/flow_matching_r2_alpha05_seeds/44/C2/best_latent_flow_matcher.pt", "cuda:1"),
        (44, "R2_A"): ("outputs/checkpoints/probabilistic/flow_matching_r2_alpha05_seeds/44/R2_A/best_latent_flow_matcher.pt", "cuda:1"),
    }

    # Parallel evaluation across GPUs: 3 on cuda:0, 3 on cuda:1
    tasks_gpu0 = [
        (42, "C2", checkpoints[(42, "C2")][0], "cuda:0"),
        (42, "R2_A", checkpoints[(42, "R2_A")][0], "cuda:0"),
        (43, "C2", checkpoints[(43, "C2")][0], "cuda:0"),
    ]
    tasks_gpu1 = [
        (43, "R2_A", checkpoints[(43, "R2_A")][0], "cuda:1"),
        (44, "C2", checkpoints[(44, "C2")][0], "cuda:1"),
        (44, "R2_A", checkpoints[(44, "R2_A")][0], "cuda:1"),
    ]

    tasks = [
        (42, "C2", checkpoints[(42, "C2")][0], "cuda:0"),
        (42, "R2_A", checkpoints[(42, "R2_A")][0], "cuda:0"),
        (43, "C2", checkpoints[(43, "C2")][0], "cuda:0"),
        (43, "R2_A", checkpoints[(43, "R2_A")][0], "cuda:0"),
        (44, "C2", checkpoints[(44, "C2")][0], "cuda:0"),
        (44, "R2_A", checkpoints[(44, "R2_A")][0], "cuda:0"),
    ]

    results = {}
    for t in tasks:
        seed, branch, data = evaluate_single_checkpoint(t)
        results[(seed, branch)] = data
        torch.cuda.empty_cache()

    # Perform Paired Analysis across Seeds 42, 43, 44
    seeds = [42, 43, 44]
    metric_keys = [
        ("primary1_h10_ens_vrmse", "Primary 1 (h=10 Ens VRMSE)"),
        ("primary2_h10_ens_spec_rel_err", "Primary 2 (h=10 Ens Spectrum Rel L2 Error)"),
        ("primary3_h10_indiv_spec_rel_err", "Primary 3 (h=10 Indiv-Member Spectrum Rel L2 Error)"),
        ("primary4_h5_ens_vrmse", "Primary 4 (h=5 Ens VRMSE)"),
        ("secondary_h10_samp_vrmse", "Secondary (h=10 Sample VRMSE)"),
        ("secondary_h10_samp_div_rms", "Secondary (h=10 Sample RMS Divergence)"),
        ("secondary_h10_samp_vort_rmse", "Secondary (h=10 Sample Vorticity RMSE)"),
    ]

    paired_report = {
        "metadata": {
            "seeds": seeds,
            "branches": ["C2", "R2_A"],
            "total_physical_trajectories_per_seed": 6,
            "total_paired_observations": 18,
            "noise_scale": 0.5,
            "ensemble_K": 8,
        },
        "by_seed_trajectory": {},
        "statistical_analysis": {},
    }

    # Detailed per-trajectory deltas
    for s in seeds:
        paired_report["by_seed_trajectory"][str(s)] = {}
        c2_trajs = results[(s, "C2")]
        r2a_trajs = results[(s, "R2_A")]
        for t_name in sorted(c2_trajs.keys()):
            c2_t = c2_trajs[t_name]
            r2a_t = r2a_trajs[t_name]
            t_entry = {
                "source_file": c2_t["source_file"],
                "traj_idx": c2_t["traj_idx"],
                "cluster_id": c2_t["cluster_id"],
                "re": c2_t["re"],
                "sc": c2_t["sc"],
                "metrics": {},
            }
            for m_key, m_label in metric_keys:
                c2_val = c2_t[m_key]
                r2a_val = r2a_t[m_key]
                delta = r2a_val - c2_val
                rel_change = (delta / c2_val * 100) if abs(c2_val) > 1e-8 else 0.0
                t_entry["metrics"][m_key] = {
                    "C2": c2_val,
                    "R2_A": r2a_val,
                    "delta": delta,
                    "relative_change_pct": rel_change,
                }
            paired_report["by_seed_trajectory"][str(s)][t_name] = t_entry

    # Compute Statistical Summaries
    for m_key, m_label in metric_keys:
        deltas_by_seed = {s: [] for s in seeds}
        c2_values_all = []
        r2a_values_all = []
        for s in seeds:
            for t_name in sorted(results[(s, "C2")].keys()):
                c2_val = results[(s, "C2")][t_name][m_key]
                r2a_val = results[(s, "R2_A")][t_name][m_key]
                c2_values_all.append(c2_val)
                r2a_values_all.append(r2a_val)
                deltas_by_seed[s].append(r2a_val - c2_val)

        stats_summary = compute_paired_statistics(deltas_by_seed)
        stats_summary["label"] = m_label
        stats_summary["baseline_mean_C2"] = float(np.mean(c2_values_all))
        stats_summary["treatment_mean_R2_A"] = float(np.mean(r2a_values_all))
        stats_summary["overall_relative_change_pct"] = (stats_summary["mean_delta"] / stats_summary["baseline_mean_C2"]) * 100

        # Also breakdown discovery vs replication
        discovery_deltas = deltas_by_seed[42]
        replication_deltas = deltas_by_seed[43] + deltas_by_seed[44]
        stats_summary["breakdown"] = {
            "discovery_seed42": {
                "mean_delta": float(np.mean(discovery_deltas)),
                "std_delta": float(np.std(discovery_deltas, ddof=1)),
                "n": len(discovery_deltas),
            },
            "replication_seeds43_44": {
                "mean_delta": float(np.mean(replication_deltas)),
                "std_delta": float(np.std(replication_deltas, ddof=1)),
                "n": len(replication_deltas),
                "standard_se": float(np.std(replication_deltas, ddof=1) / math.sqrt(len(replication_deltas))),
            },
        }

        paired_report["statistical_analysis"][m_key] = stats_summary

    out_p = Path("outputs/metrics/fm_r2_multiseed_trajectory_paired_analysis.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(paired_report, f, indent=2)

    print(f"\nSuccessfully written full trajectory paired analysis to {out_p}")

if __name__ == "__main__":
    main()
