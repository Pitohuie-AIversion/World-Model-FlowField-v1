import os
import re
from typing import Dict, List, Optional, Tuple


def parse_shear_flow_filename(filename: str) -> Dict[str, float]:
    """Extract Reynolds and Schmidt parameters from shear_flow HDF5 filename.

    Example filename: 'shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5'
    """
    clean_name = os.path.basename(filename).replace(".hdf5", "").replace(".h5", "")
    re_match = re.search(r"Reynolds_([0-9a-zA-Z\.\+\-]+)", clean_name)
    sc_match = re.search(r"Schmidt_([0-9a-zA-Z\.\+\-]+)", clean_name)

    if not re_match or not sc_match:
        raise ValueError(f"Could not parse Re and Sc from filename: {filename}")

    re_val = float(re_match.group(1))
    sc_val = float(sc_match.group(1))

    return {"re": re_val, "sc": sc_val}


class SplitManager:
    """Manages the three experimental dataset splits defined for shear_flow V1."""

    @staticmethod
    def get_official_split(
        train_files: List[str],
        valid_files: List[str],
        test_files: List[str],
    ) -> Dict[str, List[str]]:
        """Strategy 1: Official Split directly utilizing The Well's designated partitions."""
        return {
            "train": sorted(train_files),
            "valid": sorted(valid_files),
            "test": sorted(test_files),
        }

    @staticmethod
    def cluster_initial_conditions(
        file_paths: List[str],
        data_root: Optional[str] = None,
        tolerance: float = 1e-4,
    ) -> List[Dict]:
        """Cluster trajectories across files by their physical initial condition (t=0).

        Returns:
            List of cluster dictionaries:
                [{'cluster_id': int, 'members': [{'file_path': str, 'traj_idx': int, 're': float, 'sc': float}]}]
        """
        import h5py
        import numpy as np

        clusters = []
        for path in sorted(file_paths):
            params = parse_shear_flow_filename(path)
            re_val = params["re"]
            sc_val = params["sc"]

            actual_path = os.path.join(data_root, path) if data_root and not os.path.isabs(path) else path

            with h5py.File(actual_path, "r") as h5:
                vel_ds = h5["t1_fields/velocity"] if "t1_fields/velocity" in h5 else h5.get("velocity")
                if vel_ds is None:
                    continue
                n_trajs = vel_ds.shape[0]
                vel0_all = np.asarray(vel_ds[:, 0], dtype=np.float32)

                for traj_idx in range(n_trajs):
                    v0 = vel0_all[traj_idx]
                    matched = False
                    for c in clusters:
                        diff = np.max(np.abs(v0 - c["representative"]))
                        if diff < tolerance:
                            c["members"].append({
                                "file_path": path,
                                "traj_idx": traj_idx,
                                "re": re_val,
                                "sc": sc_val,
                            })
                            matched = True
                            break
                    if not matched:
                        new_cid = len(clusters)
                        clusters.append({
                            "cluster_id": new_cid,
                            "representative": v0,
                            "members": [{
                                "file_path": path,
                                "traj_idx": traj_idx,
                                "re": re_val,
                                "sc": sc_val,
                            }],
                        })

        # Remove representative numpy array before JSON serialization
        for c in clusters:
            del c["representative"]

        return clusters

    @staticmethod
    def get_grouped_split(
        all_files: List[str],
        data_root: Optional[str] = None,
        train_ratio: float = 0.77,
        valid_ratio: float = 0.11,
        seed: int = 42,
        tolerance: float = 1e-4,
    ) -> Dict[str, List[Dict]]:
        """Strategy 2: Grouped Split (Zero-IC-Leakage Split).

        Clusters all trajectories by initial condition (t=0). Partitions unique
        IC clusters into train, valid, and test sets. All trajectories belonging
        to the same cluster (regardless of Sc or file) are strictly placed in the
        same split partition, guaranteeing zero initial condition data leakage.
        """
        import random

        clusters = SplitManager.cluster_initial_conditions(all_files, data_root=data_root, tolerance=tolerance)
        num_clusters = len(clusters)

        # Deterministic shuffle of clusters
        rng = random.Random(seed)
        shuffled_clusters = list(clusters)
        rng.shuffle(shuffled_clusters)

        n_train = max(1, int(round(num_clusters * train_ratio)))
        n_valid = max(1, int(round(num_clusters * valid_ratio)))
        if n_train + n_valid >= num_clusters:
            n_valid = max(1, (num_clusters - n_train) // 2)

        train_clusters = shuffled_clusters[:n_train]
        valid_clusters = shuffled_clusters[n_train : n_train + n_valid]
        test_clusters = shuffled_clusters[n_train + n_valid :]

        def flatten_members(cluster_list):
            members = []
            for c in cluster_list:
                cid = c["cluster_id"]
                for m in c["members"]:
                    item = dict(m)
                    item["cluster_id"] = cid
                    members.append(item)
            return sorted(members, key=lambda x: (x["file_path"], x["traj_idx"]))

        return {
            "train": flatten_members(train_clusters),
            "valid": flatten_members(valid_clusters),
            "test": flatten_members(test_clusters),
            "metadata": {
                "total_trajectories": sum(len(c["members"]) for c in clusters),
                "total_clusters": num_clusters,
                "train_clusters": len(train_clusters),
                "valid_clusters": len(valid_clusters),
                "test_clusters": len(test_clusters),
                "seed": seed,
            },
        }

    @staticmethod
    def get_parameter_holdout_re_split(
        all_files: List[str],
        holdout_re: float = 1e5,
        valid_ratio: float = 0.1,
    ) -> Dict[str, List[str]]:
        """Strategy 3A: Reynolds Parameter Holdout Split.

        Trajectories matching holdout_re are reserved exclusively for out-of-distribution
        dynamics generalization testing. Remaining parameters are split into train/valid.
        """
        train_pool = []
        holdout_test = []

        for f in all_files:
            params = parse_shear_flow_filename(f)
            if abs(params["re"] - holdout_re) / max(holdout_re, 1e-4) < 1e-4:
                holdout_test.append(f)
            else:
                train_pool.append(f)

        train_pool = sorted(train_pool)
        n_valid = max(1, int(len(train_pool) * valid_ratio)) if len(train_pool) > 1 else 0

        return {
            "train": train_pool[:-n_valid] if n_valid > 0 else train_pool,
            "valid": train_pool[-n_valid:] if n_valid > 0 else [],
            "test": sorted(holdout_test),
        }

    @staticmethod
    def get_parameter_holdout_sc_split(
        all_files: List[str],
        holdout_sc: float = 1.0,
        valid_ratio: float = 0.1,
    ) -> Dict[str, List[str]]:
        """Strategy 3B: Schmidt Parameter Holdout Split.

        Trajectories matching holdout_sc are reserved exclusively for out-of-distribution
        scalar transport/diffusion generalization testing.
        """
        train_pool = []
        holdout_test = []

        for f in all_files:
            params = parse_shear_flow_filename(f)
            if abs(params["sc"] - holdout_sc) / max(holdout_sc, 1e-4) < 1e-4:
                holdout_test.append(f)
            else:
                train_pool.append(f)

        train_pool = sorted(train_pool)
        n_valid = max(1, int(len(train_pool) * valid_ratio)) if len(train_pool) > 1 else 0

        return {
            "train": train_pool[:-n_valid] if n_valid > 0 else train_pool,
            "valid": train_pool[-n_valid:] if n_valid > 0 else [],
            "test": sorted(holdout_test),
        }

    @staticmethod
    def get_parameter_holdout_split(
        all_files: List[str],
        holdout_re: Optional[float] = None,
        holdout_sc: Optional[float] = 1.0,
        valid_ratio: float = 0.1,
    ) -> Dict[str, List[str]]:
        """Strategy 3: General Parameter Holdout Split (backward compatible)."""
        train_pool = []
        holdout_test = []

        for f in all_files:
            params = parse_shear_flow_filename(f)
            is_holdout_re = holdout_re is not None and abs(params["re"] - holdout_re) / max(holdout_re, 1e-4) < 1e-4
            is_holdout_sc = holdout_sc is not None and abs(params["sc"] - holdout_sc) / max(holdout_sc, 1e-4) < 1e-4

            if is_holdout_re or is_holdout_sc:
                holdout_test.append(f)
            else:
                train_pool.append(f)

        train_pool = sorted(train_pool)
        n_valid = max(1, int(len(train_pool) * valid_ratio)) if len(train_pool) > 1 else 0

        return {
            "train": train_pool[:-n_valid] if n_valid > 0 else train_pool,
            "valid": train_pool[-n_valid:] if n_valid > 0 else [],
            "test": sorted(holdout_test),
        }
