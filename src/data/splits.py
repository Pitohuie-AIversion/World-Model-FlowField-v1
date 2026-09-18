"""Dataset split strategies: Official, Grouped (IC-leakage-free), and Parameter Holdout."""

from typing import Dict, List, Tuple
import re


def parse_shear_flow_filename(filename: str) -> Dict[str, float]:
    """Extract Reynolds and Schmidt parameters from shear_flow HDF5 filename.

    Example filename: 'shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5'
    """
    clean_name = filename.replace(".hdf5", "").replace(".h5", "")
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
    def get_parameter_holdout_split(
        all_files: List[str],
        holdout_re: float = 1e5,
        holdout_sc: float = 10.0,
        valid_ratio: float = 0.1,
    ) -> Dict[str, List[str]]:
        """Strategy 3: Parameter Holdout Split.

        All trajectories containing holdout_re OR holdout_sc are reserved strictly
        for zero-shot generalization testing and never seen during training.
        Remaining parameter combinations are split into train and valid sets.
        """
        train_pool = []
        holdout_test = []

        for f in all_files:
            params = parse_shear_flow_filename(f)
            # Check if matching holdout condition (using float tolerance)
            is_holdout_re = abs(params["re"] - holdout_re) / holdout_re < 1e-4
            is_holdout_sc = abs(params["sc"] - holdout_sc) / max(holdout_sc, 1e-4) < 1e-4

            if is_holdout_re or is_holdout_sc:
                holdout_test.append(f)
            else:
                train_pool.append(f)

        train_pool = sorted(train_pool)
        n_valid = max(1, int(len(train_pool) * valid_ratio))

        return {
            "train": train_pool[:-n_valid],
            "valid": train_pool[-n_valid:],
            "test": sorted(holdout_test),
        }
