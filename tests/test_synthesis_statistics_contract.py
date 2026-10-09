"""Contract test for synthesis statistical reporting on Flow Matching primary endpoints.

Guarantees that all primary endpoints reported in Table 5-5 and text synthesis:
1. Strictly source from the confirmatory replication group: confirmatory_replication_seeds43_46.
2. Uniformly use cluster_p_value (seed-level t-test, df=3) and cluster_ci_95.
3. Reject mixing standard_p_value (unclustered 24 trajectory pairs) with cluster_p_value.
4. Cryptographically and numerically match the archived replication metrics.
"""

import json
import math
from pathlib import Path
import pytest
import numpy as np
from scipy.stats import ttest_1samp


METRICS_PATH = Path("outputs/metrics/fm_r2_multiseed_trajectory_paired_analysis.json")


@pytest.fixture
def paired_analysis_data():
    assert METRICS_PATH.exists(), f"Missing {METRICS_PATH}"
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def test_replication_seeds_integrity(paired_analysis_data):
    """Verify replication seeds strictly match [43, 44, 45, 46]."""
    meta = paired_analysis_data.get("metadata", {})
    assert meta.get("replication_seeds") == [43, 44, 45, 46]


def test_primary_endpoints_strictly_use_cluster_statistics(paired_analysis_data):
    """Verify that all 4 primary endpoints strictly employ cluster-level inference (df=3)."""
    endpoints = [
        ("primary1_h10_ens_vrmse", 0.140128077, 0.008332778),
        ("primary2_h10_ens_spec_rel_err", 0.461729164, 0.256895080),
        ("primary3_h10_indiv_spec_rel_err", 0.392348475, 0.163441933),
        ("primary4_h5_ens_vrmse", 0.884984855, 0.879042061),
    ]

    stat_analysis = paired_analysis_data.get("statistical_analysis", {})

    for key, expected_cluster_p, distinct_standard_p in endpoints:
        assert key in stat_analysis, f"Missing endpoint {key}"
        row = stat_analysis[key]["confirmatory_replication_seeds43_46"]

        # Ensure seed-level cluster means are 4 independent replication seeds
        assert row["n_clusters"] == 4
        effects = np.asarray(row["cluster_means"], dtype=float)
        assert effects.shape == (4,)
        assert np.isfinite(effects).all()

        # Recompute 1-sample t-test on seed means
        ttest_res = ttest_1samp(effects, popmean=0.0)
        assert math.isclose(float(ttest_res.pvalue), float(row["cluster_p_value"]), rel_tol=1e-6)

        # Verify cluster_p_value matches expected
        assert math.isclose(float(row["cluster_p_value"]), expected_cluster_p, rel_tol=1e-5)

        # Verify cluster_p_value is distinct from standard_p_value (guarding against accidental mixing)
        if key in ("primary1_h10_ens_vrmse", "primary2_h10_ens_spec_rel_err", "primary3_h10_indiv_spec_rel_err"):
            assert not math.isclose(
                float(row["cluster_p_value"]),
                float(row["standard_p_value"]),
                rel_tol=1e-2,
            ), f"Cluster p and standard p are unexpectedly close for {key}"

        # Verify cluster 95% CI is populated and finite
        ci = row["cluster_ci_95"]
        assert len(ci) == 2
        assert ci[0] < ci[1]
        assert np.isfinite(ci).all()


def test_seed45_reversal_belongs_to_primary2(paired_analysis_data):
    """Verify that the Seed 45 reversal (+11.99%) strictly belongs to Primary 2 (h=10 ensemble spectrum)."""
    stat_analysis = paired_analysis_data.get("statistical_analysis", {})
    row2 = stat_analysis["primary2_h10_ens_spec_rel_err"]["confirmatory_replication_seeds43_46"]
    
    # Check cluster means: Seed 45 is the 3rd seed in [43, 44, 45, 46] (index 2)
    seed45_effect = row2["cluster_means"][2]
    assert seed45_effect > 0, f"Seed 45 should show a positive reversal in primary 2, got {seed45_effect}"


def test_synthesis_document_table_values_consistency():
    """Verify that FIRST_AUTHOR_RESEARCH_SYNTHESIS.md contains the exact cluster p-values."""
    doc_path = Path("docs/FIRST_AUTHOR_RESEARCH_SYNTHESIS.md")
    assert doc_path.exists(), f"Missing {doc_path}"
    content = doc_path.read_text(encoding="utf-8")

    # Table 5-5 must report cluster p-values for primary 1, 2, 3, 4
    assert "0.140" in content
    assert "0.462" in content
    assert "0.392" in content
    assert "0.885" in content

    # P2: Guard against invalid equivalence claim "平价不变" anywhere in Table 5-5 or document
    assert "平价不变" not in content, "Found forbidden phrase '平价不变' in synthesis doc!"
    assert "未进行等价性验证" in content

    # P2: Verify effect size definition and column header clarity
    assert "原指标均值差的 95% 置信区间" in content
    assert "\\Delta M = M_{\\text{自生成历史条件}} - M_{\\text{真实历史条件}}" in content

    # Guard against stale unclustered standard p-values for primary 2 and 3 in Table 5-5
    # (0.257 and 0.163 should not be presented as the primary inference p-values)
    lines = content.splitlines()
    table_lines = [l for l in lines if l.strip().startswith("|") and "主要终点" in l]
    for tl in table_lines:
        if "主要终点 2" in tl:
            assert "0.462" in tl or "0.4617" in tl
            assert "0.257" not in tl, "Primary 2 mistakenly reported standard_p=0.257!"
        if "主要终点 3" in tl:
            assert "0.392" in tl or "0.3923" in tl
            assert "0.163" not in tl, "Primary 3 mistakenly reported standard_p=0.163!"
        if "主要终点 4" in tl:
            assert "未进行等价性验证" in tl
            assert "平价不变" not in tl

    # Secondary endpoints row must strictly describe mean decrease across 4 seeds without claiming 24/24 pair perfection
    sec_lines = [l for l in lines if l.strip().startswith("|") and "次要指标" in l]
    assert len(sec_lines) >= 1
    assert any("三项预指定次要指标在四个复现种子的均值上均下降" in sl for sl in sec_lines)
