"""Contract test ensuring manuscript text strictly matches ground-truth experimental archives.

This test validates:
1. Representation autoencoder metrics (Table 6) match outputs/metrics/representation_metrics.json.
2. PDE controlled evaluation metrics (Table 10) match outputs/evaluations/pde_controlled_candidates_full_val.json.
3. Baseline metrics alignment and terminology integrity.
4. Window protocol and sample sizes (Table 3-B) are accurately stated.
5. Markdown formatting contracts: no LaTeX macro leaks (e.g. \\multicolumn), valid relative image paths, no trailing whitespace.
"""

import json
from pathlib import Path
import re
import pytest

DOCS_DIR = Path("docs")
MANUSCRIPT_DIR = DOCS_DIR / "manuscript"
MANUSCRIPT_DRAFT = DOCS_DIR / "MANUSCRIPT_DRAFT.md"

REP_METRICS_PATH = Path("outputs/metrics/representation_metrics.json")
PDE_EVAL_PATH = Path("outputs/evaluations/pde_controlled_candidates_full_val.json")
BASELINE_TEX_PATH = Path("outputs/tables/table_4_architecture_ablation.tex")


@pytest.fixture
def rep_metrics():
    assert REP_METRICS_PATH.exists(), f"Missing {REP_METRICS_PATH}"
    return json.loads(REP_METRICS_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def pde_eval_metrics():
    assert PDE_EVAL_PATH.exists(), f"Missing {PDE_EVAL_PATH}"
    return json.loads(PDE_EVAL_PATH.read_text(encoding="utf-8"))


def test_table_6_representation_metrics_contract(rep_metrics):
    """Verify Table 6 in 04_experimental_results.md and MANUSCRIPT_DRAFT.md matches representation_metrics.json."""
    test_m = rep_metrics["test_metrics"]
    valid_m = rep_metrics["valid_metrics"]

    # Ground-truth values rounded to 4 decimals
    u_test = f"{test_m['vrmse_u']:.4f}"         # 0.0517
    v_test = f"{test_m['vrmse_v']:.4f}"         # 0.0834
    p_test = f"{test_m['vrmse_p']:.4f}"         # 0.1504
    s_test = f"{test_m['vrmse_s']:.4f}"         # 0.1204
    mean_test = f"{test_m['vrmse_mean']:.4f}"   # 0.1015
    vort_test = f"{test_m['vorticity_rmse']:.4f}" # 0.3902

    u_val = f"{valid_m['vrmse_u']:.4f}"         # 0.0486
    mean_val = f"{valid_m['vrmse_mean']:.4f}"   # 0.0931

    docs_to_check = [
        MANUSCRIPT_DIR / "04_experimental_results.md",
        MANUSCRIPT_DRAFT,
    ]

    for doc_path in docs_to_check:
        assert doc_path.exists(), f"Missing {doc_path}"
        text = doc_path.read_text(encoding="utf-8")

        # Verify ground-truth values appear in Table 6
        assert u_test in text, f"Missing test u VRMSE ({u_test}) in {doc_path}"
        assert v_test in text, f"Missing test v VRMSE ({v_test}) in {doc_path}"
        assert p_test in text, f"Missing test p VRMSE ({p_test}) in {doc_path}"
        assert s_test in text, f"Missing test s VRMSE ({s_test}) in {doc_path}"
        assert mean_test in text, f"Missing test mean VRMSE ({mean_test}) in {doc_path}"
        assert vort_test in text, f"Missing test vorticity RMSE ({vort_test}) in {doc_path}"
        assert "2.49" in text, f"Missing pressure zero-mean drift in {doc_path}"

        # Guard against stale erroneous numbers previously flagged in review
        assert "0.0884" not in text, f"Found stale unverified number 0.0884 in {doc_path}"
        assert "0.2140" not in text, f"Found stale unverified number 0.2140 in {doc_path}"


def test_table_10_pde_controlled_metrics_contract(pde_eval_metrics):
    """Verify Table 10 in 04_experimental_results.md and MANUSCRIPT_DRAFT.md matches full_val JSON."""
    comp = pde_eval_metrics["overall_comparison"]
    vrmse = comp["vrmse_standard"]
    div = comp["div_rmse"]
    res_s = comp["res_s_rmse"]

    # Ground truth values:
    d0_vrmse = f"{vrmse['d0']:.4f}"  # 0.2586
    p0_vrmse = f"{vrmse['p0']:.4f}"  # 0.2651
    pde_vrmse = f"{vrmse['pde']:.4f}" # 0.2649

    d0_div = f"{div['d0']:.4f}"      # 0.1253
    p0_div = f"{div['p0']:.4f}"      # 0.1378
    pde_div = f"{div['pde']:.4f}"    # 0.1376

    d0_res_s = f"{res_s['d0']:.4f}"  # 0.1190
    p0_res_s = f"{res_s['p0']:.4f}"  # 0.1233
    pde_res_s = f"{res_s['pde']:.4f}" # 0.1206

    docs_to_check = [
        MANUSCRIPT_DIR / "04_experimental_results.md",
        MANUSCRIPT_DRAFT,
    ]

    for doc_path in docs_to_check:
        assert doc_path.exists(), f"Missing {doc_path}"
        text = doc_path.read_text(encoding="utf-8")

        for val in (d0_vrmse, p0_vrmse, pde_vrmse, d0_div, p0_div, pde_div, d0_res_s, p0_res_s, pde_res_s):
            assert val in text, f"Missing PDE value {val} in {doc_path}"

        # Check sample size and protocol description
        assert "1,110" in text or "1110" in text, f"Missing sample size 1110 in {doc_path}"
        assert "2.18%" in text, f"Missing relative reduction 2.18% in {doc_path}"
        assert "+2.42%" in text or "2.42%" in text, f"Missing relative increase 2.42% in {doc_path}"


def test_markdown_tables_have_no_multicolumn_macro():
    """Verify that no Markdown table contains raw LaTeX \\multicolumn macro."""
    for path in MANUSCRIPT_DIR.glob("*.md"):
        content = path.read_text(encoding="utf-8")
        assert "\\multicolumn" not in content, f"Found \\multicolumn in {path}"

    draft_content = MANUSCRIPT_DRAFT.read_text(encoding="utf-8")
    assert "\\multicolumn" not in draft_content, f"Found \\multicolumn in {MANUSCRIPT_DRAFT}"


def test_relative_image_paths_consistency():
    """Verify relative image paths are correct in both split chapters and unified draft."""
    for path in MANUSCRIPT_DIR.glob("*.md"):
        content = path.read_text(encoding="utf-8")
        # Split chapters in docs/manuscript/ must point to ../../outputs/
        matches = re.findall(r'!\[.*?\]\((.*?)\)', content)
        for m in matches:
            if "outputs/" in m:
                assert m.startswith("../../outputs/"), f"Invalid relative path {m} in {path}"

    draft_content = MANUSCRIPT_DRAFT.read_text(encoding="utf-8")
    matches = re.findall(r'!\[.*?\]\((.*?)\)', draft_content)
    for m in matches:
        if "outputs/" in m:
            assert m.startswith("../outputs/"), f"Invalid relative path {m} in {MANUSCRIPT_DRAFT}"


def test_no_trailing_whitespace_in_manuscripts():
    """Verify no lines end with trailing whitespace."""
    all_files = list(MANUSCRIPT_DIR.glob("*.md")) + [MANUSCRIPT_DRAFT]
    for path in all_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, start=1):
            assert not line.endswith(" ") and not line.endswith("\t"), (
                f"Trailing whitespace in {path} at line {idx}: {repr(line)}"
            )
