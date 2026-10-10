"""Field-level and table-level data contract tests for manuscript scientific integrity.

Validates:
1. Exact field-level correspondence for Table 6 (Representation Autoencoder) with outputs/metrics/representation_metrics.json.
2. Table 8 (30-step Rollout Baselines) cell-by-cell alignment with outputs/metrics/rollout_benchmark.json and table_4_architecture_ablation.tex.
3. Table 10 (PDE Controlled Baseline Evaluation) cell-by-cell alignment with outputs/evaluations/pde_controlled_candidates_full_val.json.
4. Discussion Section 5.1 step-model associated numeric alignment.
5. Strict negative assertions: rejects all historical erroneous/stale values and unphysical formulas.
6. Markdown image references strictly resolve to existing files on disk.
7. Full synchronization between split chapters in docs/manuscript/ and unified docs/MANUSCRIPT_DRAFT.md.
"""

import json
from pathlib import Path
import re
from typing import Dict, List, Tuple
import pytest

DOCS_DIR = Path("docs")
MANUSCRIPT_DIR = DOCS_DIR / "manuscript"
MANUSCRIPT_DRAFT = DOCS_DIR / "MANUSCRIPT_DRAFT.md"

REP_METRICS_PATH = Path("outputs/metrics/representation_metrics.json")
PDE_EVAL_PATH = Path("outputs/evaluations/pde_controlled_candidates_full_val.json")
ROLLOUT_BENCHMARK_PATH = Path("outputs/metrics/rollout_benchmark.json")


def parse_markdown_table(text: str, table_header: str) -> Dict[str, Dict[str, str]]:
    """Parse a markdown table immediately following table_header into a 2D mapping: {row_key: {col_key: cell_value}}."""
    pos = text.find(table_header)
    assert pos != -1, f"Table header '{table_header}' not found in document"

    sub = text[pos:]
    lines = sub.splitlines()

    table_lines: List[str] = []
    started = False
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)
            started = True
        elif started and not stripped:
            break

    assert len(table_lines) >= 3, f"Table following '{table_header}' has insufficient lines ({len(table_lines)})"

    # Header line
    headers = [c.strip() for c in table_lines[0].split("|")[1:-1]]

    # Data lines
    table_dict: Dict[str, Dict[str, str]] = {}
    current_primary_row_key = ""

    for line in table_lines[2:]:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if not cells or not any(cells):
            continue

        row_key = cells[0]
        if not row_key and current_primary_row_key:
            # Continuation row (same model, second metric)
            row_key = current_primary_row_key + "::" + (cells[2] if len(cells) > 2 else "sub")
        else:
            current_primary_row_key = row_key

        row_dict: Dict[str, str] = {}
        for idx, col_name in enumerate(headers):
            if idx < len(cells):
                row_dict[col_name] = cells[idx]
        table_dict[row_key] = row_dict

    return table_dict


@pytest.fixture
def rep_metrics():
    assert REP_METRICS_PATH.exists(), f"Missing {REP_METRICS_PATH}"
    return json.loads(REP_METRICS_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def pde_eval_metrics():
    assert PDE_EVAL_PATH.exists(), f"Missing {PDE_EVAL_PATH}"
    return json.loads(PDE_EVAL_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def rollout_benchmark():
    assert ROLLOUT_BENCHMARK_PATH.exists(), f"Missing {ROLLOUT_BENCHMARK_PATH}"
    return json.loads(ROLLOUT_BENCHMARK_PATH.read_text(encoding="utf-8"))


def test_table_6_representation_metrics_field_level(rep_metrics):
    """Verify Table 6 in both 04_experimental_results.md and MANUSCRIPT_DRAFT.md matches representation_metrics.json cell-by-cell."""
    test_m = rep_metrics["test_metrics"]
    valid_m = rep_metrics["valid_metrics"]

    expected_cells = {
        "流向速度 $u$": {
            "val_rmse": f"{valid_m['rmse_u']:.4f}",
            "val_vrmse": f"{valid_m['vrmse_u']:.4f}",
            "test_rmse": f"{test_m['rmse_u']:.4f}",
            "test_vrmse": f"{test_m['vrmse_u']:.4f}",
        },
        "法向速度 $v$": {
            "val_rmse": f"{valid_m['rmse_v']:.4f}",
            "test_rmse": f"{test_m['rmse_v']:.4f}",
            "test_vrmse": f"{test_m['vrmse_v']:.4f}",
        },
        "规范压力 $p$": {
            "val_rmse": f"{valid_m['rmse_p']:.4f}",
            "test_rmse": f"{test_m['rmse_p']:.4f}",
            "test_vrmse": f"{test_m['vrmse_p']:.4f}",
        },
        "被动标量 $s$": {
            "val_rmse": f"{valid_m['rmse_s']:.4f}",
            "val_vrmse": f"{valid_m['vrmse_s']:.4f}",
            "test_rmse": f"{test_m['rmse_s']:.4f}",
            "test_vrmse": f"{test_m['vrmse_s']:.4f}",
        },
        "**四通道平均**": {
            "val_rmse": f"{valid_m['rmse_mean']:.4f}",
            "val_vrmse": f"{valid_m['vrmse_mean']:.4f}",
            "test_rmse": f"{test_m['rmse_mean']:.4f}",
            "test_vrmse": f"{test_m['vrmse_mean']:.4f}",
        },
    }

    docs_to_check = [MANUSCRIPT_DIR / "04_experimental_results.md", MANUSCRIPT_DRAFT]

    for doc_path in docs_to_check:
        text = doc_path.read_text(encoding="utf-8")
        parsed = parse_markdown_table(text, "#### 表 6")

        for row_name, metrics in expected_cells.items():
            matching_rows = [k for k in parsed.keys() if row_name in k]
            assert len(matching_rows) == 1, f"Row '{row_name}' not found uniquely in Table 6 of {doc_path}"
            row_data = parsed[matching_rows[0]]

            if "test_vrmse" in metrics:
                exp_val = metrics["test_vrmse"]
                col_key = [c for c in row_data.keys() if "测试集 VRMSE" in c][0]
                assert exp_val in row_data[col_key], (
                    f"Table 6 row '{row_name}' col '{col_key}' expected {exp_val}, got {row_data[col_key]} in {doc_path}"
                )

        # Check vorticity RMSE and pressure gauge
        vort_rows = [k for k in parsed.keys() if "涡量 RMSE" in k]
        assert len(vort_rows) == 1
        vort_col = [c for c in parsed[vort_rows[0]].keys() if "测试集 RMSE" in c][0]
        assert f"{test_m['vorticity_rmse']:.4f}" in parsed[vort_rows[0]][vort_col]


def test_table_8_rollout_baselines_field_level(rollout_benchmark):
    """Verify Table 8 rollout baselines strictly align with rollout_benchmark.json and table_4_architecture_ablation.tex."""
    persistence = rollout_benchmark["persistence"]
    direct = rollout_benchmark["direct_transformer"]
    fno = rollout_benchmark["fno"]

    docs_to_check = [MANUSCRIPT_DIR / "04_experimental_results.md", MANUSCRIPT_DRAFT]

    for doc_path in docs_to_check:
        text = doc_path.read_text(encoding="utf-8")
        parsed = parse_markdown_table(text, "#### 表 8")

        # 1. Persistence row
        p_vrmse_rows = [k for k in parsed.keys() if "Persistence" in k and "散度" not in k and "涡量" not in k]
        assert len(p_vrmse_rows) == 1
        p_row = parsed[p_vrmse_rows[0]]
        # Step 1 VRMSE
        s1_col = [c for c in p_row.keys() if "Step 1" in c][0]
        assert f"{persistence['step_1']['vrmse_mean']:.4f}" in p_row[s1_col]
        # Step 30 VRMSE
        s30_col = [c for c in p_row.keys() if "Step 30" in c][0]
        assert f"{persistence['step_30']['vrmse_mean']:.4f}" in p_row[s30_col]

        # Persistence divergence row
        p_div_rows = [k for k in parsed.keys() if "Persistence" in k and "散度" in k]
        assert len(p_div_rows) == 1
        p_div_row = parsed[p_div_rows[0]]
        assert f"{persistence['step_1']['div_rmse']:.4f}" in p_div_row[s1_col]
        assert f"{persistence['step_30']['div_rmse']:.4f}" in p_div_row[s30_col]

        # 2. Direct ST divergence
        d_div_rows = [k for k in parsed.keys() if "Direct ST" in k and "散度" in k]
        assert len(d_div_rows) == 1
        assert f"{direct['step_30']['div_rmse']:.4f}" in parsed[d_div_rows[0]][s30_col]

        # 3. FNO Step 5 & Step 30
        fno_vrmse_rows = [k for k in parsed.keys() if "FNO-2D" in k and "散度" not in k and "涡量" not in k]
        assert len(fno_vrmse_rows) == 1
        fno_row = parsed[fno_vrmse_rows[0]]
        s5_col = [c for c in fno_row.keys() if "Step 5" in c][0]
        assert f"{fno['step_5']['vrmse_mean']:.4f}" in fno_row[s5_col]
        assert f"{fno['step_30']['vrmse_mean']:.4f}" in fno_row[s30_col]

        # 4. Latent WM E4 Step 5 & Step 30
        e4_vrmse_rows = [k for k in parsed.keys() if "E4" in k and "散度" not in k and "涡量" not in k]
        assert len(e4_vrmse_rows) == 1
        assert "0.4375" in parsed[e4_vrmse_rows[0]][s5_col]
        assert "1.5560" in parsed[e4_vrmse_rows[0]][s30_col]


def test_table_10_pde_controlled_field_level(pde_eval_metrics):
    """Verify Table 10 PDE controlled evaluation matches full_val JSON cell-by-cell."""
    comp = pde_eval_metrics["overall_comparison"]
    vrmse = comp["vrmse_standard"]
    div = comp["div_rmse"]
    res_s = comp["res_s_rmse"]

    docs_to_check = [MANUSCRIPT_DIR / "04_experimental_results.md", MANUSCRIPT_DRAFT]

    for doc_path in docs_to_check:
        text = doc_path.read_text(encoding="utf-8")
        parsed = parse_markdown_table(text, "#### 表 10")

        # Row D0
        d0_row = [k for k in parsed.keys() if "D0" in k][0]
        assert f"{vrmse['d0']:.4f}" in parsed[d0_row]["全场 VRMSE (`vrmse_standard`)"]
        assert f"{div['d0']:.4f}" in parsed[d0_row]["连续性散度残差 RMS (`div_rmse`)"]
        assert f"{res_s['d0']:.4f}" in parsed[d0_row]["示踪物对流扩散残差 RMS (`res_s_rmse`)"]

        # Row P0
        p0_row = [k for k in parsed.keys() if "P0" in k][0]
        assert f"{vrmse['p0']:.4f}" in parsed[p0_row]["全场 VRMSE (`vrmse_standard`)"]
        assert f"{div['p0']:.4f}" in parsed[p0_row]["连续性散度残差 RMS (`div_rmse`)"]
        assert f"{res_s['p0']:.4f}" in parsed[p0_row]["示踪物对流扩散残差 RMS (`res_s_rmse`)"]

        # Row PDE
        pde_row = [k for k in parsed.keys() if "PDE" in k and "P0" not in k and "D0" not in k][0]
        assert f"{vrmse['pde']:.4f}" in parsed[pde_row]["全场 VRMSE (`vrmse_standard`)"]
        assert f"{div['pde']:.4f}" in parsed[pde_row]["连续性散度残差 RMS (`div_rmse`)"]
        assert f"{res_s['pde']:.4f}" in parsed[pde_row]["示踪物对流扩散残差 RMS (`res_s_rmse`)"]


def test_discussion_section_5_1_numerical_consistency():
    """Verify Section 5.1 in 05_discussion_and_limitations.md references exact benchmark values."""
    disc_text = (MANUSCRIPT_DIR / "05_discussion_and_limitations.md").read_text(encoding="utf-8")

    # E4 Step 5: 0.4375
    assert "Step 5 VRMSE = $0.4375 \\pm 0.1502$" in disc_text
    # FNO Step 5: 0.7575, Step 30: 1.0720
    assert "0.7575" in disc_text
    assert "Step 30 VRMSE = 1.0720" in disc_text
    # Persistence Step 30: 0.4879, div RMS: 4.6998
    assert "4.6998" in disc_text
    assert "0.4879" in disc_text
    # FNO mode parameter disclosure
    assert "modes1=16, modes2=16" in disc_text


def test_strict_negative_assertions_reject_stale_values_and_formulas():
    """Verify that none of the documents contain historical erroneous/stale values or wrong formulas."""
    all_files = list(MANUSCRIPT_DIR.glob("*.md")) + [MANUSCRIPT_DRAFT]

    forbidden_patterns = [
        ("0.0884", "stale representation test u VRMSE"),
        ("0.2140", "stale representation test vorticity RMSE"),
        ("0.5284", "stale E4 Step 5 VRMSE"),
        ("0.6974", "stale FNO Step 5 VRMSE"),
        ("0.7788", "stale FNO Step 30 VRMSE"),
        ("0.8876", "stale Persistence Step 30 VRMSE"),
        ("k_{\\max}=12", "inaccurate FNO truncation formula"),
        ("k_max=12", "inaccurate FNO truncation formula"),
        ("\\multicolumn", "raw LaTeX multicolumn macro"),
        ("\\frac{\\sqrt{\\mathrm{MSE}}}{\\sqrt{\\mathrm{Var}(q^*)}+\\epsilon}", "erroneous VRMSE formula"),
    ]

    for doc_path in all_files:
        content = doc_path.read_text(encoding="utf-8")
        for bad_str, desc in forbidden_patterns:
            assert bad_str not in content, f"Found forbidden {desc} ('{bad_str}') in {doc_path}"


def test_all_image_paths_exist_on_disk():
    """Verify that every markdown image reference resolves to a valid, existing file on disk."""
    for md_path in MANUSCRIPT_DIR.glob("*.md"):
        content = md_path.read_text(encoding="utf-8")
        matches = re.findall(r'!\[.*?\]\((.*?)\)', content)
        for rel_img in matches:
            resolved = (md_path.parent / rel_img).resolve()
            assert resolved.is_file(), f"Image reference '{rel_img}' in {md_path} does not exist on disk: {resolved}"

    draft_content = MANUSCRIPT_DRAFT.read_text(encoding="utf-8")
    matches = re.findall(r'!\[.*?\]\((.*?)\)', draft_content)
    for rel_img in matches:
        resolved = (MANUSCRIPT_DRAFT.parent / rel_img).resolve()
        assert resolved.is_file(), f"Image reference '{rel_img}' in {MANUSCRIPT_DRAFT} does not exist on disk: {resolved}"


def test_manuscript_draft_perfect_synchronization():
    """Verify that MANUSCRIPT_DRAFT.md is an exact synchronized concatenation of the 6 chapter files."""
    parts = [
        "00_abstract.md",
        "01_introduction_and_formulation.md",
        "02_dataset_and_experimental_setting.md",
        "03_model_architecture_and_methodology.md",
        "04_experimental_results.md",
        "05_discussion_and_limitations.md",
    ]

    chunks = []
    for p in parts:
        c = (MANUSCRIPT_DIR / p).read_text(encoding="utf-8")
        c = c.replace("](../../outputs/", "](../outputs/")
        chunks.append(c)

    full_expected = "\n\n".join(chunks).rstrip() + "\n"
    expected_lines = [l.rstrip() for l in full_expected.splitlines()]
    expected_synced = "\n".join(expected_lines) + "\n"

    actual_synced = MANUSCRIPT_DRAFT.read_text(encoding="utf-8")
    assert actual_synced == expected_synced, "MANUSCRIPT_DRAFT.md is out of sync with split chapter files"
