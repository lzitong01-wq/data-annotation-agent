"""导出模块：五种格式、空结果、非法格式拦截。"""

import json

import pytest

from data_annotation_agent.core.exporter import EXPORTERS, export_results

EXTENSIONS = {
    "csv": ".csv",
    "jsonl": ".jsonl",
    "json": ".json",
    "markdown": ".md",
    "excel": ".xlsx",
}


@pytest.fixture
def results(make_result):
    return [
        make_result("s1", "疾病", confidence=0.8, annotator_id="m1", difficulty="simple", rationale="理由1"),
        make_result("s2", "症状", confidence=0.6, annotator_id="m2", difficulty="medium", rationale="理由2"),
    ]


def test_exporter_registry_covers_five_formats():
    assert set(EXPORTERS) == set(EXTENSIONS)


@pytest.mark.parametrize("fmt", list(EXTENSIONS))
def test_each_format_produces_non_empty_file(results, fmt, tmp_path):
    path = tmp_path / f"out{EXTENSIONS[fmt]}"
    export_results(results, fmt, str(path))
    assert path.exists()
    assert path.stat().st_size > 0


@pytest.mark.parametrize("fmt", list(EXTENSIONS))
def test_empty_results_still_produce_files(fmt, tmp_path):
    path = tmp_path / f"empty{EXTENSIONS[fmt]}"
    export_results([], fmt, str(path))
    assert path.exists()


def test_csv_has_bom_for_excel(results, tmp_path):
    path = tmp_path / "out.csv"
    export_results(results, "csv", str(path))
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_csv_contains_both_rows(results, tmp_path):
    path = tmp_path / "out.csv"
    export_results(results, "csv", str(path))
    lines = path.read_text(encoding="utf-8-sig").strip().splitlines()
    assert len(lines) == 3  # 表头 + 2 行


def test_jsonl_has_one_record_per_line(results, tmp_path):
    path = tmp_path / "out.jsonl"
    export_results(results, "jsonl", str(path))
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["label"] == "疾病"


def test_json_is_a_list(results, tmp_path):
    path = tmp_path / "out.json"
    export_results(results, "json", str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload[0]["sample_id"] == "s1"


def test_markdown_contains_table_header(results, tmp_path):
    path = tmp_path / "out.md"
    export_results(results, "markdown", str(path))
    text = path.read_text(encoding="utf-8")
    assert "| 样本 | 标签 |" in text
    assert "疾病" in text


def test_excel_is_readable_workbook(results, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "out.xlsx"
    export_results(results, "excel", str(path))
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    assert ws.max_row == 3  # 表头 + 2 行
    assert ws.cell(row=1, column=1).value == "样本"


def test_invalid_format_is_rejected(results, tmp_path):
    with pytest.raises(ValueError, match="不支持的导出格式"):
        export_results(results, "pdf", str(tmp_path / "x.pdf"))
