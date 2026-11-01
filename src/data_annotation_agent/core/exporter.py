"""
结果导出模块
把标注结果导出为多种格式:CSV / JSONL / Excel(xlsx)/ Markdown
供 UI 与编程方式复用
"""
import csv
import json
from typing import Any, Dict, List

from .models import AnnotationResult


def _result_to_dict(r: AnnotationResult) -> Dict[str, Any]:
    """把单个标注结果转为可序列化字典"""
    return {
        "sample_id": r.sample_id,
        "label": r.label,
        "confidence": r.confidence,
        "annotator_id": r.annotator_id,
        "difficulty": r.difficulty,
        "rationale": r.rationale,
        "created_at": r.created_at.isoformat() if r.created_at else "",
    }


def _rows(results: List[AnnotationResult]) -> List[Dict[str, Any]]:
    return [_result_to_dict(r) for r in results]


def export_csv(results: List[AnnotationResult], path: str) -> str:
    """导出为 CSV(UTF-8 with BOM,便于 Excel 直接打开中文)"""
    rows = _rows(results)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        if not rows:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "label", "confidence", "annotator_id", "difficulty", "rationale", "created_at"])
        else:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return path


def export_jsonl(results: List[AnnotationResult], path: str) -> str:
    """导出为 JSONL(每行一条 JSON)"""
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(_result_to_dict(r), ensure_ascii=False) + "\n")
    return path


def export_json(results: List[AnnotationResult], path: str) -> str:
    """导出为 JSON 数组"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_rows(results), f, ensure_ascii=False, indent=2)
    return path


def export_markdown(results: List[AnnotationResult], path: str) -> str:
    """导出为 Markdown 表格"""
    lines = ["# 标注结果", "", "| 样本 | 标签 | 置信度 | 标注者 | 难度 | 理由 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        rationale = (r.rationale or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {r.sample_id} | {r.label} | {r.confidence:.3f} | {r.annotator_id} | {r.difficulty} | {rationale} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def export_excel(results: List[AnnotationResult], path: str) -> str:
    """导出为 Excel(xlsx),依赖 openpyxl"""
    try:
        import openpyxl
    except ImportError:
        raise ImportError("导出 Excel 需要 openpyxl,请先安装: pip install openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "标注结果"
    headers = ["样本", "标签", "置信度", "标注者", "难度", "理由", "创建时间"]
    ws.append(headers)
    for r in results:
        ws.append([
            r.sample_id, r.label, r.confidence, r.annotator_id,
            r.difficulty, r.rationale,
            r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "",
        ])
    # 表头加粗
    for cell in ws[1]:
        cell.font = openpyxl.styles.Font(bold=True)
    wb.save(path)
    return path


# 格式 -> 导出函数映射
EXPORTERS = {
    "csv": export_csv,
    "jsonl": export_jsonl,
    "json": export_json,
    "markdown": export_markdown,
    "excel": export_excel,
}


def export_results(results: List[AnnotationResult], fmt: str, path: str) -> str:
    """按格式导出标注结果,返回文件路径"""
    fmt = fmt.lower()
    if fmt not in EXPORTERS:
        raise ValueError(f"不支持的导出格式: {fmt}(可选: {', '.join(EXPORTERS)})")
    return EXPORTERS[fmt](results, path)
