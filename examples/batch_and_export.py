"""分批标注与中途修改：断点续标 + 批间改指南 + 多格式导出。

运行：
    pip install -e .
    python examples/batch_and_export.py
"""

from pathlib import Path

from data_annotation_agent import AnnotationPipeline, export_results, get_backend
from data_annotation_agent.core.models import AnnotationGuideline

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    pipeline = AnnotationPipeline(backend=get_backend("mock"))
    pipeline.create_config(
        "ecommerce",
        "ecommerce_sentiment",
        llm_backend="mock",
        llm_models=["mock-1"],
    )

    # 第一批
    pipeline.annotate_batch(["这个手机很好用，强烈推荐", "质量太差，非常失望"], append=True)
    print(f"第一批标注后：{len(pipeline.results)} 条")

    # 批间修改指南（长任务里最常见的中途干预）
    pipeline.update_guideline(
        AnnotationGuideline(
            title="电商情感标注指南 v2",
            rules=[
                "含明确褒义词（好、赞、推荐）判为正面",
                "含明确贬义词（差、烂、失望）判为负面",
                "纯客观陈述、无情感倾向判为中性",
            ],
            boundary_cases=["褒贬同时出现时，以对商品本身的评价为准"],
        )
    )
    print("已更新标注指南，继续标注下一批")

    # 第二批（断点续标，不清空已有结果）
    pipeline.annotate_batch(["包装完好，物流正常"], append=True)
    print(f"第二批标注后：{len(pipeline.results)} 条\n")

    # 导出三种格式
    OUTPUT_DIR.mkdir(exist_ok=True)
    for fmt, filename in [("csv", "annotations.csv"), ("jsonl", "annotations.jsonl"), ("markdown", "annotations.md")]:
        path = OUTPUT_DIR / filename
        export_results(pipeline.results, fmt, str(path))
        print(f"已导出 {fmt:<8} → {path.relative_to(OUTPUT_DIR.parent)}")


if __name__ == "__main__":
    main()
