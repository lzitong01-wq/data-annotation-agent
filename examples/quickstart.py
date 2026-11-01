"""最小可运行示例：mock 模式下跑完一条完整标注链路（无需 API Key）。

运行：
    pip install -e .
    python examples/quickstart.py
"""

from data_annotation_agent import AnnotationPipeline, get_backend
from data_annotation_agent.core.models import AgentPresets, AnnotatorPreset, InspectorPreset

SAMPLES = [
    "患者出现持续高烧和剧烈头痛",
    "医生开具阿司匹林进行治疗",
    "确诊为病毒性肺炎",
    "患者服用抗生素后出现恶心",
]


def main() -> None:
    pipeline = AnnotationPipeline(backend=get_backend("mock"))
    industry = pipeline.registry.get_industry("medical")
    obj = pipeline.registry.get_object("medical_ner")

    pipeline.create_config(
        industry_id="medical",
        object_id="medical_ner",
        guideline=pipeline.guideline_agent.draft(industry, obj),  # 先起草一版指南
        presets=AgentPresets(
            annotator=AnnotatorPreset.RIGOROUS,  # 三个 AI 独立标注
            inspector=InspectorPreset.DEEP,      # 深度检验
        ),
        llm_backend="mock",
        llm_models=["mock-1", "mock-2", "mock-3"],
    )

    out = pipeline.run_full_pipeline(SAMPLES)

    print(f"样本数: {len(SAMPLES)}   多 AI 标注结果数: {len(SAMPLES) * 3}")
    print(f"裁决后最终结果数: {len(out['results'])}\n")

    for result in out["results"]:
        print(f"  {result.sample_id:<18} → {result.label:<6} "
              f"conf={result.confidence:.2f}  by={result.annotator_id}")

    agreement = out["inspection"]["agreement"]
    print(f"\n一致性方法: {agreement.get('method')}")
    print(f"一致性指标: {agreement.get('fleiss_kappa', agreement.get('value'))}")
    print(f"强度解读:   {agreement.get('interpretation')}")
    print(f"分歧样本:   {out['inspection']['n_disagreements']} 条")
    print(f"待人工复核: {len(out['review_list'])} 条")

    if out["updated_guideline"]:
        print(f"指南已按分歧优化到 v{out['updated_guideline'].version}")
    elif out["inspection"]["n_disagreements"]:
        print("存在分歧，但未配置标注指南，已跳过指南优化")
    else:
        print("本轮无分歧，指南未改动")


if __name__ == "__main__":
    main()
