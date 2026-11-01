"""
端到端测试:用模拟模式跑通完整流程
验证:选行业→选对象→选AI→选指南→选预设→标注→检验→复核→反馈闭环
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import (
    Registry, GoldenData, GoldSet, GoldSample, AnnotationGuideline,
    AgentPresets, AnnotatorPreset, InspectorPreset, AdjudicatorPreset,
    ReviewerPreset, MonitorPreset
)
from core.pipeline import AnnotationPipeline
from llm.backend import get_backend


def test_full_pipeline():
    """测试完整流程"""
    print("=" * 60)
    print("测试:数据标注智能体完整流程(模拟模式)")
    print("=" * 60)

    # 1. 创建 pipeline(模拟后端)
    pipeline = AnnotationPipeline(backend=get_backend("mock"))
    registry = pipeline.registry

    # 2. 选择行业和对象
    industry = registry.get_industry("medical")
    obj = registry.get_object("medical_ner")
    print(f"\n[1] 选择行业: {industry.name}")
    print(f"[2] 选择对象: {obj.name}")
    print(f"    标签: {obj.label_schema['labels']}")

    # 3. 创建金标(参考集)
    golden = GoldenData()
    golden.reference_set = GoldSet(name="参考集", samples=[
        GoldSample(data="患者出现持续高烧和剧烈头痛", label="症状"),
        GoldSample(data="医生开具阿司匹林进行治疗", label="药物"),
        GoldSample(data="确诊为病毒性肺炎", label="疾病"),
    ])
    print(f"\n[3] 金标参考集: {golden.summary()}")

    # 4. 起草标注指南
    guideline = pipeline.guideline_agent.draft(industry, obj)
    print(f"\n[4] 起草标注指南: {guideline.title}")
    print(f"    规则: {guideline.rules[:2]}")

    # 5. 配置预设(严谨模式:多AI互检)
    presets = AgentPresets(
        annotator=AnnotatorPreset.RIGOROUS,   # 多AI互检
        inspector=InspectorPreset.DEEP,       # 深度检验
        adjudicator=AdjudicatorPreset.MAJORITY,  # 多数投票
        reviewer=ReviewerPreset.DIFFICULT,    # 疑难复核
        monitor=MonitorPreset.BALANCED,       # 平衡监控
    )
    pipeline.create_config(
        industry.id, obj.id, golden_data=golden, guideline=guideline,
        presets=presets, llm_backend="mock", llm_models=["mock-model-1", "mock-model-2", "mock-model-3"]
    )
    print(f"\n[5] 配置预设: 标注={presets.annotator.value}, 检验={presets.inspector.value}")

    # 6. 成本预估
    cost = pipeline.estimate_cost(n_samples=5, n_models=3)
    print(f"\n[6] 成本预估: {cost['cost_est_usd']} USD, {cost['time_est_minutes']} 分钟")

    # 7. 标注样本
    samples = [
        "患者出现持续高烧和剧烈头痛",
        "医生开具阿司匹林进行治疗",
        "确诊为病毒性肺炎",
        "患者服用抗生素后出现恶心",
        "检查结果显示肺部有阴影",
    ]
    results = pipeline.annotate_samples(samples)
    print(f"\n[7] 标注完成: {len(results)} 条结果(多AI互检)")
    for r in results:
        print(f"    {r.sample_id[:20]}... → {r.label} (conf={r.confidence:.2f}, by={r.annotator_id})")

    # 8. 一致性检验 + AI互检
    inspection = pipeline.inspect_results()
    agreement = inspection.get("agreement", {})
    print(f"\n[8] 一致性检验: {agreement.get('method', '无')}")
    if "value" in agreement:
        print(f"    一致性: {agreement.get('value')} ({agreement.get('interpretation', '')})")
    if "fleiss_kappa" in agreement:
        print(f"    Fleiss' κ: {agreement.get('fleiss_kappa')}")
        print(f"    Krippendorff's α: {agreement.get('krippendorff_alpha')}")
    print(f"    分歧样本数: {inspection.get('n_disagreements', 0)}")

    # 9. 裁决(聚合多AI结果)
    final_results = pipeline.adjudicate_results()
    print(f"\n[9] 裁决完成: {len(final_results)} 条最终结果")
    for r in final_results:
        print(f"    {r.sample_id[:20]}... → {r.label}")

    # 10. 复核(选择疑难样本)
    review_list = pipeline.review_results(inspection.get("disagreements", []))
    print(f"\n[10] 复核列表: {len(review_list)} 条")
    for item in review_list:
        print(f"    {item.get('sample_id', '')[:20]}... (原因: {item.get('reason', '')})")

    # 11. 反馈闭环(分歧回流到指南)
    updated = pipeline.feedback_loop(inspection.get("disagreements", []))
    if updated:
        print(f"\n[11] 反馈闭环: 指南优化到版本 {updated.version}")
    else:
        print(f"\n[11] 反馈闭环: 无分歧或无需优化")

    # 12. 监控
    monitor_result = pipeline.monitor_progress(
        kappa=agreement.get("value", 0) if "value" in agreement else agreement.get("fleiss_kappa", 0),
        n_labeled=len(final_results), total=len(samples)
    )
    print(f"\n[12] 监控: should_stop={monitor_result['should_stop']}, 原因={monitor_result['reason']}")

    print("\n" + "=" * 60)
    print("✅ 完整流程测试通过!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_full_pipeline()
