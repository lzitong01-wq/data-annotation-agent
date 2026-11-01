"""
单元测试:覆盖各核心模块
agreement / monitor / models / annotator / guideline_agent / inspector / adjudicator / reviewer / exporter / pipeline 增量标注
脚本式测试(不引入 pytest),直接 python tests/test_units.py 运行
"""
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import (
    Registry, GoldenData, GoldSet, GoldSample, AnnotationGuideline,
    AgentPresets, AnnotatorPreset, InspectorPreset, AdjudicatorPreset,
    ReviewerPreset, MonitorPreset, AnnotationResult, Industry, ResearchObject
)
from core.pipeline import AnnotationPipeline
from core.exporter import export_results, EXPORTERS
from llm.backend import get_backend
from quality.agreement import cohens_kappa, fleiss_kappa, krippendorff_alpha, interpret_kappa, compute_agreement
from quality.monitor import MonitorAgent, CostEstimator
from agents.annotator import AnnotatorAgent
from agents.inspector import InspectorAgent
from agents.adjudicator import AdjudicatorAgent
from agents.reviewer import ReviewerAgent
from agents.guideline_agent import GuidelineAgent

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {detail}")


def make_result(sample_id, label, confidence=0.8, annotator_id="m1", difficulty="simple", rationale=""):
    return AnnotationResult(
        sample_id=sample_id, label=label, confidence=confidence,
        annotator_id=annotator_id, difficulty=difficulty, rationale=rationale,
    )


# ============ 1. agreement 一致性指标 ============
def test_agreement():
    print("\n[1] agreement 一致性指标")
    # Cohen's κ:完全一致 = 1.0
    k = cohens_kappa(["A", "B", "A"], ["A", "B", "A"])
    check("Cohen's κ 完全一致=1.0", abs(k - 1.0) < 1e-9, f"got {k}")
    # 完全不一致(反向配对,边际均衡时 κ=-1)
    k2 = cohens_kappa(["A", "B"], ["B", "A"])
    check("Cohen's κ 完全不一致<0", k2 < 0, f"got {k2}")
    # Fleiss' κ:完全一致 = 1.0
    fk = fleiss_kappa([["A", "A", "A"], ["B", "B", "B"]])
    check("Fleiss' κ 完全一致=1.0", abs(fk - 1.0) < 1e-9, f"got {fk}")
    # Krippendorff's α
    ka = krippendorff_alpha([["A", "A", "A"], ["B", "B", "B"]])
    check("Krippendorff's α 完全一致=1.0", abs(ka - 1.0) < 1e-9, f"got {ka}")
    # interpret_kappa 分级
    check("interpret_kappa 0.9=几乎完美", "几乎完美" in interpret_kappa(0.9))
    check("interpret_kappa 0.7=显著", "显著" in interpret_kappa(0.7))
    check("interpret_kappa 0.3=一般", "一般" in interpret_kappa(0.3))
    # compute_agreement
    agg = compute_agreement([["A", "A"], ["B", "B"]])
    check("compute_agreement 返回 method", "method" in agg)
    check("compute_agreement 返回 value", "value" in agg)


# ============ 2. monitor 监控与成本 ============
def test_monitor():
    print("\n[2] monitor 监控与成本")
    m = MonitorAgent()
    # 高一致性 + 稳定窗口(需多次记录) → 停止
    for _ in range(3):
        m.record(kappa=0.9, n_labeled=10, total=10)
    stop = m.should_stop(MonitorPreset.BALANCED, kappa=0.9, n_labeled=10, total=10)
    check("BALANCED 高κ+稳定窗口 → 停止", stop["should_stop"] is True, f"got {stop}")
    # 低一致性 → 不停止
    stop2 = m.should_stop(MonitorPreset.BALANCED, kappa=0.2, n_labeled=5, total=10)
    check("BALANCED 低κ → 不停止", stop2["should_stop"] is False, f"got {stop2}")
    # record 记录
    m.record(kappa=0.7, n_labeled=3, total=10)
    summary = m.get_summary()
    check("record 后 summary 有记录", summary.get("n_records", 0) >= 1, f"got {summary}")
    # CostEstimator
    ce = CostEstimator()
    est = ce.estimate(n_samples=100, n_models=2)
    check("成本预估含 token", "total_tokens_est" in est)
    check("成本预估含成本", "cost_est_usd" in est)
    check("成本预估含时间", "time_est_minutes" in est)


# ============ 3. models Registry 扩充 ============
def test_models():
    print("\n[3] models Registry 扩充")
    r = Registry()
    industries = r.list_industries()
    check("行业数>=8", len(industries) >= 8, f"got {len(industries)}")
    # 每个行业都有对象(修复'通用'无对象问题)
    for ind in industries:
        objs = r.list_objects_by_industry(ind.id)
        check(f"行业 {ind.name} 有对象", len(objs) > 0, f"got {len(objs)}")
    # 新增行业对象存在
    check("金融意图识别存在", r.get_object("finance_intent") is not None)
    check("法律实体识别存在", r.get_object("legal_ner") is not None)
    check("教育试题分类存在", r.get_object("education_question_class") is not None)
    check("工业缺陷分类存在", r.get_object("industrial_defect_class") is not None)
    check("自动驾驶场景分类存在", r.get_object("autonomous_scene_class") is not None)
    # 通用行业对象
    general_objs = r.list_objects_by_industry("general")
    check("通用行业有对象", len(general_objs) >= 5, f"got {len(general_objs)}")
    # 对象 industry_id 显式关联
    obj = r.get_object("medical_ner")
    check("medical_ner.industry_id=medical", obj.industry_id == "medical", f"got {obj.industry_id}")


# ============ 4. annotator 标注 ============
def test_annotator():
    print("\n[4] annotator 标注")
    backend = get_backend("mock")
    ann = AnnotatorAgent(backend=backend, models=["m1", "m2"])
    obj = Registry().get_object("medical_ner")
    object_info = {"name": obj.name, "labels": obj.label_schema["labels"]}
    guideline = AnnotationGuideline(title="指南", rules=["识别医疗实体"])
    ref = GoldSet(name="参考", samples=[GoldSample(data="发烧", label="症状")])
    # FAST:单AI
    res = ann.annotate("患者发烧", AnnotatorPreset.FAST, guideline, None, object_info)
    check("FAST 返回1条", len(res) == 1, f"got {len(res)}")
    # STANDARD:单AI+参考集
    res2 = ann.annotate("患者发烧", AnnotatorPreset.STANDARD, guideline, ref, object_info)
    check("STANDARD 返回1条", len(res2) == 1, f"got {len(res2)}")
    # RIGOROUS:多AI互检
    res3 = ann.annotate("患者发烧", AnnotatorPreset.RIGOROUS, guideline, ref, object_info)
    check("RIGOROUS 返回多AI", len(res3) >= 2, f"got {len(res3)}")
    # 难度分级
    check("conf 0.9=simple", AnnotatorAgent._classify_difficulty(0.9) == "simple")
    check("conf 0.6=medium", AnnotatorAgent._classify_difficulty(0.6) == "medium")
    check("conf 0.3=difficult", AnnotatorAgent._classify_difficulty(0.3) == "difficult")


# ============ 5. guideline_agent 指南 ============
def test_guideline_agent():
    print("\n[5] guideline_agent 指南")
    backend = get_backend("mock")
    ga = GuidelineAgent(backend=backend)
    ind = Registry().get_industry("medical")
    obj = Registry().get_object("medical_ner")
    g = ga.draft(ind, obj)
    check("draft 生成指南", g is not None and g.title != "")
    check("draft 规则为列表", isinstance(g.rules, list))
    # 无分歧 → 不优化
    g2 = ga.optimize(g, [])
    check("无分歧不优化", g2 is g)
    # 有分歧 → 优化(版本+0.1,Decimal 精确)
    g3 = ga.optimize(g, [{"sample_id": "s1", "labels": ["A", "B"]}])
    check("有分歧优化版本", g3.version == "1.1", f"got {g3.version}")
    check("优化后 source=optimized", g3.source == "optimized")
    check("优化后 updated_at 刷新", g3.updated_at >= g3.created_at)


# ============ 6. inspector 检验 ============
def test_inspector():
    print("\n[6] inspector 检验")
    ins = InspectorAgent()
    results = [
        make_result("s1", "疾病", annotator_id="m1"),
        make_result("s1", "疾病", annotator_id="m2"),
        make_result("s2", "症状", annotator_id="m1"),
        make_result("s2", "症状", annotator_id="m2"),
    ]
    # LIGHT:两标注者 Cohen's κ
    light = ins.inspect(results, InspectorPreset.LIGHT)
    check("LIGHT 返回 agreement", "agreement" in light)
    # STANDARD:Fleiss' κ
    std = ins.inspect(results, InspectorPreset.STANDARD)
    check("STANDARD 返回 agreement", "agreement" in std)
    # DEEP:多指标
    deep = ins.inspect(results, InspectorPreset.DEEP)
    check("DEEP 返回 agreement", "agreement" in deep)
    # 分歧检测
    dis = ins.find_disagreements(results)
    check("一致样本无分歧", len(dis) == 0, f"got {len(dis)}")


# ============ 7. adjudicator 裁决 ============
def test_adjudicator():
    print("\n[7] adjudicator 裁决")
    adj = AdjudicatorAgent()
    results = [
        make_result("s1", "疾病", annotator_id="m1"),
        make_result("s1", "疾病", annotator_id="m2"),
        make_result("s1", "症状", annotator_id="m3"),
    ]
    # 多数投票
    maj = adj.majority_vote(results)
    check("多数投票=疾病", maj.label == "疾病", f"got {maj.label}")
    # 加权
    w = adj.weighted_aggregate(results, weights={"m1": 1.0, "m2": 1.0, "m3": 0.1})
    check("加权聚合返回结果", w is not None)
    # 人工裁决
    human = adj.human_adjudicate(results, human_label="疾病")
    check("人工裁决=疾病", human.label == "疾病", f"got {human.label}")
    # adjudicate 分发
    r = adj.adjudicate(results, AdjudicatorPreset.MAJORITY)
    check("adjudicate MAJORITY", r.label == "疾病")


# ============ 8. reviewer 复核 ============
def test_reviewer():
    print("\n[8] reviewer 复核")
    rev = ReviewerAgent()
    results = [
        make_result("s1", "疾病", difficulty="simple"),
        make_result("s2", "症状", difficulty="difficult"),
        make_result("s3", "药物", difficulty="medium"),
    ]
    # DIFFICULT:只复核疑难
    diff = rev.select_for_review(results, ReviewerPreset.DIFFICULT, disagreements=[])
    check("DIFFICULT 只选疑难", all(i.get("difficulty") == "difficult" for i in diff), f"got {diff}")
    # FULL:全量
    full = rev.select_for_review(results, ReviewerPreset.FULL)
    check("FULL 全量复核", len(full) == len(results), f"got {len(full)}")
    # SAMPLE:抽检
    sample = rev.select_for_review(results, ReviewerPreset.SAMPLE)
    check("SAMPLE 抽检>=1", len(sample) >= 1, f"got {len(sample)}")


# ============ 9. exporter 导出 ============
def test_exporter():
    print("\n[9] exporter 导出")
    results = [
        make_result("s1", "疾病", confidence=0.8, annotator_id="m1", difficulty="simple", rationale="理由1"),
        make_result("s2", "症状", confidence=0.6, annotator_id="m2", difficulty="medium", rationale="理由2"),
    ]
    d = tempfile.mkdtemp()
    for fmt in ["csv", "jsonl", "json", "markdown", "excel"]:
        p = os.path.join(d, f"out.{fmt}")
        export_results(results, fmt, p)
        check(f"导出 {fmt} 生成文件", os.path.exists(p) and os.path.getsize(p) > 0, f"got {p}")
    # 空结果
    for fmt in ["csv", "jsonl", "json", "markdown", "excel"]:
        p = os.path.join(d, f"empty.{fmt}")
        export_results([], fmt, p)
        check(f"空结果导出 {fmt}", os.path.exists(p), f"got {p}")
    # 非法格式拦截
    try:
        export_results(results, "pdf", os.path.join(d, "x.pdf"))
        check("非法格式拦截", False)
    except ValueError:
        check("非法格式拦截", True)
    # 格式映射完整
    check("EXPORTERS 含5种格式", set(EXPORTERS.keys()) == {"csv", "jsonl", "json", "markdown", "excel"})


# ============ 10. pipeline 增量标注 ============
def test_pipeline_incremental():
    print("\n[10] pipeline 增量标注")
    p = AnnotationPipeline(backend=get_backend("mock"))
    reg = p.registry
    ind = reg.get_industry("medical")
    obj = reg.get_object("medical_ner")
    p.create_config(
        ind.id, obj.id,
        presets=AgentPresets(annotator=AnnotatorPreset.STANDARD),
        llm_backend="mock", llm_models=["m1"],
    )
    # 第一批
    r1 = p.annotate_batch(["患者高烧"], append=True)
    check("第一批结果数=1", len(p.results) == 1, f"got {len(p.results)}")
    # 断点续标:追加第二批
    r2 = p.annotate_batch(["患者咳嗽"], append=True)
    check("追加后结果数=2", len(p.results) == 2, f"got {len(p.results)}")
    # 批间修改指南
    g = AnnotationGuideline(title="新指南", rules=["新规则"])
    p.update_guideline(g)
    check("批间修改指南", p.config.guideline.title == "新指南")
    # 批间修改预设
    p.update_presets(AgentPresets(annotator=AnnotatorPreset.RIGOROUS))
    check("批间修改预设", p.config.presets.annotator.value == "rigorous")
    # 批间修改金标
    p.update_golden_data(GoldenData(reference_set=GoldSet(name="参考", samples=[GoldSample(data="x", label="y")])))
    check("批间修改金标", p.config.golden_data.reference_set.name == "参考")
    # 重标(覆盖)
    r3 = p.annotate_batch(["患者头痛"], append=False)
    check("重标后结果数=1", len(p.results) == 1, f"got {len(p.results)}")
    # 清空
    p.reset_results()
    check("清空后结果数=0", len(p.results) == 0, f"got {len(p.results)}")
    check("清空后反馈日志=0", len(p.feedback_log) == 0, f"got {len(p.feedback_log)}")


def main():
    print("=" * 60)
    print("单元测试:覆盖各核心模块")
    print("=" * 60)
    test_agreement()
    test_monitor()
    test_models()
    test_annotator()
    test_guideline_agent()
    test_inspector()
    test_adjudicator()
    test_reviewer()
    test_exporter()
    test_pipeline_incremental()
    print("\n" + "=" * 60)
    print(f"结果: {PASS} 通过, {FAIL} 失败")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
