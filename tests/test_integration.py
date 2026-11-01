"""端到端集成测试：模拟模式跑通 标注 → 检验 → 裁决 → 复核 → 反馈闭环 → 导出。"""

import pytest

from data_annotation_agent.core.exporter import export_results
from data_annotation_agent.core.models import (
    AdjudicatorPreset,
    AgentPresets,
    AnnotatorPreset,
    GoldSample,
    GoldSet,
    GoldenData,
    InspectorPreset,
    MonitorPreset,
    ReviewerPreset,
)
from data_annotation_agent.core.pipeline import AnnotationPipeline

SAMPLES = [
    "患者出现持续高烧和剧烈头痛",
    "医生开具阿司匹林进行治疗",
    "确诊为病毒性肺炎",
    "患者服用抗生素后出现恶心",
    "检查结果显示肺部有阴影",
]

MODELS = ["mock-1", "mock-2", "mock-3"]


@pytest.fixture
def pipeline(mock_backend, registry):
    """严谨模式 + 深度检验 + 多 AI 互检的完整配置。"""
    p = AnnotationPipeline(backend=mock_backend)
    industry = registry.get_industry("medical")
    obj = registry.get_object("medical_ner")

    golden = GoldenData(
        reference_set=GoldSet(
            name="参考集",
            samples=[
                GoldSample(data="持续高烧", label="症状"),
                GoldSample(data="确诊为病毒性肺炎", label="疾病"),
            ],
        )
    )
    p.create_config(
        industry.id,
        obj.id,
        golden_data=golden,
        guideline=p.guideline_agent.draft(industry, obj),
        presets=AgentPresets(
            annotator=AnnotatorPreset.RIGOROUS,
            inspector=InspectorPreset.DEEP,
            adjudicator=AdjudicatorPreset.MAJORITY,
            reviewer=ReviewerPreset.DIFFICULT,
            monitor=MonitorPreset.BALANCED,
        ),
        llm_backend="mock",
        llm_models=MODELS,
    )
    return p


def test_full_pipeline_yields_one_result_per_sample(pipeline):
    out = pipeline.run_full_pipeline(SAMPLES)
    assert len(out["results"]) == len(SAMPLES)
    assert {r.sample_id for r in out["results"]} == set(SAMPLES)


def test_multi_ai_annotation_is_collapsed_by_adjudication(pipeline):
    pipeline.annotate_samples(SAMPLES)
    assert len(pipeline.results) == len(SAMPLES) * len(MODELS)

    pipeline.adjudicate_results()
    assert len(pipeline.results) == len(SAMPLES)


def test_inspection_reports_agreement_and_sample_count(pipeline):
    out = pipeline.run_full_pipeline(SAMPLES)
    inspection = out["inspection"]
    assert "agreement" in inspection
    assert inspection["n_samples"] == len(SAMPLES)


def test_all_final_labels_are_non_empty(pipeline):
    out = pipeline.run_full_pipeline(SAMPLES)
    assert all(result.label for result in out["results"])


def test_review_list_is_returned(pipeline):
    out = pipeline.run_full_pipeline(SAMPLES)
    assert isinstance(out["review_list"], list)


def test_feedback_loop_bumps_version_only_when_disagreements_exist(pipeline):
    out = pipeline.run_full_pipeline(SAMPLES)
    if out["inspection"]["n_disagreements"]:
        assert out["updated_guideline"] is not None
        assert out["updated_guideline"].version == "1.1"
    else:
        assert out["updated_guideline"] is None


def test_monitor_progress_returns_stop_decision(pipeline):
    out = pipeline.run_full_pipeline(SAMPLES)
    agreement = out["inspection"]["agreement"]
    kappa = agreement.get("value", agreement.get("fleiss_kappa", 0.0))
    decision = pipeline.monitor_progress(kappa=kappa, n_labeled=len(out["results"]), total=len(SAMPLES))
    assert "should_stop" in decision
    assert "reason" in decision


def test_human_review_can_override_final_labels(pipeline):
    pipeline.annotate_samples(SAMPLES)
    pipeline.adjudicate_results()
    target = SAMPLES[0]
    pipeline.apply_review({target: "人工标签"})
    overridden = [r for r in pipeline.results if r.sample_id == target][0]
    assert overridden.label == "人工标签"
    assert overridden.annotator_id == "human"


def test_pipeline_results_can_be_exported(pipeline, tmp_path):
    out = pipeline.run_full_pipeline(SAMPLES)
    path = tmp_path / "annotations.csv"
    export_results(out["results"], "csv", str(path))
    assert path.exists()
    assert path.stat().st_size > 0
