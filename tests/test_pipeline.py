"""流程编排：配置校验、分批标注、批间修改、难度路由、成本预估。"""

import pytest

from data_annotation_agent.core.models import (
    AgentPresets,
    AnnotationGuideline,
    AnnotatorPreset,
    GoldSample,
    GoldSet,
    GoldenData,
)
from data_annotation_agent.core.pipeline import AnnotationPipeline


@pytest.fixture
def pipeline(mock_backend):
    return AnnotationPipeline(backend=mock_backend)


@pytest.fixture
def configured(pipeline):
    pipeline.create_config("medical", "medical_ner", llm_backend="mock", llm_models=["m1"])
    return pipeline


def test_invalid_industry_is_rejected(pipeline):
    with pytest.raises(ValueError, match="无效的行业/对象"):
        pipeline.create_config("does_not_exist", "medical_ner")


def test_invalid_object_is_rejected(pipeline):
    with pytest.raises(ValueError, match="无效的行业/对象"):
        pipeline.create_config("medical", "does_not_exist")


def test_annotating_without_config_raises(pipeline):
    with pytest.raises(ValueError, match="请先创建配置"):
        pipeline.annotate_batch(["患者高烧"])


def test_append_batch_accumulates_results(configured):
    configured.annotate_batch(["患者高烧"], append=True)
    assert len(configured.results) == 1
    configured.annotate_batch(["患者咳嗽"], append=True)
    assert len(configured.results) == 2


def test_overwrite_batch_replaces_results(configured):
    configured.annotate_batch(["患者高烧", "患者咳嗽"], append=True)
    configured.annotate_batch(["患者头痛"], append=False)
    assert len(configured.results) == 1
    assert configured.results[0].sample_id == "患者头痛"


def test_reset_clears_results_and_feedback_log(configured):
    configured.annotate_batch(["患者高烧"])
    configured.feedback_log.append({"type": "dummy"})
    configured.reset_results()
    assert configured.results == []
    assert configured.feedback_log == []


def test_guideline_can_be_updated_between_batches(configured):
    configured.update_guideline(AnnotationGuideline(title="新指南", rules=["新规则"]))
    assert configured.config.guideline.title == "新指南"


def test_presets_can_be_updated_between_batches(configured):
    configured.update_presets(AgentPresets(annotator=AnnotatorPreset.RIGOROUS))
    assert configured.config.presets.annotator is AnnotatorPreset.RIGOROUS


def test_golden_data_can_be_updated_between_batches(configured):
    golden = GoldenData(reference_set=GoldSet(name="参考", samples=[GoldSample(data="x", label="y")]))
    configured.update_golden_data(golden)
    assert configured.config.golden_data.reference_set.name == "参考"


def test_updates_require_config(pipeline):
    with pytest.raises(ValueError, match="请先创建配置"):
        pipeline.update_guideline(AnnotationGuideline(title="x"))


def test_route_by_difficulty_buckets(configured, make_result):
    configured.results = [
        make_result("s1", "疾病", confidence=0.9, difficulty="simple"),
        make_result("s2", "症状", confidence=0.6, difficulty="medium"),
        make_result("s3", "药物", confidence=0.2, difficulty="difficult"),
    ]
    routed = configured.route_by_difficulty()
    assert routed["stats"]["simple"] == 1
    assert routed["stats"]["medium"] == 1
    assert routed["stats"]["difficult"] == 1
    assert routed["stats"]["total"] == 3


def test_estimate_cost_uses_configured_model(configured):
    cost = configured.estimate_cost(n_samples=5, n_models=3)
    assert cost["n_samples"] == 5
    assert cost["n_models"] == 3
    assert cost["total_tokens_est"] > 0


def test_llm_connection_test_succeeds_for_mock(pipeline):
    result = pipeline.test_llm_connection("mock")
    assert result["ok"] is True


def test_llm_connection_test_reports_failure(pipeline):
    result = pipeline.test_llm_connection("custom", base_url="", api_key="", model="")
    assert result["ok"] is False


def test_inspect_on_empty_results(pipeline):
    out = pipeline.inspect_results()
    assert out["agreement"]["method"] == "无"
    assert out["disagreements"] == []


def test_object_info_exposes_label_schema(configured):
    info = configured.get_object_info()
    assert info["id"] == "medical_ner"
    assert "labels" in info["label_schema"]
