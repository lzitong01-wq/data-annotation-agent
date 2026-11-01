"""标注 Agent：单 AI / 多 AI 模式、难度分级、可解释输出。"""

import pytest

from data_annotation_agent.agents.annotator import AnnotatorAgent
from data_annotation_agent.core.models import (
    AnnotationGuideline,
    AnnotatorPreset,
    GoldSample,
    GoldSet,
)


@pytest.fixture
def guideline():
    return AnnotationGuideline(title="医疗实体识别指南", rules=["识别疾病、症状、药物"])


@pytest.fixture
def reference_set():
    return GoldSet(
        name="参考集",
        samples=[GoldSample(data="持续高烧", label="症状"), GoldSample(data="确诊肺炎", label="疾病")],
    )


@pytest.fixture
def agent(mock_backend):
    return AnnotatorAgent(backend=mock_backend, models=["m1", "m2"])


def test_fast_mode_returns_single_result(agent, guideline, object_info):
    results = agent.annotate("患者发烧", AnnotatorPreset.FAST, guideline, None, object_info)
    assert len(results) == 1


def test_standard_mode_returns_single_result(agent, guideline, reference_set, object_info):
    results = agent.annotate("患者发烧", AnnotatorPreset.STANDARD, guideline, reference_set, object_info)
    assert len(results) == 1


def test_rigorous_mode_returns_one_result_per_model(agent, guideline, reference_set, object_info):
    results = agent.annotate("患者发烧", AnnotatorPreset.RIGOROUS, guideline, reference_set, object_info)
    assert len(results) == 2
    assert {r.annotator_id for r in results} == {"m1", "m2"}


def test_custom_preset_falls_back_to_single_ai(agent, guideline, object_info):
    results = agent.annotate("患者发烧", AnnotatorPreset.CUSTOM, guideline, None, object_info)
    assert len(results) == 1


def test_results_carry_explainability_fields(agent, guideline, object_info):
    result = agent.annotate("患者发烧", AnnotatorPreset.FAST, guideline, None, object_info)[0]
    assert result.label != ""
    assert 0.0 <= result.confidence <= 1.0
    assert result.rationale != ""


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(1.0, "simple"), (0.8, "simple"), (0.6, "medium"), (0.5, "medium"), (0.3, "difficult"), (0.0, "difficult")],
)
def test_difficulty_classification(confidence, expected):
    assert AnnotatorAgent._classify_difficulty(confidence) == expected


def test_difficulty_field_is_derived_from_confidence(agent, guideline, object_info):
    result = agent.annotate("患者发烧", AnnotatorPreset.FAST, guideline, None, object_info)[0]
    assert result.difficulty == AnnotatorAgent._classify_difficulty(result.confidence)


def test_model_without_dedicated_backend_falls_back(mock_backend, guideline, object_info):
    dedicated = object()
    agent = AnnotatorAgent(backend=mock_backend, models=["m1"], backends={"other": dedicated})
    assert agent._get_backend_for("m1") is mock_backend
    assert agent._get_backend_for("other") is dedicated


def test_multi_ai_backends_are_used_per_model(mock_backend, guideline, object_info):
    """每个模型走各自的后端实例。"""
    second = AnnotatorAgent(backend=mock_backend, models=["m2"]).backend
    agent = AnnotatorAgent(backend=mock_backend, models=["m1", "m2"], backends={"m2": second})
    results = agent.annotate("患者发烧", AnnotatorPreset.RIGOROUS, guideline, None, object_info)
    assert len(results) == 2
