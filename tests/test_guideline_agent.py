"""标注指南 Agent：起草、分歧驱动的版本化优化。"""

import pytest

from data_annotation_agent.agents.guideline_agent import GuidelineAgent
from data_annotation_agent.core.models import AnnotationGuideline


@pytest.fixture
def agent(mock_backend):
    return GuidelineAgent(backend=mock_backend)


@pytest.fixture
def guideline(agent, registry):
    return agent.draft(registry.get_industry("medical"), registry.get_object("medical_ner"))


def test_draft_produces_titled_guideline(agent, registry):
    g = agent.draft(registry.get_industry("medical"), registry.get_object("medical_ner"))
    assert g.title != ""
    assert isinstance(g.rules, list)
    assert g.source == "drafted"


def test_draft_carries_label_schema_from_object(agent, registry):
    obj = registry.get_object("medical_ner")
    g = agent.draft(registry.get_industry("medical"), obj)
    assert g.label_schema == obj.label_schema


def test_optimize_without_disagreements_returns_same_object(agent, guideline):
    assert agent.optimize(guideline, []) is guideline


def test_optimize_bumps_version(agent, guideline):
    updated = agent.optimize(guideline, [{"sample_id": "s1", "labels": ["疾病", "症状"]}])
    assert updated.version == "1.1"
    assert updated.source == "optimized"


def test_optimize_twice_reaches_version_1_2(agent, guideline):
    once = agent.optimize(guideline, [{"sample_id": "s1", "labels": ["A", "B"]}])
    twice = agent.optimize(once, [{"sample_id": "s2", "labels": ["A", "B"]}])
    assert twice.version == "1.2"


def test_optimize_refreshes_timestamp(agent, guideline):
    updated = agent.optimize(guideline, [{"sample_id": "s1", "labels": ["A", "B"]}])
    assert updated.updated_at >= updated.created_at


def test_optimize_preserves_original_guideline(agent, guideline):
    """优化返回副本，原指南不应被就地改写。"""
    before = guideline.version
    agent.optimize(guideline, [{"sample_id": "s1", "labels": ["A", "B"]}])
    assert guideline.version == before


def test_guideline_renders_all_sections_to_prompt_text():
    g = AnnotationGuideline(
        title="测试指南",
        description="用于验证渲染",
        rules=["规则一"],
        boundary_cases=["边界一"],
        examples=[{"input": "输入示例", "output": "输出示例"}],
    )
    text = g.to_prompt_text()
    assert "测试指南" in text
    assert "## 标注规则" in text
    assert "## 边界情况处理" in text
    assert "## 示例" in text
