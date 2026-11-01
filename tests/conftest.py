"""pytest 公共夹具。

项目采用 src 布局，`pyproject.toml` 里的 ``pythonpath = ["src"]``
保证测试无需安装即可导入 ``data_annotation_agent``。
"""

import pytest

from data_annotation_agent.core.models import AnnotationResult, Registry
from data_annotation_agent.llm.backend import get_backend


@pytest.fixture
def mock_backend():
    """无需 API Key 的模拟后端。"""
    return get_backend("mock")


@pytest.fixture
def registry():
    """预置了默认行业与研究对象的新注册表。"""
    return Registry()


@pytest.fixture
def object_info(registry):
    """标注 Agent 需要的任务描述（含标签体系）。"""
    obj = registry.get_object("medical_ner")
    return {
        "name": obj.name,
        "description": obj.description,
        "label_schema": obj.label_schema,
        "data_type": obj.data_type,
        "task_type": obj.task_type,
    }


@pytest.fixture
def make_result():
    """构造标注结果的小工厂。"""

    def _make(sample_id, label, confidence=0.8, annotator_id="m1",
              difficulty="simple", rationale=""):
        return AnnotationResult(
            sample_id=sample_id,
            label=label,
            confidence=confidence,
            annotator_id=annotator_id,
            difficulty=difficulty,
            rationale=rationale,
        )

    return _make
