"""注册表：默认行业 / 研究对象、行业关联、自定义注册。"""

import pytest

from data_annotation_agent.core.models import (
    DataType,
    Industry,
    Registry,
    ResearchObject,
    TaskType,
)


def test_registry_has_at_least_eight_industries():
    assert len(Registry().list_industries()) >= 8


def test_every_industry_has_at_least_one_object():
    registry = Registry()
    for industry in registry.list_industries():
        assert registry.list_objects_by_industry(industry.id), f"行业「{industry.name}」没有任何研究对象"


@pytest.mark.parametrize(
    "object_id",
    [
        "medical_ner",
        "finance_intent",
        "legal_ner",
        "education_question_class",
        "industrial_defect_class",
        "autonomous_scene_class",
    ],
)
def test_expected_objects_are_registered(object_id):
    assert Registry().get_object(object_id) is not None


def test_general_industry_has_generic_objects():
    assert len(Registry().list_objects_by_industry("general")) >= 5


def test_object_industry_link_is_explicit():
    obj = Registry().get_object("medical_ner")
    assert obj.industry_id == "medical"


def test_unknown_ids_return_none():
    registry = Registry()
    assert registry.get_industry("does_not_exist") is None
    assert registry.get_object("does_not_exist") is None


def test_register_custom_industry_and_object():
    registry = Registry()
    registry.register_industry(Industry(id="energy", name="能源", description="电力与新能源"))

    obj = ResearchObject(
        id="energy_defect_class",
        name="光伏板缺陷分类",
        data_type=DataType.IMAGE,
        task_type=TaskType.IMAGE_CLASSIFICATION,
        label_schema={"labels": ["正常", "隐裂", "断栅"]},
        industry_id="energy",
    )
    registry.register_object(obj)

    assert registry.get_industry("energy").name == "能源"
    assert registry.list_objects_by_industry("energy") == [obj]


def test_label_schema_is_carried_on_objects():
    obj = Registry().get_object("ecommerce_sentiment")
    assert obj.label_schema["labels"] == ["正面", "负面", "中性"]
