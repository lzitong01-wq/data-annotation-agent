"""界面联动：切换行业时，研究对象下拉框只保留该行业下的任务。"""

import pytest

from data_annotation_agent.ui.app import AnnotationUI


@pytest.fixture
def ui():
    return AnnotationUI()


def test_object_choices_are_filtered_by_industry(ui):
    assert ui._object_names_for_industry("医疗健康") == ["医疗实体识别", "症状严重度分级"]
    assert ui._object_names_for_industry("金融") == ["金融意图识别", "金融风险文本分类"]


def test_every_industry_yields_at_least_one_object(ui):
    for industry in ui.registry.list_industries():
        assert ui._object_names_for_industry(industry.name), f"行业「{industry.name}」联动后为空"


def test_filtered_choices_never_exceed_registry(ui):
    total = len(ui.registry.list_objects())
    for industry in ui.registry.list_industries():
        assert len(ui._object_names_for_industry(industry.name)) <= total


def test_unknown_industry_falls_back_to_full_list(ui):
    assert len(ui._object_names_for_industry("不存在的行业")) == len(ui.registry.list_objects())


def test_industry_change_returns_choices_and_first_value(ui):
    update, _ = ui._on_industry_change("金融")
    assert update["choices"] == ["金融意图识别", "金融风险文本分类"]
    assert update["value"] == "金融意图识别"


def test_industry_change_syncs_info_panel(ui):
    _, info = ui._on_industry_change("法律")
    assert "合同条款分类" in info


def test_each_industry_selects_a_valid_object(ui):
    """联动后默认选中的第一个对象，必须属于该行业。"""
    for industry in ui.registry.list_industries():
        update, _ = ui._on_industry_change(industry.name)
        first = update["value"]
        assert first in ui._object_names_for_industry(industry.name)
        obj = next(o for o in ui.registry.list_objects() if o.name == first)
        assert obj.industry_id == industry.id


def test_switching_industry_replaces_previous_choices(ui):
    """从医疗切到金融后，医疗的对象不应残留。"""
    ui._on_industry_change("医疗健康")
    update, _ = ui._on_industry_change("金融")
    assert "医疗实体识别" not in update["choices"]
