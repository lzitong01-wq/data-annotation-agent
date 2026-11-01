"""检验 Agent：一致性计算与分歧样本定位。"""

from data_annotation_agent.agents.inspector import InspectorAgent
from data_annotation_agent.core.models import InspectorPreset


def _consistent(make_result):
    return [
        make_result("s1", "疾病", annotator_id="m1"),
        make_result("s1", "疾病", annotator_id="m2"),
        make_result("s2", "症状", annotator_id="m1"),
        make_result("s2", "症状", annotator_id="m2"),
    ]


def _conflicted(make_result):
    return [
        make_result("s1", "疾病", annotator_id="m1"),
        make_result("s1", "症状", annotator_id="m2"),
        make_result("s2", "药物", annotator_id="m1"),
        make_result("s2", "药物", annotator_id="m2"),
    ]


def test_light_preset_returns_agreement(make_result):
    out = InspectorAgent().inspect(_consistent(make_result), InspectorPreset.LIGHT)
    assert "agreement" in out
    assert out["agreement"]["method"] == "Cohen's κ"


def test_standard_preset_returns_agreement(make_result):
    out = InspectorAgent().inspect(_consistent(make_result), InspectorPreset.STANDARD)
    assert "agreement" in out
    assert out["n_samples"] == 2


def test_deep_preset_returns_agreement(make_result):
    out = InspectorAgent().inspect(_conflicted(make_result), InspectorPreset.DEEP)
    assert "agreement" in out
    assert out["n_disagreements"] == 1


def test_consistent_results_have_no_disagreements(make_result):
    assert InspectorAgent().find_disagreements(_consistent(make_result)) == []


def test_disagreement_exposes_conflicting_labels(make_result):
    disagreements = InspectorAgent().find_disagreements(_conflicted(make_result))
    assert len(disagreements) == 1
    assert disagreements[0]["sample_id"] == "s1"
    assert set(disagreements[0]["labels"]) == {"疾病", "症状"}


def test_cross_check_reports_sample_count(make_result):
    out = InspectorAgent().cross_check(_conflicted(make_result))
    assert out["n_samples"] == 2
    assert out["n_disagreements"] == 1


def test_light_preset_without_multiple_raters_is_flagged(make_result):
    single = [make_result("s1", "疾病", annotator_id="m1")]
    out = InspectorAgent().inspect(single, InspectorPreset.LIGHT)
    assert out["agreement"]["method"] == "无"
