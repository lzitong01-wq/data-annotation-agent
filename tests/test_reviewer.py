"""复核 Agent：抽检 / 疑难 / 全量选择与应用人工修正。"""

from data_annotation_agent.agents.reviewer import ReviewerAgent
from data_annotation_agent.core.models import ReviewerPreset


def test_difficult_preset_selects_low_confidence(make_result):
    results = [
        make_result("s1", "疾病", confidence=0.9),
        make_result("s2", "症状", confidence=0.3),
    ]
    picked = ReviewerAgent().select_for_review(results, ReviewerPreset.DIFFICULT, disagreements=[])
    assert [p["sample_id"] for p in picked] == ["s2"]
    assert picked[0]["reason"] == "低置信度"


def test_difficult_preset_includes_disagreement_samples(make_result):
    results = [make_result("s1", "疾病", confidence=0.9)]
    disagreements = [{"sample_id": "s1", "labels": ["疾病", "症状"]}]
    picked = ReviewerAgent().select_for_review(results, ReviewerPreset.DIFFICULT, disagreements)
    assert any(p["reason"] == "多标注者分歧" for p in picked)


def test_difficult_preset_deduplicates_samples(make_result):
    results = [make_result("s1", "疾病", confidence=0.2)]
    disagreements = [{"sample_id": "s1", "labels": ["疾病", "症状"]}]
    picked = ReviewerAgent().select_for_review(results, ReviewerPreset.DIFFICULT, disagreements)
    assert len(picked) == 1


def test_full_preset_returns_every_sample(make_result):
    results = [make_result(f"s{i}", "疾病") for i in range(4)]
    assert len(ReviewerAgent().select_for_review(results, ReviewerPreset.FULL)) == 4


def test_sample_preset_returns_at_least_one(make_result):
    results = [make_result(f"s{i}", "疾病") for i in range(3)]
    assert len(ReviewerAgent().select_for_review(results, ReviewerPreset.SAMPLE)) >= 1


def test_apply_review_overrides_label(make_result):
    results = [make_result("s1", "疾病"), make_result("s2", "症状")]
    updated = ReviewerAgent().apply_review(results, {"s1": "药物"})
    assert updated[0].label == "药物"
    assert updated[0].confidence == 1.0
    assert updated[0].annotator_id == "human"


def test_apply_review_keeps_untouched_results(make_result):
    results = [make_result("s1", "疾病"), make_result("s2", "症状")]
    updated = ReviewerAgent().apply_review(results, {"s1": "药物"})
    assert updated[1].label == "症状"
    assert updated[1].annotator_id == "m1"


def test_apply_review_with_no_decisions_is_identity(make_result):
    results = [make_result("s1", "疾病")]
    updated = ReviewerAgent().apply_review(results, {})
    assert len(updated) == 1 and updated[0].label == "疾病"
