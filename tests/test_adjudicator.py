"""裁决 Agent：多数投票、加权聚合、人工裁决。"""

import pytest

from data_annotation_agent.agents.adjudicator import AdjudicatorAgent
from data_annotation_agent.core.models import AdjudicatorPreset


@pytest.fixture
def agent():
    return AdjudicatorAgent()


@pytest.fixture
def three_raters(make_result):
    return [
        make_result("s1", "疾病", annotator_id="m1"),
        make_result("s1", "疾病", annotator_id="m2"),
        make_result("s1", "症状", annotator_id="m3"),
    ]


def test_majority_vote_picks_most_common_label(agent, three_raters):
    assert agent.majority_vote(three_raters).label == "疾病"


def test_weighted_aggregate_can_override_majority(agent, three_raters):
    """加权聚合下，高权重标注者的少数意见可以胜出。"""
    result = agent.weighted_aggregate(three_raters, weights={"m1": 0.1, "m2": 0.1, "m3": 5.0})
    assert result.label == "症状"


def test_weighted_aggregate_defaults_to_confidence(make_result):
    agent = AdjudicatorAgent()
    results = [
        make_result("s1", "疾病", confidence=0.2, annotator_id="m1"),
        make_result("s1", "症状", confidence=0.9, annotator_id="m2"),
    ]
    assert agent.weighted_aggregate(results).label == "症状"


def test_human_adjudication_wins(agent, three_raters):
    result = agent.human_adjudicate(three_raters, human_label="药物")
    assert result.label == "药物"
    assert result.annotator_id == "human"
    assert result.confidence == 1.0


def test_dispatch_majority(agent, three_raters):
    assert agent.adjudicate(three_raters, AdjudicatorPreset.MAJORITY).label == "疾病"


def test_dispatch_human_requires_label(agent, three_raters):
    with pytest.raises(ValueError):
        agent.adjudicate(three_raters, AdjudicatorPreset.HUMAN)


def test_dispatch_human_with_label(agent, three_raters):
    assert agent.adjudicate(three_raters, AdjudicatorPreset.HUMAN, human_label="药物").label == "药物"


def test_dispatch_weighted(agent, three_raters):
    result = agent.adjudicate(
        three_raters, AdjudicatorPreset.WEIGHTED, weights={"m1": 0.1, "m2": 0.1, "m3": 5.0}
    )
    assert result.label == "症状"


def test_empty_results_raise(agent):
    with pytest.raises(ValueError):
        agent.majority_vote([])
    with pytest.raises(ValueError):
        agent.weighted_aggregate([])
