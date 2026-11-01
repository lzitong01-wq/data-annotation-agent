"""一致性指标：Cohen's κ / Fleiss' κ / Krippendorff's α / 分级解读。"""

import pytest

from data_annotation_agent.quality.agreement import (
    cohens_kappa,
    compute_agreement,
    fleiss_kappa,
    interpret_kappa,
    krippendorff_alpha,
)


def test_cohens_kappa_perfect_agreement():
    assert cohens_kappa(["A", "B", "A"], ["A", "B", "A"]) == pytest.approx(1.0)


def test_cohens_kappa_complete_disagreement_is_negative():
    assert cohens_kappa(["A", "B"], ["B", "A"]) < 0


def test_cohens_kappa_length_mismatch_raises():
    with pytest.raises(ValueError):
        cohens_kappa(["A"], ["A", "B"])


def test_cohens_kappa_empty_input():
    assert cohens_kappa([], []) == 0.0


def test_fleiss_kappa_perfect_agreement():
    assert fleiss_kappa([["A", "A", "A"], ["B", "B", "B"]]) == pytest.approx(1.0)


def test_fleiss_kappa_single_rater_returns_zero():
    assert fleiss_kappa([["A"], ["B"]]) == 0.0


def test_krippendorff_alpha_perfect_agreement():
    assert krippendorff_alpha([["A", "A", "A"], ["B", "B", "B"]]) == pytest.approx(1.0)


def test_krippendorff_alpha_drops_samples_with_missing_values():
    # 第二行含缺失值会被丢弃，只剩第一行完全一致 → 1.0
    assert krippendorff_alpha([["A", "A"], [None, "B"]]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kappa", "expected"),
    [
        (0.9, "几乎完美"),
        (0.7, "显著"),
        (0.5, "中等"),
        (0.3, "一般"),
        (0.1, "微弱"),
        (-0.1, "无一致性"),
    ],
)
def test_interpret_kappa_levels(kappa, expected):
    assert expected in interpret_kappa(kappa)


def test_compute_agreement_two_raters_uses_cohens_kappa():
    agg = compute_agreement([["A", "A"], ["B", "B"]])
    assert agg["method"] == "Cohen's κ"
    assert agg["n_raters"] == 2
    assert agg["value"] == pytest.approx(1.0)


def test_compute_agreement_three_raters_reports_fleiss_and_alpha():
    agg = compute_agreement([["A", "A", "A"], ["B", "B", "B"]])
    assert agg["n_raters"] == 3
    assert agg["fleiss_kappa"] == pytest.approx(1.0)
    assert "krippendorff_alpha" in agg


def test_compute_agreement_single_rater_is_flagged():
    agg = compute_agreement([["A"], ["B"]])
    assert agg["method"] == "无"
    assert agg["n_raters"] == 1
