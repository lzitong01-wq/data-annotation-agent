"""过程监控（四档停止条件）与成本预估。"""

import pytest

from data_annotation_agent.core.models import MonitorPreset
from data_annotation_agent.quality.monitor import CostEstimator, MonitorAgent


def test_balanced_stops_when_kappa_high_and_stable():
    m = MonitorAgent()
    for _ in range(3):
        m.record(kappa=0.9, n_labeled=10, total=10)
    assert m.should_stop(MonitorPreset.BALANCED, kappa=0.9, n_labeled=10, total=10)["should_stop"] is True


def test_balanced_continues_when_kappa_low():
    m = MonitorAgent()
    assert m.should_stop(MonitorPreset.BALANCED, kappa=0.2, n_labeled=5, total=10)["should_stop"] is False


def test_balanced_waits_for_stability_window():
    """κ 已达阈值但历史不足 3 次，不应立即停止。"""
    m = MonitorAgent()
    m.record(kappa=0.9)
    result = m.should_stop(MonitorPreset.BALANCED, kappa=0.9, n_labeled=1, total=10)
    assert result["should_stop"] is False
    assert "尚未稳定" in result["reason"]


def test_conservative_threshold_is_stricter_than_balanced():
    m = MonitorAgent()
    for _ in range(3):
        m.record(kappa=0.7)
    # 0.7 达到平衡模式阈值(0.6)，但未达保守模式阈值(0.8)
    assert m.should_stop(MonitorPreset.CONSERVATIVE, kappa=0.7)["should_stop"] is False
    assert m.should_stop(MonitorPreset.BALANCED, kappa=0.7)["should_stop"] is True


def test_lenient_stops_only_when_all_samples_labeled():
    m = MonitorAgent()
    assert m.should_stop(MonitorPreset.LENIENT, kappa=0.0, n_labeled=3, total=10)["should_stop"] is False
    assert m.should_stop(MonitorPreset.LENIENT, kappa=0.0, n_labeled=10, total=10)["should_stop"] is True


def test_manual_never_auto_stops():
    m = MonitorAgent()
    assert m.should_stop(MonitorPreset.MANUAL, kappa=1.0, n_labeled=10, total=10)["should_stop"] is False


def test_summary_reflects_latest_record():
    m = MonitorAgent()
    m.record(kappa=0.5, n_labeled=1, total=10)
    m.record(kappa=0.7, n_labeled=3, total=10)
    summary = m.get_summary()
    assert summary["n_records"] == 2
    assert summary["latest_kappa"] == 0.7
    assert summary["n_labeled"] == 3


def test_summary_of_empty_history():
    assert MonitorAgent().get_summary() == {"n_records": 0}


def test_cost_estimator_scales_with_samples_and_models():
    est = CostEstimator().estimate(n_samples=100, n_models=2)
    assert est["total_tokens_est"] == 100 * 2 * CostEstimator.AVG_TOKENS_PER_SAMPLE
    assert est["cost_est_usd"] > 0
    assert est["time_est_minutes"] == pytest.approx(6.7)


def test_cost_estimator_mock_model_is_free():
    est = CostEstimator().estimate(n_samples=50, n_models=1, model="mock-model")
    assert est["cost_est_usd"] == 0.0


def test_cost_estimator_explainability_increases_tokens():
    base = CostEstimator().estimate(n_samples=10, n_models=1)
    verbose = CostEstimator().estimate(n_samples=10, n_models=1, enable_explainability=True)
    assert verbose["total_tokens_est"] > base["total_tokens_est"]
