from __future__ import annotations

import numpy as np
import pytest

from nemoroute.evaluation.metrics import bootstrap_ci, routing_metrics
from nemoroute.evaluation.system import oracle_tier, strategy_outcomes, summarize
from nemoroute.labeling.label import UNRESOLVED


def test_routing_metrics_hand_computed():
    y_true = [1, 1, 2, 2, 3, 3]
    y_pred = [1, 2, 2, 1, 3, 2]
    m = routing_metrics(y_true, y_pred, [1, 2, 3], n_bootstrap=200, seed=0)
    assert m.accuracy == pytest.approx(3 / 6)
    assert m.under_routing_rate == pytest.approx(2 / 6)
    assert m.over_routing_rate == pytest.approx(1 / 6)
    assert m.confusion.tolist() == [[1, 1, 0], [1, 1, 0], [0, 1, 1]]
    assert m.precision[2] == pytest.approx(1 / 3)
    assert m.recall[3] == pytest.approx(1 / 2)
    assert m.accuracy_ci[0] <= m.accuracy <= m.accuracy_ci[1]


def test_bootstrap_ci_degenerate():
    assert bootstrap_ci(np.ones(10), n_samples=50) == (1.0, 1.0)


def test_strategy_outcomes_add_router_overhead_and_use_cached_answers():
    responses = {("a", 1): {"cost_usd": 0.001, "latency_s": 1.0}, ("a", 3): {"cost_usd": 0.01, "latency_s": 5.0}}
    judgments = {
        ("a", 1): {"mean_score": 3.0, "scores": {"accuracy": 3, "completeness": 3, "structure": 3, "actionability": 3}},
        ("a", 3): {"mean_score": 4.5, "scores": {"accuracy": 5, "completeness": 4, "structure": 5, "actionability": 4}},
    }
    routed = strategy_outcomes({"a": 1}, responses, judgments, 4.0, 4, {"a": (0.0001, 0.2)})
    assert routed[0].cost_usd == pytest.approx(0.0011)
    assert routed[0].latency_s == pytest.approx(1.2)
    assert routed[0].passed is False
    ultra = strategy_outcomes({"a": 3}, responses, judgments, 4.0, 4)
    assert ultra[0].passed is True
    summary = summarize("all-ultra", ultra, 100, 0.95, 0)
    assert summary.mean_cost_usd == pytest.approx(0.01)
    assert summary.tier_mix == {3: 1}


def test_oracle_sends_unresolved_to_top_tier():
    assert oracle_tier(UNRESOLVED, 3) == 3
    assert oracle_tier(2, 3) == 2
