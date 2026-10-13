from __future__ import annotations

from nemoroute.labeling.label import UNRESOLVED, collapse_warning, compute_labels, label_for_query, passes

GOOD = {"accuracy": 5, "completeness": 4, "structure": 4, "actionability": 4}
WEAK = {"accuracy": 3, "completeness": 3, "structure": 4, "actionability": 3}
HIGH_MEAN_LOW_ACC = {"accuracy": 3, "completeness": 5, "structure": 5, "actionability": 5}


def j(scores):
    return {"scores": scores}


def test_passes_requires_mean_and_accuracy():
    assert passes(GOOD, 4.0, 4)
    assert not passes(WEAK, 4.0, 4)
    assert not passes(HIGH_MEAN_LOW_ACC, 4.0, 4)
    assert passes(HIGH_MEAN_LOW_ACC, 4.0, 3)


def test_label_is_cheapest_passing_tier():
    assert label_for_query({1: j(WEAK), 2: j(GOOD), 3: j(GOOD)}, [1, 2, 3], 4.0, 4) == 2
    assert label_for_query({1: j(GOOD), 2: j(WEAK), 3: j(GOOD)}, [1, 2, 3], 4.0, 4) == 1


def test_no_passing_tier_is_unresolved_not_tier_3():
    assert label_for_query({1: j(WEAK), 2: j(WEAK), 3: j(WEAK)}, [1, 2, 3], 4.0, 4) == UNRESOLVED


def test_incomplete_judgments_are_not_labeled():
    assert label_for_query({1: j(GOOD), 2: j(GOOD)}, [1, 2, 3], 4.0, 4) is None


def test_compute_labels_skips_incomplete_and_records_thresholds():
    queries = [{"id": "a", "query": "qa", "intended_tier": 1}, {"id": "b", "query": "qb", "intended_tier": 3}]
    judgments = [{"query_id": "a", "tier": t, "scores": GOOD} for t in (1, 2, 3)]
    judgments += [{"query_id": "b", "tier": 1, "scores": GOOD}]
    labels = compute_labels(queries, judgments, [1, 2, 3], 4.0, 4)
    assert [r["query_id"] for r in labels] == ["a"]
    assert labels[0]["label"] == 1 and labels[0]["pass_threshold"] == 4.0


def test_collapse_warning():
    labels = [{"label": 1}] * 7 + [{"label": 2}] * 3
    assert collapse_warning(labels, 0.6) is not None
    assert collapse_warning(labels, 0.8) is None
