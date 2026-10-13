from __future__ import annotations

from collections import Counter

import pytest

from nemoroute.labeling.label import UNRESOLVED
from nemoroute.split import assert_disjoint, stratified_split


def make_labels():
    labels = []
    for i in range(100):
        label = [1, 2, 3, UNRESOLVED][i % 4] if i < 80 else 2
        labels.append({"query_id": f"q-{i:04d}", "label": label})
    return labels


def test_split_is_deterministic_for_seed():
    a = stratified_split(make_labels(), 0.2, 42)
    b = stratified_split(list(reversed(make_labels())), 0.2, 42)
    assert a == b
    c = stratified_split(make_labels(), 0.2, 7)
    assert {r["query_id"] for r in a[1]} != {r["query_id"] for r in c[1]}


def test_split_is_stratified_and_excludes_unresolved_from_train():
    train, test = stratified_split(make_labels(), 0.2, 42)
    test_counts = Counter(str(r["label"]) for r in test)
    assert test_counts == {"1": 4, "2": 8, "3": 4, UNRESOLVED: 4}
    assert all(r["label"] != UNRESOLVED for r in train)
    assert not {r["query_id"] for r in train} & {r["query_id"] for r in test}


def test_assert_disjoint_raises_on_leak():
    assert_disjoint({"a", "b"}, {"c"}, "calibration")
    with pytest.raises(AssertionError):
        assert_disjoint({"a", "cal-0001"}, {"cal-0001"}, "calibration")
