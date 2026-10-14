from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from nemoroute.labeling.label import UNRESOLVED


def stratified_split(
    labels: list[dict[str, Any]], test_frac: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strata: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in sorted(labels, key=lambda r: r["query_id"]):
        strata[str(record["label"])].append(record)
    rng = random.Random(seed)
    train, test = [], []
    for key in sorted(strata):
        items = strata[key][:]
        rng.shuffle(items)
        n_test = round(len(items) * test_frac)
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    train = [r for r in train if r["label"] != UNRESOLVED]
    return sorted(train, key=lambda r: r["query_id"]), sorted(test, key=lambda r: r["query_id"])


def assert_disjoint(split_ids: set[str], forbidden_ids: set[str], what: str) -> None:
    overlap = split_ids & forbidden_ids
    if overlap:
        raise AssertionError(f"{len(overlap)} {what} ids leaked into the split, e.g. {sorted(overlap)[:3]}")
