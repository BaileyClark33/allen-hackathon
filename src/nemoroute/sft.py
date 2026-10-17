from __future__ import annotations

import json
import random
from string import Template
from typing import Any

from nemoroute.labeling.label import UNRESOLVED
from nemoroute.split import assert_disjoint


def sft_record(template: Template, query_id: str, query: str, tier: int) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "label": tier,
        "messages": [
            {"role": "user", "content": template.substitute(query=query)},
            {"role": "assistant", "content": json.dumps({"tier": tier})},
        ],
    }


def build_sft(
    train: list[dict[str, Any]],
    template: Template,
    forbidden_ids: dict[str, set[str]],
    val_frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Chat-format SFT records from the train split, with a stratified validation holdout for early stopping."""
    usable = [r for r in train if r["label"] != UNRESOLVED]
    ids = {r["query_id"] for r in usable}
    for what, forbidden in forbidden_ids.items():
        assert_disjoint(ids, forbidden, what)
    by_label: dict[int, list[dict[str, Any]]] = {}
    for r in sorted(usable, key=lambda r: r["query_id"]):
        by_label.setdefault(int(r["label"]), []).append(r)
    rng = random.Random(seed)
    fit, val = [], []
    for label in sorted(by_label):
        rows = by_label[label][:]
        rng.shuffle(rows)
        n_val = round(len(rows) * val_frac)
        val += [sft_record(template, r["query_id"], r["query"], label) for r in rows[:n_val]]
        fit += [sft_record(template, r["query_id"], r["query"], label) for r in rows[n_val:]]
    rng.shuffle(fit)
    return fit, sorted(val, key=lambda r: r["query_id"])
