from __future__ import annotations

from collections import Counter
from typing import Any

UNRESOLVED = "unresolved"
Label = int | str


def passes(scores: dict[str, int], pass_threshold: float, min_accuracy: int) -> bool:
    mean = sum(scores.values()) / len(scores)
    return mean >= pass_threshold and scores.get("accuracy", 0) >= min_accuracy


def label_for_query(
    judgments_by_tier: dict[int, dict[str, Any]],
    tiers: list[int],
    pass_threshold: float,
    min_accuracy: int,
) -> Label | None:
    if any(t not in judgments_by_tier for t in tiers):
        return None
    for tier in sorted(tiers):
        if passes(judgments_by_tier[tier]["scores"], pass_threshold, min_accuracy):
            return tier
    return UNRESOLVED


def compute_labels(
    queries: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    tiers: list[int],
    pass_threshold: float,
    min_accuracy: int,
) -> list[dict[str, Any]]:
    by_query: dict[str, dict[int, dict[str, Any]]] = {}
    for j in judgments:
        by_query.setdefault(j["query_id"], {})[j["tier"]] = j
    labels = []
    for q in queries:
        label = label_for_query(by_query.get(q["id"], {}), tiers, pass_threshold, min_accuracy)
        if label is None:
            continue
        labels.append(
            {
                "query_id": q["id"],
                "query": q["query"],
                "intended_tier": q.get("intended_tier"),
                "label": label,
                "pass_threshold": pass_threshold,
                "min_accuracy": min_accuracy,
            }
        )
    return labels


def label_distribution(labels: list[dict[str, Any]]) -> Counter:
    return Counter(r["label"] for r in labels)


def collapse_warning(labels: list[dict[str, Any]], max_frac: float) -> str | None:
    if not labels:
        return None
    dist = label_distribution(labels)
    label, count = dist.most_common(1)[0]
    frac = count / len(labels)
    if frac > max_frac:
        return f"Label '{label}' covers {frac:.0%} of {len(labels)} labeled queries (limit {max_frac:.0%})."
    return None


def render_distribution(labels: list[dict[str, Any]], tiers: list[int]) -> str:
    keys: list[Label] = [*tiers, UNRESOLVED]
    total = len(labels)
    lines = [f"Label distribution ({total} labeled queries):"]
    dist = label_distribution(labels)
    for key in keys:
        n = dist.get(key, 0)
        lines.append(f"  {str(key):<11} {n:>4}  {n / total if total else 0:>6.1%}")
    lines.append("\nLabel vs intended tier (rows = intended, cols = label):")
    lines.append(f"  {'':<10}" + "".join(f"{str(k):>12}" for k in keys))
    for intended in tiers:
        row = Counter(r["label"] for r in labels if r.get("intended_tier") == intended)
        lines.append(f"  intended {intended}" + "".join(f"{row.get(k, 0):>12}" for k in keys))
    return "\n".join(lines)
