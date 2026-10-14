from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from nemoroute.human_agreement import agreement
from nemoroute.labeling.label import UNRESOLVED, compute_labels, label_distribution


@dataclass
class GridPoint:
    pass_threshold: float
    min_accuracy: int
    kappa: float
    percent_agreement: float
    max_label_frac: float
    unresolved_rate: float
    distribution: dict[str, int]


def evaluate_grid(
    queries: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    human: list[dict[str, Any]],
    tiers: list[int],
    pass_grid: list[float],
    accuracy_grid: list[int],
) -> list[GridPoint]:
    points = []
    for threshold in pass_grid:
        for min_acc in accuracy_grid:
            labels = compute_labels(queries, judgments, tiers, threshold, min_acc)
            dist = label_distribution(labels)
            total = len(labels) or 1
            stats = agreement(human, judgments, threshold, min_acc) if human else {}
            points.append(
                GridPoint(
                    pass_threshold=threshold,
                    min_accuracy=min_acc,
                    kappa=stats.get("cohen_kappa", float("nan")),
                    percent_agreement=stats.get("percent_agreement", float("nan")),
                    max_label_frac=max(dist.values()) / total if dist else 0.0,
                    unresolved_rate=dist.get(UNRESOLVED, 0) / total,
                    distribution={str(k): v for k, v in sorted(dist.items(), key=lambda kv: str(kv[0]))},
                )
            )
    return points


def best_point(points: list[GridPoint]) -> GridPoint | None:
    scored = [p for p in points if not math.isnan(p.kappa)]
    if not scored:
        return None
    return max(scored, key=lambda p: (round(p.kappa, 4), -p.max_label_frac, -p.unresolved_rate))


def render_grid(points: list[GridPoint], best: GridPoint | None) -> str:
    header = f"{'pass_thr':>8} {'min_acc':>7} {'kappa':>7} {'agree':>7} {'max_frac':>8} {'unres':>6}  distribution"
    lines = [header, "-" * len(header)]
    for p in points:
        mark = "  <== best" if best is p else ""
        lines.append(
            f"{p.pass_threshold:>8.2f} {p.min_accuracy:>7} {p.kappa:>7.3f} {p.percent_agreement:>7.1%} "
            f"{p.max_label_frac:>8.1%} {p.unresolved_rate:>6.1%}  {p.distribution}{mark}"
        )
    return "\n".join(lines)
