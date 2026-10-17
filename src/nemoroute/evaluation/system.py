from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nemoroute.evaluation.metrics import bootstrap_ci
from nemoroute.labeling.label import UNRESOLVED, passes


@dataclass
class ItemOutcome:
    query_id: str
    tier: int | str
    cost_usd: float
    latency_s: float
    mean_score: float
    passed: bool


@dataclass
class StrategySummary:
    strategy: str
    n: int
    mean_cost_usd: float
    cost_ci: tuple[float, float]
    total_cost_usd: float
    p50_latency_s: float
    p95_latency_s: float
    mean_score: float
    score_ci: tuple[float, float]
    pass_rate: float
    pass_ci: tuple[float, float]
    tier_mix: dict[int | str, int]


def oracle_tier(label: Any, max_tier: int) -> int:
    return max_tier if label == UNRESOLVED else int(label)


def strategy_outcomes(
    chosen: dict[str, int | str],
    responses: dict[tuple[str, int], dict[str, Any]],
    judgments: dict[tuple[str, int], dict[str, Any]],
    pass_threshold: float,
    min_accuracy: int,
    router_overhead: dict[str, tuple[float, float]] | None = None,
) -> list[ItemOutcome]:
    outcomes = []
    for qid, tier in chosen.items():
        response = responses[(qid, tier)]
        judgment = judgments[(qid, tier)]
        extra_cost, extra_latency = (router_overhead or {}).get(qid, (0.0, 0.0))
        outcomes.append(
            ItemOutcome(
                query_id=qid,
                tier=tier,
                cost_usd=response["cost_usd"] + extra_cost,
                latency_s=response["latency_s"] + extra_latency,
                mean_score=judgment["mean_score"],
                passed=passes(judgment["scores"], pass_threshold, min_accuracy),
            )
        )
    return outcomes


def summarize(
    strategy: str, outcomes: list[ItemOutcome], n_bootstrap: int, ci: float, seed: int
) -> StrategySummary:
    cost = np.array([o.cost_usd for o in outcomes])
    latency = np.array([o.latency_s for o in outcomes])
    score = np.array([o.mean_score for o in outcomes])
    passed = np.array([float(o.passed) for o in outcomes])
    mix: dict[int | str, int] = {}
    for o in outcomes:
        mix[o.tier] = mix.get(o.tier, 0) + 1
    empty = len(outcomes) == 0
    return StrategySummary(
        strategy=strategy,
        n=len(outcomes),
        mean_cost_usd=float("nan") if empty else float(cost.mean()),
        cost_ci=bootstrap_ci(cost, n_samples=n_bootstrap, ci=ci, seed=seed),
        total_cost_usd=float(cost.sum()),
        p50_latency_s=float("nan") if empty else float(np.percentile(latency, 50)),
        p95_latency_s=float("nan") if empty else float(np.percentile(latency, 95)),
        mean_score=float("nan") if empty else float(score.mean()),
        score_ci=bootstrap_ci(score, n_samples=n_bootstrap, ci=ci, seed=seed),
        pass_rate=float("nan") if empty else float(passed.mean()),
        pass_ci=bootstrap_ci(passed, n_samples=n_bootstrap, ci=ci, seed=seed),
        tier_mix=dict(sorted(mix.items(), key=lambda kv: str(kv[0]))),
    )
