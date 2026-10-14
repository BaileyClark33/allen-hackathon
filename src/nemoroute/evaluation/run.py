from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemoroute.concurrency import TaskReport, run_tasks
from nemoroute.config import Config
from nemoroute.evaluation.metrics import RoutingMetrics, routing_metrics
from nemoroute.evaluation.system import StrategySummary, oracle_tier, strategy_outcomes, summarize
from nemoroute.jsonl import append_jsonl, index_by, read_jsonl
from nemoroute.labeling.label import UNRESOLVED
from nemoroute.routers.base import Router


@dataclass
class EvalResult:
    router_name: str
    n_items: int
    n_unresolved: int
    routing: RoutingMetrics
    summaries_resolved: list[StrategySummary]
    summaries_all: list[StrategySummary]
    fallback_count: int


def predictions_path(results_dir: Path, router_name: str) -> Path:
    return results_dir / f"predictions_{router_name}.jsonl"


def pending_items(items: list[dict[str, Any]], preds_path: Path) -> list[dict[str, Any]]:
    done = set(index_by(read_jsonl(preds_path), "query_id"))
    return [item for item in items if item["query_id"] not in done]


def run_predictions(router: Router, items: list[dict[str, Any]], preds_path: Path, concurrency: int) -> TaskReport:
    def worker(item: dict[str, Any]) -> dict[str, Any]:
        decision = router.route(item["query"])
        return {
            "query_id": item["query_id"],
            "router": router.name,
            "tier": decision.tier,
            "reasoning": decision.reasoning,
            "cost_usd": decision.cost_usd,
            "latency_s": round(decision.latency_s, 3),
            "fallback": decision.fallback,
        }

    return run_tasks(
        items,
        worker,
        lambda _item, record: append_jsonl(preds_path, record),
        concurrency,
        describe=lambda item: f"route {item['query_id']}",
    )


def evaluate(
    cfg: Config,
    router_name: str,
    items: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    pass_threshold: float,
    min_accuracy: int,
) -> EvalResult:
    tiers = cfg.tiers()
    max_tier, min_tier = max(tiers), min(tiers)
    preds = index_by(predictions, "query_id")
    missing = [i["query_id"] for i in items if i["query_id"] not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} items have no router prediction yet, e.g. {missing[:3]}")
    resp_idx = index_by(responses, "query_id", "tier")
    judg_idx = index_by(judgments, "query_id", "tier")
    ev = cfg.evaluation
    resolved = [i for i in items if i["label"] != UNRESOLVED]
    routing = routing_metrics(
        [int(i["label"]) for i in resolved],
        [preds[i["query_id"]]["tier"] for i in resolved],
        tiers,
        ev.bootstrap_samples,
        ev.ci,
        ev.seed,
    )

    def summaries(subset: list[dict[str, Any]]) -> list[StrategySummary]:
        ids = [i["query_id"] for i in subset]
        overhead = {qid: (preds[qid]["cost_usd"], preds[qid]["latency_s"]) for qid in ids}
        strategies = {
            "all-nano": ({qid: min_tier for qid in ids}, None),
            "all-ultra": ({qid: max_tier for qid in ids}, None),
            router_name: ({qid: preds[qid]["tier"] for qid in ids}, overhead),
            "oracle": ({i["query_id"]: oracle_tier(i["label"], max_tier) for i in subset}, None),
        }
        out = []
        for name, (chosen, extra) in strategies.items():
            outcomes = strategy_outcomes(chosen, resp_idx, judg_idx, pass_threshold, min_accuracy, extra)
            out.append(summarize(name, outcomes, ev.bootstrap_samples, ev.ci, ev.seed))
        return out

    return EvalResult(
        router_name=router_name,
        n_items=len(items),
        n_unresolved=len(items) - len(resolved),
        routing=routing,
        summaries_resolved=summaries(resolved),
        summaries_all=summaries(items),
        fallback_count=sum(1 for i in items if preds[i["query_id"]].get("fallback")),
    )
