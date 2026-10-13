from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from nemoroute.config import LockedThresholds
from nemoroute.evaluation.run import EvalResult
from nemoroute.evaluation.system import StrategySummary


def _ci(lo: float, hi: float, scale: float = 1.0, fmt: str = ".2f") -> str:
    return f"[{lo * scale:{fmt}}, {hi * scale:{fmt}}]"


def summary_table(summaries: list[StrategySummary], baseline: str = "all-ultra") -> str:
    base = next((s for s in summaries if s.strategy == baseline), None)
    header = (
        "| Strategy | n | Cost / query (USD) | 95% CI | Cost as % of all-Ultra | p50 latency (s) | p95 latency (s) "
        "| Mean score | 95% CI | Pass rate | 95% CI | Tier mix |"
    )
    lines = [header, "|" + "---|" * 12]
    for s in summaries:
        relative = f"{s.mean_cost_usd / base.mean_cost_usd:.0%}" if base and base.mean_cost_usd else "n/a"
        mix = ", ".join(f"T{t}:{n}" for t, n in s.tier_mix.items())
        lines.append(
            f"| {s.strategy} | {s.n} | {s.mean_cost_usd:.5f} | {_ci(*s.cost_ci, fmt='.5f')} | {relative} "
            f"| {s.p50_latency_s:.2f} | {s.p95_latency_s:.2f} | {s.mean_score:.2f} | {_ci(*s.score_ci)} "
            f"| {s.pass_rate:.1%} | {_ci(*s.pass_ci, scale=100, fmt='.1f')}% | {mix} |"
        )
    return "\n".join(lines)


def threshold_notice(history: list[dict], current: tuple[float, int]) -> str | None:
    if not history:
        return None
    first = history[0]
    if (first["pass_threshold"], first["min_accuracy"]) != current:
        return (
            f"**Thresholds changed after test evaluation.** First test evaluation used "
            f"pass_threshold={first['pass_threshold']}, min_accuracy={first['min_accuracy']} "
            f"({first['ts']}); this report uses pass_threshold={current[0]}, min_accuracy={current[1]}."
        )
    return None


def render_report(
    result: EvalResult,
    tiers: list[int],
    thresholds: LockedThresholds | None,
    pass_threshold: float,
    min_accuracy: int,
    title: str,
    notice: str | None,
    charts: dict[str, str],
) -> str:
    r = result.routing
    lines = [f"# {title}", ""]
    if notice:
        lines += [notice, ""]
    if thresholds:
        lines.append(
            f"Locked thresholds: pass_threshold={thresholds.pass_threshold}, "
            f"min_accuracy={thresholds.min_accuracy} (locked {thresholds.locked_at}, source: {thresholds.source})."
        )
    else:
        lines.append(
            f"**Unlocked thresholds (smoke test only):** pass_threshold={pass_threshold}, min_accuracy={min_accuracy}."
        )
    lines += [
        "",
        f"Items: {result.n_items}. Unresolved (no tier passed): {result.n_unresolved} "
        f"({result.n_unresolved / result.n_items:.1%})." if result.n_items else "Items: 0.",
        f"Router fallbacks (unparseable output, sent to the top tier): {result.fallback_count}.",
        "",
        f"## Routing ({result.router_name}, resolved items only, n={r.n})",
        "",
        f"- Accuracy: {r.accuracy:.1%} (95% CI {_ci(*r.accuracy_ci, scale=100, fmt='.1f')}%)",
        f"- Under-routing rate (sent to a tier below the label): {r.under_routing_rate:.1%}",
        f"- Over-routing rate (sent to a tier above the label): {r.over_routing_rate:.1%}",
        "",
        "| Tier | Precision | Recall |",
        "|---|---|---|",
    ]
    lines += [f"| {t} | {r.precision[t]:.2f} | {r.recall[t]:.2f} |" for t in tiers]
    lines += ["", "Confusion matrix (rows = label, columns = predicted):", ""]
    lines.append("| | " + " | ".join(f"Pred {t}" for t in tiers) + " |")
    lines.append("|---|" + "---|" * len(tiers))
    for i, t in enumerate(tiers):
        lines.append(f"| Label {t} | " + " | ".join(str(v) for v in r.confusion[i]) + " |")
    lines += [
        "",
        "## System comparison: resolved items (headline)",
        "",
        "Cost and latency for the router strategy include the router call. All strategies reuse the same "
        "cached answers and judge scores, so they are compared on identical outputs.",
        "",
        summary_table(result.summaries_resolved),
        "",
        "## System comparison: all items (oracle sends unresolved items to Ultra)",
        "",
        summary_table(result.summaries_all),
        "",
    ]
    if charts:
        lines += ["## Charts", ""]
        lines += [f"![{name}]({rel})" for name, rel in charts.items()]
        lines.append("")
    return "\n".join(lines)


def write_csv(result: EvalResult, path: Path) -> None:
    rows = []
    for scope, summaries in (("resolved", result.summaries_resolved), ("all", result.summaries_all)):
        for s in summaries:
            row = asdict(s)
            row["scope"] = scope
            row["tier_mix"] = str(row["tier_mix"])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
