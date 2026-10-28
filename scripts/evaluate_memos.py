from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd

from nemoroute.budget import add_budget_args, confirm_or_exit, resolve_limit
from nemoroute.cli import base_parser, make_client, print_run_summary, setup
from nemoroute.client import ChatClient
from nemoroute.concurrency import run_tasks
from nemoroute.config import Config, active_thresholds, memo_thresholds
from nemoroute.evaluation.charts import cost_quality_chart, tier_mix_chart
from nemoroute.evaluation.report import summary_table
from nemoroute.evaluation.system import ItemOutcome, StrategySummary, summarize
from nemoroute.human_agreement import render_agreement_section, write_agreement_section
from nemoroute.jsonl import append_jsonl, index_by, read_jsonl
from nemoroute.labeling.judge import judge_answer
from nemoroute.labeling.label import passes
from nemoroute.pricing import estimate
from nemoroute.prompts import load_prompt
from nemoroute.routers.base import Router
from nemoroute.routers.registry import build_router
from nemoroute.usage_log import utc_now

STRATEGIES = ("routed", "all_ultra", "all_nano")
MEMO_CRITERIA = ["accuracy", "completeness", "structure", "actionability", "grounding"]
MEMO_DIR = "data/memo_bench"

# Plan + search run once per question with Nano (the shared evidence), so all three strategies see
# identical sources. Each memo then adds: route (routed only), one extract, one synthesis. Routed
# synthesis is priced at Ultra as a worst case; the judge is one extra call per memo.
EVIDENCE_ROLE = "tier1"
BASIS = {
    "routed": "route 1 + extract 1 (Nano) + synthesis 1 (Ultra worst case)",
    "all_ultra": "extract 1 + synthesis 1 (Ultra)",
    "all_nano": "extract 1 + synthesis 1 (Nano)",
}
ROLE_CALLS = {
    "routed": {"router": 1, "tier1": 1, "tier3": 1},
    "all_ultra": {"tier3": 2},
    "all_nano": {"tier1": 2},
}


def make_search(cfg: Config):
    from nemoroute.search import TavilySearchClient

    return TavilySearchClient(cfg)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    return asdict(obj) if is_dataclass(obj) else dict(vars(obj))


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts = []
    for i, row in enumerate(read_jsonl(path), 1):
        query_id = row.get("id") or row.get("query_id")
        question = row.get("query") or row.get("question")
        if not query_id or not str(question or "").strip():
            raise SystemExit(f"{path}: row {i} needs id/query_id and query/question: {row!r}")
        prompts.append({"id": str(query_id), "query": str(question).strip()})
    if len({p["id"] for p in prompts}) != len(prompts):
        raise SystemExit(f"{path}: duplicate prompt ids")
    return prompts


def pending_memos(prompts: list[dict[str, str]], memos: list[dict]) -> list[tuple[dict, str]]:
    done = set(index_by(memos, "query_id", "strategy"))
    return [(p, s) for p in prompts for s in STRATEGIES if (p["id"], s) not in done]


def pending_judgments(
    prompts: list[dict[str, str]], memos: list[dict], judgments: list[dict]
) -> list[tuple[dict, str]]:
    answered = set(index_by(memos, "query_id", "strategy"))
    judged = set(index_by(judgments, "query_id", "strategy"))
    return [(p, s) for p in prompts for s in STRATEGIES if (p["id"], s) in answered and (p["id"], s) not in judged]


def pending_evidence(todo_memos: list[tuple[dict, str]], evidence_rows: list[dict]) -> list[dict]:
    have = {row["query_id"] for row in evidence_rows}
    todo: dict[str, dict] = {}
    for prompt, _strategy in todo_memos:
        if prompt["id"] not in have:
            todo.setdefault(prompt["id"], prompt)
    return list(todo.values())


def evidence_record(prompt: dict[str, str], evidence: Any) -> dict[str, Any]:
    return {
        "ts": utc_now(),
        "query_id": prompt["id"],
        "question": prompt["query"],
        "role": EVIDENCE_ROLE,
        "queries": list(_field(evidence, "queries", []) or []),
        "sources": [_as_dict(s) for s in (_field(evidence, "sources", []) or [])],
        "plan_step": _as_dict(_field(evidence, "plan_step")),
        "search_step": _as_dict(_field(evidence, "search_step")),
    }


def evidence_from_record(row: dict[str, Any]) -> Any:
    from nemoroute.agent.memo import Evidence, StepTrace
    from nemoroute.search import Source

    return Evidence(
        queries=list(row["queries"]),
        sources=[Source(**s) for s in row["sources"]],
        plan_step=StepTrace(**row["plan_step"]),
        search_step=StepTrace(**row["search_step"]),
    )


def source_mismatches(memos: list[dict]) -> list[str]:
    """Query ids whose strategies did not all see the same source URLs, in the same order."""
    urls: dict[str, set[tuple[str, ...]]] = {}
    for memo in memos:
        key = tuple(_field(s, "url", "") for s in memo.get("sources") or [])
        urls.setdefault(memo["query_id"], set()).add(key)
    return sorted(qid for qid, seen in urls.items() if len(seen) > 1)


def memo_record(prompt: dict[str, str], strategy: str, result: Any) -> dict[str, Any]:
    return {
        "ts": utc_now(),
        "query_id": prompt["id"],
        "question": prompt["query"],
        "strategy": strategy,
        "memo_md": _field(result, "memo_md", "") or "",
        "sources": [_as_dict(s) for s in (_field(result, "sources", []) or [])],
        "steps": [_as_dict(s) for s in (_field(result, "steps", []) or [])],
        "total_cost_usd": _field(result, "total_cost_usd", 0.0),
        "total_latency_s": _field(result, "total_latency_s", 0.0),
        "route_tier": _field(result, "route_tier"),
        "route_reasoning": _field(result, "route_reasoning"),
    }


def answer_block(memo_md: str, sources: list[Any]) -> str:
    blocks = [
        f"[{i}] {_field(s, 'title', '')}\n{_field(s, 'url', '')}\n{_field(s, 'snippet', '')}"
        for i, s in enumerate(sources, 1)
    ]
    return f"{memo_md.strip()}\n\nSource snippets:\n\n" + ("\n\n".join(blocks) if blocks else "(none)")


def judgment_record(
    client: ChatClient, template: Template, prompt: dict[str, str], strategy: str, memo: dict[str, Any]
) -> dict[str, Any]:
    memo_md = str(memo.get("memo_md") or "")
    answer = answer_block(memo_md, memo.get("sources") or []) if memo_md.strip() else ""
    record = judge_answer(
        client, template, MEMO_CRITERIA, prompt["id"], prompt["query"], strategy, answer, role="judge"
    )
    return {**record, "strategy": strategy, "ts": utc_now()}


def run_evidence_stage(
    cfg: Config, client: ChatClient, search: Any, prompts: list[dict], path: Path, concurrency: int
):
    def worker(prompt: dict) -> dict[str, Any]:
        from nemoroute.agent.memo import gather_evidence

        return evidence_record(prompt, gather_evidence(prompt["query"], cfg, client, search, role=EVIDENCE_ROLE))

    return run_tasks(
        prompts,
        worker,
        lambda _prompt, record: append_jsonl(path, record),
        concurrency,
        describe=lambda prompt: f"evidence {prompt['id']}",
    )


def run_memo_stage(
    cfg: Config,
    client: ChatClient,
    search: Any,
    router: Router,
    pairs: list[tuple[dict, str]],
    evidence_idx: dict[str, dict],
    path: Path,
    concurrency: int,
):
    def worker(pair: tuple[dict, str]) -> dict[str, Any]:
        prompt, strategy = pair
        from nemoroute.agent.memo import run_memo

        evidence = evidence_from_record(evidence_idx[prompt["id"]])
        result = run_memo(prompt["query"], strategy, cfg, client, search, router, evidence=evidence)
        return memo_record(prompt, strategy, result)

    return run_tasks(
        pairs,
        worker,
        lambda _pair, record: append_jsonl(path, record),
        concurrency,
        describe=lambda pair: f"memo {pair[0]['id']} {pair[1]}",
    )


def run_judge_stage(
    client: ChatClient,
    template: Template,
    pairs: list[tuple[dict, str]],
    memo_idx: dict[tuple[str, str], dict],
    path: Path,
    concurrency: int,
):
    def worker(pair: tuple[dict, str]) -> dict[str, Any]:
        prompt, strategy = pair
        return judgment_record(client, template, prompt, strategy, memo_idx[(prompt["id"], strategy)])

    return run_tasks(
        pairs,
        worker,
        lambda _pair, record: append_jsonl(path, record),
        concurrency,
        describe=lambda pair: f"judge {pair[0]['id']} {pair[1]}",
    )


def build_summaries(
    cfg: Config,
    prompts: list[dict[str, str]],
    memos: list[dict],
    judgments: list[dict],
    pass_threshold: float,
    min_accuracy: int,
) -> list[StrategySummary]:
    memo_idx = index_by(memos, "query_id", "strategy")
    judgment_idx = index_by(judgments, "query_id", "strategy")
    max_tier, min_tier = max(cfg.tiers()), min(cfg.tiers())
    summaries = []
    for strategy in STRATEGIES:
        outcomes = []
        for prompt in prompts:
            key = (prompt["id"], strategy)
            memo, judgment = memo_idx.get(key), judgment_idx.get(key)
            if memo is None or judgment is None:
                continue
            scores = judgment["scores"]
            tier = memo.get("route_tier")
            if strategy == "all_ultra":
                tier = max_tier
            elif strategy == "all_nano":
                tier = min_tier
            outcomes.append(
                ItemOutcome(
                    query_id=prompt["id"],
                    tier=tier if tier is not None else "n/a",
                    cost_usd=memo["total_cost_usd"],
                    latency_s=memo["total_latency_s"],
                    mean_score=sum(scores.values()) / len(scores),
                    passed=passes(scores, pass_threshold, min_accuracy),
                )
            )
        summaries.append(
            summarize(strategy, outcomes, cfg.evaluation.bootstrap_samples, cfg.evaluation.ci, cfg.evaluation.seed)
        )
    return summaries


def paired_tier_table(memos: list[dict], judgments: list[dict]) -> str:
    """routed vs all_ultra on the same questions, split by the routed synthesis tier. On tier-3 rows both
    synthesize with Ultra, so a gap there comes from the earlier steps (routed extracts with Nano)."""
    memo_idx = index_by(memos, "query_id", "strategy")
    judgment_idx = index_by(judgments, "query_id", "strategy")

    def mean_of(values: list[float]) -> float:
        return sum(values) / len(values)

    lines = [
        "| Routed tier | n | Mean: routed | Mean: all_ultra | routed better / tie / worse "
        "| Accuracy: routed | Accuracy: all_ultra | Grounding: routed | Grounding: all_ultra |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    tiers = sorted({m.get("route_tier") for (q, s), m in memo_idx.items() if s == "routed"} - {None})
    for tier in tiers:
        qids = [
            q for (q, s), m in memo_idx.items()
            if s == "routed" and m.get("route_tier") == tier
            and (q, "routed") in judgment_idx and (q, "all_ultra") in judgment_idx
        ]
        if not qids:
            continue
        r = [judgment_idx[(q, "routed")]["scores"] for q in qids]
        u = [judgment_idx[(q, "all_ultra")]["scores"] for q in qids]
        rm = [mean_of([x[c] for c in MEMO_CRITERIA]) for x in r]
        um = [mean_of([x[c] for c in MEMO_CRITERIA]) for x in u]
        better = sum(a > b for a, b in zip(rm, um))
        tie = sum(a == b for a, b in zip(rm, um))
        lines.append(
            f"| T{tier} | {len(qids)} | {mean_of(rm):.2f} | {mean_of(um):.2f} | {better} / {tie} / {len(qids) - better - tie} "
            f"| {mean_of([x['accuracy'] for x in r]):.2f} | {mean_of([x['accuracy'] for x in u]):.2f} "
            f"| {mean_of([x['grounding'] for x in r]):.2f} | {mean_of([x['grounding'] for x in u]):.2f} |"
        )
    return "\n".join(lines)


def locked_pass_table(locked_summaries: list[StrategySummary]) -> str:
    lines = ["| Strategy | n | Pass rate | 95% CI |", "|---|---|---|---|"]
    for s in locked_summaries:
        lines.append(f"| {s.strategy} | {s.n} | {s.pass_rate:.1%} | [{s.pass_ci[0]:.1%}, {s.pass_ci[1]:.1%}] |")
    return "\n".join(lines)


def render_report(
    cfg: Config,
    prompts: list[dict[str, str]],
    summaries: list[StrategySummary],
    locked_summaries: list[StrategySummary],
    memo_rule: tuple[float, int],
    locked_rule: tuple[float, int],
    title: str,
    missing: list[str],
    incomplete: list[str] | None = None,
    paired: str | None = None,
) -> str:
    locked = cfg.labeling.locked
    memo_line = (
        f"Memo pass rule (headline): mean of the five criteria >= {memo_rule[0]} and accuracy >= {memo_rule[1]}."
    )
    if locked:
        locked_line = (
            f"Locked labeling thresholds: pass_threshold={locked.pass_threshold}, "
            f"min_accuracy={locked.min_accuracy} (locked {locked.locked_at}, source: {locked.source}); "
            "unchanged, reported as a second pass rate below."
        )
    else:
        locked_line = (
            f"**Unlocked labeling thresholds (smoke test only):** pass_threshold={locked_rule[0]}, "
            f"min_accuracy={locked_rule[1]}."
        )
    n_complete = sum(s.n for s in summaries)
    lines = [
        f"# {title}",
        "",
        memo_line,
        "",
        locked_line,
        "",
        f"Frozen prompts: {len(prompts)} in data/memo_bench/prompts.jsonl. "
        f"Complete memo/judgment pairs: {n_complete} of {len(prompts) * len(STRATEGIES)}.",
        *([f"No complete memo/judgment pairs for: {', '.join(missing)}."] if missing else []),
        *(
            [f"Excluded (no memo or no judgment after the resume pass, see logs): {', '.join(incomplete)}."]
            if incomplete
            else []
        ),
        "",
        "Method:",
        "- Prompts are frozen before any memo runs; every prompt is run under all three strategies.",
        "- Evidence is shared: for each question, Nano plans the search queries once and the cached search "
        "results are frozen to data/memo_bench/evidence.jsonl. All three strategies get identical sources "
        "(checked before judging).",
        "- `routed` uses the Nano router to pick the synthesis tier (extract is Nano); `all_ultra` and "
        "`all_nano` use that model for extract and synthesis. The shared plan step is Nano for every "
        "strategy, so `all_ultra` is all-Ultra except for planning.",
        "- The judge is blind: it sees the question, the memo and numbered source snippets, never the strategy or model.",
        "- Confidence intervals are 95% bootstrap CIs over complete memos.",
        "- Memo cost sums every model call in the pipeline (route, plan, extract, synthesize); each memo "
        "includes the shared plan call's cost. The judge is evaluation overhead and is not included in the memo cost.",
        "",
        summary_table(summaries, baseline="all_ultra", unit="memo"),
        "",
        f"Pass rate above uses the memo rule (mean >= {memo_rule[0]}, accuracy >= {memo_rule[1]}). This is the "
        "labeling config's pre-calibration default; calibration tightened routing labels to accuracy >= 5, "
        "a bar a multi-claim memo rarely meets, so that stricter rule is reported separately below.",
        "",
        f"Pass rate under the locked labeling rule (mean >= {locked_rule[0]}, accuracy >= {locked_rule[1]}):",
        "",
        locked_pass_table(locked_summaries),
        "",
        *(
            [
                "Paired view: routed vs all_ultra on the same questions, by the tier the router picked. On T3 rows "
                "both strategies synthesize with Ultra, so a gap there points to the extract step (Nano for routed, "
                "Ultra for all_ultra), plus sampling noise.",
                "",
                paired,
                "",
            ]
            if paired
            else []
        ),
        "![Cost vs quality](charts/cost_vs_quality_memos.png)",
        "",
        "![Routed tier mix](charts/tier_mix_memos.png)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = base_parser("Memo benchmark: routed vs all-Ultra vs all-Nano, judged blind with Ultra.")
    add_budget_args(parser)
    parser.add_argument(
        "--report-only", action="store_true", help="re-render the report and charts from existing outputs; no API calls"
    )
    args = parser.parse_args()
    cfg, usage_log = setup(args)
    limit = resolve_limit(args, cfg.budget.dry_run_limit)

    memo_dir = cfg.path(MEMO_DIR)
    prompts_path = memo_dir / "prompts.jsonl"
    if not prompts_path.exists():
        raise SystemExit(f"Missing {prompts_path}. Run scripts/build_memo_bench.py first.")
    prompts = load_prompts(prompts_path)
    if limit is not None:
        prompts = prompts[:limit]

    evidence_path = memo_dir / "evidence.jsonl"
    memos_path = memo_dir / "memos.jsonl"
    judgments_path = memo_dir / "memo_judgments.jsonl"
    evidence_rows = read_jsonl(evidence_path)
    memos = read_jsonl(memos_path)
    judgments = read_jsonl(judgments_path)
    todo_memos = pending_memos(prompts, memos)
    todo_evidence = pending_evidence(todo_memos, evidence_rows)
    judged = set(index_by(judgments, "query_id", "strategy"))
    # Include memos that will be written this run: they are judged straight after.
    planned_judgments = [(p, s) for p in prompts for s in STRATEGIES if (p["id"], s) not in judged]
    print(
        f"Memo benchmark: {len(prompts)} prompts x {len(STRATEGIES)} strategies; pending evidence "
        f"{len(todo_evidence)}, pending memos {len(todo_memos)}, pending judgments {len(planned_judgments)}."
    )
    if args.report_only:
        print("--report-only: skipping all pending work; re-rendering from existing outputs.")
        todo_memos, todo_evidence, planned_judgments = [], [], []

    calls: dict[str, int] = {}
    if todo_evidence:
        print(f"Shared evidence: {len(todo_evidence)} questions x plan 1 (Nano) + cached search")
        calls[EVIDENCE_ROLE] = len(todo_evidence)
    if todo_memos or planned_judgments:
        memo_counts = Counter(s for _p, s in todo_memos)
        judge_counts = Counter(s for _p, s in planned_judgments)
        print("Cost basis per memo (observed token averages when available, else config defaults):")
        for strategy in STRATEGIES:
            n, n_judge = memo_counts[strategy], judge_counts[strategy]
            if not n and not n_judge:
                continue
            print(f"  {strategy:<9} {n} memos: {BASIS[strategy]}; judge {n_judge}")
            for role, per_memo in ROLE_CALLS[strategy].items():
                calls[role] = calls.get(role, 0) + n * per_memo
            calls["judge"] = calls.get("judge", 0) + n_judge

    est = estimate(cfg, calls, usage_log)
    confirm_or_exit(
        cfg,
        est,
        usage_log,
        args.yes,
        f"memo benchmark ({len(todo_evidence)} evidence, {len(todo_memos)} memos, "
        f"{len(planned_judgments)} judgments)",
    )

    if est.total_calls:
        client = make_client(cfg, usage_log, args.max_cost)
        stopped = False
        if todo_memos:
            search = make_search(cfg)
            if todo_evidence:
                report = run_evidence_stage(
                    cfg, client, search, todo_evidence, evidence_path, cfg.evaluation.concurrency
                )
                print(f"Evidence: {report.done} done, {report.failed} failed")
                stopped = report.stopped_by_budget
            evidence_idx = {row["query_id"]: row for row in read_jsonl(evidence_path)}
            ready = [(p, s) for p, s in todo_memos if p["id"] in evidence_idx]
            if len(ready) < len(todo_memos):
                print(f"Skipping {len(todo_memos) - len(ready)} memos with no evidence yet (rerun to resume).")
            if ready and not stopped:
                router = build_router(cfg, client, step="memo_route")
                report = run_memo_stage(
                    cfg, client, search, router, ready, evidence_idx, memos_path, cfg.evaluation.concurrency
                )
                print(f"Memos: {report.done} done, {report.failed} failed")
                stopped = report.stopped_by_budget
        if not stopped:
            memos = read_jsonl(memos_path)
            mismatched = source_mismatches(memos)
            if mismatched:
                raise SystemExit(
                    f"Shared-source check failed: strategies saw different sources for {', '.join(mismatched)}. "
                    "Not judging. Inspect data/memo_bench/memos.jsonl and evidence.jsonl."
                )
            judgments = read_jsonl(judgments_path)
            todo_judgments = pending_judgments(prompts, memos, judgments)
            if todo_judgments:
                template = load_prompt(cfg, "memo_judge")
                memo_idx = index_by(memos, "query_id", "strategy")
                report = run_judge_stage(
                    client, template, todo_judgments, memo_idx, judgments_path, cfg.evaluation.concurrency
                )
                print(f"Judgments: {report.done} done, {report.failed} failed")
        print_run_summary(usage_log, client)

    memos = read_jsonl(memos_path)
    judgments = read_jsonl(judgments_path)
    mismatched = source_mismatches(memos)
    if mismatched:
        raise SystemExit(f"Shared-source check failed for {', '.join(mismatched)}; not writing a report.")
    memo_rule = memo_thresholds(cfg)
    locked_rule = active_thresholds(cfg)
    all_summaries = build_summaries(cfg, prompts, memos, judgments, *memo_rule)
    summaries = [s for s in all_summaries if s.n]
    if not summaries:
        raise SystemExit("No complete memo/judgment pairs yet; nothing to report.")
    locked_summaries = [s for s in build_summaries(cfg, prompts, memos, judgments, *locked_rule) if s.n]
    missing = [s.strategy for s in all_summaries if not s.n]
    complete = set(index_by(memos, "query_id", "strategy")) & set(index_by(judgments, "query_id", "strategy"))
    incomplete = [f"{p['id']}/{st}" for p in prompts for st in STRATEGIES if (p["id"], st) not in complete]

    title = "Memo benchmark: routed vs all-Ultra vs all-Nano"
    if limit is not None:
        title += f" (first {len(prompts)} prompts only)"
    results_dir = cfg.path(cfg.paths.results_dir)
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    cost_quality_chart(summaries, charts_dir / "cost_vs_quality_memos.png", unit="memo")
    routed = next((s for s in summaries if s.strategy == "routed"), None)
    if routed is not None:
        tier_mix_chart(
            routed.tier_mix, cfg.tiers(), charts_dir / "tier_mix_memos.png",
            f"Routed memos: synthesis tier picked by the Nano router (n={routed.n})",
        )
    md = render_report(
        cfg, prompts, summaries, locked_summaries, memo_rule, locked_rule, title, missing, incomplete,
        paired_tier_table(memos, judgments),
    )
    (results_dir / "memo_results.md").write_text(md)
    agreement_json = results_dir / "human_agreement_memo.json"
    if agreement_json.exists():
        report = results_dir / "memo_results.md"
        write_agreement_section(report, render_agreement_section(json.loads(agreement_json.read_text())))
    locked_pass = {s.strategy: s for s in locked_summaries}
    rows = []
    for summary in summaries:
        row = asdict(summary)
        row["tier_mix"] = str(row["tier_mix"])
        row["pass_rule"] = f"mean>={memo_rule[0]},acc>={memo_rule[1]}"
        row["locked_pass_rate"] = locked_pass[summary.strategy].pass_rate
        row["locked_pass_ci"] = locked_pass[summary.strategy].pass_ci
        rows.append(row)
    pd.DataFrame(rows).to_csv(results_dir / "memo_results.csv", index=False)
    print("\n" + md)
    print(f"Wrote {results_dir / 'memo_results.md'}")


if __name__ == "__main__":
    main()
