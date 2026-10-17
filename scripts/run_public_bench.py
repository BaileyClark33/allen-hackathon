from __future__ import annotations

import json

import pandas as pd

from nemoroute.benchmarks.mmlu_pro import answer_mcq, download, mcq_outcomes, oracle_choice, sample_items
from nemoroute.budget import add_budget_args, confirm_or_exit, resolve_limit
from nemoroute.cli import base_parser, make_client, print_run_summary, setup
from nemoroute.concurrency import run_tasks
from nemoroute.evaluation.charts import cost_quality_chart
from nemoroute.evaluation.report import summary_table
from nemoroute.evaluation.run import pending_items, predictions_path, run_predictions
from nemoroute.evaluation.system import summarize
from nemoroute.jsonl import append_jsonl, index_by, read_jsonl, write_jsonl
from nemoroute.pricing import estimate
from nemoroute.prompts import load_prompt
from nemoroute.reference import REFERENCE
from nemoroute.routers.registry import ROUTERS, build_router


def main() -> None:
    parser = base_parser(
        "Public benchmark: a seeded MMLU-Pro business/economics sample, scored by exact match (no judge). "
        "Every tier answers every question; strategies are compared on those cached answers. Never used for training."
    )
    parser.add_argument("--router", choices=sorted(ROUTERS), default=None, help="override router.type")
    parser.add_argument("--reference", action="store_true", help="also answer with the closed reference model")
    parser.add_argument("--categories", default="business,economics")
    parser.add_argument("--n", type=int, default=300, help="sample size (frozen on first run)")
    parser.add_argument("--seed", type=int, default=5)
    add_budget_args(parser)
    args = parser.parse_args()
    cfg, usage_log = setup(args)
    if args.router:
        cfg.router.type = args.router
    limit = resolve_limit(args, cfg.budget.dry_run_limit)

    root = cfg.path("data/public/mmlu_pro")
    items_path = root / "items.jsonl"
    if not items_path.exists():
        rows = pd.read_parquet(download(cfg.path("data/public/mmlu_pro_test.parquet"))).to_dict("records")
        write_jsonl(items_path, sample_items(rows, args.categories.split(","), args.n, args.seed))
        (root / "sample_meta.json").write_text(json.dumps(
            {"categories": args.categories, "n": args.n, "seed": args.seed}, indent=2))
    items = read_jsonl(items_path)
    if limit is not None:
        items = items[:limit]

    tiers = cfg.tiers()
    answer_tiers: list[int | str] = [*tiers, *([REFERENCE] if args.reference else [])]
    responses_path = root / "responses.jsonl"
    done = set(index_by(read_jsonl(responses_path), "query_id", "tier"))
    tasks = [(i, t) for i in items for t in answer_tiers if (i["query_id"], t) not in done]

    client = make_client(cfg, usage_log, args.max_cost)
    router = build_router(cfg, client, step="public_bench_route")
    out_dir = cfg.path(cfg.paths.results_dir) / "public_bench"
    preds_path = predictions_path(out_dir, router.name)
    route_todo = pending_items(items, preds_path)

    def role_for(tier: int | str) -> str:
        return REFERENCE if tier == REFERENCE else cfg.tier_roles[tier]

    calls: dict[str, int] = {}
    for _item, tier in tasks:
        calls[role_for(tier)] = calls.get(role_for(tier), 0) + 1
    calls[router.role] = len(route_todo)
    est = estimate(cfg, calls, usage_log)
    confirm_or_exit(cfg, est, usage_log, args.yes, f"MMLU-Pro: {len(tasks)} answers, route {len(route_todo)}")

    system_prompt = load_prompt(cfg, "mcq").template
    if tasks:
        report = run_tasks(
            tasks,
            lambda task: answer_mcq(client, system_prompt, task[0], task[1], role_for(task[1])),
            lambda _task, record: append_jsonl(responses_path, record),
            cfg.evaluation.concurrency,
            describe=lambda task: f"mcq {task[0]['query_id']} tier {task[1]}",
        )
        print(f"Answered {report.done}, failed {report.failed}")
    if route_todo:
        report = run_predictions(router, route_todo, preds_path, cfg.evaluation.concurrency)
        print(f"Routed {report.done}, failed {report.failed}")
    if tasks or route_todo:
        print_run_summary(usage_log, client)

    resp = index_by(read_jsonl(responses_path), "query_id", "tier")
    preds = index_by(read_jsonl(preds_path), "query_id")
    ids = [i["query_id"] for i in items if all((i["query_id"], t) in resp for t in tiers) and i["query_id"] in preds]
    if not ids:
        raise SystemExit("No fully answered and routed items yet.")
    ev = cfg.evaluation
    strategies = {f"all-{cfg.roles[cfg.tier_roles[t]].model}": ({q: t for q in ids}, None) for t in tiers}
    strategies[router.name] = (
        {q: preds[q]["tier"] for q in ids}, {q: (preds[q]["cost_usd"], preds[q]["latency_s"]) for q in ids}
    )
    strategies["oracle"] = ({q: oracle_choice(resp, q, tiers) for q in ids}, None)
    if all((q, REFERENCE) in resp for q in ids):
        strategies[REFERENCE] = ({q: REFERENCE for q in ids}, None)
    summaries = [
        summarize(name, mcq_outcomes(chosen, resp, extra), ev.bootstrap_samples, ev.ci, ev.seed)
        for name, (chosen, extra) in strategies.items()
    ]

    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    cost_quality_chart(summaries, charts_dir / f"cost_vs_accuracy_{router.name}.png", y_label="Accuracy (exact match)")
    unparsed = sum(1 for q in ids for t in tiers if resp[(q, t)]["choice"] is None)
    md = "\n".join([
        f"# Public benchmark: MMLU-Pro ({args.categories}), router {router.name}",
        "",
        f"Items: {len(ids)} (seeded sample, frozen in data/public/mmlu_pro/items.jsonl; never used for training). "
        f"Scored by exact match on the final letter, so no LLM judge is involved. "
        f"Unparseable tier answers (counted wrong): {unparsed}.",
        "",
        "In the table, Mean score and Pass rate both equal accuracy. Router cost includes the routing call.",
        "",
        summary_table(summaries),
        "",
        f"![Cost vs accuracy](charts/cost_vs_accuracy_{router.name}.png)",
        "",
    ])
    (out_dir / f"results_{router.name}.md").write_text(md)
    print("\n" + md)
    print(f"Wrote {out_dir / f'results_{router.name}.md'}")


if __name__ == "__main__":
    main()
