from __future__ import annotations

from nemoroute.budget import add_budget_args, confirm_or_exit, resolve_limit
from nemoroute.cli import base_parser, make_client, print_run_summary, setup
from nemoroute.config import active_thresholds, require_locked
from nemoroute.datasets import dataset_paths
from nemoroute.evaluation.charts import confusion_chart, cost_quality_chart, latency_chart
from nemoroute.evaluation.report import render_report, threshold_notice, write_csv
from nemoroute.evaluation.run import (
    evaluate,
    other_router_predictions,
    pending_items,
    predictions_path,
    run_predictions,
)
from nemoroute.jsonl import append_jsonl, index_by, read_jsonl
from nemoroute.labeling.label import compute_labels
from nemoroute.pricing import estimate
from nemoroute.reference import JUDGE_ROLES, reference_paths, tier_judgments_path
from nemoroute.routers.registry import ROUTERS, build_router
from nemoroute.usage_log import utc_now


def main() -> None:
    parser = base_parser("Evaluate the router against all-Nano, all-Ultra and oracle baselines.")
    parser.add_argument(
        "--smoke-dataset",
        choices=["calibration"],
        default=None,
        help="pipeline smoke test on calibration labels (unlocked thresholds allowed; never the test set)",
    )
    parser.add_argument("--router", choices=sorted(ROUTERS), default=None, help="override router.type")
    parser.add_argument(
        "--judge",
        choices=JUDGE_ROLES,
        default="judge",
        help="whose scores to report: the labeling judge (Ultra) or the reference model as a second judge "
        "(run scripts/run_reference.py --judge reference_judge first). Labels always come from the labeling judge.",
    )
    add_budget_args(parser)
    args = parser.parse_args()
    cfg, usage_log = setup(args)
    if args.router:
        cfg.router.type = args.router
    limit = resolve_limit(args, cfg.budget.dry_run_limit)
    results_dir = cfg.path(cfg.paths.results_dir)

    if args.smoke_dataset:
        source = dataset_paths(cfg, args.smoke_dataset)
        threshold, min_acc = active_thresholds(cfg)
        items = compute_labels(read_jsonl(source.queries), read_jsonl(source.judgments), cfg.tiers(), threshold, min_acc)
        out_dir = results_dir / "smoke"
        locked = cfg.labeling.locked
        title = f"Smoke test on {args.smoke_dataset} set (NOT the held-out test set)"
    else:
        locked = require_locked(cfg)
        threshold, min_acc = locked.pass_threshold, locked.min_accuracy
        source = dataset_paths(cfg, "full")
        items = read_jsonl(cfg.path(cfg.paths.splits_dir) / "test.jsonl")
        if not items:
            raise SystemExit("No test split found. Run scripts/split_dataset.py first.")
        out_dir = results_dir if args.judge == "judge" else results_dir / f"judge_{args.judge}"
        title = "Held-out test set results" + ("" if args.judge == "judge" else f" (scored by {args.judge})")
    if limit is not None:
        items = items[:limit]
        title += f" (first {len(items)} items only)"

    client = make_client(cfg, usage_log, args.max_cost)
    router = build_router(cfg, client, step="evaluate_route")
    if router.name != "zero_shot_nano":
        out_dir = out_dir / router.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Routing decisions don't depend on the judge, so every judge view shares one predictions file per router.
    preds_dir = results_dir / "smoke" if args.smoke_dataset else results_dir
    preds_path = predictions_path(preds_dir, router.name)
    todo = pending_items(items, preds_path)
    est = estimate(cfg, {router.role: len(todo)}, usage_log)
    confirm_or_exit(cfg, est, usage_log, args.yes, f"route {len(todo)} items with {router.name}")
    if todo:
        report = run_predictions(router, todo, preds_path, cfg.evaluation.concurrency)
        print(f"Routed {report.done}, failed {report.failed}")
        print_run_summary(usage_log, client)

    ref = reference_paths(cfg)
    tier_judgments = source.judgments if args.smoke_dataset else tier_judgments_path(cfg, args.judge)
    judged = set(index_by(read_jsonl(tier_judgments), "query_id", "tier"))
    missing = [i["query_id"] for i in items for t in cfg.tiers() if (i["query_id"], t) not in judged]
    if missing:
        raise SystemExit(
            f"{len(missing)} tier answers have no {args.judge} score yet, e.g. {missing[:3]}. "
            "Run scripts/run_reference.py --judge reference_judge --full first."
        )
    result = evaluate(
        cfg, router.name, items, read_jsonl(preds_path), read_jsonl(source.responses),
        read_jsonl(tier_judgments), threshold, min_acc,
        reference_responses=None if args.smoke_dataset else read_jsonl(ref.responses),
        reference_judgments=None if args.smoke_dataset else read_jsonl(ref.judgments(args.judge)),
        other_routers=other_router_predictions(preds_dir, router.name),
    )

    notice = None
    if not args.smoke_dataset and args.judge == "judge":
        history_path = results_dir / "threshold_history.jsonl"
        history = read_jsonl(history_path)
        current = (threshold, min_acc)
        if not history or (history[-1]["pass_threshold"], history[-1]["min_accuracy"]) != current:
            append_jsonl(history_path, {"ts": utc_now(), "pass_threshold": threshold, "min_accuracy": min_acc})
            history = read_jsonl(history_path)
        notice = threshold_notice(history, current)

    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    confusion_chart(result.routing.confusion, cfg.tiers(), f"{router.name}: label vs predicted tier",
                    charts_dir / "confusion_matrix.png")
    cost_quality_chart(result.summaries_resolved, charts_dir / "cost_vs_quality.png")
    latency_chart(result.summaries_resolved, charts_dir / "latency.png")
    charts = {
        "Confusion matrix": "charts/confusion_matrix.png",
        "Cost vs quality": "charts/cost_vs_quality.png",
        "Latency": "charts/latency.png",
    }
    report_md = render_report(result, cfg.tiers(), locked, threshold, min_acc, title, notice, charts)
    (out_dir / "results.md").write_text(report_md)
    write_csv(result, out_dir / "results.csv")
    print("\n" + report_md)
    print(f"Wrote {out_dir / 'results.md'}")


if __name__ == "__main__":
    main()
