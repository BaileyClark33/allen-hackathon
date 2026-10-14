from __future__ import annotations

from collections import Counter

from nemoroute.budget import add_budget_args, confirm_or_exit, resolve_limit
from nemoroute.cli import base_parser, make_client, print_run_summary, setup
from nemoroute.config import Config, active_thresholds
from nemoroute.datasets import DatasetPaths, dataset_paths
from nemoroute.jsonl import write_jsonl
from nemoroute.labeling.label import collapse_warning, compute_labels, render_distribution
from nemoroute.labeling.pipeline import (
    load_stage_files,
    pending_answers,
    pending_judgments,
    planned_calls,
    run_answer_stage,
    run_judge_stage,
    select_queries,
)
from nemoroute.pricing import estimate
from nemoroute.prompts import load_prompt


def write_labels(cfg: Config, paths: DatasetPaths) -> list[dict]:
    queries, _, judgments = load_stage_files(paths)
    threshold, min_acc = active_thresholds(cfg)
    labels = compute_labels(queries, judgments, cfg.tiers(), threshold, min_acc)
    write_jsonl(paths.labels, labels)
    status = "locked" if cfg.labeling.locked else "UNLOCKED defaults"
    print(f"\nThresholds ({status}): pass_threshold={threshold}, min_accuracy={min_acc}")
    print(render_distribution(labels, cfg.tiers()))
    return labels


def main() -> None:
    parser = base_parser("Answer each query with Nano/Super/Ultra, judge with Ultra, and derive tier labels.")
    parser.add_argument("--dataset", required=True, choices=["calibration", "full"])
    parser.add_argument("--relabel", action="store_true", help="recompute labels from saved judgments (no API calls)")
    parser.add_argument("--ack-imbalance", action="store_true", help="allow --full despite a collapsed label distribution")
    add_budget_args(parser)
    args = parser.parse_args()
    cfg, usage_log = setup(args)
    paths = dataset_paths(cfg, args.dataset)

    if args.relabel:
        write_labels(cfg, paths)
        return

    limit = resolve_limit(args, cfg.budget.dry_run_limit)
    if args.dataset == "full" and cfg.labeling.locked is None:
        raise SystemExit("Thresholds are not locked. Tune and lock them on the calibration set first.")
    if limit is None:
        existing_labels = compute_labels(*_labels_inputs(cfg, paths))
        if not existing_labels:
            raise SystemExit("No labeled queries yet. Run a dry run (--limit) and review the distribution before --full.")
        warning = collapse_warning(existing_labels, cfg.labeling.collapse_warn_frac)
        if warning and not args.ack_imbalance:
            raise SystemExit(f"Class-balance check failed: {warning}\nReview before a full run, or pass --ack-imbalance.")

    queries, responses, judgments = load_stage_files(paths)
    selected = select_queries(queries, limit)
    answers = pending_answers(selected, responses, cfg.tiers())
    judges = pending_judgments(selected, responses, judgments, cfg.tiers())
    print(f"Dataset '{args.dataset}': {len(queries)} queries, processing {len(selected)}.")
    print(f"Pending answers: {len(answers)} | pending judgments: {len(judges)} (+{len(answers)} after answering)")
    est = estimate(cfg, planned_calls(cfg, answers, judges), usage_log)
    confirm_or_exit(cfg, est, usage_log, args.yes, f"label {len(selected)} queries ({args.dataset})")

    if est.total_calls:
        client = make_client(cfg, usage_log, args.max_cost)
        report = run_answer_stage(cfg, client, paths, load_prompt(cfg, "answer").template, answers)
        print(f"Answers: {report.done} done, {report.failed} failed")
        if not report.stopped_by_budget:
            queries, responses, judgments = load_stage_files(paths)
            judges = pending_judgments(selected, responses, judgments, cfg.tiers())
            report = run_judge_stage(cfg, client, paths, load_prompt(cfg, "judge"), judges)
            print(f"Judgments: {report.done} done, {report.failed} failed")
        _, responses, _ = load_stage_files(paths)
        truncated = Counter(r["tier"] for r in responses if r.get("finish_reason") == "length")
        if truncated:
            print(f"Answers truncated at max_tokens, by tier: {dict(truncated)}")
        print_run_summary(usage_log, client)

    labels = write_labels(cfg, paths)
    warning = collapse_warning(labels, cfg.labeling.collapse_warn_frac)
    if warning:
        print(f"\nCLASS-BALANCE WARNING: {warning}")


def _labels_inputs(cfg: Config, paths: DatasetPaths) -> tuple:
    queries, _, judgments = load_stage_files(paths)
    threshold, min_acc = active_thresholds(cfg)
    return queries, judgments, cfg.tiers(), threshold, min_acc


if __name__ == "__main__":
    main()
