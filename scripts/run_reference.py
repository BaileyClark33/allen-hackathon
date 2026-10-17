from __future__ import annotations

from nemoroute.budget import add_budget_args, confirm_or_exit, resolve_limit
from nemoroute.cli import base_parser, make_client, print_run_summary, setup
from nemoroute.config import require_locked
from nemoroute.datasets import dataset_paths
from nemoroute.jsonl import read_jsonl
from nemoroute.pricing import estimate
from nemoroute.prompts import load_prompt
from nemoroute.reference import (
    JUDGE_ROLES,
    REFERENCE,
    pending_judge_jobs,
    pending_reference_answers,
    reference_paths,
    run_judge_jobs,
    run_reference_answers,
    tier_judgments_path,
)


def main() -> None:
    parser = base_parser(
        "Answer the frozen test set with the closed reference model and grade it blind with the same rubric. "
        "With --judge reference_judge, also re-grade the Nemotron tier answers using the reference model as judge."
    )
    parser.add_argument("--judge", choices=JUDGE_ROLES, default="judge")
    add_budget_args(parser)
    args = parser.parse_args()
    cfg, usage_log = setup(args)
    require_locked(cfg)
    limit = resolve_limit(args, cfg.budget.dry_run_limit)

    items = read_jsonl(cfg.path(cfg.paths.splits_dir) / "test.jsonl")
    if not items:
        raise SystemExit("No test split found. Run scripts/split_dataset.py first.")
    if limit is not None:
        items = items[:limit]
    ref = reference_paths(cfg)
    full = dataset_paths(cfg, "full")
    ref_judged_path = ref.judgments(args.judge)
    tier_judged_path = tier_judgments_path(cfg, args.judge)

    answer_todo = pending_reference_answers(items, read_jsonl(ref.responses))
    unjudged_refs = pending_judge_jobs(items, read_jsonl(ref.responses), read_jsonl(ref_judged_path), [REFERENCE], ref_judged_path)
    tier_jobs = []
    if args.judge != "judge":
        tier_jobs = pending_judge_jobs(
            items, read_jsonl(full.responses), read_jsonl(tier_judged_path), cfg.tiers(), tier_judged_path
        )
    calls = {"reference": len(answer_todo), args.judge: len(answer_todo) + len(unjudged_refs) + len(tier_jobs)}
    est = estimate(cfg, calls, usage_log)
    confirm_or_exit(cfg, est, usage_log, args.yes, f"reference answers + {args.judge} on {len(items)} test items")

    client = make_client(cfg, usage_log, args.max_cost)
    if answer_todo:
        report = run_reference_answers(cfg, client, load_prompt(cfg, "answer").template, answer_todo, ref.responses)
        print(f"Reference answers: {report.done} done, {report.failed} failed")
    jobs = pending_judge_jobs(items, read_jsonl(ref.responses), read_jsonl(ref_judged_path), [REFERENCE], ref_judged_path)
    jobs += tier_jobs
    if jobs:
        report = run_judge_jobs(cfg, client, load_prompt(cfg, "judge"), jobs, args.judge)
        print(f"Judgments ({args.judge}): {report.done} done, {report.failed} failed")
    print_run_summary(usage_log, client)
    print(f"Next: python scripts/evaluate.py --full --judge {args.judge}")


if __name__ == "__main__":
    main()
