from __future__ import annotations

from collections import Counter

from nemoroute.budget import add_budget_args, confirm_or_exit, resolve_limit
from nemoroute.cli import base_parser, make_client, print_run_summary, setup
from nemoroute.dataset.generate import cap_by_limit, estimate_generator_calls, needed_per_tier, run_generation
from nemoroute.datasets import dataset_paths
from nemoroute.jsonl import read_jsonl
from nemoroute.pricing import estimate
from nemoroute.prompts import load_prompt


def main() -> None:
    parser = base_parser("Generate consulting/business-strategy queries with Nemotron Ultra.")
    parser.add_argument("--dataset", required=True, choices=["calibration", "full"])
    add_budget_args(parser)
    args = parser.parse_args()
    cfg, usage_log = setup(args)
    limit = resolve_limit(args, cfg.budget.dry_run_limit)
    if args.dataset == "full" and "calibration" in cfg.datasets:
        cal = read_jsonl(dataset_paths(cfg, "calibration").queries)
        if len(cal) < cfg.datasets["calibration"].target:
            print(f"Warning: calibration set has {len(cal)}/{cfg.datasets['calibration'].target} queries.")
        if cfg.labeling.locked is None:
            raise SystemExit("Lock thresholds on the calibration set before generating the full set.")

    paths = dataset_paths(cfg, args.dataset)
    existing = read_jsonl(paths.queries)
    target = cfg.datasets[args.dataset].target
    needed = cap_by_limit(needed_per_tier(existing, target, cfg.tiers()), limit)
    print(f"Dataset '{args.dataset}': {len(existing)}/{target} queries. This run adds up to {sum(needed.values())}: {needed}")
    calls = estimate_generator_calls(needed, cfg.generation.batch_size)
    est = estimate(cfg, {"generator": calls}, usage_log)
    confirm_or_exit(cfg, est, usage_log, args.yes, f"generate {sum(needed.values())} queries ({args.dataset})")
    if est.total_calls == 0:
        return

    client = make_client(cfg, usage_log, args.max_cost)
    added = run_generation(cfg, client, paths, load_prompt(cfg, "generate"), limit)
    final = read_jsonl(paths.queries)
    print(f"\nAdded {sum(added.values())} queries {dict(sorted(added.items()))}. "
          f"Dataset now has {len(final)}: {dict(sorted(Counter(r['intended_tier'] for r in final).items()))}")
    print_run_summary(usage_log, client)


if __name__ == "__main__":
    main()
