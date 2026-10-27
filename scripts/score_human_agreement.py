from __future__ import annotations

import json

from nemoroute.cli import base_parser, setup
from nemoroute.config import Config, active_thresholds, memo_thresholds
from nemoroute.datasets import dataset_paths, human_files
from nemoroute.human_agreement import (
    agreement,
    read_human_labels,
    render_agreement_section,
    write_agreement_section,
)
from nemoroute.jsonl import read_jsonl


def thresholds_for(cfg: Config, target: str) -> tuple[float, int]:
    """Memos are scored against the memo benchmark's headline rule; answers against the labeling rule."""
    return memo_thresholds(cfg) if target == "memo" else active_thresholds(cfg)


def main() -> None:
    parser = base_parser("Compare human pass/fail labels with the LLM judge.")
    parser.add_argument("--target", required=True, choices=["calibration", "test", "memo"])
    args = parser.parse_args()
    cfg, _ = setup(args)
    csv_path, key_path = human_files(cfg, args.target)
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found. Run scripts/sample_for_human.py --target {args.target} first.")
    if args.target == "memo":
        memo_dir = cfg.path("data/memo_bench")
        judgments = [
            {**j, "tier": j.get("strategy", j.get("tier"))}
            for j in read_jsonl(memo_dir / "memo_judgments.jsonl")
        ]
    else:
        dataset = "calibration" if args.target == "calibration" else "full"
        judgments = read_jsonl(dataset_paths(cfg, dataset).judgments)
    human = read_human_labels(csv_path, key_path)
    threshold, min_acc = thresholds_for(cfg, args.target)
    stats = agreement(human, judgments, threshold, min_acc)
    stats.update({"target": args.target, "pass_threshold": threshold, "min_accuracy": min_acc,
                  "thresholds_locked": cfg.labeling.locked is not None,
                  "pass_rule": "memo_bench" if args.target == "memo" else "labeling"})
    print(json.dumps(stats, indent=2))
    out = cfg.path(cfg.paths.results_dir) / f"human_agreement_{args.target}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    print(f"Saved to {out}")
    if args.target == "memo":
        report = cfg.path(cfg.paths.results_dir) / "memo_results.md"
        if report.exists():
            write_agreement_section(report, render_agreement_section(stats))
            print(f"Updated {report}")
        else:
            print(f"{report} not found; wrote the JSON only. Run scripts/evaluate_memos.py first.")


if __name__ == "__main__":
    main()
