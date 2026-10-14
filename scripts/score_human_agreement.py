from __future__ import annotations

import json

from nemoroute.cli import base_parser, setup
from nemoroute.config import active_thresholds
from nemoroute.datasets import dataset_paths, human_files
from nemoroute.human_agreement import agreement, read_human_labels
from nemoroute.jsonl import read_jsonl


def main() -> None:
    parser = base_parser("Compare human pass/fail labels with the LLM judge.")
    parser.add_argument("--target", required=True, choices=["calibration", "test"])
    args = parser.parse_args()
    cfg, _ = setup(args)
    csv_path, key_path = human_files(cfg, args.target)
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found. Run scripts/sample_for_human.py --target {args.target} first.")
    dataset = "calibration" if args.target == "calibration" else "full"
    judgments = read_jsonl(dataset_paths(cfg, dataset).judgments)
    human = read_human_labels(csv_path, key_path)
    threshold, min_acc = active_thresholds(cfg)
    stats = agreement(human, judgments, threshold, min_acc)
    stats.update({"target": args.target, "pass_threshold": threshold, "min_accuracy": min_acc,
                  "thresholds_locked": cfg.labeling.locked is not None})
    print(json.dumps(stats, indent=2))
    out = cfg.path(cfg.paths.results_dir) / f"human_agreement_{args.target}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
