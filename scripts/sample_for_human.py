from __future__ import annotations

from nemoroute.cli import base_parser, setup
from nemoroute.config import require_locked
from nemoroute.datasets import dataset_paths, human_files
from nemoroute.human_agreement import answer_candidates, stratified_sample, write_human_files
from nemoroute.jsonl import read_jsonl
from nemoroute.labeling.label import passes


def main() -> None:
    parser = base_parser("Export a blind sample of (query, answer) pairs for human pass/fail labeling.")
    parser.add_argument("--target", required=True, choices=["calibration", "test"])
    parser.add_argument("--force", action="store_true", help="overwrite an existing CSV")
    args = parser.parse_args()
    cfg, _ = setup(args)
    csv_path, key_path = human_files(cfg, args.target)

    if args.target == "calibration":
        cal = dataset_paths(cfg, "calibration")
        candidates = answer_candidates(read_jsonl(cal.queries), read_jsonl(cal.responses), read_jsonl(cal.judgments))
        total = cfg.tuning.human_per_tier * len(cfg.tiers())
        sample = stratified_sample(candidates, lambda c: c["tier"], total, cfg.tuning.seed)
    else:
        locked = require_locked(cfg)
        full = dataset_paths(cfg, "full")
        test_ids = {r["query_id"] for r in read_jsonl(cfg.path(cfg.paths.splits_dir) / "test.jsonl")}
        if not test_ids:
            raise SystemExit("No test split found. Run scripts/split_dataset.py first.")
        candidates = answer_candidates(
            read_jsonl(full.queries), read_jsonl(full.responses), read_jsonl(full.judgments), test_ids
        )

        def stratum(c: dict) -> tuple[int, bool]:
            return c["tier"], passes(c["judgment"]["scores"], locked.pass_threshold, locked.min_accuracy)

        sample = stratified_sample(candidates, stratum, cfg.human_agreement.sample_size, cfg.human_agreement.seed)

    write_human_files(sample, csv_path, key_path, args.force)
    print(f"Wrote {len(sample)} items to {csv_path}")
    print("Fill the human_pass column with 1 (pass) or 0 (fail). Model identity and judge scores are hidden.")
    print(f"The key mapping items to queries/tiers is in {key_path}; do not open it while labeling.")


if __name__ == "__main__":
    main()
