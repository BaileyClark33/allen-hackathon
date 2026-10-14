from __future__ import annotations

import json
from collections import Counter

from nemoroute.cli import base_parser, setup
from nemoroute.config import require_locked
from nemoroute.datasets import dataset_paths
from nemoroute.jsonl import read_jsonl, write_jsonl
from nemoroute.labeling.label import compute_labels
from nemoroute.split import assert_disjoint, stratified_split


def main() -> None:
    parser = base_parser("Seeded stratified train/test split of the full dataset (after thresholds are locked).")
    parser.add_argument("--force", action="store_true", help="overwrite an existing test split")
    args = parser.parse_args()
    cfg, _ = setup(args)
    locked = require_locked(cfg)
    full = dataset_paths(cfg, "full")
    splits_dir = cfg.path(cfg.paths.splits_dir)
    test_path = splits_dir / "test.jsonl"
    if test_path.exists() and not args.force:
        raise SystemExit(f"{test_path} already exists. The test set is frozen; pass --force only if you mean it.")

    queries = read_jsonl(full.queries)
    labels = compute_labels(queries, read_jsonl(full.judgments), cfg.tiers(), locked.pass_threshold, locked.min_accuracy)
    if len(labels) != len(queries):
        raise SystemExit(f"Only {len(labels)}/{len(queries)} full-set queries are fully labeled. Finish labeling first.")
    write_jsonl(full.labels, labels)

    calibration_ids = {q["id"] for q in read_jsonl(dataset_paths(cfg, "calibration").queries)}
    candidates = [r for r in labels if r["query_id"] not in calibration_ids]
    train, test = stratified_split(candidates, cfg.split.test_frac, cfg.split.seed)
    assert_disjoint({r["query_id"] for r in train} | {r["query_id"] for r in test}, calibration_ids, "calibration")
    assert_disjoint({r["query_id"] for r in train}, {r["query_id"] for r in test}, "train/test overlapping")

    write_jsonl(splits_dir / "train.jsonl", train)
    write_jsonl(test_path, test)
    meta = {
        "seed": cfg.split.seed,
        "test_frac": cfg.split.test_frac,
        "pass_threshold": locked.pass_threshold,
        "min_accuracy": locked.min_accuracy,
        "train": dict(Counter(str(r["label"]) for r in train)),
        "test": dict(Counter(str(r["label"]) for r in test)),
    }
    (splits_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
