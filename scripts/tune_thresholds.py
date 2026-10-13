from __future__ import annotations

from nemoroute.cli import base_parser, setup
from nemoroute.config import write_locked_thresholds
from nemoroute.datasets import dataset_paths
from nemoroute.human_agreement import read_human_labels
from nemoroute.jsonl import read_jsonl
from nemoroute.labeling.tuning import best_point, evaluate_grid, render_grid


def main() -> None:
    parser = base_parser("Tune pass thresholds on the calibration set only (no API calls), then lock them.")
    parser.add_argument("--lock", nargs=2, metavar=("PASS_THRESHOLD", "MIN_ACCURACY"), help="write these values to config")
    parser.add_argument("--force", action="store_true", help="overwrite thresholds that are already locked")
    args = parser.parse_args()
    cfg, _ = setup(args)
    paths = dataset_paths(cfg, "calibration")

    if args.lock:
        if cfg.labeling.locked is not None and not args.force:
            raise SystemExit(f"Thresholds already locked: {cfg.labeling.locked}. Pass --force to change them.")
        if cfg.labeling.locked is not None and read_jsonl(cfg.path(cfg.paths.splits_dir) / "test.jsonl"):
            print("WARNING: a test split already exists. Changing thresholds now must be reported in results.md.")
        locked = write_locked_thresholds(float(args.lock[0]), int(args.lock[1]), "calibration", args.config)
        print(f"Locked in config.yaml: {locked}")
        return

    queries = read_jsonl(paths.queries)
    judgments = read_jsonl(paths.judgments)
    human = []
    if paths.human_csv.exists() and paths.human_key.exists():
        human = read_human_labels(paths.human_csv, paths.human_key)
    print(f"Calibration: {len(queries)} queries, {len(judgments)} judgments, {len(human)} human-labeled answers.")
    if not human:
        print("No human labels yet: kappa is unavailable. Export with "
              "`python scripts/sample_for_human.py --target calibration` and fill in human_pass.")
    points = evaluate_grid(
        queries, judgments, human, cfg.tiers(), cfg.tuning.pass_threshold_grid, cfg.tuning.min_accuracy_grid
    )
    best = best_point(points)
    print(render_grid(points, best))
    if best:
        print(
            f"\nRecommended: pass_threshold={best.pass_threshold}, min_accuracy={best.min_accuracy} "
            f"(kappa {best.kappa:.3f}). To lock after review:\n"
            f"  python scripts/tune_thresholds.py --lock {best.pass_threshold} {best.min_accuracy}"
        )


if __name__ == "__main__":
    main()
