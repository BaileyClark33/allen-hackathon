from __future__ import annotations

import json
from collections import Counter

from nemoroute.cli import base_parser, setup
from nemoroute.config import require_locked
from nemoroute.datasets import dataset_paths
from nemoroute.jsonl import read_jsonl, write_jsonl
from nemoroute.prompts import load_prompt
from nemoroute.sft import build_sft


def main() -> None:
    parser = base_parser("Export the train split as chat-format SFT data for the fine-tuned router. No API calls.")
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    cfg, _ = setup(args)
    require_locked(cfg)
    splits_dir = cfg.path(cfg.paths.splits_dir)
    train = read_jsonl(splits_dir / "train.jsonl")
    test = read_jsonl(splits_dir / "test.jsonl")
    if not train or not test:
        raise SystemExit("No train/test split found. Run scripts/split_dataset.py first.")
    forbidden = {
        "test": {r["query_id"] for r in test},
        "calibration": {q["id"] for q in read_jsonl(dataset_paths(cfg, "calibration").queries)},
        "public benchmark": {r["query_id"] for r in read_jsonl(cfg.path("data/public/mmlu_pro/items.jsonl"))},
    }
    fit, val = build_sft(train, load_prompt(cfg, "router_ft"), forbidden, args.val_frac, args.seed)

    out_dir = cfg.path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", fit)
    write_jsonl(out_dir / "val.jsonl", val)
    meta = {
        "prompt": "router_ft",
        "seed": args.seed,
        "val_frac": args.val_frac,
        "train": dict(Counter(str(r["label"]) for r in fit)),
        "val": dict(Counter(str(r["label"]) for r in val)),
    }
    (out_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"Wrote {out_dir / 'train.jsonl'} and {out_dir / 'val.jsonl'}")


if __name__ == "__main__":
    main()
