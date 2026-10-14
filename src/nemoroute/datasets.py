from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nemoroute.config import Config
from nemoroute.jsonl import read_jsonl


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    root: Path

    @property
    def queries(self) -> Path:
        return self.root / "queries.jsonl"

    @property
    def responses(self) -> Path:
        return self.root / "responses.jsonl"

    @property
    def judgments(self) -> Path:
        return self.root / "judgments.jsonl"

    @property
    def labels(self) -> Path:
        return self.root / "labels.jsonl"

    @property
    def human_csv(self) -> Path:
        return self.root / f"{self.name}_human.csv"

    @property
    def human_key(self) -> Path:
        return self.root / f"{self.name}_human_key.jsonl"


def dataset_paths(cfg: Config, name: str) -> DatasetPaths:
    if name not in cfg.datasets:
        raise SystemExit(f"Unknown dataset '{name}'. Options: {sorted(cfg.datasets)}")
    return DatasetPaths(name=name, root=cfg.dataset_dir(name))


def human_files(cfg: Config, target: str) -> tuple[Path, Path]:
    if target == "calibration":
        paths = dataset_paths(cfg, "calibration")
        return paths.human_csv, paths.human_key
    splits_dir = cfg.path(cfg.paths.splits_dir)
    return splits_dir / "test_human.csv", splits_dir / "test_human_key.jsonl"


def all_query_texts(cfg: Config) -> list[str]:
    texts = []
    for name in cfg.datasets:
        texts.extend(r["query"] for r in read_jsonl(dataset_paths(cfg, name).queries))
    return texts
