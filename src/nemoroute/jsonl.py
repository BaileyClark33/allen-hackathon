from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable

_write_lock = threading.Lock()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _write_lock, open(path, "a") as f:
        f.write(line + "\n")
        f.flush()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def index_by(records: Iterable[dict[str, Any]], *keys: str) -> dict[Any, dict[str, Any]]:
    if len(keys) == 1:
        return {r[keys[0]]: r for r in records}
    return {tuple(r[k] for k in keys): r for r in records}
