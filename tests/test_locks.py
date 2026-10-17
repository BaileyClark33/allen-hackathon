from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fakes import make_config, write_config

from nemoroute.config import ROOT, load_config, require_locked, write_locked_thresholds
from nemoroute.datasets import dataset_paths
from nemoroute.jsonl import read_jsonl, write_jsonl

SCRIPTS = ROOT / "scripts"


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args], capture_output=True, text=True, cwd=ROOT, timeout=120
    )


def test_write_locked_thresholds_only_touches_locked_line(tmp_path):
    config_path = tmp_path / "config.yaml"
    original = (ROOT / "config.yaml").read_text()
    config_path.write_text(original)
    write_locked_thresholds(3.75, 4, "calibration", config_path)
    updated = config_path.read_text()
    changed = [(a, b) for a, b in zip(original.splitlines(), updated.splitlines()) if a != b]
    assert len(original.splitlines()) == len(updated.splitlines())
    assert len(changed) == 1 and changed[0][0].strip() == "locked: null"
    cfg = load_config(config_path)
    assert cfg.labeling.locked.pass_threshold == 3.75
    assert cfg.labeling.locked.min_accuracy == 4
    assert cfg.labeling.locked.source == "calibration"


def test_require_locked_refuses_when_unlocked(tmp_path):
    with pytest.raises(SystemExit):
        require_locked(make_config(tmp_path))


@pytest.mark.parametrize(
    "script,args",
    [
        ("evaluate.py", []),
        ("split_dataset.py", []),
        ("label_queries.py", ["--dataset", "full", "--limit", "1"]),
        ("generate_queries.py", ["--dataset", "full", "--limit", "1"]),
        ("run_reference.py", ["--limit", "1"]),
    ],
)
def test_scripts_refuse_without_locked_thresholds(tmp_path, script, args):
    config_path = write_config(tmp_path, make_config(tmp_path))
    result = run_script(script, "--config", str(config_path), *args)
    assert result.returncode != 0
    assert "lock" in (result.stderr + result.stdout).lower()


def test_split_never_includes_calibration_queries(tmp_path):
    cfg = make_config(tmp_path)
    config_path = write_config(tmp_path, cfg)
    write_locked_thresholds(4.0, 4, "calibration", config_path)
    cfg = load_config(config_path)
    scores = {"accuracy": 5, "completeness": 5, "structure": 5, "actionability": 5}
    full = dataset_paths(cfg, "full")
    cal = dataset_paths(cfg, "calibration")
    full_queries = [{"id": f"q-{i:04d}", "query": f"q{i}", "intended_tier": 1} for i in range(20)]
    full_queries.append({"id": "cal-0001", "query": "leaked", "intended_tier": 1})
    write_jsonl(full.queries, full_queries)
    write_jsonl(full.judgments, [{"query_id": q["id"], "tier": t, "scores": scores} for q in full_queries for t in (1, 2, 3)])
    write_jsonl(cal.queries, [{"id": "cal-0001", "query": "leaked", "intended_tier": 1}])
    result = run_script("split_dataset.py", "--config", str(config_path))
    assert result.returncode == 0, result.stderr
    splits = Path(cfg.paths.splits_dir)
    ids = {r["query_id"] for r in read_jsonl(splits / "train.jsonl") + read_jsonl(splits / "test.jsonl")}
    assert "cal-0001" not in ids and len(ids) == 20
    again = run_script("split_dataset.py", "--config", str(config_path))
    assert again.returncode != 0 and "frozen" in again.stderr + again.stdout
