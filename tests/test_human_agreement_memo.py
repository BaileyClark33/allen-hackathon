from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from collections import Counter

import pytest
from fakes import make_config

from nemoroute.config import LockedThresholds
from nemoroute.datasets import human_files
from nemoroute.human_agreement import (
    HUMAN_COLUMNS,
    agreement,
    memo_candidates,
    read_human_labels,
    render_agreement_section,
    stratified_sample,
    write_agreement_section,
    write_human_files,
)
from nemoroute.jsonl import read_jsonl

STRATEGIES = ["routed", "all_ultra", "all_nano"]
HIGH = {"accuracy": 5, "completeness": 5, "structure": 5, "actionability": 5, "grounding": 5}


def memo_rows(per_strategy: int = 12) -> tuple[list[dict], list[dict]]:
    memos, judgments = [], []
    for strategy in STRATEGIES:
        for i in range(per_strategy):
            query_id = f"{strategy}-{i:02d}"
            memos.append({
                "query_id": query_id,
                "strategy": strategy,
                "question": f"Question {query_id}",
                "memo_md": f"Memo {query_id}",
            })
            judgments.append({"query_id": query_id, "strategy": strategy, "scores": HIGH})
    return memos, judgments


def sample_memos(seed: int = 99) -> list[dict]:
    memos, judgments = memo_rows()
    candidates = memo_candidates(memos, judgments)
    return stratified_sample(candidates, lambda c: c["strategy"], 30, seed)


def stats(**overrides) -> dict:
    base = {
        "n": 30,
        "percent_agreement": 0.8,
        "cohen_kappa": 0.5,
        "both_pass": 20,
        "both_fail": 4,
        "human_pass_judge_fail": 3,
        "human_fail_judge_pass": 3,
        "pass_threshold": 4.0,
        "min_accuracy": 5,
    }
    base.update(overrides)
    return base


def test_memo_sample_is_30_and_stratified_by_strategy():
    sample = sample_memos()
    assert len(sample) == 30
    assert Counter(c["strategy"] for c in sample) == {"routed": 10, "all_ultra": 10, "all_nano": 10}
    assert sample == sample_memos()


def test_memo_candidates_fall_back_and_keep_one_row_per_key():
    memos = [
        {"query_id": "a", "strategy": "routed", "question": "Q", "memo": "old"},
        {"query_id": "a", "strategy": "routed", "question": "Q", "memo_md": "new"},
        {"query_id": "b", "strategy": "all_nano", "query": "Q2", "answer": "answer text"},
        {"query_id": "c", "strategy": "all_ultra", "question": "unjudged"},
    ]
    judgments = [
        {"query_id": "a", "strategy": "routed", "scores": HIGH},
        {"query_id": "b", "strategy": "all_nano", "scores": HIGH},
    ]
    candidates = {c["query_id"]: c for c in memo_candidates(memos, judgments)}
    assert set(candidates) == {"a", "b"}
    assert candidates["a"]["memo_md"] == "new"
    assert candidates["a"]["answer"] == "new"
    assert candidates["b"]["question"] == "Q2"
    assert candidates["b"]["memo_md"] == "answer text"
    assert candidates["a"]["tier"] == "routed"


def test_memo_csv_is_blind_and_key_carries_strategy(tmp_path):
    sample = sample_memos()
    csv_path = tmp_path / "memo_human.csv"
    key_path = tmp_path / "memo_human_key.jsonl"
    write_human_files(sample, csv_path, key_path, overwrite=False)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == HUMAN_COLUMNS
        rows = list(reader)
    assert len(rows) == 30
    assert all(row["human_pass"] == "" for row in rows)
    assert "strategy" not in ",".join(HUMAN_COLUMNS)
    assert "model" not in ",".join(HUMAN_COLUMNS)
    assert "scores" not in csv_path.read_text().lower()

    keys = read_jsonl(key_path)
    assert len(keys) == 30
    assert {k["strategy"] for k in keys} == set(STRATEGIES)
    assert {(k["query_id"], k["strategy"]) for k in keys} == {(c["query_id"], c["strategy"]) for c in sample}

    with pytest.raises(SystemExit):
        write_human_files(sample, csv_path, key_path, overwrite=False)


def test_write_human_files_accepts_strategy_without_tier(tmp_path):
    csv_path = tmp_path / "memo_human.csv"
    key_path = tmp_path / "memo_human_key.jsonl"
    item = {"query_id": "m-1", "strategy": "routed", "query": "Q", "answer": "memo"}
    write_human_files([item], csv_path, key_path, overwrite=False)
    key = read_jsonl(key_path)[0]
    assert key["strategy"] == "routed" and "tier" not in key


def test_agreement_works_with_memo_strategy_keys(tmp_path):
    sample = sample_memos()
    csv_path = tmp_path / "memo_human.csv"
    key_path = tmp_path / "memo_human_key.jsonl"
    write_human_files(sample, csv_path, key_path, overwrite=False)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        row["human_pass"] = "1" if i % 2 == 0 else "0"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HUMAN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    human = read_human_labels(csv_path, key_path)
    assert {row["tier"] for row in human} == set(STRATEGIES)
    _, judgments = memo_rows()
    judged = [{**j, "tier": j["strategy"]} for j in judgments]
    result = agreement(human, judged, 4.0, 5)

    assert result["n"] == 30
    assert result["percent_agreement"] == 0.5
    assert result["both_pass"] == 15
    assert result["human_fail_judge_pass"] == 15
    assert result["cohen_kappa"] == pytest.approx(0.0)


def test_memo_report_section_is_replaced_not_duplicated(tmp_path):
    report = tmp_path / "memo_results.md"
    report.write_text("# Memo results\n\n## Summary\n\nExisting results.\n")

    write_agreement_section(report, render_agreement_section(stats()))
    first = report.read_text()
    assert first.count("## Human agreement") == 1
    assert "## Summary" in first

    write_agreement_section(report, render_agreement_section(stats()))
    assert report.read_text() == first

    write_agreement_section(report, render_agreement_section(stats(percent_agreement=0.5, cohen_kappa=0.1)))
    updated = report.read_text()
    assert updated.count("## Human agreement") == 1
    assert "50.0%" in updated
    assert "80.0%" not in updated


def test_human_files_memo_paths(tmp_path, monkeypatch):
    import nemoroute.config as config_module

    monkeypatch.setattr(config_module, "ROOT", tmp_path)
    cfg = make_config(tmp_path)
    csv_path, key_path = human_files(cfg, "memo")
    assert csv_path == tmp_path / "data" / "memo_bench" / "memo_human.csv"
    assert key_path == tmp_path / "data" / "memo_bench" / "memo_human_key.jsonl"


def test_memo_agreement_uses_memo_rule_not_locked_rule(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "score_human_agreement.py"
    spec = importlib.util.spec_from_file_location("score_human_agreement", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = make_config(tmp_path)
    cfg.labeling.locked = LockedThresholds(
        pass_threshold=4.0, min_accuracy=5, locked_at="2026-10-21T00:00:00+00:00", source="calibration"
    )

    assert module.thresholds_for(cfg, "memo") == (cfg.memo_bench.pass_threshold, cfg.memo_bench.min_accuracy)
    assert module.thresholds_for(cfg, "memo") == (4.0, 4)
    assert module.thresholds_for(cfg, "test") == (4.0, 5)
    assert module.thresholds_for(cfg, "calibration") == (4.0, 5)
