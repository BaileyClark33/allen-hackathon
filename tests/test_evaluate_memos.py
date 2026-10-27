from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
from fakes import FakeClient, make_config, write_config

from nemoroute.config import LockedThresholds
from nemoroute.jsonl import read_jsonl, write_jsonl

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_memos.py"
spec = importlib.util.spec_from_file_location("evaluate_memos", SCRIPT)
evaluate_memos = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_memos)

HIGH = {"accuracy": 5, "completeness": 4, "structure": 4, "actionability": 4, "grounding": 4, "rationale": "ok"}
LOW = {"accuracy": 3, "completeness": 3, "structure": 3, "actionability": 3, "grounding": 3, "rationale": "weak"}


@dataclass
class Source:
    url: str
    title: str
    snippet: str


@dataclass
class StepTrace:
    name: str
    tier: int
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float


@dataclass
class MemoResult:
    question: str
    strategy: str
    memo_md: str
    sources: list[Source]
    steps: list[StepTrace]
    total_cost_usd: float
    total_latency_s: float
    route_tier: int | None
    route_reasoning: str | None


class RecordingClient(FakeClient):
    def __init__(self, responder):
        super().__init__(responder)
        self.messages: list[tuple[str, list[dict[str, str]]]] = []

    def chat(self, role, messages, *, step, **kwargs):
        self.messages.append((role, messages))
        return super().chat(role, messages, step=step, **kwargs)


def install_memo_stub(monkeypatch, calls, costs, latencies):
    module = types.ModuleType("nemoroute.agent.memo")

    def run_memo(question, strategy, cfg, client, search, router):
        calls.append((question, strategy))
        tier = {"routed": 2, "all_ultra": 3, "all_nano": 1}[strategy]
        return MemoResult(
            question=question,
            strategy=strategy,
            memo_md=f"Memo about {question}.",
            sources=[Source(url="https://example.com/source", title="Example source", snippet="A sourced fact.")],
            steps=[
                StepTrace(
                    name="synthesize",
                    tier=tier,
                    model="fake",
                    input_tokens=10,
                    output_tokens=5,
                    cost_usd=costs[strategy],
                    latency_s=latencies[strategy],
                )
            ],
            total_cost_usd=costs[strategy],
            total_latency_s=latencies[strategy],
            route_tier=tier if strategy == "routed" else None,
            route_reasoning="needs analysis" if strategy == "routed" else None,
        )

    module.run_memo = run_memo
    parent = types.ModuleType("nemoroute.agent")
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, "nemoroute.agent", parent)
    monkeypatch.setitem(sys.modules, "nemoroute.agent.memo", module)


def prepare(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    cfg.labeling.locked = LockedThresholds(
        pass_threshold=4.0, min_accuracy=4, locked_at="2026-10-21T00:00:00+00:00", source="calibration"
    )
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "router.txt").write_text("$query")
    (prompts_dir / "memo_judge.txt").write_text("$query\n\n$answer")
    cfg.paths.prompts_dir = str(prompts_dir)
    config_path = write_config(tmp_path, cfg)

    memo_dir = tmp_path / "memo_bench"
    memo_dir.mkdir()
    write_jsonl(
        memo_dir / "prompts.jsonl",
        [{"id": "m0", "query": "question 0"}, {"query_id": "m1", "question": "question 1"}],
    )
    monkeypatch.setattr(evaluate_memos, "MEMO_DIR", str(memo_dir))
    return config_path, memo_dir


def run_main(monkeypatch, config_path, client):
    monkeypatch.setattr(
        sys, "argv", ["evaluate_memos", "--config", str(config_path), "--limit", "2", "--yes"]
    )
    monkeypatch.setattr(evaluate_memos, "make_client", lambda cfg, usage_log, max_cost: client)
    monkeypatch.setattr(evaluate_memos, "make_search", lambda cfg: object())
    monkeypatch.setattr(evaluate_memos, "print_run_summary", lambda usage_log, client: None)
    evaluate_memos.main()


def test_evaluate_memos_end_to_end_and_resume(tmp_path, monkeypatch, capsys):
    config_path, memo_dir = prepare(tmp_path, monkeypatch)
    calls: list[tuple[str, str]] = []
    costs = {"routed": 0.01, "all_ultra": 0.05, "all_nano": 0.002}
    latencies = {"routed": 10.0, "all_ultra": 30.0, "all_nano": 5.0}
    install_memo_stub(monkeypatch, calls, costs, latencies)
    client = RecordingClient(lambda role, messages: json.dumps(HIGH))

    run_main(monkeypatch, config_path, client)
    out = capsys.readouterr().out
    assert "judge 2" in out
    assert "Lifetime logged spend" in out

    memos = read_jsonl(memo_dir / "memos.jsonl")
    judgments = read_jsonl(memo_dir / "memo_judgments.jsonl")
    assert len(memos) == 6 and len(judgments) == 6
    assert len({(m["query_id"], m["strategy"]) for m in memos}) == 6
    assert len({(j["query_id"], j["strategy"]) for j in judgments}) == 6
    assert all(isinstance(m["sources"][0], dict) and m["sources"][0]["url"] for m in memos)
    assert all(isinstance(m["steps"][0], dict) and m["steps"][0]["tier"] in {1, 2, 3} for m in memos)
    assert len(calls) == 6
    assert [role for role, _messages in client.messages] == ["judge"] * 6

    for _role, messages in client.messages:
        content = messages[-1]["content"]
        assert "Memo about question" in content
        assert "[1] Example source" in content
        assert "https://example.com/source" in content
        assert "A sourced fact." in content
        assert "routed" not in content and "all_ultra" not in content and "all_nano" not in content

    results_dir = tmp_path / "results"
    md = (results_dir / "memo_results.md").read_text()
    assert "Locked thresholds" in md
    assert "95% CI" in md
    assert "Cost / memo (USD)" in md
    assert "Mean score" in md and "Pass rate" in md and "Tier mix" in md
    assert "20%" in md
    assert "[4.20, 4.20]" in md
    assert "30.00" in md
    assert "blind" in md and "cached search" in md and "every model call" in md
    assert (results_dir / "charts" / "cost_vs_quality_memos.png").stat().st_size > 0
    assert (results_dir / "memo_results.csv").read_text().count("\n") == 4

    run_main(monkeypatch, config_path, client)
    assert len(read_jsonl(memo_dir / "memos.jsonl")) == 6
    assert len(read_jsonl(memo_dir / "memo_judgments.jsonl")) == 6
    assert len(calls) == 6
    assert len(client.messages) == 6


def test_memo_summaries_aggregate_cost_latency_and_pass(tmp_path):
    cfg = make_config(tmp_path)
    prompts = [{"id": f"m{i}", "query": f"q{i}"} for i in range(3)]
    costs = {"routed": [0.008, 0.01, 0.012], "all_ultra": [0.04, 0.05, 0.06], "all_nano": [0.001, 0.002, 0.003]}
    latencies = {"routed": [8.0, 10.0, 12.0], "all_ultra": [20.0, 30.0, 40.0], "all_nano": [4.0, 5.0, 6.0]}
    memos, judgments = [], []
    for i, prompt in enumerate(prompts):
        for strategy in ("routed", "all_ultra", "all_nano"):
            memos.append(
                {
                    "query_id": prompt["id"],
                    "strategy": strategy,
                    "total_cost_usd": costs[strategy][i],
                    "total_latency_s": latencies[strategy][i],
                    "route_tier": 2 if strategy == "routed" else None,
                }
            )
            scores = {c: (LOW if strategy == "all_nano" else HIGH)[c] for c in evaluate_memos.MEMO_CRITERIA}
            judgments.append(
                {
                    "query_id": prompt["id"],
                    "strategy": strategy,
                    "scores": scores,
                    "mean_score": sum(scores.values()) / len(scores),
                }
            )

    summaries = evaluate_memos.build_summaries(cfg, prompts, memos, judgments, 4.0, 4)
    by_name = {s.strategy: s for s in summaries}
    assert by_name["routed"].mean_cost_usd == pytest.approx(0.01)
    assert by_name["routed"].p50_latency_s == pytest.approx(10.0)
    assert by_name["routed"].p95_latency_s == pytest.approx(11.8)
    assert by_name["routed"].mean_score == pytest.approx(4.2)
    assert by_name["routed"].pass_rate == 1.0
    assert by_name["routed"].tier_mix == {2: 3}
    assert by_name["all_ultra"].mean_cost_usd == pytest.approx(0.05)
    assert by_name["all_ultra"].p50_latency_s == pytest.approx(30.0)
    assert by_name["all_ultra"].p95_latency_s == pytest.approx(39.0)
    assert by_name["all_ultra"].tier_mix == {3: 3}
    assert by_name["all_nano"].mean_cost_usd == pytest.approx(0.002)
    assert by_name["all_nano"].mean_score == pytest.approx(3.0)
    assert by_name["all_nano"].pass_rate == 0.0
    assert by_name["all_nano"].tier_mix == {1: 3}
    for summary in summaries:
        assert len(summary.cost_ci) == 2 and summary.cost_ci[0] <= summary.mean_cost_usd <= summary.cost_ci[1]
        assert len(summary.score_ci) == 2 and len(summary.pass_ci) == 2
