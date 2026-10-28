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
# Passes the memo rule (mean 4.2, accuracy 4) but not the locked rule (accuracy >= 5).
MID = {"accuracy": 4, "completeness": 5, "structure": 4, "actionability": 4, "grounding": 4, "rationale": "ok"}
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


@dataclass
class Evidence:
    queries: list[str]
    sources: list[Source]
    plan_step: StepTrace
    search_step: StepTrace


def install_memo_stub(monkeypatch, calls, costs, latencies, gather_calls=None):
    module = types.ModuleType("nemoroute.agent.memo")
    gather_calls = gather_calls if gather_calls is not None else []

    def gather_evidence(question, cfg, client, search, role="tier1"):
        # Nondeterministic like a temperature > 0 planner: every call yields a different source URL.
        gather_calls.append(question)
        n = len(gather_calls)
        return Evidence(
            queries=[f"{question} angle {n}"],
            sources=[Source(url=f"https://example.com/source-{n}", title="Example source", snippet="A sourced fact.")],
            plan_step=StepTrace("plan", int(role[-1]), "fake", 10, 5, 0.0, 0.1),
            search_step=StepTrace("search", 0, "tavily", 0, 0, 0.0, 0.2),
        )

    def run_memo(question, strategy, cfg, client, search, router, *, evidence=None):
        calls.append((question, strategy))
        if evidence is None:
            evidence = gather_evidence(question, cfg, client, search)
        tier = {"routed": 2, "all_ultra": 3, "all_nano": 1}[strategy]
        return MemoResult(
            question=question,
            strategy=strategy,
            memo_md=f"Memo about {question}.",
            sources=list(evidence.sources),
            steps=[
                evidence.plan_step,
                evidence.search_step,
                StepTrace(
                    name="synthesize",
                    tier=tier,
                    model="fake",
                    input_tokens=10,
                    output_tokens=5,
                    cost_usd=costs[strategy],
                    latency_s=latencies[strategy],
                ),
            ],
            total_cost_usd=costs[strategy],
            total_latency_s=latencies[strategy],
            route_tier=tier if strategy == "routed" else None,
            route_reasoning="needs analysis" if strategy == "routed" else None,
        )

    module.Evidence = Evidence
    module.StepTrace = StepTrace
    module.gather_evidence = gather_evidence
    module.run_memo = run_memo
    parent = types.ModuleType("nemoroute.agent")
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, "nemoroute.agent", parent)
    monkeypatch.setitem(sys.modules, "nemoroute.agent.memo", module)


def prepare(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    cfg.labeling.locked = LockedThresholds(
        pass_threshold=4.0, min_accuracy=5, locked_at="2026-10-21T00:00:00+00:00", source="calibration"
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
    gather_calls: list[str] = []
    install_memo_stub(monkeypatch, calls, costs, latencies, gather_calls)
    client = RecordingClient(lambda role, messages: json.dumps(MID))

    run_main(monkeypatch, config_path, client)
    out = capsys.readouterr().out
    assert "judge 2" in out
    assert "pending evidence 2" in out
    assert "Lifetime logged spend" in out

    memos = read_jsonl(memo_dir / "memos.jsonl")
    judgments = read_jsonl(memo_dir / "memo_judgments.jsonl")
    assert len(memos) == 6 and len(judgments) == 6
    assert len({(m["query_id"], m["strategy"]) for m in memos}) == 6
    assert len({(j["query_id"], j["strategy"]) for j in judgments}) == 6
    assert all(isinstance(m["sources"][0], dict) and m["sources"][0]["url"] for m in memos)
    assert all(isinstance(m["steps"][0], dict) and m["steps"][0]["tier"] in {1, 2, 3} for m in memos)
    assert len(calls) == 6
    assert gather_calls == ["question 0", "question 1"]
    evidence = read_jsonl(memo_dir / "evidence.jsonl")
    assert [e["query_id"] for e in evidence] == ["m0", "m1"]
    assert all(e["role"] == "tier1" and e["plan_step"]["name"] == "plan" for e in evidence)
    for qid in ("m0", "m1"):
        urls = {tuple(s["url"] for s in m["sources"]) for m in memos if m["query_id"] == qid}
        assert len(urls) == 1
    assert evaluate_memos.source_mismatches(memos) == []
    assert [s["name"] for s in memos[0]["steps"]] == ["plan", "search", "synthesize"]
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
    assert "Memo pass rule (headline): mean of the five criteria >= 4.0 and accuracy >= 4." in md
    assert "Locked labeling thresholds: pass_threshold=4.0, min_accuracy=5" in md
    assert "| 100.0% |" in md
    assert "Pass rate under the locked labeling rule (mean >= 4.0, accuracy >= 5):" in md
    assert "| routed | 2 | 0.0% | [0.0%, 0.0%] |" in md
    assert "Evidence is shared" in md
    assert "Excluded" not in md
    assert "| T2 | 2 | 4.20 | 4.20 | 0 / 2 / 0 | 4.00 | 4.00 | 4.00 | 4.00 |" in md
    assert "95% CI" in md
    assert "Cost / memo (USD)" in md
    assert "Mean score" in md and "Pass rate" in md and "Tier mix" in md
    assert "20%" in md
    assert "[4.20, 4.20]" in md
    assert "30.00" in md
    assert "blind" in md and "cached search" in md and "every model call" in md
    assert (results_dir / "charts" / "cost_vs_quality_memos.png").stat().st_size > 0
    assert (results_dir / "charts" / "tier_mix_memos.png").stat().st_size > 0
    assert "charts/tier_mix_memos.png" in md
    assert (results_dir / "memo_results.csv").read_text().count("\n") == 4

    run_main(monkeypatch, config_path, client)
    assert len(read_jsonl(memo_dir / "memos.jsonl")) == 6
    assert len(read_jsonl(memo_dir / "memo_judgments.jsonl")) == 6
    assert len(calls) == 6
    assert len(gather_calls) == 2
    assert len(read_jsonl(memo_dir / "evidence.jsonl")) == 2
    assert len(client.messages) == 6


def test_existing_evidence_is_reused_not_regenerated(tmp_path, monkeypatch):
    config_path, memo_dir = prepare(tmp_path, monkeypatch)
    frozen = {
        "query_id": "m0",
        "question": "question 0",
        "role": "tier1",
        "queries": ["frozen query"],
        "sources": [{"url": "https://example.com/frozen", "title": "Frozen", "snippet": "Frozen fact."}],
        "plan_step": {"name": "plan", "tier": 1, "model": "fake", "input_tokens": 1, "output_tokens": 1,
                      "cost_usd": 0.0, "latency_s": 0.0},
        "search_step": {"name": "search", "tier": 0, "model": "tavily", "input_tokens": 0, "output_tokens": 0,
                        "cost_usd": 0.0, "latency_s": 0.0},
    }
    write_jsonl(memo_dir / "evidence.jsonl", [frozen])
    calls: list[tuple[str, str]] = []
    gather_calls: list[str] = []
    costs = {"routed": 0.01, "all_ultra": 0.05, "all_nano": 0.002}
    install_memo_stub(monkeypatch, calls, costs, {k: 1.0 for k in costs}, gather_calls)

    run_main(monkeypatch, config_path, RecordingClient(lambda role, messages: json.dumps(HIGH)))

    assert gather_calls == ["question 1"]
    memos = read_jsonl(memo_dir / "memos.jsonl")
    assert {m["sources"][0]["url"] for m in memos if m["query_id"] == "m0"} == {"https://example.com/frozen"}
    assert read_jsonl(memo_dir / "evidence.jsonl")[0] == frozen


def test_shared_source_gate_blocks_report(tmp_path, monkeypatch):
    config_path, memo_dir = prepare(tmp_path, monkeypatch)
    install_memo_stub(monkeypatch, [], {}, {})
    memos, judgments = [], []
    for qid, question in (("m0", "question 0"), ("m1", "question 1")):
        for strategy in evaluate_memos.STRATEGIES:
            url = "https://example.com/other" if (qid, strategy) == ("m1", "all_nano") else "https://example.com/a"
            memos.append({"query_id": qid, "question": question, "strategy": strategy, "memo_md": "m",
                          "sources": [{"url": url, "title": "t", "snippet": "s"}], "steps": [],
                          "total_cost_usd": 0.01, "total_latency_s": 1.0, "route_tier": 2})
            judgments.append({"query_id": qid, "strategy": strategy,
                              "scores": {c: HIGH[c] for c in evaluate_memos.MEMO_CRITERIA}})
    write_jsonl(memo_dir / "memos.jsonl", memos)
    write_jsonl(memo_dir / "memo_judgments.jsonl", judgments)

    with pytest.raises(SystemExit, match="Shared-source check failed for m1"):
        run_main(monkeypatch, config_path, RecordingClient(lambda role, messages: json.dumps(HIGH)))
    assert not (tmp_path / "results" / "memo_results.md").exists()


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


def test_report_only_makes_no_calls_and_lists_excluded_pairs(tmp_path, monkeypatch):
    config_path, memo_dir = prepare(tmp_path, monkeypatch)
    calls: list[tuple[str, str]] = []
    gather_calls: list[str] = []
    install_memo_stub(monkeypatch, calls, {k: 0.01 for k in evaluate_memos.STRATEGIES}, {k: 1.0 for k in evaluate_memos.STRATEGIES}, gather_calls)
    client = RecordingClient(lambda role, messages: json.dumps(HIGH))
    run_main(monkeypatch, config_path, client)
    memos = [m for m in read_jsonl(memo_dir / "memos.jsonl") if (m["query_id"], m["strategy"]) != ("m1", "all_nano")]
    write_jsonl(memo_dir / "memos.jsonl", memos)
    n_calls, n_judge = len(calls), len(client.messages)

    monkeypatch.setattr(
        sys, "argv", ["evaluate_memos", "--config", str(config_path), "--limit", "2", "--report-only"]
    )
    evaluate_memos.main()

    assert (len(calls), len(client.messages)) == (n_calls, n_judge)
    md = (tmp_path / "results" / "memo_results.md").read_text()
    assert "Excluded (no memo or no judgment after the resume pass, see logs): m1/all_nano." in md
