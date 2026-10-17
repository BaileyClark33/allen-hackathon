from __future__ import annotations

import json

import pytest
from fakes import FakeClient, make_config
from test_report import build_fixture

from nemoroute.anthropic_backend import request_kwargs
from nemoroute.benchmarks.mmlu_pro import extract_choice, mcq_outcomes, oracle_choice, sample_items
from nemoroute.client import TokenFactoryClient
from nemoroute.evaluation.report import render_report
from nemoroute.evaluation.run import evaluate
from nemoroute.labeling.judge import judge_schema
from nemoroute.labeling.label import UNRESOLVED
from nemoroute.pricing import estimate
from nemoroute.reference import REFERENCE, pending_judge_jobs
from nemoroute.usage_log import CallRecord, UsageLog


def test_per_model_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "nebius-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    cfg = make_config(tmp_path)
    client = TokenFactoryClient(cfg, UsageLog(tmp_path / "calls.jsonl"))
    assert client._key_for(cfg.models["nano"]) == "nebius-key"
    assert client._key_for(cfg.models["reference"]) == "anthropic-key"
    a = client._client_for("http://vm:8000/v1/", "k1")
    assert client._client_for("http://vm:8000/v1/", "k2") is not a
    assert client._client_for("http://vm:8000/v1/", "k1") is a


def test_anthropic_request_shape(tmp_path):
    cfg = make_config(tmp_path)
    schema = judge_schema(cfg.labeling.criteria)
    messages = [{"role": "system", "content": "be a consultant"}, {"role": "user", "content": "q"}]
    kwargs = request_kwargs(cfg.models["reference"], cfg.roles["reference_judge"], messages, schema)
    assert kwargs["system"] == "be a consultant"
    assert kwargs["messages"] == [{"role": "user", "content": "q"}]
    assert "temperature" not in kwargs and "thinking" not in kwargs
    assert kwargs["output_config"]["effort"] == "medium"
    fmt = kwargs["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert "minimum" not in json.dumps(fmt["schema"]) and fmt["schema"]["required"] == schema["required"]
    no_schema = request_kwargs(cfg.models["reference"], cfg.roles["reference"], messages[1:], None)
    assert "system" not in no_schema and "format" not in no_schema["output_config"]


def test_credit_budget_counts_only_token_factory_spend(tmp_path):
    cfg = make_config(tmp_path)
    log = UsageLog(tmp_path / "calls.jsonl")
    base = dict(ts="2026-10-16T00:00:00+00:00", run_id="r", step="s", input_tokens=1, output_tokens=1,
                latency_s=0.1, ok=True)
    log.record(CallRecord(role="tier1", model="nano", cost_usd=0.5, **base))
    log.record(CallRecord(role="reference", model="claude", cost_usd=2.0, billing="anthropic", **base))
    assert log.total_spend() == 0.5
    assert log.total_spend(billing=None) == 2.5
    est = estimate(cfg, {"tier1": 10, "reference": 10})
    assert 0 < est.total_for("token_factory") < est.total_for("anthropic")
    assert "billed outside Token Factory (anthropic)" in est.render()


def test_mmlu_choice_parsing_and_sampling():
    assert extract_choice("work...\nAnswer: C", 4) == "C"
    assert extract_choice("so the answer is (B).", 4) == "B"
    assert extract_choice("Answer: **D**", 10) == "D"
    assert extract_choice("Answer: J", 4) is None
    assert extract_choice("", 4) is None
    rows = [
        {"question_id": i, "question": f"q{i}", "options": ["x", "y", "z"], "answer": "B",
         "category": "business" if i % 2 else "economics"}
        for i in range(40)
    ]
    sample = sample_items(rows, ["business", "economics"], 11, seed=3)
    assert len(sample) == 11 and sample == sample_items(rows, ["business", "economics"], 11, seed=3)
    assert sum(r["category"] == "business" for r in sample) == 6
    assert sample[0]["query_id"].startswith("mmlu-") and "B. y" in sample[0]["query"]


def test_mmlu_oracle_and_outcomes():
    resp = {
        ("a", 1): {"correct": False, "cost_usd": 0.1, "latency_s": 1.0},
        ("a", 2): {"correct": True, "cost_usd": 0.2, "latency_s": 2.0},
        ("a", 3): {"correct": True, "cost_usd": 0.3, "latency_s": 3.0},
        ("b", 1): {"correct": False, "cost_usd": 0.1, "latency_s": 1.0},
        ("b", 2): {"correct": False, "cost_usd": 0.2, "latency_s": 2.0},
        ("b", 3): {"correct": False, "cost_usd": 0.3, "latency_s": 3.0},
    }
    assert oracle_choice(resp, "a", [1, 2, 3]) == 2
    assert oracle_choice(resp, "b", [1, 2, 3]) == 3
    outcomes = mcq_outcomes({"a": 2, "b": 1}, resp, {"a": (0.01, 0.5)})
    assert [o.passed for o in outcomes] == [True, False]
    assert outcomes[0].cost_usd == pytest.approx(0.21) and outcomes[0].latency_s == 2.5


def test_evaluation_adds_reference_and_other_routers(tmp_path):
    cfg = make_config(tmp_path)
    items, preds, responses, judgments = build_fixture()
    hi = {"accuracy": 5, "completeness": 5, "structure": 5, "actionability": 5}
    ref_resp = [{"query_id": i["query_id"], "tier": REFERENCE, "cost_usd": 0.02, "latency_s": 9.0} for i in items]
    ref_judg = [{"query_id": i["query_id"], "tier": REFERENCE, "scores": hi, "mean_score": 5.0} for i in items]
    alt_preds = [{**p, "tier": i["label"] if i["label"] != UNRESOLVED else 3} for p, i in zip(preds, items)]
    result = evaluate(
        cfg, "zero_shot_nano", items, preds, responses, judgments, 4.0, 4,
        reference_responses=ref_resp, reference_judgments=ref_judg,
        other_routers={"alt_router": alt_preds},
    )
    by_name = {s.strategy: s for s in result.summaries_resolved}
    assert by_name[REFERENCE].pass_rate == 1.0 and by_name[REFERENCE].tier_mix == {REFERENCE: 9}
    assert by_name["alt_router"].pass_rate == 1.0
    assert result.other_routing["alt_router"].accuracy == 1.0
    md = render_report(result, cfg.tiers(), None, 4.0, 4, "Test", None, {})
    assert "| reference |" in md and "reference:9" in md and "| alt_router | 100.0%" in md

    partial = evaluate(cfg, "zero_shot_nano", items, preds, responses, judgments, 4.0, 4,
                       reference_responses=ref_resp[:3], reference_judgments=ref_judg[:3],
                       other_routers={"alt_router": alt_preds[:3]})
    names = {s.strategy for s in partial.summaries_resolved}
    assert REFERENCE not in names and "alt_router" not in names


def test_reference_judge_jobs_skip_done_and_unanswered(tmp_path):
    items = [{"query_id": "a", "query": "qa"}, {"query_id": "b", "query": "qb"}]
    answers = [{"query_id": "a", "tier": 1, "answer": "x"}, {"query_id": "a", "tier": 2, "answer": "y"},
               {"query_id": "b", "tier": 1, "answer": "z"}]
    judged = [{"query_id": "a", "tier": 1}]
    jobs = pending_judge_jobs(items, answers, judged, [1, 2], tmp_path / "out.jsonl")
    assert [(j.query_id, j.tier) for j in jobs] == [("a", 2), ("b", 1)]

