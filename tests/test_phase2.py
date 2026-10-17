from __future__ import annotations

import json
from string import Template

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
from nemoroute.routers.registry import build_router
from nemoroute.sft import build_sft
from nemoroute.usage_log import CallRecord, UsageLog


def test_finetuned_router_uses_its_own_role_prompt_and_schema(tmp_path):
    cfg = make_config(tmp_path)
    cfg.router.type = "finetuned"
    client = FakeClient(lambda role, messages: '{"tier": 1}')
    router = build_router(cfg, client)
    decision = router.route("What is EBITDA?")
    assert router.name == "finetuned_nano" and router.role == "router_ft"
    assert decision.tier == 1 and not decision.fallback
    assert client.calls == [("router_ft", "route")]
    assert cfg.model_for_role("router_ft").billing != "token_factory"


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


def _labels(n_per_tier: int) -> list[dict]:
    rows = []
    for tier in (1, 2, 3):
        rows += [{"query_id": f"q-{tier}{i:03d}", "query": f"query {tier} {i}", "label": tier} for i in range(n_per_tier)]
    return rows + [{"query_id": "q-u000", "query": "hard", "label": UNRESOLVED}]


def test_sft_export_is_stratified_and_leak_checked():
    template = Template("Q: $query")
    fit, val = build_sft(_labels(10), template, {"test": {"q-9999"}}, val_frac=0.2, seed=1)
    assert len(fit) == 24 and len(val) == 6
    assert sorted(r["label"] for r in val) == [1, 1, 2, 2, 3, 3]
    assert all(r["query_id"] != "q-u000" for r in fit + val)
    first = fit[0]
    assert first["messages"][0]["content"].startswith("Q: query")
    assert json.loads(first["messages"][1]["content"]) == {"tier": first["label"]}
    assert build_sft(_labels(10), template, {}, 0.2, 1) == (fit, val)
    with pytest.raises(AssertionError):
        build_sft(_labels(10), template, {"test": {"q-1003"}}, 0.2, 1)


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
    ft_preds = [{**p, "tier": i["label"] if i["label"] != UNRESOLVED else 3} for p, i in zip(preds, items)]
    result = evaluate(
        cfg, "zero_shot_nano", items, preds, responses, judgments, 4.0, 4,
        reference_responses=ref_resp, reference_judgments=ref_judg,
        other_routers={"finetuned_nano": ft_preds},
    )
    by_name = {s.strategy: s for s in result.summaries_resolved}
    assert by_name[REFERENCE].pass_rate == 1.0 and by_name[REFERENCE].tier_mix == {REFERENCE: 9}
    assert by_name["finetuned_nano"].pass_rate == 1.0
    assert result.other_routing["finetuned_nano"].accuracy == 1.0
    md = render_report(result, cfg.tiers(), None, 4.0, 4, "Test", None, {})
    assert "| reference |" in md and "reference:9" in md and "| finetuned_nano | 100.0%" in md

    partial = evaluate(cfg, "zero_shot_nano", items, preds, responses, judgments, 4.0, 4,
                       reference_responses=ref_resp[:3], reference_judgments=ref_judg[:3],
                       other_routers={"finetuned_nano": ft_preds[:3]})
    names = {s.strategy for s in partial.summaries_resolved}
    assert REFERENCE not in names and "finetuned_nano" not in names


def test_reference_judge_jobs_skip_done_and_unanswered(tmp_path):
    items = [{"query_id": "a", "query": "qa"}, {"query_id": "b", "query": "qb"}]
    answers = [{"query_id": "a", "tier": 1, "answer": "x"}, {"query_id": "a", "tier": 2, "answer": "y"},
               {"query_id": "b", "tier": 1, "answer": "z"}]
    judged = [{"query_id": "a", "tier": 1}]
    jobs = pending_judge_jobs(items, answers, judged, [1, 2], tmp_path / "out.jsonl")
    assert [(j.query_id, j.tier) for j in jobs] == [("a", 2), ("b", 1)]


def test_app_falls_back_to_zero_shot_when_finetuned_router_is_down(tmp_path):
    import importlib.util

    from nemoroute.config import ROOT
    from nemoroute.routers.zero_shot import ZeroShotRouter

    spec = importlib.util.spec_from_file_location("streamlit_app", ROOT / "app" / "streamlit_app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)

    def responder(role, messages):
        if role == "router_ft":
            raise ConnectionError("vm down")
        return '{"tier": 2, "reasoning": "framework"}'

    client = FakeClient(responder)
    ft = ZeroShotRouter(client, Template("$query"), [1, 2, 3], role="router_ft", name="finetuned_nano")
    zs = ZeroShotRouter(client, Template("$query"), [1, 2, 3])
    decision, note = app.route_with_fallback(ft, zs, "q")
    assert decision.tier == 2 and "unreachable" in note
    assert app.route_with_fallback(zs, zs, "q")[1] is None
