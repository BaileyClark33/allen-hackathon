from __future__ import annotations

import json
from string import Template

import pytest
from fakes import FakeClient

from nemoroute.dataset.dedup import dedup
from nemoroute.dataset.generate import cap_by_limit, last_id_number, needed_per_tier
from nemoroute.human_agreement import agreement, stratified_sample
from nemoroute.parsing import JsonParseError, extract_json, split_reasoning
from nemoroute.routers.zero_shot import ZeroShotRouter


def test_split_reasoning_handles_think_tags():
    assert split_reasoning("<think>hmm</think>Answer") == ("Answer", "hmm")
    assert split_reasoning("hmm</think>\nAnswer") == ("Answer", "hmm")
    assert split_reasoning("Plain") == ("Plain", "")
    assert split_reasoning(None) == ("", "")


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"tier": 2}\n```') == {"tier": 2}
    assert extract_json('Sure! {"tier": 3, "reasoning": "x"} done') == {"tier": 3, "reasoning": "x"}
    with pytest.raises(JsonParseError):
        extract_json("no json here")


def test_dedup_exact_and_near_duplicates():
    existing = ["What is the difference between gross margin and contribution margin?"]
    candidates = [
        "what is the difference between gross margin and contribution margin",
        "What's the difference between gross margin and contribution margin?",
        "How should a SaaS startup price its enterprise tier?",
        "How should a SaaS startup price its enterprise tier?",
    ]
    assert dedup(candidates, existing, 0.85) == ["How should a SaaS startup price its enterprise tier?"]


def test_generation_balances_tiers_and_respects_limit():
    existing = [{"intended_tier": 1}] * 5
    needed = needed_per_tier(existing, 60, [1, 2, 3])
    assert needed == {1: 15, 2: 20, 3: 20}
    assert cap_by_limit(needed, 20) == {1: 7, 2: 7, 3: 6}
    assert cap_by_limit({1: 1, 2: 0, 3: 10}, 5) == {1: 1, 2: 0, 3: 4}


def test_last_id_number_survives_removed_rows():
    assert last_id_number([]) == 0
    assert last_id_number([{"id": "cal-0001"}, {"id": "cal-0007"}, {"id": "cal-0003"}]) == 7


def test_router_parses_and_falls_back_to_top_tier():
    template = Template("$query")
    good = ZeroShotRouter(FakeClient(lambda r, m: '{"tier": 2, "reasoning": "framework"}'), template, [1, 2, 3])
    decision = good.route("q")
    assert decision.tier == 2 and not decision.fallback and decision.cost_usd > 0
    bad = ZeroShotRouter(FakeClient(lambda r, m: "tier two I think"), template, [1, 2, 3])
    assert bad.route("q").tier == 3 and bad.route("q").fallback
    out_of_range = ZeroShotRouter(FakeClient(lambda r, m: json.dumps({"tier": 7})), template, [1, 2, 3])
    assert out_of_range.route("q").fallback


def test_stratified_sample_covers_strata():
    candidates = [{"query_id": f"q{i}", "tier": 1 + i % 3} for i in range(30)]
    sample = stratified_sample(candidates, lambda c: c["tier"], 9, seed=1)
    assert len(sample) == 9
    assert sorted(c["tier"] for c in sample) == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert sample == stratified_sample(candidates, lambda c: c["tier"], 9, seed=1)


def test_agreement_stats():
    hi = {"accuracy": 5, "completeness": 5, "structure": 5, "actionability": 5}
    lo = {"accuracy": 2, "completeness": 2, "structure": 2, "actionability": 2}
    judgments = [
        {"query_id": "a", "tier": 1, "scores": hi},
        {"query_id": "b", "tier": 1, "scores": lo},
        {"query_id": "c", "tier": 1, "scores": hi},
        {"query_id": "d", "tier": 1, "scores": lo},
    ]
    human = [
        {"query_id": "a", "tier": 1, "human_pass": True},
        {"query_id": "b", "tier": 1, "human_pass": False},
        {"query_id": "c", "tier": 1, "human_pass": False},
        {"query_id": "d", "tier": 1, "human_pass": False},
    ]
    stats = agreement(human, judgments, 4.0, 4)
    assert stats["n"] == 4
    assert stats["percent_agreement"] == pytest.approx(0.75)
    assert stats["human_fail_judge_pass"] == 1
    assert stats["cohen_kappa"] == pytest.approx(0.5)
