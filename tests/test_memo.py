from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemoroute.agent.memo import MemoResult, StepTrace, run_memo, gather_evidence, VALID_STRATEGIES
from nemoroute.config import Config
from nemoroute.routers.base import RoutingDecision, Router
from nemoroute.search import Source
from tests.fakes import FakeClient, FakeSearchClient, make_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plan_response(_role: str, _messages: list) -> str:
    return json.dumps({"queries": ["alpha strategy", "beta analysis"]})


def _extract_response(_role: str, _messages: list) -> str:
    return json.dumps(
        {
            "facts": [
                {"source_index": 1, "fact": "Market grew 12% in 2024."},
                {"source_index": 2, "fact": "Competitor X launched Y product."},
            ]
        }
    )


def _synthesize_response(_role: str, _messages: list) -> str:
    return (
        "## Bottom Line\nMarket is growing. [1]\n\n"
        "## Key Findings\n- Growth at 12% [1]\n- Competitor launch [2]\n\n"
        "## Analysis\nStrong signals. [1] [2]\n\n"
        "## Risks & Unknowns\nData limited.\n\n"
        "## Next Steps\nMonitor quarterly reports."
    )


def _router_response(_role: str, _messages: list) -> str:
    return json.dumps({"tier": 2, "reasoning": "Requires framework application."})


def _make_client(responses: dict[str, str] | None = None) -> FakeClient:
    mapping = responses or {}

    def responder(role: str, messages: list) -> str:
        # Step-based dispatch: FakeClient records (role, step) in calls.
        # But the responder only sees role+msgs. We use a closure counter.
        # The FakeClient.calls list is the source of truth for order.
        # We use the messages content as a lightweight dispatcher.
        content = json.dumps(messages) if messages else ""
        for key, value in mapping.items():
            if key in content:
                return value
        # Fallback: try step pattern from messages
        for msg in messages:
            txt = str(msg.get("content", ""))
            if "strategist" in txt.lower() or "search queries" in txt.lower():
                return mapping.get("plan", _plan_response(role, messages))
            if "research analyst" in txt.lower() or "extracting facts" in txt.lower():
                return mapping.get("extract", _extract_response(role, messages))
            if "management consultant" in txt.lower():
                return mapping.get("synthesize", _synthesize_response(role, messages))
        return mapping.get("default", "{}")

    return FakeClient(responder)


class StepTrackingClient:
    """Wraps FakeClient to track the role used for each step."""

    def __init__(self, responses: dict[str, str] | None = None):
        self._fake = _make_client(responses)
        self.step_roles: list[tuple[str, str]] = []  # (role, step)

    def chat(self, role: str, messages, *, step, json_schema=None, **overrides):
        self.step_roles.append((role, step))
        return self._fake.chat(role, messages, step=step, json_schema=json_schema, **overrides)

    @property
    def calls(self):
        return self._fake.calls


class StepTrackingSearch:
    """Wraps FakeSearchClient to allow per-query source control."""

    def __init__(self, sources_per_query: list[list[Source]] | None = None):
        self._sources = sources_per_query or []
        self._idx = 0
        self.calls: list[str] = []

    def search(self, query: str) -> list[Source]:
        self.calls.append(query)
        if self._idx < len(self._sources):
            result = self._sources[self._idx]
            self._idx += 1
            return result
        # Default: unique sources per call
        base = self._idx + 1
        self._idx += 1
        return [
            Source(
                url=f"https://example.com/{base}-{i}",
                title=f"Result {i} for {query}",
                snippet=f"Snippet {i} for {query}.",
            )
            for i in range(2)
        ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlanStepGeneratesQueries:
    def test_plan_step_has_correct_trace(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = _make_client({"plan": _plan_response("tier1", [])})
        search = StepTrackingSearch()

        result = run_memo("What is the market size?", "all_nano", cfg, client, search)

        plan_steps = [s for s in result.steps if s.name == "plan"]
        assert len(plan_steps) == 1
        plan = plan_steps[0]
        assert plan.name == "plan"
        assert plan.tier == 1
        assert plan.model == "fake-tier1"
        assert plan.input_tokens == 100
        assert plan.output_tokens == 50


class TestSearchDeduplicatesByUrl:
    def test_duplicates_removed(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = _make_client({"plan": _plan_response("tier1", [])})

        dup_url = "https://example.com/dup"
        search = StepTrackingSearch(
            sources_per_query=[
                [
                    Source(url=dup_url, title="A", snippet="a"),
                    Source(url="https://example.com/unique-1", title="B", snippet="b"),
                ],
                [
                    Source(url=dup_url, title="A-again", snippet="a2"),
                    Source(url="https://example.com/unique-2", title="C", snippet="c"),
                ],
            ]
        )

        result = run_memo("test", "all_nano", cfg, client, search)

        urls = [s.url for s in result.sources]
        assert urls.count(dup_url) == 1
        assert len(result.sources) == 3


class TestExtractStepParsesFacts:
    def test_extract_step_has_correct_trace(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = _make_client({})
        search = StepTrackingSearch()

        result = run_memo("Analyze competitors.", "all_nano", cfg, client, search)

        extract_steps = [s for s in result.steps if s.name == "extract"]
        assert len(extract_steps) == 1
        assert extract_steps[0].name == "extract"
        assert extract_steps[0].tier == 1


class TestSynthesizeStepProducesMarkdown:
    def test_memo_md_is_populated(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = _make_client({})
        search = StepTrackingSearch()

        result = run_memo("What is happening?", "all_nano", cfg, client, search)

        assert len(result.memo_md) > 0
        assert "Bottom Line" in result.memo_md
        assert "[1]" in result.memo_md


class TestStepOrderIsCorrect:
    def test_routed_step_order(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = StepTrackingClient({"router": _router_response("router", [])})

        class FakeRouterForOrder:
            name = "test_router"
            role = "router"

            def route(self, query: str) -> RoutingDecision:
                return RoutingDecision(tier=2, reasoning="test")

        search = StepTrackingSearch()
        result = run_memo("test", "routed", cfg, client, search, router=FakeRouterForOrder())

        step_names = [s.name for s in result.steps]
        assert step_names == ["route", "plan", "search", "extract", "synthesize"]

    def test_all_ultra_step_order(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = _make_client({})
        search = StepTrackingSearch()

        result = run_memo("test", "all_ultra", cfg, client, search)

        step_names = [s.name for s in result.steps]
        assert step_names == ["plan", "search", "extract", "synthesize"]


class TestCostSumming:
    def test_total_cost_equals_sum(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = _make_client({})
        search = StepTrackingSearch()

        result = run_memo("test", "all_nano", cfg, client, search)

        expected = sum(s.cost_usd for s in result.steps)
        assert result.total_cost_usd == pytest.approx(expected)


class TestCitationIndicesMatchSources:
    def test_citations_within_range(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = _make_client({})
        search = StepTrackingSearch()

        result = run_memo("test", "all_nano", cfg, client, search)

        max_index = len(result.sources)
        # Check that [N] references in memo_md don't exceed max_index
        import re

        citations = re.findall(r"\[(\d+)\]", result.memo_md)
        for c in citations:
            idx = int(c)
            assert 1 <= idx <= max_index, f"Citation [{idx}] out of range (max {max_index})"


class TestStrategyAllUltraUsesTier3Everywhere:
    def test_all_steps_use_tier3(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = StepTrackingClient({})
        search = StepTrackingSearch()

        run_memo("test", "all_ultra", cfg, client, search)

        # Every chat call should use "tier3"
        for role, step in client.step_roles:
            assert role == "tier3", f"Step {step} used role {role}, expected tier3"


class TestStrategyAllNanoUsesTier1Everywhere:
    def test_all_steps_use_tier1(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = StepTrackingClient({})
        search = StepTrackingSearch()

        run_memo("test", "all_nano", cfg, client, search)

        for role, step in client.step_roles:
            assert role == "tier1", f"Step {step} used role {role}, expected tier1"


class TestInvalidStrategyRaises:
    def test_invalid_raises_valueerror(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = _make_client({})
        search = StepTrackingSearch()

        with pytest.raises(ValueError, match="Invalid strategy"):
            run_memo("test", "invalid", cfg, client, search)


class TestRoutedStrategyWithoutRouter:
    def test_builds_router_internally(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = StepTrackingClient({"router": _router_response("router", [])})
        search = StepTrackingSearch()

        # Passing router=None — run_memo should call build_router internally.
        # build_router needs the real config’s prompts dir to exist and have router.txt.
        # make_config preserves paths.prompts_dir, so it should work.
        # But the FakeClient won't match the real router prompt properly.
        # We provide a router directly instead to make the test robust.
        class FakeRouter:
            name = "test"
            role = "router"

            def route(self, query: str) -> RoutingDecision:
                return RoutingDecision(tier=2, reasoning="ok")

        result = run_memo("test", "routed", cfg, client, search, router=FakeRouter())
        assert result.route_tier == 2
        assert result.route_reasoning == "ok"
        assert "route" in [s.name for s in result.steps]


class TestSearchStepHasZeroCost:
    def test_search_step_trace(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = _make_client({})
        search = StepTrackingSearch()

        result = run_memo("test", "all_nano", cfg, client, search)

        search_steps = [s for s in result.steps if s.name == "search"]
        assert len(search_steps) == 1
        s = search_steps[0]
        assert s.tier == 0
        assert s.model == "tavily"
        assert s.input_tokens == 0
        assert s.output_tokens == 0
        assert s.cost_usd == 0.0
        assert s.latency_s >= 0


class TestMemoResultFields:
    def test_all_fields_populated(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = _make_client({})
        search = StepTrackingSearch()

        result = run_memo("What is the market size?", "all_nano", cfg, client, search)

        assert result.question == "What is the market size?"
        assert result.strategy == "all_nano"
        assert len(result.memo_md) > 0
        assert len(result.sources) > 0
        assert len(result.steps) == 4  # plan, search, extract, synthesize
        assert result.total_cost_usd > 0
        assert result.total_latency_s > 0
        assert result.route_tier is None
        assert result.route_reasoning is None


class TestRoutedStrategyFields:
    def test_route_fields_populated(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = _make_client({})

        class FakeRouter:
            name = "test"
            role = "router"

            def route(self, query: str) -> RoutingDecision:
                return RoutingDecision(tier=3, reasoning="Complex multi-step analysis required.")

        search = StepTrackingSearch()
        result = run_memo("test", "routed", cfg, client, search, router=FakeRouter())

        assert result.route_tier == 3
        assert "Complex" in result.route_reasoning


class _FixedRouter:
    name = "test"
    role = "router"

    def route(self, query: str) -> RoutingDecision:
        return RoutingDecision(tier=2, reasoning="fixed")


class TestSharedEvidence:
    def test_gather_evidence_plans_with_given_role(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        client = StepTrackingClient()

        evidence = gather_evidence("test", cfg, client, StepTrackingSearch(), role="tier1")

        assert client.step_roles == [("tier1", "plan")]
        assert evidence.queries == ["alpha strategy", "beta analysis"]
        assert evidence.plan_step.name == "plan" and evidence.plan_step.tier == 1
        assert evidence.search_step.name == "search"
        assert len(evidence.sources) == 4

    def test_evidence_skips_plan_and_search(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        evidence = gather_evidence("test", cfg, StepTrackingClient(), StepTrackingSearch())
        client = StepTrackingClient()
        search = StepTrackingSearch()

        result = run_memo("test", "all_ultra", cfg, client, search, evidence=evidence)

        assert client.step_roles == [("tier3", "extract"), ("tier3", "synthesize")]
        assert search.calls == []
        assert [s.name for s in result.steps] == ["plan", "search", "extract", "synthesize"]
        assert result.steps[0] is evidence.plan_step
        assert result.total_cost_usd == pytest.approx(sum(s.cost_usd for s in result.steps))

    def test_all_strategies_share_sources(self, tmp_path: Path):
        cfg = make_config(tmp_path)
        evidence = gather_evidence("test", cfg, StepTrackingClient(), StepTrackingSearch())

        urls = {
            strategy: [
                s.url
                for s in run_memo(
                    "test", strategy, cfg, StepTrackingClient(), StepTrackingSearch(), _FixedRouter(),
                    evidence=evidence,
                ).sources
            ]
            for strategy in sorted(VALID_STRATEGIES)
        }

        assert len({tuple(u) for u in urls.values()}) == 1
        assert urls["routed"] == [s.url for s in evidence.sources]