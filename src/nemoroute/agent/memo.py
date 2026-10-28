from __future__ import annotations

import json
import time
from dataclasses import dataclass

from nemoroute.client import ChatClient, ChatResult
from nemoroute.config import Config
from nemoroute.parsing import extract_json
from nemoroute.prompts import load_prompt
from nemoroute.routers.base import RoutingDecision, Router
from nemoroute.routers.registry import build_router
from nemoroute.search import SearchClient, Source

VALID_STRATEGIES = frozenset({"routed", "all_ultra", "all_nano"})

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_index": {"type": "integer", "minimum": 1},
                    "fact": {"type": "string"},
                },
                "required": ["source_index", "fact"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


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
class Evidence:
    """Plan + search output for one question. Passing the same Evidence to every strategy makes
    them see identical sources, which the memo benchmark needs for a fair comparison."""

    queries: list[str]
    sources: list[Source]
    plan_step: StepTrace
    search_step: StepTrace


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


def _step_from_chat(result: ChatResult, name: str, tier: int) -> StepTrace:
    return StepTrace(
        name=name,
        tier=tier,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_s=result.latency_s,
    )


def _search_step_from_latency(latency_s: float) -> StepTrace:
    return StepTrace(
        name="search",
        tier=0,
        model="tavily",
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        latency_s=latency_s,
    )


def _step_role(strategy: str, routed_tier: int | None, step: str) -> str:
    """Determine which role to use for a given step in a given strategy."""
    if strategy == "all_ultra":
        return "tier3"
    if strategy == "all_nano":
        return "tier1"
    # routed
    if step in ("plan", "extract"):
        return "tier1"
    if step == "synthesize":
        return f"tier{routed_tier}"
    raise ValueError(f"Unknown step: {step}")


def _step_tier(strategy: str, routed_tier: int | None, step: str) -> int:
    role = _step_role(strategy, routed_tier, step)
    return int(role.replace("tier", ""))


def _format_sources(sources: list[Source]) -> str:
    lines: list[str] = []
    for i, src in enumerate(sources, start=1):
        lines.append(f"[{i}] {src.title}")
        lines.append(f"    URL: {src.url}")
        lines.append(f"    {src.snippet}")
    return "\n\n".join(lines)


def _format_facts(facts: list[dict], sources: list[Source]) -> str:
    # Keep only facts that point at a real source so the memo has no dangling citations.
    lines = [
        f"[{fact['source_index']}] {fact['fact']}"
        for fact in facts
        if 1 <= fact["source_index"] <= len(sources)
    ]
    return "\n".join(lines)


def _parse_queries(content: str) -> list[str]:
    parsed = extract_json(content)
    queries = parsed.get("queries", [])
    if not isinstance(queries, list) or len(queries) == 0:
        raise ValueError("Plan step did not return any queries")
    return [str(q) for q in queries]


def _parse_facts(content: str) -> list[dict]:
    parsed = extract_json(content)
    facts = parsed.get("facts", [])
    if not isinstance(facts, list):
        raise ValueError("Extract step did not return a facts list")
    return [
        {"source_index": int(f["source_index"]), "fact": str(f["fact"])}
        for f in facts
    ]


def gather_evidence(
    question: str,
    cfg: Config,
    client: ChatClient,
    search: SearchClient,
    role: str = "tier1",
) -> Evidence:
    """Plan search queries with `role`, run them, and dedupe sources by URL."""
    plan_prompt = load_prompt(cfg, "plan").substitute(max_searches=str(cfg.agent.max_searches))
    plan_messages = [
        {"role": "system", "content": plan_prompt},
        {"role": "user", "content": question},
    ]
    plan_result = client.chat(role, plan_messages, step="plan", json_schema=PLAN_SCHEMA)
    queries = _parse_queries(plan_result.content)[: cfg.agent.max_searches]
    plan_step = _step_from_chat(plan_result, "plan", int(role.replace("tier", "")))

    search_start = time.perf_counter()
    all_sources: list[Source] = []
    seen_urls: set[str] = set()
    for query in queries:
        sources = search.search(query)
        for src in sources:
            if src.url not in seen_urls:
                seen_urls.add(src.url)
                all_sources.append(src)
    search_step = _search_step_from_latency(time.perf_counter() - search_start)
    return Evidence(queries=queries, sources=all_sources, plan_step=plan_step, search_step=search_step)


def run_memo(
    question: str,
    strategy: str,
    cfg: Config,
    client: ChatClient,
    search: SearchClient,
    router: Router | None = None,
    *,
    evidence: Evidence | None = None,
) -> MemoResult:
    """Run the memo pipeline. With `evidence`, the plan and search steps are reused instead of
    re-run, and their recorded traces (and cost) are included in this memo's steps."""
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Must be one of: {', '.join(sorted(VALID_STRATEGIES))}"
        )

    steps: list[StepTrace] = []
    route_tier: int | None = None
    route_reasoning: str | None = None

    # --- Step 1: Route (only for "routed" strategy) ---
    if strategy == "routed":
        if router is None:
            router = build_router(cfg, client, step="route")
        decision: RoutingDecision = router.route(question)
        route_tier = decision.tier
        route_reasoning = decision.reasoning
        steps.append(
            StepTrace(
                name="route",
                tier=route_tier,
                model="nano",  # the router model (role "router" maps to nano)
                input_tokens=0,
                output_tokens=0,
                cost_usd=decision.cost_usd,
                latency_s=decision.latency_s,
            )
        )

    # --- Steps 2–3: Plan + Search (reused when evidence is supplied) ---
    if evidence is None:
        evidence = gather_evidence(question, cfg, client, search, role=_step_role(strategy, route_tier, "plan"))
    all_sources = list(evidence.sources)
    steps.append(evidence.plan_step)
    steps.append(evidence.search_step)

    # --- Step 4: Extract ---
    extract_role = _step_role(strategy, route_tier, "extract")
    extract_prompt = load_prompt(cfg, "extract").safe_substitute()
    sources_text = _format_sources(all_sources)
    extract_messages = [
        {"role": "system", "content": extract_prompt},
        {"role": "user", "content": f"Question: {question}\n\nSources:\n{sources_text}"},
    ]
    extract_result = client.chat(extract_role, extract_messages, step="extract", json_schema=EXTRACT_SCHEMA)
    facts = _parse_facts(extract_result.content)
    steps.append(_step_from_chat(extract_result, "extract", _step_tier(strategy, route_tier, "extract")))

    # --- Step 5: Synthesize ---
    synthesize_role = _step_role(strategy, route_tier, "synthesize")
    synthesize_prompt = load_prompt(cfg, "synthesize").safe_substitute()
    facts_text = _format_facts(facts, all_sources)
    synthesize_messages = [
        {"role": "system", "content": synthesize_prompt},
        {"role": "user", "content": f"Question: {question}\n\nFacts:\n{facts_text}"},
    ]
    synthesize_result = client.chat(synthesize_role, synthesize_messages, step="synthesize")
    memo_md = synthesize_result.content
    steps.append(_step_from_chat(synthesize_result, "synthesize", _step_tier(strategy, route_tier, "synthesize")))

    total_cost_usd = sum(s.cost_usd for s in steps)
    total_latency_s = sum(s.latency_s for s in steps)

    return MemoResult(
        question=question,
        strategy=strategy,
        memo_md=memo_md,
        sources=all_sources,
        steps=steps,
        total_cost_usd=total_cost_usd,
        total_latency_s=total_latency_s,
        route_tier=route_tier,
        route_reasoning=route_reasoning,
    )