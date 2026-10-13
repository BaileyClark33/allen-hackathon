from __future__ import annotations

from string import Template
from typing import Any

from nemoroute.client import ChatClient
from nemoroute.parsing import JsonParseError, extract_json


def judge_schema(criteria: list[str]) -> dict[str, Any]:
    properties: dict[str, Any] = {c: {"type": "integer", "minimum": 1, "maximum": 5} for c in criteria}
    properties["rationale"] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": [*criteria, "rationale"],
        "additionalProperties": False,
    }


def parse_scores(payload: Any, criteria: list[str]) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise JsonParseError(f"Judge output is not an object: {payload!r}")
    scores = {}
    for criterion in criteria:
        try:
            value = int(payload[criterion])
        except (KeyError, TypeError, ValueError) as exc:
            raise JsonParseError(f"Judge output missing or invalid '{criterion}': {payload!r}") from exc
        if not 1 <= value <= 5:
            raise JsonParseError(f"Judge score out of range for '{criterion}': {value}")
        scores[criterion] = value
    return scores


def empty_answer_judgment(query_id: str, tier: int, criteria: list[str]) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "tier": tier,
        "scores": {c: 1 for c in criteria},
        "mean_score": 1.0,
        "rationale": "Empty answer (no content returned); scored without a judge call.",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_s": 0.0,
        "cost_usd": 0.0,
        "judge_called": False,
    }


def judge_answer(
    client: ChatClient,
    template: Template,
    criteria: list[str],
    query_id: str,
    query: str,
    tier: int,
    answer: str,
) -> dict[str, Any]:
    if not answer.strip():
        return empty_answer_judgment(query_id, tier, criteria)
    prompt = template.substitute(query=query, answer=answer)
    result = client.chat(
        "judge", [{"role": "user", "content": prompt}], step="judge", json_schema=judge_schema(criteria)
    )
    payload = extract_json(result.content)
    scores = parse_scores(payload, criteria)
    return {
        "query_id": query_id,
        "tier": tier,
        "scores": scores,
        "mean_score": sum(scores.values()) / len(scores),
        "rationale": str(payload.get("rationale", "")).strip(),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_s": round(result.latency_s, 3),
        "cost_usd": result.cost_usd,
        "judge_called": True,
    }
