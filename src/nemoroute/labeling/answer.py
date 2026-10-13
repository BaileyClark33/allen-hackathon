from __future__ import annotations

from typing import Any

from nemoroute.client import ChatClient


def answer_messages(system_prompt: str, query: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]


def answer_query(
    client: ChatClient, system_prompt: str, query_id: str, query: str, tier: int, role: str
) -> dict[str, Any]:
    result = client.chat(role, answer_messages(system_prompt, query), step="answer")
    return {
        "query_id": query_id,
        "tier": tier,
        "model": result.model,
        "answer": result.content,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_s": round(result.latency_s, 3),
        "cost_usd": result.cost_usd,
        "finish_reason": result.finish_reason,
    }
