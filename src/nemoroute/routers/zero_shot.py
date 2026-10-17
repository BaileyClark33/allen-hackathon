from __future__ import annotations

from string import Template

from nemoroute.client import ChatClient
from nemoroute.parsing import JsonParseError, extract_json
from nemoroute.routers.base import RoutingDecision

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {"tier": {"type": "integer", "enum": [1, 2, 3]}, "reasoning": {"type": "string"}},
    "required": ["tier", "reasoning"],
    "additionalProperties": False,
}

TIER_ONLY_SCHEMA = {
    "type": "object",
    "properties": {"tier": {"type": "integer", "enum": [1, 2, 3]}},
    "required": ["tier"],
    "additionalProperties": False,
}


class ZeroShotRouter:
    """Prompted JSON router. Also serves the fine-tuned router, which differs only in role, prompt and schema."""

    def __init__(
        self,
        client: ChatClient,
        template: Template,
        tiers: list[int],
        step: str = "route",
        role: str = "router",
        name: str = "zero_shot_nano",
        schema: dict = ROUTER_SCHEMA,
    ) -> None:
        self.client = client
        self.template = template
        self.tiers = tiers
        self.step = step
        self.role = role
        self.name = name
        self.schema = schema

    def route(self, query: str) -> RoutingDecision:
        prompt = self.template.substitute(query=query)
        result = self.client.chat(
            self.role, [{"role": "user", "content": prompt}], step=self.step, json_schema=self.schema
        )
        raw = {"content": result.content, "finish_reason": result.finish_reason}
        try:
            payload = extract_json(result.content)
            tier = int(payload["tier"])
            if tier not in self.tiers:
                raise ValueError(f"tier {tier} not in {self.tiers}")
            reasoning = str(payload.get("reasoning", "")).strip()
            fallback = False
        except (JsonParseError, KeyError, TypeError, ValueError) as exc:
            tier = max(self.tiers)
            reasoning = f"Router output could not be parsed ({type(exc).__name__}); defaulting to the largest tier."
            fallback = True
        return RoutingDecision(
            tier=tier,
            reasoning=reasoning,
            cost_usd=result.cost_usd,
            latency_s=result.latency_s,
            fallback=fallback,
            raw=raw,
        )
