from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class RoutingDecision:
    tier: int
    reasoning: str
    confidence: float | None = None
    cost_usd: float = 0.0
    latency_s: float = 0.0
    fallback: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


class Router(Protocol):
    name: str

    def route(self, query: str) -> RoutingDecision: ...
