from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from nemoroute.config import TOKEN_FACTORY
from nemoroute.jsonl import append_jsonl, read_jsonl


@dataclass
class CallRecord:
    ts: str
    run_id: str
    step: str
    role: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    ok: bool
    finish_reason: str | None = None
    error: str | None = None
    billing: str = TOKEN_FACTORY


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class UsageLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, rec: CallRecord) -> None:
        append_jsonl(self.path, asdict(rec))

    def rows(self) -> list[dict]:
        return read_jsonl(self.path)

    def total_spend(self, since: datetime | None = None, billing: str | None = TOKEN_FACTORY) -> float:
        """Logged spend; by default only Token Factory calls, since the credit budget covers only those."""
        total = 0.0
        for row in self.rows():
            if since is not None and datetime.fromisoformat(row["ts"]) < since:
                continue
            if billing is not None and row.get("billing", TOKEN_FACTORY) != billing:
                continue
            total += row.get("cost_usd", 0.0)
        return total

    def spend_today_utc(self) -> float:
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return self.total_spend(since=start)

    def avg_tokens_by_role(
        self, min_calls: int, exclude_steps: tuple[str, ...] = ("probe",)
    ) -> dict[str, tuple[float, float]]:
        sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
        for row in self.rows():
            if not row.get("ok") or row.get("step") in exclude_steps:
                continue
            s = sums[row["role"]]
            s[0] += row["input_tokens"]
            s[1] += row["output_tokens"]
            s[2] += 1
        return {role: (s[0] / s[2], s[1] / s[2]) for role, s in sums.items() if s[2] >= min_calls}
