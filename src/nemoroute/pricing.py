from __future__ import annotations

from dataclasses import dataclass, field

from nemoroute.config import TOKEN_FACTORY, Config, ModelConfig
from nemoroute.usage_log import UsageLog


def call_cost(model: ModelConfig, input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * model.input_usd_per_mtok + output_tokens * model.output_usd_per_mtok) / 1_000_000


@dataclass
class CostLine:
    role: str
    model_id: str
    calls: int
    input_tokens: float
    output_tokens: float
    usd: float
    source: str
    billing: str = TOKEN_FACTORY


@dataclass
class CostEstimate:
    lines: list[CostLine] = field(default_factory=list)

    @property
    def total_usd(self) -> float:
        return sum(line.usd for line in self.lines)

    def total_for(self, billing: str) -> float:
        return sum(line.usd for line in self.lines if line.billing == billing)

    @property
    def total_calls(self) -> int:
        return sum(line.calls for line in self.lines)

    def render(self) -> str:
        header = f"{'role':<10} {'model':<40} {'calls':>6} {'in tok':>10} {'out tok':>10} {'usd':>9}  basis"
        rows = [header, "-" * len(header)]
        for line in self.lines:
            rows.append(
                f"{line.role:<10} {line.model_id:<40} {line.calls:>6} {line.input_tokens:>10.0f} "
                f"{line.output_tokens:>10.0f} {line.usd:>9.4f}  {line.source}"
            )
        for billing in sorted({line.billing for line in self.lines} - {TOKEN_FACTORY}):
            rows.append(f"  of which billed outside Token Factory ({billing}): ${self.total_for(billing):.4f}")
        rows.append(f"{'total':<10} {'':<40} {self.total_calls:>6} {'':>10} {'':>10} {self.total_usd:>9.4f}")
        return "\n".join(rows)


def estimate(cfg: Config, calls_by_role: dict[str, int], usage_log: UsageLog | None = None) -> CostEstimate:
    observed = usage_log.avg_tokens_by_role(cfg.budget.min_observed_calls) if usage_log else {}
    result = CostEstimate()
    for role, calls in calls_by_role.items():
        if calls <= 0:
            continue
        if role in observed:
            avg_in, avg_out = observed[role]
            source = "observed avg"
        else:
            avg_in, avg_out = cfg.estimates[role].input, cfg.estimates[role].output
            source = "config default"
        model = cfg.model_for_role(role)
        in_tok, out_tok = avg_in * calls, avg_out * calls
        result.lines.append(
            CostLine(
                role, model.id, calls, in_tok, out_tok, call_cost(model, int(in_tok), int(out_tok)), source,
                model.billing,
            )
        )
    return result
