from __future__ import annotations

import argparse
import sys

from nemoroute.agent.memo import run_memo
from nemoroute.budget import RunBudget
from nemoroute.client import TokenFactoryClient
from nemoroute.config import load_config
from nemoroute.search import TavilySearchClient
from nemoroute.usage_log import UsageLog


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the memo agent pipeline")
    parser.add_argument("question", help="The question to research and answer")
    parser.add_argument(
        "--strategy",
        default="routed",
        choices=["routed", "all_ultra", "all_nano"],
        help="Model strategy (default: routed)",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Budget cap in USD for this run",
    )
    args = parser.parse_args()

    cfg = load_config()
    usage_log = UsageLog(cfg.path(cfg.paths.calls_log))
    client = TokenFactoryClient(cfg, usage_log, RunBudget(args.limit))
    search = TavilySearchClient(cfg)

    result = run_memo(
        question=args.question,
        strategy=args.strategy,
        cfg=cfg,
        client=client,
        search=search,
    )

    print("=" * 60)
    print(result.memo_md)
    print("=" * 60)
    print(f"\nSources ({len(result.sources)}):")
    for i, src in enumerate(result.sources, start=1):
        print(f"  [{i}] {src.title} — {src.url}")

    print(f"\n--- Cost & Latency Summary ---")
    print(f"{'Step':<14} {'Tier':>4} {'Model':>30} {'In Tok':>7} {'Out Tok':>8} {'Cost':>9} {'Latency':>9}")
    print("-" * 86)
    for s in result.steps:
        print(
            f"{s.name:<14} {s.tier:>4} {s.model:>30} {s.input_tokens:>7} {s.output_tokens:>8} "
            f"${s.cost_usd:>8.4f} {s.latency_s:>8.2f}s"
        )
    print("-" * 86)
    print(
        f"{'TOTAL':<14} {'':>4} {'':>30} {'':>7} {'':>8} "
        f"${result.total_cost_usd:>8.4f} {result.total_latency_s:>8.2f}s"
    )

    if result.route_tier is not None:
        print(f"\nStrategy: {result.strategy} | Routed to tier {result.route_tier}")
        print(f"Reasoning: {result.route_reasoning}")
    else:
        print(f"\nStrategy: {result.strategy}")


if __name__ == "__main__":
    main()