from __future__ import annotations

import argparse
import sys
import threading

from nemoroute.config import TOKEN_FACTORY, Config
from nemoroute.pricing import CostEstimate
from nemoroute.usage_log import UsageLog


class BudgetExceeded(RuntimeError):
    pass


class RunBudget:
    def __init__(self, max_cost_usd: float | None) -> None:
        self.max_cost_usd = max_cost_usd
        self.spent_usd = 0.0
        self._lock = threading.Lock()

    def add(self, cost_usd: float) -> None:
        with self._lock:
            self.spent_usd += cost_usd

    def check(self) -> None:
        if self.max_cost_usd is not None and self.spent_usd >= self.max_cost_usd:
            raise BudgetExceeded(f"Run spend ${self.spent_usd:.4f} reached --max-cost ${self.max_cost_usd:.4f}")

    @property
    def exhausted(self) -> bool:
        return self.max_cost_usd is not None and self.spent_usd >= self.max_cost_usd


def add_budget_args(parser: argparse.ArgumentParser) -> None:
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--limit", type=int, default=None, help="dry-run size (default: budget.dry_run_limit)")
    scope.add_argument("--full", action="store_true", help="process the whole dataset")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument("--max-cost", type=float, default=None, help="abort once this run spends this many USD")


def resolve_limit(args: argparse.Namespace, default_limit: int) -> int | None:
    if args.full:
        return None
    return args.limit if args.limit is not None else default_limit


def confirm_or_exit(cfg: Config, est: CostEstimate, usage_log: UsageLog, yes: bool, label: str) -> None:
    lifetime = usage_log.total_spend()
    remaining = cfg.budget.total_credits_usd - lifetime
    print(f"\n=== Cost estimate: {label} ===")
    print(est.render())
    print(
        f"\nLifetime logged spend: ${lifetime:.4f} of ${cfg.budget.total_credits_usd:.2f} "
        f"(remaining ${remaining:.4f})"
    )
    if est.total_calls == 0:
        print("Nothing to do.")
        return
    if est.total_for(TOKEN_FACTORY) > remaining:
        raise SystemExit("Estimated cost exceeds remaining credits. Aborting.")
    if yes:
        print("--yes given, proceeding.")
        return
    if not sys.stdin.isatty():
        raise SystemExit("Confirmation required but stdin is not interactive. Re-run with --yes to proceed.")
    answer = input(f"Proceed with ~${est.total_usd:.4f} of calls? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted by user.")
