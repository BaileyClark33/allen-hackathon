from __future__ import annotations

import argparse

from nemoroute.budget import RunBudget
from nemoroute.client import TokenFactoryClient
from nemoroute.config import Config, load_config
from nemoroute.usage_log import UsageLog


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser


def setup(args: argparse.Namespace) -> tuple[Config, UsageLog]:
    cfg = load_config(args.config)
    return cfg, UsageLog(cfg.path(cfg.paths.calls_log))


def make_client(cfg: Config, usage_log: UsageLog, max_cost: float | None) -> TokenFactoryClient:
    return TokenFactoryClient(cfg, usage_log, RunBudget(max_cost))


def print_run_summary(usage_log: UsageLog, client: TokenFactoryClient) -> None:
    rows = [r for r in usage_log.rows() if r.get("run_id") == client.run_id]
    ok = [r for r in rows if r.get("ok")]
    cost = sum(r["cost_usd"] for r in rows)
    print(f"\nRun {client.run_id}: {len(ok)} ok calls, {len(rows) - len(ok)} failed, actual spend ${cost:.4f}")
    print(f"Lifetime logged spend: ${usage_log.total_spend():.4f}")
