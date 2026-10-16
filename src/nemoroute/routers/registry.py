from __future__ import annotations

from typing import Callable

from nemoroute.client import ChatClient
from nemoroute.config import Config
from nemoroute.prompts import load_prompt
from nemoroute.routers.base import Router
from nemoroute.routers.zero_shot import TIER_ONLY_SCHEMA, ZeroShotRouter

RouterFactory = Callable[[Config, ChatClient, str], Router]


def _zero_shot(cfg: Config, client: ChatClient, step: str) -> Router:
    return ZeroShotRouter(client, load_prompt(cfg, "router"), cfg.tiers(), step=step)


def _finetuned(cfg: Config, client: ChatClient, step: str) -> Router:
    return ZeroShotRouter(
        client, load_prompt(cfg, "router_ft"), cfg.tiers(), step=step,
        role="router_ft", name="finetuned_nano", schema=TIER_ONLY_SCHEMA,
    )


ROUTERS: dict[str, RouterFactory] = {"zero_shot": _zero_shot, "finetuned": _finetuned}


def register_router(name: str, factory: RouterFactory) -> None:
    ROUTERS[name] = factory


def build_router(cfg: Config, client: ChatClient, step: str = "route", router_type: str | None = None) -> Router:
    router_type = router_type or cfg.router.type
    if router_type not in ROUTERS:
        raise SystemExit(f"Unknown router type '{router_type}'. Registered: {sorted(ROUTERS)}")
    return ROUTERS[router_type](cfg, client, step)
