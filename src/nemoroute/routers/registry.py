from __future__ import annotations

from typing import Callable

from nemoroute.client import ChatClient
from nemoroute.config import Config
from nemoroute.prompts import load_prompt
from nemoroute.routers.base import Router
from nemoroute.routers.zero_shot import ZeroShotRouter

RouterFactory = Callable[[Config, ChatClient, str], Router]


def _zero_shot(cfg: Config, client: ChatClient, step: str) -> Router:
    return ZeroShotRouter(client, load_prompt(cfg, "router"), cfg.tiers(), step=step)


ROUTERS: dict[str, RouterFactory] = {"zero_shot": _zero_shot}


def register_router(name: str, factory: RouterFactory) -> None:
    ROUTERS[name] = factory


def build_router(cfg: Config, client: ChatClient, step: str = "route") -> Router:
    if cfg.router.type not in ROUTERS:
        raise SystemExit(f"Unknown router type '{cfg.router.type}'. Registered: {sorted(ROUTERS)}")
    return ROUTERS[cfg.router.type](cfg, client, step)
