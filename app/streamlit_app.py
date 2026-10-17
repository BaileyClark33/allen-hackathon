from __future__ import annotations

import os

import streamlit as st

from nemoroute.app_guard import guard
from nemoroute.client import TokenFactoryClient
from nemoroute.config import Config, load_config
from nemoroute.labeling.answer import answer_messages
from nemoroute.pricing import call_cost
from nemoroute.prompts import load_prompt
from nemoroute.routers.base import Router, RoutingDecision
from nemoroute.routers.registry import build_router
from nemoroute.usage_log import UsageLog

TIER_NAMES = {1: "Tier 1 - Nemotron Nano", 2: "Tier 2 - Nemotron Super", 3: "Tier 3 - Nemotron Ultra"}
EXAMPLES = [
    "What is the difference between EBITDA and operating cash flow?",
    "Run a SWOT analysis for a mid-sized regional grocery chain facing competition from discount entrants.",
    "Estimate the US market size for at-home EV chargers in 2030 and recommend whether a utility should enter.",
]


ROUTER_LABELS = {"finetuned": "Fine-tuned Nano (LoRA)", "zero_shot": "Zero-shot Nano (prompted)"}
FT_BASE_URL_ENV = "ROUTER_FT_BASE_URL"


def load_secrets(cfg: Config) -> None:
    ft = cfg.models["nano_ft"]
    for name in (cfg.api.api_key_env, ft.api_key_env, FT_BASE_URL_ENV):
        if not name or os.environ.get(name):
            continue
        try:
            value = st.secrets.get(name)
        except Exception:
            value = None
        if value:
            os.environ[name] = value
    if ft.base_url is None and os.environ.get(FT_BASE_URL_ENV):
        ft.base_url = os.environ[FT_BASE_URL_ENV]


@st.cache_resource
def resources() -> tuple[Config, UsageLog, TokenFactoryClient, dict[str, Router], str]:
    cfg = load_config()
    load_secrets(cfg)
    app_log = UsageLog(cfg.path(cfg.paths.app_spend_log))
    client = TokenFactoryClient(cfg, app_log, run_id="app")
    # The fine-tuned router is offered only when its vLLM endpoint is configured.
    types = (["finetuned"] if cfg.models["nano_ft"].base_url else []) + ["zero_shot"]
    routers = {t: build_router(cfg, client, step="app_route", router_type=t) for t in types}
    return cfg, app_log, client, routers, load_prompt(cfg, "answer").template


def init_state() -> None:
    defaults = {"count": 0, "actual": 0.0, "ultra": 0.0, "query": ""}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def sidebar(cfg: Config, app_log: UsageLog) -> None:
    s = st.session_state
    st.sidebar.header("This session")
    saved = s.ultra - s.actual
    st.sidebar.metric("Queries", f"{s.count} / {cfg.app.session_query_cap}")
    st.sidebar.metric("Actual cost (router + answer)", f"${s.actual:.5f}")
    st.sidebar.metric("Estimated all-Ultra cost", f"${s.ultra:.5f}")
    pct = f"{saved / s.ultra:.0%}" if s.ultra else "-"
    st.sidebar.metric("Saved vs all-Ultra", f"${saved:.5f}", pct)
    st.sidebar.caption(
        "The all-Ultra estimate prices the same answer's input and output tokens at Ultra rates. "
        "Ultra's own answer length would differ, so treat it as an estimate."
    )
    st.sidebar.divider()
    st.sidebar.caption(
        f"Demo budget: ${app_log.spend_today_utc():.3f} of ${cfg.app.daily_spend_cap_usd:.2f} used today (UTC), "
        f"${app_log.total_spend():.3f} of ${cfg.app.total_spend_cap_usd:.2f} overall."
    )


def main() -> None:
    st.set_page_config(page_title="Nemotron Router", page_icon=":compass:", layout="centered")
    init_state()
    try:
        cfg, app_log, client, routers, system_prompt = resources()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    st.title("Nemotron consulting router")
    st.write(
        "A Nemotron Nano router reads your business-strategy question and sends it to the cheapest Nemotron "
        "model that can answer it well, served by Nebius Token Factory."
    )
    cols = st.columns(len(EXAMPLES))
    for col, example, label in zip(cols, EXAMPLES, ("Simple", "Medium", "Hard")):
        if col.button(f"{label} example", use_container_width=True):
            st.session_state.query = example
    query = st.text_area("Your question", key="query", height=120, max_chars=cfg.app.max_query_chars)
    choice = st.radio("Router", list(routers), format_func=ROUTER_LABELS.get, horizontal=True)
    if st.button("Route and answer", type="primary"):
        run_query(cfg, app_log, client, routers[choice], routers["zero_shot"], system_prompt, query)
    sidebar(cfg, app_log)


def route_with_fallback(router: Router, fallback: Router, query: str) -> tuple[RoutingDecision, str | None]:
    try:
        return router.route(query), None
    except Exception as exc:
        if router is fallback:
            raise
        note = f"The fine-tuned router was unreachable ({type(exc).__name__}), so the zero-shot router was used."
        return fallback.route(query), note


def run_query(
    cfg: Config, app_log: UsageLog, client: TokenFactoryClient, router: Router, fallback: Router,
    system_prompt: str, query: str,
) -> None:
    decision = guard(cfg, app_log, query, st.session_state.count)
    if not decision.allowed:
        st.warning(decision.message)
        return

    try:
        with st.spinner("Routing with Nemotron Nano..."):
            route, note = route_with_fallback(router, fallback, query)
        role = cfg.tier_roles[route.tier]
        model_name = TIER_NAMES.get(route.tier, f"Tier {route.tier}")
        with st.spinner(f"Answering with {model_name}..."):
            result = client.chat(role, answer_messages(system_prompt, query), step="app_answer",
                                 max_tokens=cfg.app.answer_max_tokens)
    except Exception as exc:
        st.error(f"The model call failed ({type(exc).__name__}). Please try again in a moment.")
        return

    ultra = cfg.model_for_role(cfg.tier_roles[max(cfg.tiers())])
    ultra_cost = call_cost(ultra, result.input_tokens, result.output_tokens)
    total_cost = route.cost_usd + result.cost_usd
    total_latency = route.latency_s + result.latency_s
    s = st.session_state
    s.count += 1
    s.actual += total_cost
    s.ultra += ultra_cost

    st.subheader(model_name)
    if note:
        st.info(note)
    if route.reasoning:
        st.caption(f"Router reasoning: {route.reasoning}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cost", f"${total_cost:.5f}")
    c2.metric("Latency", f"{total_latency:.2f}s")
    c3.metric("All-Ultra est.", f"${ultra_cost:.5f}")
    st.caption(
        f"Router: ${route.cost_usd:.6f} in {route.latency_s:.2f}s | "
        f"Answer ({result.model}): ${result.cost_usd:.5f} in {result.latency_s:.2f}s, "
        f"{result.input_tokens} in / {result.output_tokens} out tokens"
    )
    st.markdown(result.content or "_The model returned an empty answer._")


if __name__ == "__main__":
    main()
