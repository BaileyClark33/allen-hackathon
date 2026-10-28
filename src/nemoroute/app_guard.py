from __future__ import annotations

from dataclasses import dataclass

from nemoroute.config import AppConfig, Config
from nemoroute.pricing import call_cost
from nemoroute.usage_log import UsageLog


@dataclass
class GuardDecision:
    allowed: bool
    message: str = ""


def worst_case_query_cost(cfg: Config) -> float:
    """Upper bound for one memo: every LLM step at the top tier's max_tokens, sources included."""
    top_role = cfg.tier_roles[max(cfg.tiers())]
    top_model = cfg.model_for_role(top_role)
    prompt_tokens = cfg.app.max_query_chars // 2 + 600
    source_tokens = cfg.agent.max_searches * cfg.agent.results_per_search * 300
    llm_steps = 3 * call_cost(top_model, prompt_tokens + source_tokens, cfg.roles[top_role].max_tokens)
    return call_cost(top_model, prompt_tokens, cfg.roles["router"].max_tokens) + llm_steps


def check_query_allowed(
    app: AppConfig,
    query: str,
    session_count: int,
    spend_today: float,
    spend_total: float,
    worst_case: float,
) -> GuardDecision:
    if not query.strip():
        return GuardDecision(False, "Enter a question first.")
    if len(query) > app.max_query_chars:
        return GuardDecision(False, f"Please keep questions under {app.max_query_chars} characters.")
    if session_count >= app.session_query_cap:
        return GuardDecision(False, f"This demo allows {app.session_query_cap} questions per session. Thanks for trying it!")
    if spend_total + worst_case > app.total_spend_cap_usd:
        return GuardDecision(False, "The demo budget is exhausted. See the README to run it locally with your own key.")
    if spend_today + worst_case > app.daily_spend_cap_usd:
        return GuardDecision(False, "Today's demo budget has been reached. Please try again tomorrow (UTC).")
    return GuardDecision(True)


def guard(cfg: Config, app_log: UsageLog, query: str, session_count: int) -> GuardDecision:
    return check_query_allowed(
        cfg.app, query, session_count, app_log.spend_today_utc(), app_log.total_spend(), worst_case_query_cost(cfg)
    )
