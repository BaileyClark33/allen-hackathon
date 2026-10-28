from __future__ import annotations

from fakes import make_config

from nemoroute.app_guard import check_query_allowed, guard, worst_case_query_cost
from nemoroute.pricing import call_cost
from nemoroute.usage_log import CallRecord, UsageLog, utc_now


def test_guard_allows_normal_query(tmp_path):
    app = make_config(tmp_path).app
    assert check_query_allowed(app, "What is EBITDA?", 0, 0.0, 0.0, 0.005).allowed


def test_guard_blocks_session_cap(tmp_path):
    app = make_config(tmp_path).app
    d = check_query_allowed(app, "q", app.session_query_cap, 0.0, 0.0, 0.005)
    assert not d.allowed and "per session" in d.message


def test_guard_blocks_daily_cap(tmp_path):
    app = make_config(tmp_path).app
    d = check_query_allowed(app, "q", 0, app.daily_spend_cap_usd - 0.001, 1.0, 0.005)
    assert not d.allowed and "Today" in d.message


def test_guard_blocks_lifetime_cap_even_when_daily_is_fine(tmp_path):
    app = make_config(tmp_path).app
    d = check_query_allowed(app, "q", 0, 0.0, app.total_spend_cap_usd - 0.001, 0.005)
    assert not d.allowed and "exhausted" in d.message


def test_guard_blocks_long_and_empty_queries(tmp_path):
    app = make_config(tmp_path).app
    assert not check_query_allowed(app, "x" * (app.max_query_chars + 1), 0, 0, 0, 0).allowed
    assert not check_query_allowed(app, "   ", 0, 0, 0, 0).allowed


def test_guard_reads_spend_from_app_log(tmp_path):
    cfg = make_config(tmp_path)
    log = UsageLog(cfg.path(cfg.paths.app_spend_log))
    assert guard(cfg, log, "q", 0).allowed
    log.record(CallRecord(utc_now(), "app", "app_answer", "tier3", "m", 1, 1, 1.0, cfg.app.daily_spend_cap_usd, True))
    assert not guard(cfg, log, "q", 0).allowed
    assert worst_case_query_cost(cfg) > 0


def test_worst_case_prices_all_three_memo_llm_steps(tmp_path):
    cfg = make_config(tmp_path)
    top_role = cfg.tier_roles[max(cfg.tiers())]
    top_model = cfg.model_for_role(top_role)
    three_full_outputs = 3 * call_cost(top_model, 0, cfg.roles[top_role].max_tokens)
    assert worst_case_query_cost(cfg) >= three_full_outputs
