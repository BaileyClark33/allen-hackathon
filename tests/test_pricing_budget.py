from __future__ import annotations

import pytest
from fakes import make_config

from nemoroute.budget import BudgetExceeded, RunBudget
from nemoroute.pricing import call_cost, estimate
from nemoroute.usage_log import CallRecord, UsageLog


def test_call_cost_uses_per_million_prices(tmp_path):
    cfg = make_config(tmp_path)
    ultra = cfg.models["ultra"]
    assert call_cost(ultra, 1_000_000, 0) == pytest.approx(ultra.input_usd_per_mtok)
    assert call_cost(ultra, 1000, 2000) == pytest.approx((1000 * 1.0 + 2000 * 3.0) / 1e6)


def test_estimate_uses_config_defaults_then_observed_averages(tmp_path):
    cfg = make_config(tmp_path)
    log = UsageLog(tmp_path / "calls.jsonl")
    default = estimate(cfg, {"judge": 10}, log)
    assert default.lines[0].source == "config default"
    for _ in range(cfg.budget.min_observed_calls):
        log.record(CallRecord("2026-10-13T00:00:00+00:00", "r", "judge", "judge", "m", 100, 200, 1.0, 0.01, True))
    observed = estimate(cfg, {"judge": 10}, log)
    assert observed.lines[0].source == "observed avg"
    assert observed.lines[0].input_tokens == pytest.approx(1000)
    assert observed.lines[0].output_tokens == pytest.approx(2000)


def test_probe_calls_do_not_skew_estimates(tmp_path):
    cfg = make_config(tmp_path)
    log = UsageLog(tmp_path / "calls.jsonl")
    for _ in range(5):
        log.record(CallRecord("2026-10-13T00:00:00+00:00", "r", "probe", "judge", "m", 5, 5, 1.0, 0.0, True))
    assert estimate(cfg, {"judge": 1}, log).lines[0].source == "config default"


def test_run_budget_stops_at_cap():
    budget = RunBudget(0.05)
    budget.add(0.03)
    budget.check()
    budget.add(0.03)
    with pytest.raises(BudgetExceeded):
        budget.check()
