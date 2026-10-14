from __future__ import annotations

from fakes import make_config

from nemoroute.evaluation.charts import confusion_chart, cost_quality_chart, latency_chart
from nemoroute.evaluation.report import render_report, threshold_notice, write_csv
from nemoroute.evaluation.run import evaluate
from nemoroute.labeling.label import UNRESOLVED

HI = {"accuracy": 5, "completeness": 5, "structure": 4, "actionability": 4}
LO = {"accuracy": 3, "completeness": 3, "structure": 3, "actionability": 3}
COST = {1: 0.0002, 2: 0.0007, 3: 0.0025}


def build_fixture():
    items, preds, responses, judgments = [], [], [], []
    labels = [1, 1, 2, 2, 3, 3, UNRESOLVED, 1, 2, 3]
    predicted = [1, 2, 2, 1, 3, 3, 3, 1, 3, 2]
    for i, (label, pred) in enumerate(zip(labels, predicted)):
        qid = f"q-{i:04d}"
        items.append({"query_id": qid, "query": f"q{i}", "label": label})
        preds.append({"query_id": qid, "tier": pred, "cost_usd": 0.00003, "latency_s": 0.3, "fallback": False})
        for tier in (1, 2, 3):
            ok = label != UNRESOLVED and tier >= label
            responses.append({"query_id": qid, "tier": tier, "cost_usd": COST[tier], "latency_s": float(tier * 2)})
            scores = HI if ok else LO
            judgments.append({"query_id": qid, "tier": tier, "scores": scores,
                              "mean_score": sum(scores.values()) / 4})
    return items, preds, responses, judgments


def test_end_to_end_report_and_charts(tmp_path):
    cfg = make_config(tmp_path)
    items, preds, responses, judgments = build_fixture()
    result = evaluate(cfg, "zero_shot_nano", items, preds, responses, judgments, 4.0, 4)
    assert result.n_unresolved == 1
    assert result.routing.n == 9
    assert result.routing.accuracy == 5 / 9
    by_name = {s.strategy: s for s in result.summaries_resolved}
    assert by_name["oracle"].pass_rate == 1.0
    assert by_name["all-ultra"].pass_rate == 1.0
    assert by_name["all-nano"].mean_cost_usd < by_name["zero_shot_nano"].mean_cost_usd < by_name["all-ultra"].mean_cost_usd

    charts = tmp_path / "charts"
    charts.mkdir()
    confusion_chart(result.routing.confusion, cfg.tiers(), "t", charts / "c.png")
    cost_quality_chart(result.summaries_resolved, charts / "q.png")
    latency_chart(result.summaries_resolved, charts / "l.png")
    assert all((charts / f).stat().st_size > 0 for f in ("c.png", "q.png", "l.png"))

    md = render_report(result, cfg.tiers(), None, 4.0, 4, "Test", None, {"c": "charts/c.png"})
    assert "Unlocked thresholds" in md and "Under-routing rate" in md and "Unresolved" in md
    write_csv(result, tmp_path / "results.csv")
    assert (tmp_path / "results.csv").read_text().count("\n") == 9


def test_threshold_change_notice():
    history = [{"ts": "t0", "pass_threshold": 4.0, "min_accuracy": 4}]
    assert threshold_notice(history, (4.0, 4)) is None
    assert "changed after test evaluation" in threshold_notice(history, (3.75, 4))
