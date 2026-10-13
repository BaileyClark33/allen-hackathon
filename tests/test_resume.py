from __future__ import annotations

import json
from string import Template

from fakes import FakeClient, make_config

from nemoroute.datasets import dataset_paths
from nemoroute.jsonl import read_jsonl, write_jsonl
from nemoroute.labeling.pipeline import (
    load_stage_files,
    pending_answers,
    pending_judgments,
    run_answer_stage,
    run_judge_stage,
)

JUDGE_JSON = json.dumps({"accuracy": 5, "completeness": 4, "structure": 4, "actionability": 4, "rationale": "ok"})


def responder(role, messages):
    return JUDGE_JSON if role == "judge" else f"answer from {role}"


def test_labeling_resumes_without_repeating_calls(tmp_path):
    cfg = make_config(tmp_path)
    paths = dataset_paths(cfg, "calibration")
    write_jsonl(paths.queries, [{"id": f"cal-{i}", "query": f"q{i}", "intended_tier": 1} for i in range(3)])
    write_jsonl(paths.responses, [{"query_id": "cal-0", "tier": 1, "answer": "pre-existing", "cost_usd": 0, "latency_s": 0}])
    template = Template("$query $answer")
    client = FakeClient(responder)

    queries, responses, judgments = load_stage_files(paths)
    answers = pending_answers(queries, responses, cfg.tiers())
    assert len(answers) == 8
    run_answer_stage(cfg, client, paths, "system", answers)
    queries, responses, judgments = load_stage_files(paths)
    run_judge_stage(cfg, client, paths, template, pending_judgments(queries, responses, judgments, cfg.tiers()))
    first_calls = len(client.calls)
    assert first_calls == 8 + 9

    queries, responses, judgments = load_stage_files(paths)
    assert pending_answers(queries, responses, cfg.tiers()) == []
    assert pending_judgments(queries, responses, judgments, cfg.tiers()) == []
    assert len(read_jsonl(paths.judgments)) == 9


def test_failed_calls_are_retried_on_next_run(tmp_path):
    cfg = make_config(tmp_path)
    paths = dataset_paths(cfg, "calibration")
    write_jsonl(paths.queries, [{"id": "cal-0", "query": "q", "intended_tier": 1}])
    state = {"fail": True}

    def flaky(role, messages):
        if state["fail"] and role == "tier2":
            raise RuntimeError("boom")
        return "fine"

    queries, responses, _ = load_stage_files(paths)
    report = run_answer_stage(cfg, FakeClient(flaky), paths, "s", pending_answers(queries, responses, cfg.tiers()))
    assert report.failed == 1 and report.done == 2
    _, responses, _ = load_stage_files(paths)
    remaining = pending_answers(queries, responses, cfg.tiers())
    assert [(t.query_id, t.tier) for t in remaining] == [("cal-0", 2)]
    state["fail"] = False
    run_answer_stage(cfg, FakeClient(flaky), paths, "s", remaining)
    _, responses, _ = load_stage_files(paths)
    assert pending_answers(queries, responses, cfg.tiers()) == []
