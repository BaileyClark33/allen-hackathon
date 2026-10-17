from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from nemoroute.client import ChatClient
from nemoroute.concurrency import TaskReport, run_tasks
from nemoroute.config import Config
from nemoroute.datasets import dataset_paths
from nemoroute.jsonl import append_jsonl, index_by
from nemoroute.labeling.answer import answer_query
from nemoroute.labeling.judge import judge_answer

REFERENCE = "reference"
JUDGE_ROLES = ("judge", "reference_judge")


@dataclass(frozen=True)
class ReferencePaths:
    root: Path

    @property
    def responses(self) -> Path:
        return self.root / "responses.jsonl"

    def judgments(self, judge_role: str) -> Path:
        return self.root / f"judgments_{judge_role}.jsonl"


def reference_paths(cfg: Config) -> ReferencePaths:
    return ReferencePaths(cfg.path("data/reference"))


def tier_judgments_path(cfg: Config, judge_role: str) -> Path:
    """Judgments of the Nemotron tier answers. The default judge's file is the labeling file itself."""
    full = dataset_paths(cfg, "full")
    return full.judgments if judge_role == "judge" else full.root / f"judgments_{judge_role}.jsonl"


@dataclass(frozen=True)
class JudgeJob:
    query_id: str
    query: str
    tier: int | str
    answer: str
    out_path: Path


def pending_reference_answers(items: list[dict[str, Any]], responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    done = set(index_by(responses, "query_id"))
    return [i for i in items if i["query_id"] not in done]


def pending_judge_jobs(
    items: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    judged: list[dict[str, Any]],
    tiers: list[int | str],
    out_path: Path,
) -> list[JudgeJob]:
    answered = index_by(answers, "query_id", "tier")
    done = set(index_by(judged, "query_id", "tier"))
    jobs = []
    for item in items:
        for tier in tiers:
            key = (item["query_id"], tier)
            if key in answered and key not in done:
                jobs.append(JudgeJob(item["query_id"], item["query"], tier, answered[key]["answer"], out_path))
    return jobs


def run_reference_answers(
    cfg: Config, client: ChatClient, system_prompt: str, items: list[dict[str, Any]], out_path: Path
) -> TaskReport:
    return run_tasks(
        items,
        lambda i: answer_query(client, system_prompt, i["query_id"], i["query"], REFERENCE, REFERENCE),
        lambda _i, record: append_jsonl(out_path, record),
        cfg.labeling.concurrency,
        describe=lambda i: f"reference answer {i['query_id']}",
    )


def run_judge_jobs(
    cfg: Config, client: ChatClient, template: Template, jobs: list[JudgeJob], judge_role: str
) -> TaskReport:
    return run_tasks(
        jobs,
        lambda j: judge_answer(client, template, cfg.labeling.criteria, j.query_id, j.query, j.tier, j.answer, judge_role),
        lambda j, record: append_jsonl(j.out_path, record),
        cfg.labeling.concurrency,
        describe=lambda j: f"{judge_role} {j.query_id} tier {j.tier}",
    )
