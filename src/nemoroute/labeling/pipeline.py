from __future__ import annotations

from dataclasses import dataclass
from string import Template
from typing import Any

from nemoroute.client import ChatClient
from nemoroute.concurrency import TaskReport, run_tasks
from nemoroute.config import Config
from nemoroute.datasets import DatasetPaths
from nemoroute.jsonl import append_jsonl, index_by, read_jsonl
from nemoroute.labeling.answer import answer_query
from nemoroute.labeling.judge import judge_answer


@dataclass(frozen=True)
class AnswerTask:
    query_id: str
    query: str
    tier: int


@dataclass(frozen=True)
class JudgeTask:
    query_id: str
    query: str
    tier: int
    answer: str


def select_queries(queries: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return queries if limit is None else queries[:limit]


def pending_answers(
    selected: list[dict[str, Any]], responses: list[dict[str, Any]], tiers: list[int]
) -> list[AnswerTask]:
    done = set(index_by(responses, "query_id", "tier"))
    return [AnswerTask(q["id"], q["query"], t) for q in selected for t in tiers if (q["id"], t) not in done]


def pending_judgments(
    selected: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    tiers: list[int],
) -> list[JudgeTask]:
    answered = index_by(responses, "query_id", "tier")
    judged = set(index_by(judgments, "query_id", "tier"))
    tasks = []
    for q in selected:
        for t in tiers:
            key = (q["id"], t)
            if key in answered and key not in judged:
                tasks.append(JudgeTask(q["id"], q["query"], t, answered[key]["answer"]))
    return tasks


def planned_calls(cfg: Config, answers: list[AnswerTask], judges: list[JudgeTask]) -> dict[str, int]:
    calls = {role: 0 for role in cfg.tier_roles.values()}
    for task in answers:
        calls[cfg.tier_roles[task.tier]] += 1
    calls["judge"] = len(judges) + len(answers)
    return calls


def run_answer_stage(
    cfg: Config, client: ChatClient, paths: DatasetPaths, system_prompt: str, tasks: list[AnswerTask]
) -> TaskReport:
    def worker(task: AnswerTask) -> dict[str, Any]:
        return answer_query(client, system_prompt, task.query_id, task.query, task.tier, cfg.tier_roles[task.tier])

    return run_tasks(
        tasks,
        worker,
        lambda _task, record: append_jsonl(paths.responses, record),
        cfg.labeling.concurrency,
        describe=lambda t: f"answer {t.query_id} tier {t.tier}",
    )


def run_judge_stage(
    cfg: Config, client: ChatClient, paths: DatasetPaths, template: Template, tasks: list[JudgeTask]
) -> TaskReport:
    def worker(task: JudgeTask) -> dict[str, Any]:
        return judge_answer(
            client, template, cfg.labeling.criteria, task.query_id, task.query, task.tier, task.answer
        )

    return run_tasks(
        tasks,
        worker,
        lambda _task, record: append_jsonl(paths.judgments, record),
        cfg.labeling.concurrency,
        describe=lambda t: f"judge {t.query_id} tier {t.tier}",
    )


def load_stage_files(paths: DatasetPaths) -> tuple[list[dict], list[dict], list[dict]]:
    return read_jsonl(paths.queries), read_jsonl(paths.responses), read_jsonl(paths.judgments)
