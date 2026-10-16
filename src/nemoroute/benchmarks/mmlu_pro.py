from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import httpx

from nemoroute.client import ChatClient
from nemoroute.evaluation.system import ItemOutcome

MMLU_PRO_URL = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"
LETTERS = "ABCDEFGHIJ"
ANSWER_RE = re.compile(r"answer(?:\s+is)?\s*[:：]?\s*\(?\**([A-J])\b", re.IGNORECASE)


def download(path: Path, url: str = MMLU_PRO_URL) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    return path


def sample_items(rows: list[dict[str, Any]], categories: list[str], n: int, seed: int) -> list[dict[str, Any]]:
    """Seeded sample split evenly across categories, as query records the router and answer code accept."""
    rng = random.Random(seed)
    per_cat = n // len(categories)
    picked = []
    for i, cat in enumerate(sorted(categories)):
        pool = sorted((r for r in rows if r["category"] == cat), key=lambda r: r["question_id"])
        take = per_cat + (1 if i < n % len(categories) else 0)
        picked += rng.sample(pool, min(take, len(pool)))
    return sorted(
        (
            {
                "query_id": f"mmlu-{r['question_id']}",
                "query": format_question(r["question"], list(r["options"])),
                "answer": r["answer"],
                "n_options": len(r["options"]),
                "category": r["category"],
            }
            for r in picked
        ),
        key=lambda r: r["query_id"],
    )


def format_question(question: str, options: list[str]) -> str:
    lines = [question.strip(), ""]
    lines += [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    return "\n".join(lines)


def extract_choice(text: str, n_options: int) -> str | None:
    matches = [m.upper() for m in ANSWER_RE.findall(text or "")]
    valid = [m for m in matches if LETTERS.index(m) < n_options]
    return valid[-1] if valid else None


def answer_mcq(
    client: ChatClient, system_prompt: str, item: dict[str, Any], tier: int | str, role: str
) -> dict[str, Any]:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": item["query"]}]
    result = client.chat(role, messages, step="public_bench")
    choice = extract_choice(result.content, item["n_options"])
    return {
        "query_id": item["query_id"],
        "tier": tier,
        "model": result.model,
        "answer": result.content,
        "choice": choice,
        "correct": choice == item["answer"],
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_s": round(result.latency_s, 3),
        "cost_usd": result.cost_usd,
        "finish_reason": result.finish_reason,
    }


def oracle_choice(responses: dict[tuple[str, int | str], dict[str, Any]], qid: str, tiers: list[int]) -> int:
    """Cheapest tier that answered correctly; the top tier when none did."""
    for tier in sorted(tiers):
        if responses[(qid, tier)]["correct"]:
            return tier
    return max(tiers)


def mcq_outcomes(
    chosen: dict[str, int | str],
    responses: dict[tuple[str, int | str], dict[str, Any]],
    router_overhead: dict[str, tuple[float, float]] | None = None,
) -> list[ItemOutcome]:
    outcomes = []
    for qid, tier in chosen.items():
        r = responses[(qid, tier)]
        extra_cost, extra_latency = (router_overhead or {}).get(qid, (0.0, 0.0))
        outcomes.append(
            ItemOutcome(
                query_id=qid,
                tier=tier,
                cost_usd=r["cost_usd"] + extra_cost,
                latency_s=r["latency_s"] + extra_latency,
                mean_score=float(r["correct"]),
                passed=bool(r["correct"]),
            )
        )
    return outcomes
