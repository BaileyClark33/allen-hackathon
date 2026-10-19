from __future__ import annotations

import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from string import Template

from nemoroute.budget import BudgetExceeded
from nemoroute.client import ChatClient
from nemoroute.config import Config
from nemoroute.dataset.dedup import dedup
from nemoroute.datasets import DatasetPaths, all_query_texts
from nemoroute.jsonl import append_jsonl, read_jsonl
from nemoroute.parsing import JsonParseError, extract_json

QUERIES_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class BatchSpec:
    tier: int
    subdomain: str
    persona: str
    n: int


def needed_per_tier(existing: list[dict], target: int, tiers: list[int]) -> dict[int, int]:
    counts = Counter(r["intended_tier"] for r in existing)
    base, extra = divmod(target, len(tiers))
    goals = {t: base + (1 if i < extra else 0) for i, t in enumerate(tiers)}
    return {t: max(0, goals[t] - counts.get(t, 0)) for t in tiers}


def cap_by_limit(needed: dict[int, int], limit: int | None) -> dict[int, int]:
    if limit is None:
        return dict(needed)
    capped = {t: 0 for t in needed}
    remaining = limit
    while remaining > 0 and any(capped[t] < needed[t] for t in needed):
        for t in sorted(needed):
            if remaining > 0 and capped[t] < needed[t]:
                capped[t] += 1
                remaining -= 1
    return capped


def plan_batches(
    needed: dict[int, int],
    batch_size: int,
    subdomains: list[str],
    personas: list[str],
    rng: random.Random,
) -> list[BatchSpec]:
    specs = []
    for tier in sorted(needed):
        remaining = needed[tier]
        while remaining > 0:
            n = min(batch_size, remaining)
            specs.append(BatchSpec(tier, rng.choice(subdomains), rng.choice(personas), n))
            remaining -= n
    return specs


def last_id_number(existing: list[dict]) -> int:
    """Highest numeric id suffix, so new ids never collide after rows are removed."""
    return max((int(r["id"].rsplit("-", 1)[1]) for r in existing), default=0)


def estimate_generator_calls(needed: dict[int, int], batch_size: int) -> int:
    return sum(math.ceil(n / batch_size) for n in needed.values() if n > 0)


def generate_batch(
    client: ChatClient, template: Template, spec: BatchSpec, avoid: list[str]
) -> list[str]:
    avoid_text = "\n".join(f"- {q}" for q in avoid) if avoid else "- (none yet)"
    prompt = template.substitute(
        n=spec.n, persona=spec.persona, subdomain=spec.subdomain, tier=spec.tier, avoid=avoid_text
    )
    result = client.chat(
        "generator", [{"role": "user", "content": prompt}], step="generate", json_schema=QUERIES_SCHEMA
    )
    payload = extract_json(result.content)
    queries = payload.get("queries", []) if isinstance(payload, dict) else payload
    return [q for q in queries if isinstance(q, str) and q.strip()]


def run_generation(
    cfg: Config,
    client: ChatClient,
    paths: DatasetPaths,
    template: Template,
    limit: int | None,
) -> dict[int, int]:
    gen = cfg.generation
    prefix = cfg.datasets[paths.name].id_prefix
    target = cfg.datasets[paths.name].target
    rng = random.Random(gen.seed + len(read_jsonl(paths.queries)))
    added: Counter[int] = Counter()
    for _ in range(gen.max_rounds):
        existing = read_jsonl(paths.queries)
        remaining_limit = None if limit is None else limit - sum(added.values())
        needed = cap_by_limit(needed_per_tier(existing, target, cfg.tiers()), remaining_limit)
        if sum(needed.values()) == 0:
            break
        specs = plan_batches(needed, gen.batch_size, gen.subdomains, gen.personas, rng)
        known = all_query_texts(cfg)
        next_index = last_id_number(existing)
        stop = False
        with ThreadPoolExecutor(max_workers=gen.concurrency) as pool:
            futures = {}
            for spec in specs:
                same_tier = [r["query"] for r in existing if r["intended_tier"] == spec.tier]
                avoid = rng.sample(same_tier, min(12, len(same_tier)))
                futures[pool.submit(generate_batch, client, template, spec, avoid)] = spec
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    candidates = future.result()
                except BudgetExceeded as exc:
                    print(f"Stopping: {exc}")
                    stop = True
                    continue
                except JsonParseError as exc:
                    print(f"Skipping batch (tier {spec.tier}, {spec.subdomain}): {exc}")
                    continue
                kept = dedup(candidates, known, gen.dedup_cosine_threshold)
                for text in kept[: max(0, needed[spec.tier])]:
                    next_index += 1
                    append_jsonl(
                        paths.queries,
                        {
                            "id": f"{prefix}-{next_index:04d}",
                            "query": text,
                            "intended_tier": spec.tier,
                            "subdomain": spec.subdomain,
                            "persona": spec.persona,
                        },
                    )
                    known.append(text)
                    added[spec.tier] += 1
                    needed[spec.tier] -= 1
        if stop:
            break
    return dict(added)
