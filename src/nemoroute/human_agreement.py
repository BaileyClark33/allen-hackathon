from __future__ import annotations

import csv
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from sklearn.metrics import cohen_kappa_score

from nemoroute.jsonl import index_by, read_jsonl, write_jsonl
from nemoroute.labeling.label import passes

HUMAN_COLUMNS = ["item_id", "query", "answer", "human_pass", "notes"]


def stratified_sample(
    candidates: list[dict[str, Any]],
    stratum: Callable[[dict[str, Any]], Any],
    total: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for c in sorted(candidates, key=lambda c: (c["query_id"], c["tier"])):
        groups[stratum(c)].append(c)
    rng = random.Random(seed)
    for items in groups.values():
        rng.shuffle(items)
    picked: list[dict[str, Any]] = []
    keys = sorted(groups, key=str)
    while len(picked) < total and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(picked) < total:
                picked.append(groups[k].pop())
    rng.shuffle(picked)
    return picked


def answer_candidates(
    queries: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    allowed_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    qtext = {q["id"]: q["query"] for q in queries}
    judged = index_by(judgments, "query_id", "tier")
    out = []
    for r in responses:
        key = (r["query_id"], r["tier"])
        if key not in judged or (allowed_ids is not None and r["query_id"] not in allowed_ids):
            continue
        out.append({**r, "query": qtext[r["query_id"]], "judgment": judged[key]})
    return out


def memo_candidates(
    memos: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    judged = index_by(judgments, "query_id", "strategy")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for memo in memos:
        key = (memo["query_id"], memo["strategy"])
        if key not in judged:
            continue
        question = memo.get("question") or memo.get("query", "")
        text = memo.get("memo_md") or memo.get("memo") or memo.get("answer", "")
        out[key] = {
            **memo,
            "question": question,
            "query": question,
            "memo_md": text,
            "answer": text,
            "tier": memo["strategy"],  # agreement() keys on (query_id, tier)
            "judgment": judged[key],
        }
    return list(out.values())


def write_human_files(sample: list[dict[str, Any]], csv_path: Path, key_path: Path, overwrite: bool) -> None:
    if csv_path.exists() and not overwrite:
        raise SystemExit(f"{csv_path} already exists (it may contain your labels). Pass --force to overwrite.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HUMAN_COLUMNS)
        writer.writeheader()
        for i, item in enumerate(sample, start=1):
            item_id = f"h{i:03d}"
            writer.writerow({"item_id": item_id, "query": item["query"], "answer": item["answer"],
                             "human_pass": "", "notes": ""})
            key = {"item_id": item_id, "query_id": item["query_id"]}
            if "tier" in item:
                key["tier"] = item["tier"]
            if "strategy" in item:
                key["strategy"] = item["strategy"]
            keys.append(key)
    write_jsonl(key_path, keys)


def parse_pass(value: str) -> bool | None:
    v = value.strip().lower()
    if v in {"1", "y", "yes", "pass", "true", "p"}:
        return True
    if v in {"0", "n", "no", "fail", "false", "f"}:
        return False
    return None


def read_human_labels(csv_path: Path, key_path: Path) -> list[dict[str, Any]]:
    keys = index_by(read_jsonl(key_path), "item_id")
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            verdict = parse_pass(row.get("human_pass", ""))
            if verdict is None:
                continue
            key = keys[row["item_id"]]
            tier = key.get("tier", key.get("strategy"))
            rows.append({"query_id": key["query_id"], "tier": tier, "human_pass": verdict})
    return rows


def agreement(
    human: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    pass_threshold: float,
    min_accuracy: int,
) -> dict[str, Any]:
    judged = index_by(judgments, "query_id", "tier")
    rows = [
        (row["tier"], row["human_pass"],
         passes(judged[(row["query_id"], row["tier"])]["scores"], pass_threshold, min_accuracy))
        for row in human
    ]
    h = [r[1] for r in rows]
    j = [r[2] for r in rows]
    n = len(h)
    agree = sum(a == b for a, b in zip(h, j))
    kappa = float(cohen_kappa_score(h, j)) if n and len(set(h) | set(j)) > 1 else float("nan")
    by_tier = {}
    for tier in sorted({r[0] for r in rows}, key=str):
        sub = [r for r in rows if r[0] == tier]
        by_tier[tier] = {
            "n": len(sub),
            "human_pass": sum(r[1] for r in sub),
            "judge_pass": sum(r[2] for r in sub),
            "agreement": sum(r[1] == r[2] for r in sub) / len(sub),
        }
    return {
        "n": n,
        "percent_agreement": agree / n if n else float("nan"),
        "cohen_kappa": kappa,
        "both_pass": sum(a and b for a, b in zip(h, j)),
        "both_fail": sum((not a) and (not b) for a, b in zip(h, j)),
        "human_pass_judge_fail": sum(a and not b for a, b in zip(h, j)),
        "human_fail_judge_pass": sum((not a) and b for a, b in zip(h, j)),
        "by_tier": by_tier,
    }


def render_agreement_section(stats: dict[str, Any]) -> str:
    pct = "n/a" if math.isnan(stats["percent_agreement"]) else f"{stats['percent_agreement']:.1%}"
    kappa = "n/a" if math.isnan(stats["cohen_kappa"]) else f"{stats['cohen_kappa']:.3f}"
    lines = [
        "## Human agreement",
        "",
        f"Human pass/fail labels vs the judge on {stats['n']} memos "
        f"(pass threshold {stats['pass_threshold']:.2f}, min accuracy {stats['min_accuracy']}).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Percent agreement | {pct} |",
        f"| Cohen's kappa | {kappa} |",
        f"| n | {stats['n']} |",
        f"| Both pass | {stats['both_pass']} |",
        f"| Both fail | {stats['both_fail']} |",
        f"| Human pass, judge fail | {stats['human_pass_judge_fail']} |",
        f"| Human fail, judge pass | {stats['human_fail_judge_pass']} |",
    ]
    by_tier = stats.get("by_tier") or {}
    if by_tier:
        lines += [
            "",
            "By group (strategy for memos, routed tier for answers). Small strata: indicative only.",
            "",
            "| Group | n | Human pass | Judge pass | Agreement |",
            "|---|---|---|---|---|",
            *(f"| {tier} | {s['n']} | {s['human_pass']} | {s['judge_pass']} | {s['agreement']:.0%} |"
              for tier, s in by_tier.items()),
        ]
    return "\n".join(lines)


_SECTION = re.compile(r"(?ms)^## Human agreement[^\n]*\n?.*?(?=^## |\Z)")


def write_agreement_section(path: Path, section: str) -> None:
    text = path.read_text() if path.exists() else ""
    if _SECTION.search(text):
        text = _SECTION.sub(lambda _: section + "\n\n", text, count=1)
    else:
        text = text.rstrip() + ("\n\n" if text.strip() else "") + section + "\n"
    path.write_text(text.rstrip() + "\n")
