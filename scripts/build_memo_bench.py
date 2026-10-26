from __future__ import annotations

from pathlib import Path
from string import Template

from nemoroute.budget import confirm_or_exit
from nemoroute.cli import base_parser, make_client, print_run_summary, setup
from nemoroute.client import ChatClient
from nemoroute.jsonl import read_jsonl, write_jsonl
from nemoroute.parsing import extract_json
from nemoroute.pricing import estimate
from nemoroute.prompts import load_prompt

GENERATED_COUNT = 25
HANDWRITTEN_COUNT = 15

QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
    "additionalProperties": False,
}


def _question(row: str | dict) -> str:
    if isinstance(row, str):
        return row.strip()
    if isinstance(row, dict) and isinstance(row.get("question"), str):
        return row["question"].strip()
    return ""


def ensure_output_available(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"{path} already exists. The memo benchmark is frozen; archive it yourself to rebuild.")


def load_handwritten(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Add {HANDWRITTEN_COUNT} memo questions, one JSON object per line: "
            '{"question": "...", "subdomain": "optional"}'
        )
    rows = read_jsonl(path)
    if len(rows) != HANDWRITTEN_COUNT:
        raise SystemExit(f"{path} must contain exactly {HANDWRITTEN_COUNT} questions; found {len(rows)}.")
    for i, row in enumerate(rows, 1):
        if not _question(row):
            raise SystemExit(f"{path} row {i} needs a non-empty 'question' string.")
    return rows


def generate_questions(
    client: ChatClient, template: Template, subdomains: list[str], avoid: list[str]
) -> list[str]:
    prompt = template.substitute(
        n=GENERATED_COUNT,
        subdomains="\n".join(f"- {s}" for s in subdomains),
        avoid="\n".join(f"- {q}" for q in avoid) if avoid else "- (none)",
    )
    result = client.chat(
        "generator", [{"role": "user", "content": prompt}], step="memo_generate", json_schema=QUESTIONS_SCHEMA
    )
    payload = extract_json(result.content)
    questions = payload.get("questions", []) if isinstance(payload, dict) else payload
    return [q.strip() for q in questions if isinstance(q, str) and q.strip()]


def merge_prompts(generated: list[str], handwritten: list[dict]) -> list[dict]:
    if len(generated) != GENERATED_COUNT:
        raise SystemExit(f"Expected exactly {GENERATED_COUNT} generated questions, got {len(generated)}.")
    if len(handwritten) != HANDWRITTEN_COUNT:
        raise SystemExit(f"Expected exactly {HANDWRITTEN_COUNT} handwritten questions, got {len(handwritten)}.")
    records: list[dict] = []
    seen: dict[str, int] = {}
    for origin, rows in (("generated", generated), ("handwritten", handwritten)):
        for row in rows:
            question = _question(row)
            if not question:
                raise SystemExit(f"Empty or missing question in {origin} rows.")
            key = " ".join(question.lower().split())
            if key in seen:
                raise SystemExit(f"Duplicate question at rows {seen[key]} and {len(records) + 1}: {question!r}")
            seen[key] = len(records) + 1
            record = {"id": f"m{len(records) + 1:03d}", "question": question, "origin": origin}
            subdomain = row.get("subdomain") if isinstance(row, dict) else None
            if isinstance(subdomain, str) and subdomain.strip():
                record["subdomain"] = subdomain.strip()
            records.append(record)
    return records


def main() -> None:
    parser = base_parser("Build the frozen 40-prompt memo benchmark: 25 generated + 15 handwritten questions.")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument("--max-cost", type=float, default=None, help="abort once this run spends this many USD")
    args = parser.parse_args()
    cfg, usage_log = setup(args)

    root = cfg.path("data/memo_bench")
    prompts_path = root / "prompts.jsonl"
    ensure_output_available(prompts_path)
    handwritten = load_handwritten(root / "handwritten.jsonl")

    est = estimate(cfg, {"generator": 1}, usage_log)
    confirm_or_exit(cfg, est, usage_log, args.yes, f"build memo benchmark: {GENERATED_COUNT} generated questions")
    if est.total_calls == 0:
        return

    client = make_client(cfg, usage_log, args.max_cost)
    generated = generate_questions(
        client, load_prompt(cfg, "memo_generate"), cfg.generation.subdomains,
        [_question(row) for row in handwritten],
    )
    records = merge_prompts(generated, handwritten)
    ensure_output_available(prompts_path)
    write_jsonl(prompts_path, records)
    print(f"\nWrote {len(records)} prompts to {prompts_path}")
    print_run_summary(usage_log, client)


if __name__ == "__main__":
    main()
