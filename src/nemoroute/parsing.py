from __future__ import annotations

import json
import re
from typing import Any

THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class JsonParseError(ValueError):
    pass


def split_reasoning(content: str | None) -> tuple[str, str]:
    if not content:
        return "", ""
    reasoning_parts = THINK_BLOCK.findall(content)
    answer = THINK_BLOCK.sub("", content)
    if "</think>" in answer:
        head, _, answer = answer.partition("</think>")
        reasoning_parts.insert(0, head)
    if "<think>" in answer:
        answer, _, tail = answer.partition("<think>")
        reasoning_parts.append(tail)
    return answer.strip(), "\n".join(p.strip() for p in reasoning_parts).strip()


def extract_json(text: str) -> Any:
    cleaned = CODE_FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise JsonParseError(f"No valid JSON found in model output: {text[:200]!r}")
