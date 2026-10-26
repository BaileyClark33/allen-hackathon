from __future__ import annotations

import importlib.util
import json
from string import Template

import pytest
from fakes import FakeClient, make_config

from nemoroute.config import ROOT
from nemoroute.jsonl import write_jsonl
from nemoroute.prompts import load_prompt


def _load_script():
    spec = importlib.util.spec_from_file_location("build_memo_bench", ROOT / "scripts" / "build_memo_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


memo_bench = _load_script()


def generated_questions(n: int = 25) -> list[str]:
    return [f"Generated memo question {i}?" for i in range(1, n + 1)]


def handwritten_questions(n: int = 15) -> list[dict]:
    return [
        {"question": f"Handwritten memo question {i}?", "subdomain": "pricing strategy"}
        for i in range(1, n + 1)
    ]


def test_merge_generates_forty_stable_unique_prompts(tmp_path):
    cfg = make_config(tmp_path)
    client = FakeClient(lambda role, messages: json.dumps({"questions": generated_questions()}))
    template = load_prompt(cfg, "memo_generate")
    generated = memo_bench.generate_questions(client, template, cfg.generation.subdomains, ["Existing?"])
    assert client.calls == [("generator", "memo_generate")]

    records = memo_bench.merge_prompts(generated, handwritten_questions())
    assert [r["id"] for r in records] == [f"m{i:03d}" for i in range(1, 41)]
    assert [r["origin"] for r in records] == ["generated"] * 25 + ["handwritten"] * 15
    assert len({r["question"] for r in records}) == 40
    assert records[0] == {"id": "m001", "question": "Generated memo question 1?", "origin": "generated"}
    assert records[25]["subdomain"] == "pricing strategy"
    assert "subdomain" not in records[0]


def test_generate_questions_trims_extras_to_twenty_five(tmp_path):
    cfg = make_config(tmp_path)
    client = FakeClient(lambda role, messages: json.dumps({"questions": generated_questions(26)}))
    template = load_prompt(cfg, "memo_generate")
    generated = memo_bench.generate_questions(client, template, cfg.generation.subdomains, [])
    assert len(generated) == 25
    assert generated[-1] == "Generated memo question 25?"


def test_refuses_to_overwrite_existing_prompts(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"id": "m001"}\n')
    with pytest.raises(SystemExit, match="already exists"):
        memo_bench.ensure_output_available(path)
    assert path.read_text() == '{"id": "m001"}\n'


def test_rejects_missing_or_wrong_handwritten_file(tmp_path):
    with pytest.raises(SystemExit, match="Missing"):
        memo_bench.load_handwritten(tmp_path / "handwritten.jsonl")

    path = tmp_path / "handwritten.jsonl"
    write_jsonl(path, handwritten_questions(14))
    with pytest.raises(SystemExit, match="exactly 15"):
        memo_bench.load_handwritten(path)
    with pytest.raises(SystemExit, match="exactly 15"):
        memo_bench.merge_prompts(generated_questions(), handwritten_questions(14))


def test_rejects_duplicate_questions():
    handwritten = handwritten_questions()
    handwritten[0]["question"] = generated_questions()[0]
    with pytest.raises(SystemExit, match="Duplicate"):
        memo_bench.merge_prompts(generated_questions(), handwritten)


def test_memo_judge_prompt_scores_grounding_and_hides_strategy():
    text = (ROOT / "src/nemoroute/prompts/memo_judge.txt").read_text()
    assert set(Template(text).get_identifiers()) == {"query", "answer"}
    assert "$strategy" not in text and "$model" not in text
    for criterion in ("accuracy", "completeness", "structure", "actionability", "grounding"):
        assert f'"{criterion}": 1-5' in text
    assert "not reward length" in text.lower()
    assert "confident" in text.lower()
    assert "source snippets" in text.lower()
