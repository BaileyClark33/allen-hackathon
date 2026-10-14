from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml

from nemoroute.client import ChatResult
from nemoroute.config import Config, load_config


class FakeClient:
    def __init__(self, responder: Callable[[str, list[dict[str, str]]], str]) -> None:
        self.responder = responder
        self.calls: list[tuple[str, str]] = []

    def chat(self, role: str, messages: list[dict[str, str]], *, step: str,
             json_schema: dict[str, Any] | None = None, **overrides: Any) -> ChatResult:
        self.calls.append((role, step))
        return ChatResult(
            content=self.responder(role, messages),
            reasoning="",
            input_tokens=100,
            output_tokens=50,
            latency_s=0.5,
            cost_usd=0.001,
            model=f"fake-{role}",
            role=role,
            finish_reason="stop",
        )


def make_config(tmp_path: Path) -> Config:
    cfg = load_config()
    data = cfg.model_dump()
    data["paths"]["calls_log"] = str(tmp_path / "calls.jsonl")
    data["paths"]["app_spend_log"] = str(tmp_path / "app.jsonl")
    data["paths"]["splits_dir"] = str(tmp_path / "splits")
    data["paths"]["results_dir"] = str(tmp_path / "results")
    for name in data["datasets"]:
        data["datasets"][name]["dir"] = str(tmp_path / name)
    return Config.model_validate(data)


def write_config(tmp_path: Path, cfg: Config) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    return path
