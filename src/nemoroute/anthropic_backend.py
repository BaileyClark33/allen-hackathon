from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nemoroute.config import ModelConfig, RoleConfig

UNSUPPORTED_SCHEMA_KEYS = {"minimum", "maximum"}


@dataclass
class AnthropicReply:
    content: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None


def strip_schema(schema: Any) -> Any:
    """Drop numeric bounds; callers validate ranges after parsing (see judge.parse_scores)."""
    if isinstance(schema, dict):
        return {k: strip_schema(v) for k, v in schema.items() if k not in UNSUPPORTED_SCHEMA_KEYS}
    if isinstance(schema, list):
        return [strip_schema(v) for v in schema]
    return schema


def split_system(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    return system, [m for m in messages if m["role"] != "system"]


def request_kwargs(
    model: ModelConfig, role_cfg: RoleConfig, messages: list[dict[str, str]], json_schema: dict[str, Any] | None
) -> dict[str, Any]:
    # Thinking and sampling parameters are left at model defaults: current Claude models reject
    # disabled thinking and non-default temperature. Depth is set with effort instead.
    system, turns = split_system(messages)
    kwargs: dict[str, Any] = {"model": model.id, "max_tokens": role_cfg.max_tokens, "messages": turns}
    if system:
        kwargs["system"] = system
    output_config: dict[str, Any] = {}
    if model.effort:
        output_config["effort"] = model.effort
    if json_schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": strip_schema(json_schema)}
    if output_config:
        kwargs["output_config"] = output_config
    return kwargs


class AnthropicBackend:
    def __init__(self, key: str, timeout_s: float, max_retries: int) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=key, timeout=timeout_s, max_retries=max_retries)

    def chat(
        self,
        model: ModelConfig,
        role_cfg: RoleConfig,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None,
    ) -> AnthropicReply:
        response = self._client.messages.create(**request_kwargs(model, role_cfg, messages, json_schema))
        # A refusal returns no usable text; it is recorded as an empty answer, which the
        # judge stage scores as a fail without a judge call.
        text = "" if response.stop_reason == "refusal" else "".join(
            b.text for b in response.content if b.type == "text"
        )
        return AnthropicReply(
            content=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason,
        )
