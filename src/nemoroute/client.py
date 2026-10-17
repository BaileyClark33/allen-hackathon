from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import openai
from openai import OpenAI
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from nemoroute.anthropic_backend import AnthropicBackend
from nemoroute.budget import RunBudget
from nemoroute.config import Config, ModelConfig, api_key
from nemoroute.parsing import split_reasoning
from nemoroute.pricing import call_cost
from nemoroute.usage_log import CallRecord, UsageLog, utc_now

Message = dict[str, str]


@dataclass
class ChatResult:
    content: str
    reasoning: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    model: str
    role: str
    finish_reason: str | None


class ChatClient(Protocol):
    def chat(
        self,
        role: str,
        messages: list[Message],
        *,
        step: str,
        json_schema: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> ChatResult: ...


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


class RetryAfterWait:
    def __init__(self, initial: float, maximum: float) -> None:
        self.maximum = maximum
        self.fallback = wait_exponential(multiplier=initial, max=maximum)

    def __call__(self, retry_state: Any) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exc, openai.RateLimitError):
            header = exc.response.headers.get("retry-after") if exc.response is not None else None
            if header:
                try:
                    return min(float(header), self.maximum)
                except ValueError:
                    pass
        return self.fallback(retry_state)


class TokenFactoryClient:
    def __init__(
        self,
        cfg: Config,
        usage_log: UsageLog,
        run_budget: RunBudget | None = None,
        run_id: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.usage_log = usage_log
        self.run_budget = run_budget
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._key = api_key(cfg)
        self._clients: dict[tuple[str, str], OpenAI] = {}
        self._anthropic: dict[str, AnthropicBackend] = {}

    def _key_for(self, model: ModelConfig) -> str:
        return api_key(self.cfg, model.api_key_env) if model.api_key_env else self._key

    def _client_for(self, base_url: str, key: str | None = None) -> OpenAI:
        key = key or self._key
        if (base_url, key) not in self._clients:
            self._clients[(base_url, key)] = OpenAI(
                base_url=base_url, api_key=key, timeout=self.cfg.api.timeout_s, max_retries=0
            )
        return self._clients[(base_url, key)]

    def _anthropic_for(self, key: str) -> AnthropicBackend:
        if key not in self._anthropic:
            self._anthropic[key] = AnthropicBackend(key, self.cfg.api.timeout_s, self.cfg.api.max_retries)
        return self._anthropic[key]

    def _call_openai(
        self, model: ModelConfig, role: str, messages: list[Message], json_schema: dict[str, Any] | None,
        overrides: dict[str, Any],
    ) -> tuple[str, str, int, int, str | None]:
        client = self._client_for(model.base_url or self.cfg.api.base_url, self._key_for(model))
        kwargs = self._request_kwargs(role, json_schema, overrides)
        retrying = Retrying(
            stop=stop_after_attempt(self.cfg.api.max_retries + 1),
            wait=RetryAfterWait(self.cfg.api.backoff_initial_s, self.cfg.api.backoff_max_s),
            retry=retry_if_exception(is_retryable),
            reraise=True,
        )
        response = retrying(client.chat.completions.create, model=model.id, messages=messages, **kwargs)
        choice = response.choices[0]
        message = choice.message
        content, inline_reasoning = split_reasoning(message.content)
        extra = message.model_extra or {}
        reasoning = extra.get("reasoning_content") or extra.get("reasoning") or inline_reasoning
        usage = response.usage
        return (
            content,
            reasoning or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            choice.finish_reason,
        )

    def _call_anthropic(
        self, model: ModelConfig, role: str, messages: list[Message], json_schema: dict[str, Any] | None
    ) -> tuple[str, str, int, int, str | None]:
        reply = self._anthropic_for(self._key_for(model)).chat(model, self.cfg.roles[role], messages, json_schema)
        return reply.content, "", reply.input_tokens, reply.output_tokens, reply.finish_reason

    def _request_kwargs(
        self, role: str, json_schema: dict[str, Any] | None, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        role_cfg = self.cfg.roles[role]
        thinking = overrides.pop("thinking", role_cfg.thinking)
        kwargs: dict[str, Any] = {
            "max_tokens": overrides.pop("max_tokens", role_cfg.max_tokens),
            "temperature": overrides.pop("temperature", role_cfg.temperature),
        }
        top_p = overrides.pop("top_p", role_cfg.top_p)
        if top_p is not None:
            kwargs["top_p"] = top_p
        extra_body: dict[str, Any] = dict(overrides.pop("extra_body", {}))
        if self.cfg.api.thinking_control == "chat_template_kwargs":
            extra_body["chat_template_kwargs"] = {"enable_thinking": thinking}
        if extra_body:
            kwargs["extra_body"] = extra_body
        if json_schema is not None and self.cfg.api.json_mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema, "strict": True},
            }
        elif json_schema is not None and self.cfg.api.json_mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        kwargs.update(overrides)
        return kwargs

    def chat(
        self,
        role: str,
        messages: list[Message],
        *,
        step: str,
        json_schema: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> ChatResult:
        if self.run_budget is not None:
            self.run_budget.check()
        model = self.cfg.model_for_role(role)
        start = time.perf_counter()
        try:
            if model.provider == "anthropic":
                content, reasoning, input_tokens, output_tokens, finish_reason = self._call_anthropic(
                    model, role, messages, json_schema
                )
            else:
                content, reasoning, input_tokens, output_tokens, finish_reason = self._call_openai(
                    model, role, messages, json_schema, dict(overrides)
                )
        except Exception as exc:
            self.usage_log.record(
                CallRecord(
                    ts=utc_now(),
                    run_id=self.run_id,
                    step=step,
                    role=role,
                    model=model.id,
                    input_tokens=0,
                    output_tokens=0,
                    latency_s=round(time.perf_counter() - start, 3),
                    cost_usd=0.0,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}"[:500],
                    billing=model.billing,
                )
            )
            raise
        latency = time.perf_counter() - start
        cost = call_cost(model, input_tokens, output_tokens)
        if self.run_budget is not None:
            self.run_budget.add(cost)
        self.usage_log.record(
            CallRecord(
                ts=utc_now(),
                run_id=self.run_id,
                step=step,
                role=role,
                model=model.id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_s=round(latency, 3),
                cost_usd=cost,
                ok=True,
                finish_reason=finish_reason,
                billing=model.billing,
            )
        )
        return ChatResult(
            content=content,
            reasoning=reasoning,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            cost_usd=cost,
            model=model.id,
            role=role,
            finish_reason=finish_reason,
        )

    def list_models(self, verbose: bool = True, base_url: str | None = None) -> list[dict[str, Any]]:
        url = (base_url or self.cfg.api.base_url).rstrip("/") + "/models"
        response = httpx.get(
            url,
            params={"verbose": str(verbose).lower()},
            headers={"Authorization": f"Bearer {self._key}"},
            timeout=self.cfg.api.timeout_s,
        )
        response.raise_for_status()
        return response.json().get("data", [])
