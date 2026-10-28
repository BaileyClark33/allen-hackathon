from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from nemoroute.config import ROOT, Config, api_key

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_ENV = "TAVILY_API_KEY"
DEFAULT_CACHE_DIR = ROOT / "data" / "search_cache"


@dataclass(frozen=True)
class Source:
    url: str
    title: str
    snippet: str


class SearchClient(Protocol):
    def search(self, query: str) -> list[Source]: ...


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def parse_tavily_response(payload: dict[str, Any], limit: int) -> list[Source]:
    results = payload.get("results") or []
    sources: list[Source] = []
    for result in results[:limit]:
        url = result.get("url")
        if not url:
            continue
        sources.append(
            Source(
                url=url,
                title=result.get("title", ""),
                snippet=result.get("content", ""),
            )
        )
    return sources


class TavilySearchClient:
    def __init__(
        self,
        cfg: Config,
        *,
        cache_dir: Path | str | None = None,
        http_client: httpx.Client | None = None,
        key: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.agent = cfg.agent
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self._http = http_client
        self._key = key

    def search(self, query: str) -> list[Source]:
        normalized = normalize_query(query)
        cached = self._read_cache(normalized)
        if cached is not None:
            return cached
        sources = parse_tavily_response(self._fetch(self._payload(normalized)), self.agent.results_per_search)
        self._write_cache(normalized, sources)
        return sources

    def is_cached(self, query: str) -> bool:
        return self._cache_path(normalize_query(query)).exists()

    def _payload(self, normalized: str) -> dict[str, Any]:
        return {
            "api_key": self._api_key(),
            "query": normalized,
            "search_depth": self.agent.search_depth,
            "max_results": self.agent.results_per_search,
        }

    def _api_key(self) -> str:
        if self._key is None:
            self._key = api_key(self.cfg, TAVILY_ENV)
        return self._key

    def _http_client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=self.cfg.api.timeout_s)
        return self._http

    def _fetch(self, payload: dict[str, Any]) -> dict[str, Any]:
        retrying = Retrying(
            stop=stop_after_attempt(self.cfg.api.max_retries + 1),
            wait=wait_exponential(multiplier=self.cfg.api.backoff_initial_s, max=self.cfg.api.backoff_max_s),
            retry=retry_if_exception(is_retryable),
            reraise=True,
        )
        response = retrying(self._post, payload)
        return response.json()

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        response = self._http_client().post(TAVILY_URL, json=payload)
        response.raise_for_status()
        return response

    def _cache_path(self, normalized: str) -> Path:
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, normalized: str) -> list[Source] | None:
        path = self._cache_path(normalized)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
            return [Source(**entry) for entry in payload["sources"]]
        except (OSError, ValueError, KeyError, TypeError):
            return None  # a crash mid-write can leave a truncated cache file

    def _write_cache(self, normalized: str, sources: list[Source]) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {"query": normalized, "sources": [asdict(source) for source in sources]}
            target = self._cache_path(normalized)
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, target)
        except (PermissionError, OSError):
            pass  # read-only filesystem (e.g. Streamlit Cloud) — cache is optional
