from __future__ import annotations

import httpx
import pytest
from fakes import FakeSearchClient, make_config

from nemoroute.search import Source, TavilySearchClient, normalize_query, parse_tavily_response

TAVILY_BODY = {
    "query": "us ev charger market size",
    "results": [
        {"title": "EV Charging Report", "url": "https://example.com/a", "content": "Market grew 40%."},
        {"title": "Charger Outlook", "url": "https://example.com/b", "content": "Installed base rising."},
        {"title": "Analyst Note", "url": "https://example.com/c", "content": "Policy tailwinds."},
    ],
}


def make_transport(calls: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=TAVILY_BODY)

    return httpx.MockTransport(handler)


def make_client(tmp_path, calls: list[httpx.Request]) -> TavilySearchClient:
    cfg = make_config(tmp_path)
    return TavilySearchClient(
        cfg, cache_dir=tmp_path / "cache", http_client=httpx.Client(transport=make_transport(calls)), key="test-key"
    )


def test_normalize_query_collapses_whitespace_and_case():
    assert normalize_query("  US   EV Charger\nMarket ") == "us ev charger market"


def test_parse_tavily_response_maps_fields_and_limit():
    sources = parse_tavily_response(TAVILY_BODY, limit=2)
    assert sources == [
        Source(url="https://example.com/a", title="EV Charging Report", snippet="Market grew 40%."),
        Source(url="https://example.com/b", title="Charger Outlook", snippet="Installed base rising."),
    ]


def test_parse_tavily_response_skips_entries_without_url():
    payload = {"results": [{"title": "no url", "content": "x"}, TAVILY_BODY["results"][0]]}
    assert parse_tavily_response(payload, limit=5) == [Source("https://example.com/a", "EV Charging Report", "Market grew 40%.")]


def test_cache_hit_avoids_second_api_call(tmp_path):
    calls: list[httpx.Request] = []
    client = make_client(tmp_path, calls)

    first = client.search("US EV charger market size")
    assert client.is_cached("US EV charger market size")
    second = client.search("  us   ev charger MARKET size ")

    assert first == second
    assert len(first) == 3
    assert len(calls) == 1


def test_truncated_cache_is_treated_as_a_miss(tmp_path):
    calls: list[httpx.Request] = []
    client = make_client(tmp_path, calls)
    client.search("US EV charger market size")
    cache_file, = (tmp_path / "cache").glob("*.json")
    cache_file.write_text('{"query": "x", "sources": [')

    calls.clear()
    assert len(client.search("US EV charger market size")) == 3
    assert len(calls) == 1


def test_cache_survives_without_api_key_on_second_client(tmp_path, monkeypatch):
    calls: list[httpx.Request] = []
    client = make_client(tmp_path, calls)
    client.search("US EV charger market size")

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    cold = TavilySearchClient(make_config(tmp_path), cache_dir=tmp_path / "cache")
    assert cold.search("us ev charger market size") == client.search("US EV charger market size")
    assert len(calls) == 1


def test_missing_api_key_fails_loud_on_cache_miss(tmp_path, monkeypatch):
    def missing_key(*_args, **_kwargs):
        raise RuntimeError("TAVILY_API_KEY is not set")

    # api_key() reloads .env, so patch the lookup instead of deleting the env var.
    monkeypatch.setattr("nemoroute.search.api_key", missing_key)
    client = TavilySearchClient(make_config(tmp_path), cache_dir=tmp_path / "cache")
    with pytest.raises(RuntimeError):
        client.search("no cached results here")


def test_fake_search_client_returns_deterministic_sources():
    fake = FakeSearchClient()
    assert fake.search("q") == fake.search("q")
    assert fake.calls == ["q", "q"]
