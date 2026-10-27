from __future__ import annotations

import sys

from nemoroute.config import load_config
from nemoroute.search import TavilySearchClient, normalize_query


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python scripts/smoke_search.py "your search query"')

    query = " ".join(sys.argv[1:])
    cfg = load_config()
    client = TavilySearchClient(cfg)

    from_cache = client.is_cached(query)
    try:
        sources = client.search(query)
    except Exception as exc:
        raise SystemExit(f"Search failed: {type(exc).__name__}: {exc}")

    origin = "disk cache (no API call)" if from_cache else "live Tavily API"
    print(f'Query: {normalize_query(query)!r} | {len(sources)} results | source: {origin}')
    for index, source in enumerate(sources, start=1):
        print(f"\n[{index}] {source.title}\n    URL: {source.url}\n    {source.snippet}")


if __name__ == "__main__":
    main()
