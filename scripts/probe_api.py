from __future__ import annotations

import json
from typing import Any

from nemoroute.budget import confirm_or_exit
from nemoroute.cli import base_parser, make_client, print_run_summary, setup
from nemoroute.client import TokenFactoryClient
from nemoroute.parsing import extract_json
from nemoroute.pricing import estimate
from nemoroute.routers.zero_shot import ROUTER_SCHEMA

REGIONAL_US = "https://api.tokenfactory.us-central1.nebius.com/v1/"
PING = [{"role": "user", "content": "Reply with exactly one word: OK"}]
MATH = [{"role": "user", "content": "What is 17 * 23? Reply with just the number."}]
ROUTE = [{"role": "user", "content": 'Classify "What is EBITDA?" into tier 1, 2 or 3. Return JSON {"tier": n, "reasoning": "..."}.'}]


def check_catalog(client: TokenFactoryClient, model_ids: dict[str, str]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for base_url in (client.cfg.api.base_url, REGIONAL_US):
        try:
            models = client.list_models(verbose=True, base_url=base_url)
        except Exception as exc:
            print(f"  list_models failed on {base_url}: {type(exc).__name__}: {exc}")
            continue
        listed = {m["id"]: m for m in models}
        print(f"  {base_url}: {len(listed)} models listed")
        nemotron = sorted(i for i in listed if "nemotron" in i.lower())
        print(f"    Nemotron IDs: {nemotron}")
        for name, model_id in model_ids.items():
            if model_id in listed and name not in found:
                found[name] = {"base_url": base_url, **listed[model_id]}
    return found


def try_call(client: TokenFactoryClient, role: str, messages: list[dict], **overrides: Any) -> dict[str, Any]:
    try:
        r = client.chat(role, messages, step="probe", **overrides)
        try:
            parsed = extract_json(r.content)
        except ValueError:
            parsed = None
        return {
            "ok": True,
            "parsed_json": parsed,
            "content": r.content[:200],
            "reasoning_chars": len(r.reasoning),
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "latency_s": round(r.latency_s, 2),
            "finish_reason": r.finish_reason,
            "cost_usd": round(r.cost_usd, 6),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def main() -> None:
    parser = base_parser("Verify Token Factory model IDs, hosts, thinking toggle and JSON support.")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--max-cost", type=float, default=0.05)
    args = parser.parse_args()
    cfg, usage_log = setup(args)
    client = make_client(cfg, usage_log, args.max_cost)

    print("1) Model catalog (GET /v1/models?verbose=true, free)")
    model_ids = {name: m.id for name, m in cfg.models.items()}
    found = check_catalog(client, model_ids)
    for name, model_id in model_ids.items():
        info = found.get(name)
        if not info:
            print(f"  MISSING {name}: {model_id}")
            continue
        print(
            f"  OK {name}: {model_id} via {info['base_url']} | live pricing {info.get('pricing')} | "
            f"config ${cfg.models[name].input_usd_per_mtok}/${cfg.models[name].output_usd_per_mtok} per 1M | "
            f"context {info.get('context_length')}"
        )

    est = estimate(cfg, {"router": 4, "tier2": 1, "tier3": 2})
    confirm_or_exit(cfg, est, usage_log, args.yes, "API probe (upper bound, tiny calls)")

    results: dict[str, Any] = {}
    print("\n2) Reachability per model on the default host (thinking off)")
    for role in ("tier1", "tier2", "tier3"):
        results[f"ping_{role}"] = try_call(client, role, PING, max_tokens=32, thinking=False)
        print(f"  {role} ({cfg.model_for_role(role).id}): {results[f'ping_{role}']}")

    print("\n3) Thinking toggle on Nano (same prompt, thinking off vs on)")
    results["nano_think_off"] = try_call(client, "tier1", MATH, max_tokens=512, thinking=False)
    results["nano_think_on"] = try_call(client, "tier1", MATH, max_tokens=1024, thinking=True)
    print(f"  off: {results['nano_think_off']}")
    print(f"  on:  {results['nano_think_on']}")

    print("\n4) Thinking on Ultra (judge config)")
    results["ultra_think_on"] = try_call(client, "tier3", MATH, max_tokens=1024, thinking=True)
    print(f"  {results['ultra_think_on']}")

    print("\n5) Structured output (Nano router; Ultra with thinking on, as the judge runs)")
    schema_fmt = {"type": "json_schema", "json_schema": {"name": "output", "schema": ROUTER_SCHEMA, "strict": True}}
    results["json_schema"] = try_call(client, "router", ROUTE, response_format=schema_fmt)
    results["json_object"] = try_call(client, "router", ROUTE, response_format={"type": "json_object"})
    results["prompt_only"] = try_call(client, "router", ROUTE)
    results["ultra_thinking_json_schema"] = try_call(
        client, "tier3", ROUTE, response_format=schema_fmt, thinking=True, max_tokens=1024
    )
    for key in ("json_schema", "json_object", "prompt_only", "ultra_thinking_json_schema"):
        res = results[key]
        print(f"  {key}: ok={res.get('ok')} parsed={res.get('parsed_json')} "
              f"reasoning_chars={res.get('reasoning_chars')} {res.get('error', '')}")

    out = cfg.path(cfg.paths.results_dir) / "probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"catalog": {k: {kk: vv for kk, vv in v.items() if kk != "description"}
                                           for k, v in found.items()}, "calls": results}, indent=2))
    print(f"\nSaved probe details to {out}")
    print_run_summary(usage_log, client)


if __name__ == "__main__":
    main()
