from __future__ import annotations

import html
import os

import streamlit as st

from nemoroute.agent.memo import MemoResult, StepTrace, run_memo
from nemoroute.app_guard import guard
from nemoroute.client import TokenFactoryClient
from nemoroute.config import Config, load_config
from nemoroute.pricing import call_cost
from nemoroute.search import TAVILY_ENV, TavilySearchClient
from nemoroute.usage_log import UsageLog

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
<style>
  /* --- typography & tone --- */
  html, body, .stApp {
    background: #0f0f1a;
    color: #e0e0e0;
  }
  .stApp header { background: transparent; }
  h1, h2, h3, h4, h5, h6 { color: #f0c060; }

  /* --- cards --- */
  .memo-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin: 1rem 0 2rem 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }
  .memo-card h1, .memo-card h2, .memo-card h3 {
    color: #f0c060;
    margin-top: 1.2em;
  }
  .memo-card p, .memo-card li { line-height: 1.75; }

  /* --- sources --- */
  .source-item {
    border-left: 3px solid #f0c060;
    padding: 0.4rem 0 0.4rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
  }
  .source-item a { color: #7eb8ff; }

  /* --- cost table --- */
  .cost-table {
    width: 100%;
    border-collapse: collapse;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 0.78rem;
  }
  .cost-table th {
    text-align: left;
    padding: 0.35rem 0.5rem;
    border-bottom: 1px solid #3a3a5a;
    color: #999;
    font-weight: 600;
  }
  .cost-table td {
    padding: 0.3rem 0.5rem;
    border-bottom: 1px solid #2a2a3a;
  }
  .cost-table .total-row td {
    border-top: 2px solid #f0c060;
    font-weight: 700;
    color: #f0c060;
  }

  /* --- savings badge --- */
  .savings-badge {
    display: inline-block;
    background: #2a3a1a;
    color: #8be08b;
    border: 1px solid #4a6a2a;
    border-radius: 6px;
    padding: 0.25rem 0.75rem;
    font-size: 0.85rem;
    font-weight: 600;
  }

  /* --- sidebar overrides --- */
  [data-testid="stSidebar"] {
    background: #13132a;
  }
  [data-testid="stSidebar"] .stMetric label { color: #888; }
</style>
"""

TIER_NAMES: dict[int, str] = {
    1: "Tier 1 · Nemotron Nano",
    2: "Tier 2 · Nemotron Super",
    3: "Tier 3 · Nemotron Ultra",
}

EXAMPLES: list[str] = [
    "What are the key drivers of the US EV market in 2025?",
    "Analyze the competitive landscape for enterprise AI assistants.",
    "What is the state of quantum computing for pharmaceutical R&D?",
]


# ---------------------------------------------------------------------------
# Resource loading (cached once per process)
# ---------------------------------------------------------------------------

def load_secret_key(cfg: Config, env_name: str) -> str | None:
    """Try to load a secret from Streamlit secrets.toml or env vars.

    Returns the key value or None if not found.
    """
    if os.environ.get(env_name):
        return None  # already set
    try:
        value = st.secrets.get(env_name)
    except Exception:
        value = None
    if value:
        os.environ[env_name] = value
    return None


@st.cache_resource
def resources() -> tuple[Config, UsageLog, TokenFactoryClient, TavilySearchClient]:
    """Load config, secrets, and shared clients. Cached across reruns."""
    cfg = load_config()

    # Load Nebius secret
    load_secret_key(cfg, cfg.api.api_key_env)

    # Load Tavily secret  (fail fast if missing)
    load_secret_key(cfg, TAVILY_ENV)
    if not os.environ.get(TAVILY_ENV):
        raise RuntimeError(
            f"{TAVILY_ENV} is not set. "
            "Add it to .streamlit/secrets.toml or set it as an environment variable.\n\n"
            "Get a free API key at https://tavily.com"
        )

    app_log = UsageLog(cfg.path(cfg.paths.app_spend_log))
    client = TokenFactoryClient(cfg, app_log, run_id="app")
    search = TavilySearchClient(cfg)
    return cfg, app_log, client, search


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state() -> None:
    """Initialize session-level defaults."""
    defaults: dict[str, object] = {
        "memo_count": 0,
        "total_cost": 0.0,
        "last_result": None,
        "last_query": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ---------------------------------------------------------------------------
# Helpers (pure display functions)
# ---------------------------------------------------------------------------

def estimate_ultra_cost(cfg: Config, steps: list[StepTrace]) -> float:
    """Price every LLM step at Tier-3 (Ultra) rates.

    Search steps (tier 0) are free and are excluded.
    """
    ultra_model = cfg.model_for_role(cfg.tier_roles[max(cfg.tiers())])
    total = 0.0
    for step in steps:
        if step.tier == 0:
            continue  # search step — no LLM cost
        total += call_cost(ultra_model, step.input_tokens, step.output_tokens)
    return total


def build_cost_table_html(steps: list[StepTrace], cfg: Config) -> str:
    """Render an HTML cost-breakdown table with Ultra comparison row."""
    actual_total = sum(s.cost_usd for s in steps)
    ultra_total = estimate_ultra_cost(cfg, steps)

    rows: list[str] = []
    for step in steps:
        tier_label = TIER_NAMES.get(step.tier, f"Tier {step.tier}") if step.tier else "—"
        model = step.model if step.model else "—"
        in_tok = f"{step.input_tokens:,}" if step.input_tokens else "—"
        out_tok = f"{step.output_tokens:,}" if step.output_tokens else "—"
        cost = f"${step.cost_usd:.4f}"
        latency = f"{step.latency_s:.1f}s" if step.latency_s else "—"
        rows.append(
            f"<tr><td>{step.name}</td><td>{tier_label}</td><td>{model}</td>"
            f"<td style='text-align:right'>{in_tok}</td>"
            f"<td style='text-align:right'>{out_tok}</td>"
            f"<td style='text-align:right'>{cost}</td>"
            f"<td style='text-align:right'>{latency}</td></tr>"
        )

    # Total row
    rows.append(
        f"<tr class='total-row'><td><strong>TOTAL</strong></td><td></td><td></td>"
        f"<td></td><td></td>"
        f"<td style='text-align:right'><strong>${actual_total:.4f}</strong></td>"
        f"<td></td></tr>"
    )

    # Ultra comparison row
    pct_saved = f"{(1 - actual_total / ultra_total) * 100:.0f}%" if ultra_total else "—"
    rows.append(
        f"<tr><td colspan='7' style='padding-top:0.5rem;color:#999'>"
        f"All-Ultra estimate: <strong>${ultra_total:.4f}</strong> &nbsp;→&nbsp; "
        f"<span class='savings-badge'>saved {pct_saved}</span></td></tr>"
    )

    header = (
        "<thead><tr>"
        "<th>Step</th><th>Tier</th><th>Model</th>"
        "<th style='text-align:right'>In Tok</th><th style='text-align:right'>Out Tok</th>"
        "<th style='text-align:right'>Cost</th><th style='text-align:right'>Latency</th>"
        "</tr></thead>"
    )
    return f"<table class='cost-table'>{header}<tbody>{''.join(rows)}</tbody></table>"


def display_memo_result(result: MemoResult, cfg: Config) -> None:
    """Render the memo, sources, and cost breakdown."""
    if result.route_tier:
        tier_name = TIER_NAMES.get(result.route_tier, f"Tier {result.route_tier}")
        st.caption(f"Routed to {tier_name} · {result.route_reasoning or ''}")

    # --- Memo content ---
    st.markdown('<div class="memo-card">', unsafe_allow_html=True)
    # Escape $ so paired dollar amounts aren't rendered as LaTeX math.
    st.markdown(result.memo_md.replace("$", r"\$"))
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Sources ---
    if result.sources:
        st.subheader("Sources")
        for i, src in enumerate(result.sources, 1):
            url = html.escape(src.url) if src.url.startswith(("http://", "https://")) else "#"
            st.markdown(
                f'<div class="source-item">'
                f'<strong>[{i}]</strong> '
                f'<a href="{url}" target="_blank" rel="noopener noreferrer">{html.escape(src.title)}</a>'
                f' — {html.escape(src.snippet[:200])}</div>',
                unsafe_allow_html=True,
            )

    # --- Cost breakdown ---
    st.divider()
    st.subheader("Cost Breakdown")
    st.markdown(build_cost_table_html(result.steps, cfg), unsafe_allow_html=True)

    # --- Download ---
    safe_name = result.question[:50].replace(" ", "_").replace("/", "_")
    st.download_button(
        label="Download as Markdown",
        data=result.memo_md,
        file_name=f"memo_{safe_name}.md",
        mime="text/markdown",
        type="secondary",
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(cfg: Config, app_log: UsageLog) -> None:
    """Budget panel and session stats."""
    s = st.session_state
    st.sidebar.header("This Session")

    remaining = max(cfg.app.session_query_cap - s.memo_count, 0)
    st.sidebar.metric("Memos run", s.memo_count, f"{remaining} remaining")

    st.sidebar.metric("Total cost (this session)", f"${float(s.total_cost):.5f}")

    st.sidebar.divider()
    st.sidebar.caption(
        f"Demo budget: \\${app_log.spend_today_utc():.3f} of "
        f"\\${cfg.app.daily_spend_cap_usd:.2f} used today (UTC), "
        f"\\${app_log.total_spend():.3f} of "
        f"\\${cfg.app.total_spend_cap_usd:.2f} overall."
    )
    st.sidebar.caption(
        "The all-Ultra estimate prices the same token usage at Ultra rates. "
        "Ultra's own output would differ, so treat it as directional."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="NemoRoute · Research Memo Agent",
        page_icon="📋",
        layout="wide",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    # --- Load resources (fail fast on missing keys) ---
    try:
        cfg, app_log, client, search = resources()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    # --- Header ---
    st.title("Nemotron Research Memo Agent")
    st.markdown(
        "Ask any business research question and get a sourced consulting memo in seconds. "
        "NemoRoute intelligently routes each step of the pipeline — planning, extraction, and synthesis — "
        "to the cheapest capable Nemotron model, then compiles findings into a polished, "
        "citation-backed markdown memo."
    )

    # --- Example buttons ---
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, use_container_width=True, key=f"ex_{example[:20]}"):
            st.session_state.last_query = example

    # --- Query input ---
    query = st.text_area(
        "Your question",
        key="last_query",
        height=100,
        max_chars=cfg.app.max_query_chars,
        placeholder="e.g. What are the key drivers of the US EV market in 2025?",
    )

    # --- Run button ---
    if st.button("Research & Write Memo", type="primary"):
        run_memo_flow(cfg, app_log, client, search, query)

    # Keep the last memo on screen across reruns (typing the next question, opening the sidebar).
    if st.session_state.last_result is not None:
        display_memo_result(st.session_state.last_result, cfg)

    # --- Sidebar ---
    render_sidebar(cfg, app_log)

    # --- Footer ---
    st.divider()
    st.caption(
        "Powered by [Nemotron 3](https://build.nvidia.com/models) (Nano · Super · Ultra) via "
        "[Nebius Token Factory](https://nebius.com) · Web search by [Tavily](https://tavily.com) · "
        "Built for the Nebius + NVIDIA Global AI Hackathon 2026"
    )


def run_memo_flow(
    cfg: Config,
    app_log: UsageLog,
    client: TokenFactoryClient,
    search: TavilySearchClient,
    query: str,
) -> None:
    """Validate, guard, run the memo pipeline, and display the result."""
    # --- Guard clause: early exit on empty input ---
    if not query or not query.strip():
        st.warning("Enter a question first.")
        return

    # --- Budget guard ---
    decision = guard(cfg, app_log, query, st.session_state.memo_count)
    if not decision.allowed:
        st.warning(decision.message)
        return

    # --- Run pipeline ---
    try:
        with st.status("Researching & writing memo…", expanded=True) as status:
            result = run_memo(
                question=query.strip(),
                strategy="routed",
                cfg=cfg,
                client=client,
                search=search,
            )
            # Populate status with step summary
            for step in result.steps:
                st.write(f"✓ {step.name} ({step.model}) — {step.latency_s:.1f}s")
            status.update(label="Memo complete", state="complete")
    except Exception as exc:
        st.error(f"Pipeline failed ({type(exc).__name__}): {exc}")
        return

    # --- Update session state ---
    s = st.session_state
    s.memo_count += 1
    s.total_cost += result.total_cost_usd
    s.last_result = result


if __name__ == "__main__":
    main()