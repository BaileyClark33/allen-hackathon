"""Evidence page — benchmark results and charts for NemoRoute.

Displays routing benchmark results, public MMLU-Pro benchmark, and associated
chart images from the results/ directory.

Guard clauses ensure graceful handling of missing files.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
PUBLIC_BENCH_DIR = RESULTS_DIR / "public_bench"
CHARTS_DIR = RESULTS_DIR / "charts"
PUBLIC_CHARTS_DIR = PUBLIC_BENCH_DIR / "charts"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Evidence · NemoRoute",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS — share the main app's dark theme
# ---------------------------------------------------------------------------

CSS = """
<style>
  html, body, .stApp {
    background: #0f0f1a;
    color: #e0e0e0;
  }
  .stApp header { background: transparent; }
  h1, h2, h3, h4, h5, h6 { color: #f0c060; }

  .evidence-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin: 1rem 0 2rem 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }
  .evidence-card h1, .evidence-card h2, .evidence-card h3 {
    color: #f0c060;
  }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("Benchmark Evidence")

# ---------------------------------------------------------------------------
# Routing benchmark
# ---------------------------------------------------------------------------

st.header("Routing Benchmark (Held-Out Test Set)")

routing_results = RESULTS_DIR / "results.md"
if not routing_results.exists():
    st.warning("Routing benchmark results not found. Run the benchmark first.")
else:
    with st.container():
        st.markdown(
            f'<div class="evidence-card">{routing_results.read_text()}</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# MMLU-Pro public benchmark
# ---------------------------------------------------------------------------

st.header("MMLU-Pro Public Benchmark")

pub_results = PUBLIC_BENCH_DIR / "results_zero_shot_nano.md"
if not pub_results.exists():
    st.info("Public benchmark results not yet available.")
else:
    with st.container():
        st.markdown(
            f'<div class="evidence-card">{pub_results.read_text()}</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

st.header("Charts")

col1, col2 = st.columns(2)

with col1:
    chart_path = CHARTS_DIR / "cost_vs_quality.png"
    if chart_path.exists():
        st.image(str(chart_path), caption="Cost vs Quality")
    else:
        st.caption("Cost vs Quality chart not found.")

with col2:
    chart_path = CHARTS_DIR / "latency.png"
    if chart_path.exists():
        st.image(str(chart_path), caption="Latency by Tier")
    else:
        st.caption("Latency chart not found.")

confusion_path = CHARTS_DIR / "confusion_matrix.png"
if confusion_path.exists():
    st.image(str(confusion_path), caption="Routing Confusion Matrix")
else:
    st.caption("Confusion matrix chart not found.")

pub_chart_path = PUBLIC_CHARTS_DIR / "cost_vs_accuracy_zero_shot_nano.png"
if pub_chart_path.exists():
    st.image(
        str(pub_chart_path),
        caption="MMLU-Pro: Zero-Shot Nano (Public Benchmark)",
    )
else:
    st.caption("Public benchmark chart not found.")

# ---------------------------------------------------------------------------
# Memo benchmark
# ---------------------------------------------------------------------------

st.divider()

memo_results = RESULTS_DIR / "memo_results.md"
if memo_results.exists():
    st.header("Memo Benchmark")
    with st.container(border=True):
        st.markdown(memo_results.read_text())
else:
    st.info("Memo benchmark results coming soon.")