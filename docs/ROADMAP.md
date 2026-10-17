# Roadmap: Nemotron Research Memo Agent

## Context
Hackathon: Nebius x NVIDIA Global AI Hackathon, Best Apps and Agents track.
- Submissions close Thu Oct 30, 10:00 AM PT. **Our internal deadline is Wed Oct 29.**
- Required for submission:
  - a hosted demo URL, live through Dec 15
  - a YouTube video under 3 minutes
  - a public MIT repo with a README
  - a Devpost description, plus feedback on the Nebius/NVIDIA tools

Decisions already made:
- **Product:** a research memo agent for junior consultants and analysts. It takes a business question, does live web research (Tavily), and writes a cited memo.
- **Models:** Nemotron 3 Nano / Super / Ultra on Token Factory, for every runtime call.
- **Routing:** the existing Nano router picks the tier for the memo's analysis and synthesis steps. Mechanical steps always run on Nano.
- **Benchmark:** three levels.
  - (a) Routing accuracy on single questions, using the existing pipeline.
  - (b) Memo quality: routed vs all-Ultra vs all-Nano, graded by Ultra.
  - (c) A public anchor: an MMLU-Pro business and economics sample scored by exact match, so no judge is involved. Already built.
- **Frontier comparison:** Nemotron Ultra for now. A Claude Opus reference model and second judge are already built (`run_reference.py`) and stay dormant until we have a paid `ANTHROPIC_API_KEY`. They only change the evaluation, never the app.
- **Hosting:** Streamlit Community Cloud, with a dedicated low-limit Token Factory key.
- **Dropped:** fine-tuning. Token Factory can't fine-tune Nemotron, and a self-hosted model can't be kept live cheaply through judging.

### What's already on `main` (as of Oct 16)
| Area | Status | Use it in |
|---|---|---|
| Phase 1 pipeline: generate → answer → judge → label → split → evaluate, plus the Streamlit Q&A app | Built, not yet run | A1–A3, B3 starting point |
| Per-model API keys, providers and billing tags; the budget guard counts only Token Factory spend | Built | Everything |
| `run_public_bench.py`: MMLU-Pro business/economics, 300 questions, exact match | Built, not yet run | **A3b** |
| `run_reference.py` + `anthropic_backend.py`: Opus reference answers and a second judge | Built, dormant | **A7** (optional) |
| `judge_answer(..., role=...)`: any role can act as judge | Built | A4 (memo judge) |
| Fine-tuned router (`training/`, `export_sft.py`, `finetuned` router, app router selector) | **Removed** in this PR, recoverable from git history (commit `ec388d4`) | none |

Two people:
- **Person A, Data & Evaluation:** owns the dataset, router evaluation and memo benchmark.
- **Person B, Agent & App:** owns Tavily, the memo pipeline, the UI and deployment.

Decide who is A and who is B. Each chunk below has a goal, an exit check anyone can verify, and a prompt to paste into Claude Code or opencode.

## The A↔B contract (agree on day 1, don't change without telling each other)
Person B builds this. Person A's memo benchmark calls it.

```python
# src/nemoroute/agent/memo.py
@dataclass
class Source:      url: str; title: str; snippet: str
@dataclass
class StepTrace:   name: str; tier: int; model: str; input_tokens: int; output_tokens: int; cost_usd: float; latency_s: float
@dataclass
class MemoResult:
    question: str; strategy: str           # "routed" | "all_ultra" | "all_nano"
    memo_md: str                           # cites sources as [1], [2] ...
    sources: list[Source]
    steps: list[StepTrace]
    total_cost_usd: float; total_latency_s: float
    route_tier: int | None; route_reasoning: str | None

def run_memo(question: str, strategy: str, cfg: Config, client: ChatClient,
             search: SearchClient, router: Router | None = None) -> MemoResult: ...
```

Rules:
- Search results are **cached per question** (`data/search_cache/`), so all three strategies see identical sources. That keeps the comparison fair and saves Tavily credits.
- Searches per memo are capped (config: `agent.max_searches`, default 4).

## Budget ($100 of Token Factory credits)
| Chunk | Cap (`--max-cost`) |
|---|---|
| A1 calibration | $8 |
| A2 full labeling | $35 |
| A3b MMLU-Pro public benchmark | $3 |
| A4 memo benchmark | $12 |
| B dev/smoke testing | $5 |
| Demo key (separate, Oct 29 – Dec 15) | $15 |
| Reserve | $22 |

Every paid script prints lifetime spend (from `logs/calls.jsonl`) and asks for confirmation before running. Check that number before each run. The log is per machine, so post your spend in the team chat after each chunk.

## Timeline at a glance
| Dates | Person A (Data & Eval) | Person B (Agent & App) |
|---|---|---|
| Oct 16–17 | **0** Setup (both) · **A1** Calibration + lock thresholds | **0** Setup · **B1** Tavily search client |
| Oct 18–21 | **A2** Full dataset + split | **B2** Memo agent pipeline |
| Oct 21–22 | **A3** Router evaluation · **A3b** MMLU-Pro | **B2** finish · CLI |
| Oct 22–25 | **A4** Memo benchmark (needs B2) | **B3** Streamlit memo UI |
| Oct 25–26 | **A5** Human agreement + results page data | **B4** Deploy to Streamlit Cloud |
| Oct 26–27 | **A6** README results + charts | **B5** Results tab + polish |
| Oct 28 | **J1** Video + Devpost write-up (both) | |
| Oct 29 | **J2** Final check + submit (both) | |

---

## Chunk 0: Setup (both, Oct 16, ~1 hr)
**Goal:** both machines can run tests and call Token Factory, and keys are sorted.

**Exit check:**
- `pytest` passes.
- `python scripts/probe_api.py` reports Nano, Super and Ultra OK.
- Three keys exist and are kept out of git:
  - a dev Token Factory key
  - a *separate* demo key with a low limit
  - a Tavily key (free tier or Builder Program)
- The A↔B contract above is pasted into an issue or a shared doc.

No prompt needed. Follow the README Setup section, plus `NEBIUS_API_KEY` and `TAVILY_API_KEY` in `.env`.

---

## Person A: Data & Evaluation

### A1. Calibration run and threshold lock (Oct 16–17)
**Goal:** a 60-query calibration set, labeled by the judge and checked by a human, with locked pass thresholds.

**Exit check:**
- `data/calibration/` has 60 queries with three-tier answers and judgments.
- You personally labeled the human sample.
- `config.yaml` → `labeling.locked` is set.
- Spend for this chunk is ≤ $8.

**Prompt:**
> Read README.md, config.yaml and scripts/ in this repo. I'm running the calibration stage of the pipeline. Walk me through it one command at a time, in this order:
> 1. `generate_queries.py --dataset calibration --limit 20`, then the rest with `--full`
> 2. `label_queries.py --dataset calibration` (dry run first)
> 3. `sample_for_human.py --target calibration`
> 4. `tune_thresholds.py`
> 5. `tune_thresholds.py --lock <values>`
>
> Before each paid command, show me the cost estimate. Use `--max-cost 8` cumulatively. After labeling, show me the label distribution, and tell me if any tier has more than 60% of queries (`collapse_warn_frac`). Help me read the tuning grid and recommend thresholds that balance agreement with my human labels against a sensible tier mix. Don't change any code unless a command fails. If one does, diagnose it and propose a minimal fix with a test.

### A2. Full dataset and frozen split (Oct 18–21)
**Goal:** ~600 labeled queries, a stratified 80/20 split, and a frozen test set.

**Exit check:**
- `label_queries.py --dataset full --full` completes.
- No tier has more than 60% of queries.
- The unresolved rate is reported.
- `split_dataset.py` passes its calibration-overlap check.
- `data/splits/` exists.
- Spend for this chunk is ≤ $35.

**Prompt:**
> Read README.md and config.yaml. Thresholds are locked. Run the full-dataset stage:
> 1. `generate_queries.py --dataset full --full`
> 2. `label_queries.py --dataset full --limit 20` (check class balance)
> 3. `label_queries.py --dataset full --full --max-cost 35`
> 4. `split_dataset.py`
>
> After labeling, report the label distribution, the label-vs-intended-tier confusion table, and the unresolved rate. If tier 1 or tier 3 has under 15% of queries, stop and propose a generation tweak (subdomains, personas, or prompt wording in `src/nemoroute/prompts/generate.txt`) before spending more. If a run is interrupted, resume it. The pipeline skips completed work.

### A3. Router evaluation (Oct 21–22)
**Goal:** a routing-level results table with confidence intervals.

**Exit check:**
- `results/results.md` and the charts exist.
- They cover: router accuracy, under-routing rate, cost per query and pass rate versus all-Nano, all-Ultra and oracle.
- Spend is about $1, mostly routing calls.

**Prompt:**
> Run `python scripts/evaluate.py --full` and summarize `results/results.md` for me in five bullets: router accuracy, under-routing rate, cost vs all-Ultra (% saved), pass rate vs all-Ultra (with CIs), and anything suspicious. If under-routing is above 15%, suggest concrete edits to `src/nemoroute/prompts/router.txt`. Don't apply them. List them so I can decide, and remember that changing the router prompt after seeing test results must be disclosed in results.md.

### A3b. Public benchmark: MMLU-Pro (Oct 22)
**Goal:** a judge-free, public result showing what routing costs in accuracy and saves in money.

**Exit check:**
- `results/public_bench/results_zero_shot_nano.md` exists, with exact-match accuracy, cost and CIs for all-Nano, all-Ultra, the router and the oracle.
- The 300-question sample is frozen in `data/public/mmlu_pro/items.jsonl`.
- Spend is ≤ $3.

**Prompt:**
> Read `scripts/run_public_bench.py` and `src/nemoroute/benchmarks/mmlu_pro.py`. Run `python scripts/run_public_bench.py --limit 20` first and show me the cost estimate and the answer-parse rate (how many answers had no extractable letter). If more than 5% fail to parse, propose a fix to `src/nemoroute/prompts/mcq.txt` or `extract_choice` before the full run. Then run `--full --max-cost 3` and summarize the results table in three bullets: router accuracy vs all-Ultra accuracy, % cost saved, and how close the router is to the oracle.

### A4. Memo benchmark (Oct 22–25; needs B2's `run_memo`)
**Goal:** a frozen 40-prompt memo benchmark comparing routed, all-Ultra and all-Nano memos, with CIs.

**Exit check:**
- `data/memo_bench/prompts.jsonl` has 40 prompts: 25 generated and 15 written by us, frozen before any memo runs.
- `scripts/evaluate_memos.py --full` writes `results/memo_results.md`, with mean score, pass rate, cost and latency per strategy (bootstrap CIs).
- Tests pass using fakes.
- Spend is ≤ $12.

**Prompt:**
> Read README.md, `src/nemoroute/evaluation/` (metrics.py, system.py, report.py), `src/nemoroute/labeling/judge.py`, `src/nemoroute/prompts/judge.txt`, and `src/nemoroute/agent/memo.py` (the `run_memo` contract). Build a memo-level benchmark, reusing the existing code wherever possible:
> 0. Reuse `judge_answer(..., role=...)` in `labeling/judge.py` (it already supports any judge role) and the strategy/summary helpers in `evaluation/system.py`.
> 1. `scripts/build_memo_bench.py`: generate 25 memo-style research questions with the generator role (diverse across the `generation.subdomains` in config), merge with my hand-written `data/memo_bench/handwritten.jsonl`, and write a frozen `data/memo_bench/prompts.jsonl`. Refuse to overwrite it once it exists.
> 2. `src/nemoroute/prompts/memo_judge.txt`: the existing rubric plus a fifth criterion, `grounding` (claims supported by the cited sources; citations real and relevant). The judge sees the question, the memo and the source snippets, but not the strategy or model.
> 3. `scripts/evaluate_memos.py`: for each prompt × strategy (routed, all_ultra, all_nano), call `run_memo`, judge it blind, and write append-only JSONL with resume support, like the labeling pipeline. Use the same cost-estimate-then-confirm flow, plus `--limit` and `--max-cost`.
> 4. Summarize with `evaluation/metrics.bootstrap_ci` into `results/memo_results.md`.
>
> Add tests under `tests/` using `tests/fakes.py`. No real API calls in tests. Run `pytest` and show me the result. Then do a `--limit 3` live dry run and show me the costs before the full run.

### A5. Human agreement on memos (Oct 25–26)
**Goal:** evidence that the Ultra judge agrees with a human on memo quality.

**Exit check:** 30 memos have human pass/fail labels, and percent agreement plus Cohen's kappa appear in `results/memo_results.md`.

**Prompt:**
> Extend `src/nemoroute/human_agreement.py` and `scripts/sample_for_human.py` / `score_human_agreement.py` so they also work on the memo benchmark outputs (`--target memo`). Sample 30 memos stratified by strategy. Export a CSV with blank `human_pass` columns, and hide the strategy and model. Add tests with fakes, then run `pytest`.

### A6. README results (Oct 26–27)
**Goal:** the README "Results" section filled in with real numbers and charts.

**Exit check:**
- The README shows the headline claim ("routed memos score X vs all-Ultra Y at Z% of the cost", with CIs), the routing table, two charts, and the limitations (self-grading, judge-dependent labels).
- Every number matches `results/*.md`.

**Prompt:**
> Using `results/results.md` and `results/memo_results.md`, rewrite the README Results section: a one-sentence headline claim with CIs, a compact table per benchmark, two charts (cost vs quality per strategy; tier mix), and an honest Limitations section. Don't invent numbers. Every figure must come from those files. Also update the README intro so it describes the research memo agent rather than just the router.

### A7. Optional: frontier reference (only if we get a paid Anthropic API key)
**Goal:** a "vs frontier" row and a second-judge bias check.

**Exit check:**
- `results/results.md` has a `reference` row.
- `results/judge_reference_judge/results.md` exists.
- The Anthropic spend estimate was approved before running. Roughly $30–40 for the full second-judge pass, so consider `--limit`.

**Prompt:**
> Read `scripts/run_reference.py`, `src/nemoroute/reference.py` and the README section "Optional: frontier reference and public benchmark". With `ANTHROPIC_API_KEY` set, run `run_reference.py --limit 20`, show me both the Token Factory and Anthropic cost estimates, then run the full commands in the README order. Summarize how far the routed system is from the reference model, and whether the reference judge's scores agree with Ultra's (mean difference per strategy). If budget is tight, run the second judge on a stratified 60-item subset instead of the full test set.

---

## Person B: Agent & App

### B1. Tavily search client (Oct 16–17)
**Goal:** a cached, tested web-search wrapper.

**Exit check:**
- `pytest` passes, with new tests using a fake.
- `python scripts/smoke_search.py "US EV charger market size"` prints ≤ 5 results with URL, title and snippet.
- A second run is served from `data/search_cache/`, with no API call.

**Prompt:**
> Read README.md, `src/nemoroute/client.py`, `src/nemoroute/config.py` and `tests/fakes.py` to learn the patterns used here. Add a Tavily web-search client:
> - `src/nemoroute/search.py` with a `SearchClient` protocol, a `TavilySearchClient` using `TAVILY_API_KEY` (via httpx or the `tavily-python` package; add it to requirements.txt), and a disk cache keyed by the normalized query under `data/search_cache/`.
> - Retries like `client.py`.
> - Config section `agent:` with `max_searches: 4`, `results_per_search: 5`, `search_depth: basic`.
> - Return `Source(url, title, snippet)` objects.
> - A `FakeSearchClient` in `tests/fakes.py`, tests for caching and parsing, and `scripts/smoke_search.py`.
>
> Keep the code style consistent with the repo. Run `pytest`.

### B2. Memo agent pipeline (Oct 18–22)
**Goal:** `run_memo` implementing the A↔B contract, plus a CLI.

**Exit check:**
- `python scripts/run_memo.py "Should a regional grocer launch a private-label meal kit?" --strategy routed` prints a memo with ≥ 3 numbered citations that map to real URLs in `sources`, and a step trace whose costs sum to `total_cost_usd`.
- The same command with `--strategy all_ultra` and `--strategy all_nano` works.
- `pytest` passes.
- One routed memo costs under $0.02.

**Prompt:**
> Read README.md, `src/nemoroute/client.py`, `routers/`, `labeling/answer.py`, `prompts/`, `search.py`, and `tests/fakes.py`. Implement `src/nemoroute/agent/memo.py` exactly to this contract: [paste the A↔B contract]. Pipeline steps, each logged as a StepTrace through the existing `TokenFactoryClient` (so `usage_log` and pricing work):
> 1. **route:** existing router classifies the question → tier T. Skipped for `all_*` strategies.
> 2. **plan:** Nano writes up to `agent.max_searches` search queries (JSON output).
> 3. **search:** `SearchClient`, cached.
> 4. **extract:** Nano condenses the sources into numbered facts with source indices.
> 5. **synthesize:** tier T (routed), Ultra (all_ultra) or Nano (all_nano) writes the memo with sections Bottom line / Key findings / Analysis / Risks & unknowns / Next steps, citing [n].
>
> For all_ultra and all_nano, *every* step uses that model. Put the prompts in `src/nemoroute/prompts/` as .txt files, like the existing ones. Add `scripts/run_memo.py`, and tests using `FakeClient` and `FakeSearchClient` covering the step order, cost summing, citation-index validation and strategy model selection. Run `pytest`, then a single live routed run, and show me the memo and the trace.

### B3. Streamlit memo UI (Oct 22–25)
**Goal:** the demo app is the memo agent, designed for a junior analyst.

**Exit check:**
- `streamlit run app/streamlit_app.py` locally gives:
  - a question box and 3 example questions
  - live step-by-step progress (route → plan → search → extract → write)
  - the rendered memo with clickable sources
  - a cost panel: per-step model/tier/cost, the total, and an "all-Ultra estimate" with % saved
  - a "Copy as Markdown" / download button
- `app_guard` caps still apply. Two people test it, and nothing breaks on a bad question or an API error.

**Prompt:**
> (Note: the app on `main` is the original single-router Q&A app. The fine-tuned router selector was removed.) Read `app/streamlit_app.py`, `src/nemoroute/app_guard.py` and `src/nemoroute/agent/memo.py`. Rebuild the Streamlit app around `run_memo` with `strategy="routed"`, keeping the existing secret loading, `guard()` caps and session cost sidebar. Use `st.status` to show each step live, a cost table per step (tier, model, tokens, cost, latency), and an all-Ultra estimate that prices each step's tokens at Ultra rates (state the caveat in the UI, as the current app does). Render the memo with a numbered source list of clickable links, and add a download-as-.md button. Write the copy for a junior consulting analyst. Keep the app working when `TAVILY_API_KEY` is missing (show a clear message). Have it pass `agent.max_searches` and `app.answer_max_tokens` to bound cost. Run it locally and fix any errors.

### B4. Deploy (Oct 25–26)
**Goal:** a public URL that stays up and can't drain credits.

**Exit check:**
- The Streamlit Cloud URL works in an incognito window on a phone.
- The secrets are the demo Token Factory key (low limit) and the Tavily key.
- A query after the session cap is refused politely.
- The URL is in the README.

**Prompt:**
> Prepare this repo for Streamlit Community Cloud: make sure `requirements.txt` installs the package (`-e .` works there, or switch to `.`), the app's entry point is `app/streamlit_app.py`, secrets are read from `st.secrets` (`NEBIUS_API_KEY`, `TAVILY_API_KEY`), and writable paths (logs, search cache) fall back to a temp dir if the repo directory is read-only. Add a short "Deploying" section to the README covering the dedicated low-limit key. Don't commit any secrets.

### B5. Results tab and polish (Oct 26–27)
**Goal:** judges can see the evidence inside the app.

**Exit check:** an "Evidence" page shows A's headline numbers and charts, read from `results/` files committed to the repo. The app has a title, short explainer and footer crediting Nemotron + Token Factory + Tavily.

**Prompt:**
> Add a Streamlit multipage "Evidence" page that reads `results/memo_results.md`, `results/results.md` and the chart PNGs from `results/`, and presents the headline claim, both tables and the charts. Polish the main page: a clear title ("Nemotron Research Memo Agent"), a two-line explainer, and a footer naming the Nemotron 3 models, Nebius Token Factory and Tavily. Don't hard-code any numbers.

---

## Joint chunks

### J1. Video and Devpost write-up (Oct 28)
**Goal:** a video under 3 minutes and a complete Devpost draft.

**Exit check:**
- The YouTube video is public, under 3:00, and shows a live memo run + the cost panel + the Evidence page.
- The Devpost draft has every required field, including platform feedback and the "built during the submission period" note.

**Prompt (either person):**
> Using README.md and `results/`, draft (1) a 2:45 demo-video script with timestamps: problem (15s), live memo demo (75s), how routing works (30s), evidence (30s), close (15s); and (2) a Devpost description: inspiration, what it does, how we built it (Nemotron 3 Nano/Super/Ultra on Token Factory, Tavily), challenges, accomplishments, what we learned, what's next, and a candid feedback section on Token Factory and Nemotron. Note that the repo was created on Oct 14, inside the submission period.

### J2. Final check and submit (Oct 29)
**Exit check, all ticked:**
- [ ] Demo URL works logged out
- [ ] Repo is public and MIT, README has setup + results + demo URL
- [ ] `pytest` passes on a fresh clone
- [ ] No secrets in git history (`git log -p | grep -i "api_key"` is clean)
- [ ] Video link public
- [ ] Track = Best Apps and Agents
- [ ] Submitted before Oct 30, 10:00 AM PT

## Risks and fallbacks
- **B2 slips:** A4 starts by evaluating a stub `run_memo` with fakes, so the evaluation code is ready the moment B2 lands.
- **Labels collapse onto one tier:** A2 stops and adjusts generation before spending more.
- **Tavily credits run out:** the search cache covers the benchmark, and the demo shows a friendly message.
- **Ultra is slow in the demo:** the routed strategy uses Ultra only for synthesis, and B3 shows live progress so the wait is visible.
- **Budget overrun:** every paid script has `--max-cost`. Stop at 80% of the cap and re-plan.
